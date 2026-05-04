# mypy: allow-untyped-defs
from __future__ import annotations

import atexit
import functools
import json
import logging
import multiprocessing
import os
import queue as _queue
import io
import re
import selectors
import shutil
import socket
import sys
import tempfile
import threading
from concurrent.futures import (
    Future,
    ThreadPoolExecutor,
    TimeoutError as FuturesTimeoutError,
)
from concurrent.futures.process import BrokenProcessPool
from functools import partial
from time import time, time_ns
from typing import Any, cast, NamedTuple, TYPE_CHECKING

import torch
from torch._dynamo.device_interface import get_registered_device_interfaces
from torch._dynamo.utils import (
    counters,
    dynamo_timed,
    get_metrics_context,
    set_feature_use,
)
from torch._inductor import config
from torch._inductor.codecache import (
    _load_triton_kernel_from_source,
    code_hash,
    CodeCacheFuture,
    CppCodeCache,
    CppPythonBindingsCodeCache,
    CUDACodeCache,
    HalideCodeCache,
    LambdaFuture,
    ROCmCodeCache,
    StaticAutotunerFuture,
    torch_key,
    XPUCodeCache,
)
from torch._inductor.compile_worker.subproc_pool import (
    AnyPool,
    SubprocException,
    SubprocPool,
)
from torch._inductor.compile_worker.tracked_process_pool import (
    TrackedProcessPoolExecutor,
)
from torch._inductor.compile_worker.utils import _async_compile_initializer
from torch._inductor.runtime.compile_tasks import (
    _set_triton_libdevice_path,
    _set_triton_ptxas_path,
    _StreamingUnpickler,
    _worker_compile_triton,
)
from torch._inductor.utils import clear_on_fresh_cache
from torch._inductor.virtualized import V
from torch._utils_internal import log_triton_builds
from torch.hub import _Faketqdm, tqdm
from torch.utils._ordered_set import OrderedSet
from torch.utils._triton import has_triton_package


if TYPE_CHECKING:
    from collections.abc import Callable

    from torch._inductor.runtime.hints import HalideMeta
    from torch._inductor.runtime.triton_heuristics import CachingAutotuner

# timing metrics for time spent in the compilation
_cumulative_compile_time = 0.0
_t0: float | None = None

kernel_code_log = torch._logging.getArtifactLogger(__name__, "kernel_code")

log = logging.getLogger(__name__)

_triton_kernel_metrics: dict[str, dict[str, Any]] | None = None

size_hints_regex = re.compile(
    r"size_hints=(\{.*?\})",
)


class StreamingCompileHandle(NamedTuple):
    """Stashed on a CachingAutotuner by ``AsyncCompile.triton`` when
    ``config.streaming_compile`` is enabled.

    Three-stage pipeline (per kernel):

      1. ``router`` thread routes worker messages from a shared AF_UNIX
         listener into ``queue`` (one per kernel).
      2. ``_bg_drain_kernel`` daemon (per kernel, started when the
         streaming handle is stashed) pulls from ``queue``, does
         ``bundle + make_launcher``, and pushes each newly-made launcher
         onto ``launcher_q``. Runs in parallel with inductor codegen of
         *other* kernels.
      3. ``StreamingCompilePlugin.pre_dispatch`` (per kernel, fires at
         first ``run()``) pulls from ``launcher_q`` and benchmarks each
         launcher with the real input args. Bench[i] thus overlaps with
         the bg drain's ``make_launcher_{i+1}`` (which itself overlaps
         with the worker's compile of ``c_{i+2}``).

    Fields:
      ``queue``: worker → router results queue.
      ``sentinel``: identity-compared end-of-stream marker for ``queue``.
      ``num_configs``: number of base configs (pre-rblock-scaling); used
        by the plugin to short-circuit single-config kernels (no bench).
      ``static_triton_bundle_key``: passed to
        ``TritonBundler.put_static_autotuner`` after launchers are made;
        bg drain handles this since ``precompile`` is skipped.
      ``drain_complete``: ``threading.Event`` set when the bg drain has
        produced its last launcher and finished cleanup. The plugin
        consults this for single-config kernels (no launcher_q needed)
        and as a fallback signal.
      ``launcher_q``: bg drain → plugin per-launcher pipe. Plugin
        consumes blocking-style; bg drain pushes ``launcher_sentinel``
        (or the fallback launcher) when all launchers are made.
      ``launcher_sentinel``: identity-compared end-of-stream marker for
        ``launcher_q``.

    Timing-attribution fields (the plugin reads these at
    ``pre_dispatch`` exit and emits a single combined ``compile_time_us``
    metric so the per-kernel breakdown matches non-streaming semantics):

      ``kernel_name``: source-derived name used as the metric key.
      ``worker_done_event``: set by ``_on_task_done`` when the worker
        process-pool task finishes; the plugin briefly waits on this
        before reading ``worker_elapsed_us``.
      ``worker_elapsed_us``: single-element list[int|None] holder
        populated by ``_on_task_done`` with the worker's reported wall
        time; ``None`` if the worker errored.
      ``bg_drain_wait_ns``: single-element list[int] holder accumulated
        by ``_bg_drain_kernel`` — total time the bg-drain thread spent
        blocked on ``queue.get()`` waiting for the worker. The plugin
        adds this to the per-kernel compile-time attribution.
    """

    queue: _queue.Queue[object]
    sentinel: object
    num_configs: int
    static_triton_bundle_key: str | None
    drain_complete: threading.Event
    launcher_q: _queue.Queue[object]
    launcher_sentinel: object
    kernel_name: str
    worker_done_event: threading.Event
    worker_elapsed_us: list[int | None]
    bg_drain_wait_ns: list[int]


class _KernelLoadFailure:
    """Poison item pushed onto ``kernel_slot`` when the worker dies
    before sending the kernel.

    The parent's ``get_result`` sees this item, raises ``self.exc``,
    and the failure propagates as if ``task.result()`` had raised
    directly.
    """

    def __init__(self, exc: BaseException) -> None:
        self.exc = exc


class _BufferedFramedReader:
    """Non-blocking reader matching ``multiprocessing.connection``'s wire
    framing: 4-byte big-endian signed length prefix, then payload bytes.
    A length of ``-1`` is the "big payload" sentinel followed by an 8-byte
    big-endian unsigned length (for payloads >= 2GB).

    Used by ``_StreamingRouter`` to drain a non-blocking AF_UNIX socket and
    parse out complete framed messages without ever blocking the router
    thread on a partial read.
    """

    def __init__(self, sock: socket.socket) -> None:
        self._sock = sock
        sock.setblocking(False)
        self._buf = bytearray()

    def read_all_available(self) -> tuple[list[bytes], bool]:
        """Drain everything currently readable from the socket. Returns
        ``(complete_payloads, eof)`` — ``eof`` is ``True`` if the peer
        closed the connection or sent a corrupt frame header (treat as
        end-of-stream so the router closes the connection rather than
        looping on bad framing)."""
        eof = False
        while True:
            try:
                chunk = self._sock.recv(65536)
            except BlockingIOError:
                break
            except (ConnectionError, OSError):
                eof = True
                break
            if not chunk:
                eof = True
                break
            self._buf.extend(chunk)
        msgs: list[bytes] = []
        while True:
            try:
                msg = self._try_pop_message()
            except ValueError:
                # Corrupt frame header — give up on this connection.
                eof = True
                break
            if msg is None:
                break
            msgs.append(msg)
        return msgs, eof

    def _try_pop_message(self) -> bytes | None:
        if len(self._buf) < 4:
            return None
        header_n = int.from_bytes(self._buf[:4], "big", signed=True)
        if header_n == -1:
            if len(self._buf) < 12:
                return None
            payload_len = int.from_bytes(self._buf[4:12], "big", signed=False)
            header_bytes = 12
        elif header_n < 0:
            # Any other negative value is a malformed frame; ``payload_len
            # = header_n`` would make ``total < header_bytes`` and the
            # caller's parse loop would spin forever returning ``b""``.
            raise ValueError(f"invalid streaming frame header: {header_n}")
        else:
            payload_len = header_n
            header_bytes = 4
        total = header_bytes + payload_len
        if len(self._buf) < total:
            return None
        payload = bytes(self._buf[header_bytes:total])
        del self._buf[:total]
        return payload


class _StreamingRouter:
    """Single-listener / single-thread router that demuxes messages from many
    worker connections to per-kernel queues.

    The previous design created a fresh ``Listener`` + reader thread + tmpdir
    per kernel; for kernel-heavy graphs (densenet121: 411 kernels) this added
    ~2-3 s of E2E compile-time overhead. With the shared router that cost
    becomes O(1) per process: one listener, one thread, one tmpdir.

    Wire protocol on each worker connection:
      1. ``kernel_id`` (small int) — first message; the router uses this to
         look up which per-kernel queues to demux subsequent messages to.
      2. ``kernel`` (CachingAutotuner) — second message.
      3. Zero or more ``CompileResult`` / ``_CompileFailureMarker`` messages
         until the worker closes the connection.
    """

    def __init__(self) -> None:
        self._sock_dir = tempfile.mkdtemp(prefix="inductor_stream_")
        self.sock_addr = os.path.join(self._sock_dir, "stream.sock")
        self._listen_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._listen_sock.bind(self.sock_addr)
        self._listen_sock.listen(128)
        self._listen_sock.setblocking(False)

        # kernel_id -> (kernel_slot, stream_q, sentinel)
        self._registry: dict[
            int, tuple[_queue.Queue[object], _queue.Queue[object], object]
        ] = {}
        self._registry_lock = threading.Lock()
        self._next_id = 0

        self._sel = selectors.DefaultSelector()
        self._sel.register(
            self._listen_sock, selectors.EVENT_READ, data="LISTEN"
        )

        atexit.register(self._cleanup)
        threading.Thread(
            target=self._loop, name="inductor-stream-router", daemon=True
        ).start()

    def _cleanup(self) -> None:
        try:
            self._listen_sock.close()
        except OSError:
            pass
        shutil.rmtree(self._sock_dir, ignore_errors=True)

    def register(
        self, kernel_name: str
    ) -> tuple[str, _queue.Queue[object], _queue.Queue[object], object]:
        """Allocate a kernel_id + per-kernel queues. Returns
        ``(<addr>#<kid>, kernel_slot, stream_q, sentinel)`` — the worker's
        ``_stream_compile_triton`` parses the address to extract the
        kernel_id before connecting."""
        with self._registry_lock:
            self._next_id += 1
            kid = self._next_id
            ks: _queue.Queue[object] = _queue.Queue()
            sq: _queue.Queue[object] = _queue.Queue()
            sentinel: object = object()
            self._registry[kid] = (ks, sq, sentinel)
        return f"{self.sock_addr}#{kid}", ks, sq, sentinel

    def fail_kernel(self, kid: int, exc: BaseException) -> None:
        """Push a ``_KernelLoadFailure`` to a kernel's slot if it hasn't
        received its kernel yet; called by ``get_result`` when the worker's
        process-pool task finishes with an exception before the kernel
        arrived via the stream.

        Atomically pops the kernel from the registry under the lock so
        concurrent activity in ``_dispatch`` (which also looks up by
        ``kernel_id``) can no longer route a real kernel to the slot
        after we've decided to fail it. ``Queue.empty()`` is unreliable
        for cross-thread coordination, so we use the registry-pop as the
        single race-free arbiter of whether this kernel is still live.
        """
        with self._registry_lock:
            slot = self._registry.pop(kid, None)
        if slot is None:
            return
        ks, sq, sentinel = slot
        ks.put(_KernelLoadFailure(exc))
        sq.put(sentinel)

    def _loop(self) -> None:
        while True:
            try:
                events = self._sel.select(timeout=10.0)
            except OSError:
                return
            for key, _ in events:
                if key.data == "LISTEN":
                    self._on_accept()
                else:
                    self._on_readable(key.data)

    def _on_accept(self) -> None:
        try:
            conn_sock, _ = self._listen_sock.accept()
        except (BlockingIOError, OSError):
            return
        conn_sock.setblocking(False)
        state: dict[str, Any] = {
            "sock": conn_sock,
            "reader": _BufferedFramedReader(conn_sock),
            "kernel_id": None,
            "kernel_received": False,
        }
        self._sel.register(conn_sock, selectors.EVENT_READ, data=state)

    def _on_readable(self, state: dict[str, Any]) -> None:
        msgs, eof = state["reader"].read_all_available()
        for payload in msgs:
            try:
                obj = _StreamingUnpickler(io.BytesIO(payload)).load()
            except Exception as e:
                # Broad catch is intentional here: ``pickle.load()`` can
                # surface many failure modes (UnpicklingError, EOFError,
                # ImportError / ModuleNotFoundError when a referenced
                # class isn't importable on the parent, AttributeError,
                # TypeError, ValueError, ...). The router thread is a
                # process-wide singleton — letting any of these escape
                # would kill the thread and deadlock every other pending
                # ``kernel_slot.get()``. Drop just this connection and
                # keep serving the rest. Logged at WARNING so we still
                # see it.
                log.warning(
                    "streaming router: dropping connection (kid=%s) after "
                    "unpickle failure: %s",
                    state.get("kernel_id"),
                    e,
                )
                self._on_close(state)
                return
            self._dispatch(state, obj)
        if eof:
            self._on_close(state)

    def _dispatch(self, state: dict[str, Any], obj: object) -> None:
        if state["kernel_id"] is None:
            # First message on this connection is the kernel_id (an int).
            state["kernel_id"] = obj
            return
        slot = self._registry.get(state["kernel_id"])
        if slot is None:
            return
        ks, sq, _ = slot
        if not state["kernel_received"]:
            ks.put(obj)
            state["kernel_received"] = True
        else:
            sq.put(obj)

    def _on_close(self, state: dict[str, Any]) -> None:
        sock = state["sock"]
        try:
            self._sel.unregister(sock)
        except (KeyError, ValueError):
            pass
        try:
            sock.close()
        except OSError:
            pass
        kid = state["kernel_id"]
        if kid is None:
            return
        slot = self._registry.get(kid)
        if slot is None:
            return
        ks, sq, sentinel = slot
        if not state["kernel_received"]:
            ks.put(
                _KernelLoadFailure(
                    RuntimeError(
                        "worker disconnected before kernel arrived"
                    )
                )
            )
        sq.put(sentinel)
        with self._registry_lock:
            self._registry.pop(kid, None)


_GLOBAL_STREAMING_ROUTER: _StreamingRouter | None = None
_GLOBAL_STREAMING_ROUTER_LOCK = threading.Lock()


def _get_streaming_router() -> _StreamingRouter:
    """Lazy-init the process-wide streaming router."""
    global _GLOBAL_STREAMING_ROUTER
    if _GLOBAL_STREAMING_ROUTER is None:
        with _GLOBAL_STREAMING_ROUTER_LOCK:
            if _GLOBAL_STREAMING_ROUTER is None:
                _GLOBAL_STREAMING_ROUTER = _StreamingRouter()
    return _GLOBAL_STREAMING_ROUTER


def _setup_streaming_compile(
    kernel_name: str,
) -> tuple[str, "_queue.Queue[object]", "_queue.Queue[object]", object]:
    """Register a new streaming-compile kernel with the global router.

    Returns ``(socket_address, kernel_slot, stream_q, sentinel)``. The
    ``socket_address`` is in ``<unix-path>#<kernel_id>`` format; the
    worker's ``_stream_compile_triton`` parses it before connecting.
    """
    return _get_streaming_router().register(kernel_name)


def _bg_drain_kernel(kernel: Any, handle: StreamingCompileHandle) -> None:
    """Background drain for a single kernel — middle stage of the
    streaming-compile pipeline.

    Started as a daemon thread by ``AsyncCompile.triton``'s ``get_result``
    immediately after the streaming handle is stashed on the kernel — i.e.
    while inductor codegen is still running for *other* kernels. Pulls
    streamed compile results from ``handle.queue``, does
    ``bundle + make_launcher``, and pushes each newly-made launcher onto
    ``handle.launcher_q``. The plugin's ``pre_dispatch`` consumes
    ``launcher_q`` from the main thread at first dispatch and benchmarks
    each launcher; bench[i] therefore overlaps with the bg drain's
    ``make_launcher_{i+1}`` (which itself overlaps with the worker's
    compile of ``c_{i+2}``).

    Bench is intentionally NOT done here — the bench arguments
    (``*args, **kwargs``) only become available at first dispatch when
    the user's ``model(*inputs)`` call hits the kernel's ``run()``.
    """
    from torch._dynamo.device_interface import DeviceGuard
    from torch._inductor.runtime.compile_tasks import _CompileFailureMarker

    import time as _time

    last_make_launcher_exc: BaseException | None = None

    def _accept(launcher: Any, exc: BaseException | None) -> None:
        nonlocal last_make_launcher_exc
        if launcher is None:
            if exc is not None:
                last_make_launcher_exc = exc
            return
        kernel.launchers.append(launcher)
        handle.launcher_q.put(launcher)

    try:
        device_interface = kernel.get_device_interface()
        with DeviceGuard(device_interface, kernel.triton_meta["device"]):
            # Phase 1: drain the worker's stream as items arrive. Time
            # the ``queue.get()`` blocking wait — that wait represents
            # parent-side time spent waiting on the worker's compile, so
            # the plugin attributes it to per-kernel compile_time_us.
            while True:
                tq0 = _time.time_ns()
                item = handle.queue.get()
                handle.bg_drain_wait_ns[0] += _time.time_ns() - tq0
                if item is handle.sentinel:
                    break
                if isinstance(item, _CompileFailureMarker):
                    continue
                kernel.compile_results.append(item)
                kernel._bundle_compile_result(item)
                _accept(*kernel._make_launcher(item))

            # Phase 2: dynamic rblock scaling on the assembled set
            # (sequential parent-process compile + make launcher; rblock
            # candidates are typically few).
            if kernel._could_rblock_scale:
                for new_config in kernel._iter_rblock_scale_candidates():
                    result = kernel._precompile_config(new_config)
                    kernel.compile_results.append(result)
                    kernel._bundle_compile_result(result)
                    _accept(*kernel._make_launcher(result))

            # All-failed fallback (only if at least one compile result
            # made it through — otherwise the plugin will surface the
            # NoTritonConfigsError below). Push the fallback launcher to
            # launcher_q too so the plugin can take a uniform code path.
            if not kernel.launchers and kernel.compile_results:
                kernel._all_failed_fallback(last_make_launcher_exc)
                if kernel.launchers:
                    handle.launcher_q.put(kernel.launchers[-1])

        kernel.configs = None
        kernel._maybe_put_static_autotuner(handle.static_triton_bundle_key)
    except BaseException as e:
        # Stash the exception on the handle so the plugin can surface it
        # at first dispatch instead of hanging forever.
        kernel._streaming_drain_exc = e
        log.exception("background drain failed for %s", kernel.fn.__name__)
    finally:
        # Order matters: push launcher_sentinel BEFORE setting
        # drain_complete so the plugin (which may already be looping on
        # launcher_q.get()) sees the sentinel and exits its loop.
        handle.launcher_q.put(handle.launcher_sentinel)
        handle.drain_complete.set()


def _make_streaming_compile_plugin() -> Any:
    """Build the ``StreamingCompilePlugin`` lazily.

    Defined inside a function so the runtime-side imports
    (``CachingAutotunerPlugin``, ``DEFER``, etc.) only happen when
    ``config.streaming_compile`` is actually enabled — keeps this
    module's import graph clean for the default-off path.
    """
    from torch._inductor.runtime.triton_heuristics import (
        CachingAutotunerPlugin,
        DEFER,
    )

    class StreamingCompilePlugin(CachingAutotunerPlugin):
        """Last stage of the streaming-compile pipeline.

        Activated by ``config.streaming_compile``. ``AsyncCompile.triton``
        skips ``kernel.precompile()`` when streaming is on, stashes a
        ``StreamingCompileHandle`` on the kernel, and starts a per-kernel
        ``_bg_drain_kernel`` daemon that turns streamed compile results
        into launchers, pushing each onto ``handle.launcher_q``.

        At first ``run()`` call:

          1. For multi-config kernels: pull launchers from
             ``handle.launcher_q`` as the bg drain produces them, bench
             each one with the real input args, repeat until the bg drain
             pushes its sentinel. Bench[i] thus overlaps with the bg
             drain's ``make_launcher_{i+1}`` (and the worker's compile of
             ``c_{i+2}``). When all launchers are benched, pick the
             winner via ``_finalize_autotune_winner``.
          2. For single-config kernels: just wait for ``drain_complete``
             (no bench needed — there's nothing to compare against).
          3. Re-raise any exception caught by the bg drain so it surfaces
             to the user.
          4. Return ``DEFER`` so ``run()`` proceeds normally; with
             ``self.launchers == [winner]`` the standard
             ``autotune_to_one_config`` is gated out (its ``len > 1``
             check) and we go straight to dispatch.

        ``pre_compile`` is a defensive short-circuit: if anything ever
        calls ``CachingAutotuner.precompile()`` on a kernel that still has
        a streaming handle (shouldn't happen — ``async_compile`` skips
        that call), bypass the standard precompile flow so we don't try
        to drive ``_precompile_worker`` on a path that no longer expects
        to.
        """

        def pre_compile(self, autotuner: object) -> object:
            if hasattr(autotuner, "_streaming_compile_handle"):
                return None
            return DEFER

        def pre_dispatch(
            self,
            autotuner: object,
            *args: object,
            stream: object,
            **kwargs: object,
        ) -> object:
            import time as _time

            from torch._dynamo.device_interface import DeviceGuard
            from torch._inductor.runtime.triton_heuristics import (
                NoTritonConfigsError,
            )

            handle = getattr(autotuner, "_streaming_compile_handle", None)
            if handle is None:
                return DEFER
            del autotuner._streaming_compile_handle

            # Per-call timing-attribution accumulators. ``plugin_wait_ns``
            # is parent-thread time spent blocked on ``launcher_q.get()``
            # waiting for the bg drain (which itself waits on the worker
            # — so this rolls up to compile time). ``bench_ns`` is the
            # actual ``do_bench`` time and is the one piece of streaming
            # work attributed to *runtime autotuning*.
            plugin_wait_ns = 0
            bench_ns = 0

            # Single-config kernels can't be autotuned; just wait for the
            # bg drain to finish so ``self.launchers`` is final, then
            # proceed (run() will see ``len == 1`` and skip
            # ``autotune_to_one_config``).
            if handle.num_configs <= 1:
                tq0 = _time.time_ns()
                handle.drain_complete.wait()
                plugin_wait_ns += _time.time_ns() - tq0
            else:
                # Multi-config: pull launchers from the bg drain as they
                # arrive and bench each one. ``launcher_q.get()`` blocks
                # until either a new launcher is ready or the bg drain
                # pushes its sentinel.
                timings: dict[object, float] = {}
                device_interface = autotuner.get_device_interface()
                with DeviceGuard(
                    device_interface, autotuner.triton_meta["device"]
                ):
                    while True:
                        tq0 = _time.time_ns()
                        launcher = handle.launcher_q.get()
                        plugin_wait_ns += _time.time_ns() - tq0
                        if launcher is handle.launcher_sentinel:
                            break
                        tb0 = _time.time_ns()
                        timing = autotuner.bench(launcher, *args, **kwargs)
                        bench_ns += _time.time_ns() - tb0
                        timings[launcher] = timing
                        autotuner.coordesc_tuner.cache_benchmark_result(
                            launcher.config, timing
                        )
                # Pick the winner. If ``timings`` is empty (all-failed
                # fallback or some exotic state), leave
                # ``autotuner.launchers`` as the bg drain left it — the
                # standard ``run()`` flow will handle dispatch.
                if timings:
                    autotuner.autotune_time_taken_ns = bench_ns
                    autotuner._finalize_autotune_winner(timings)

            # Re-raise any exception caught by the bg drain.
            drain_exc = getattr(autotuner, "_streaming_drain_exc", None)
            if drain_exc is not None:
                del autotuner._streaming_drain_exc
                raise drain_exc

            if not autotuner.launchers and not autotuner.compile_results:
                raise NoTritonConfigsError(
                    f"Streaming compile produced 0 results for "
                    f"{autotuner.fn.__name__}"
                )

            # Emit the per-kernel compile_time_us metric, deferred from
            # ``_on_task_done`` so the parent-side wait (``bg drain
            # queue.get()`` + plugin's ``launcher_q.get()``) is rolled
            # into the same reported number as the worker's wall.
            # Without this, ``triton_kernel_compile_times_us`` in
            # streaming mode would only reflect the worker subprocess
            # time and silently under-count parent attribution.
            handle.worker_done_event.wait(timeout=5.0)
            worker_us = handle.worker_elapsed_us[0]
            if worker_us is not None:
                total_compile_us = (
                    worker_us
                    + handle.bg_drain_wait_ns[0] // 1000
                    + plugin_wait_ns // 1000
                )
                _add_triton_kernel_info(
                    handle.kernel_name, {"compile_time_us": total_compile_us}
                )
                get_metrics_context().add_top_n(
                    "triton_kernel_compile_times_us",
                    handle.kernel_name,
                    total_compile_us,
                )
            return DEFER

    return StreamingCompilePlugin()


def pre_fork_setup():
    """
    Setup that must be done prior to forking with a process pool.
    """
    # ensure properties have been calculated before processes
    # are forked
    caching_device_properties()

    # Computing the triton key can be slow. If we call it before fork,
    # it will be cached for the forked subprocesses.
    from torch._inductor.runtime.triton_compat import HAS_TRITON, triton_key

    if HAS_TRITON:
        triton_key()


def caching_device_properties():
    for _, device_interface in get_registered_device_interfaces():
        if device_interface.is_available():
            device_interface.Worker.get_device_properties()


def _compile_start() -> None:
    global _t0, _triton_kernel_metrics
    if _t0 is None:
        _t0 = time()
    if _triton_kernel_metrics is None:
        _triton_kernel_metrics = {}


def _compile_end() -> None:
    global _cumulative_compile_time, _t0, _triton_kernel_metrics
    if _t0 is not None:
        t1 = time()
        _cumulative_compile_time += t1 - _t0
        _t0 = None
        # print("CUMULATIVE COMPILE TIME", _cumulative_compile_time)
    if _triton_kernel_metrics:
        # Log triton kernel info
        sorted_info = dict(sorted(_triton_kernel_metrics.items()))
        torch._logging.trace_structured(
            "artifact",
            metadata_fn=lambda: {
                "name": "triton_kernel_info",
                "encoding": "json",
            },
            payload_fn=lambda: json.dumps(sorted_info),
        )
        _triton_kernel_metrics = None


def _add_triton_kernel_info(kernel_name: str, info: dict[str, Any]):
    global _triton_kernel_metrics
    # Must be called between _compile_start and _compile_end
    if _triton_kernel_metrics is not None:
        _triton_kernel_metrics[kernel_name] = info


_IS_WINDOWS = sys.platform == "win32"

log = logging.getLogger(__name__)

# Used to keep track of all process pools invoked so far.
_pool_set = OrderedSet[AnyPool]()


def shutdown_compile_workers() -> None:
    """Shut down all outstanding compile-worker pools."""
    for pool in _pool_set:
        pool.shutdown()
    AsyncCompile._ready_future = None
    after_fork()


def after_fork():
    """Reset pools to initial state without shutting them down"""
    _pool_set.clear()
    AsyncCompile.process_pool.cache_clear()


try:
    os.register_at_fork(after_in_child=after_fork)
except AttributeError:
    pass  # register_at_fork does not exist on windows


def get_compile_threads() -> int:
    """
    Temporary for internal rollout. Assign config.compile_threads lazily and return it.
    TODO: remove after rollout.
    """
    if config.compile_threads is None:
        config.compile_threads = config.decide_compile_threads()
    return config.compile_threads


@clear_on_fresh_cache
class CompiledTritonKernels:
    """
    In memory cache for storing compiled triton kernels.

    Each triton kernel is keyed by the hash of its source code. Each value stored
    in the cache is a return value of AsyncCompile.triton().

    Currently, the cache stores Future objects, but it should be generalizable for any kernels.
    """

    _cache: dict[str, CodeCacheFuture] = {}

    @staticmethod
    def key(kernel_src: str):
        """
        Generates a cache key given a triton kernel's full source code.
        This source includes the inductor meta, compilation metadata, the kernel itself, etc.
        `kernel_src` should be the exact string passed to async_compile.triton()'s first argument.
        """
        # Hashes the kernel source with torch_key into a single hash key
        return code_hash(kernel_src, extra=torch_key())

    @staticmethod
    def save(kernel_src: str, future: CodeCacheFuture):
        """
        Saves a compiled triton kernel to the cache.
        TODO: We store a LambdaFuture as that's the callable returned by async_compile.triton,
        but the real type we want to return here is actually an abstract triton kernel.

        TODO: Source code here is not just the kernel's source code, but also includes the inductor preamble, etc.
        so it could be less strict.
        """
        key = CompiledTritonKernels.key(kernel_src)
        CompiledTritonKernels._cache[key] = future

    @staticmethod
    def get(kernel_src: str) -> CodeCacheFuture | None:
        key = CompiledTritonKernels.key(kernel_src)
        return CompiledTritonKernels._cache.get(key, None)

    @staticmethod
    def cache_clear():
        CompiledTritonKernels._cache = {}

    @staticmethod
    def remove_future(kernel_src: str) -> None:
        key = CompiledTritonKernels.key(kernel_src)

        # Delete the LambdaFuture if there is one
        if key in CompiledTritonKernels._cache:
            del CompiledTritonKernels._cache[key]


class AsyncCompile:
    """
    Utilities to compile in thread pools or subprocess pools (in the case of Triton).
    """

    _ready_future: Future[Any] | None = None
    _metal_sources: list[tuple[str, str, list[str]]] | None = None

    def __init__(self) -> None:
        pass

    @staticmethod
    @functools.lru_cache(1)
    def pool() -> ThreadPoolExecutor:
        assert get_compile_threads() > 1
        return ThreadPoolExecutor(get_compile_threads())

    @staticmethod
    def _get_ready():
        """No-op function to help mark when the subprocess pool is ready."""
        return "ready"

    @staticmethod
    @functools.lru_cache(1)
    def process_pool() -> AnyPool:
        assert get_compile_threads() > 1
        AsyncCompile._ready_future = None
        log.info(
            "Creating '%s' pool with %d workers",
            config.worker_start_method,
            get_compile_threads(),
        )

        pool: AnyPool
        if config.worker_start_method == "subprocess":
            # Wrapper around ProcessPoolExecutor forks in a new process we control
            pool = SubprocPool(
                get_compile_threads(), quiesce=config.quiesce_async_compile_pool
            )
        else:
            if config.worker_start_method == "spawn":
                # Avoid creating pools in the spawned subprocs themselves:
                os.environ["TORCH_WARM_POOL"] = "0"
            pre_fork_setup()
            ctx = multiprocessing.get_context(config.worker_start_method)
            pool = TrackedProcessPoolExecutor(
                get_compile_threads(),
                mp_context=ctx,
                initializer=partial(_async_compile_initializer, os.getpid()),
            )
            # when this pool is created in a subprocess object, the normal exit handler
            # doesn't run, and we need to register our own handler.
            # exitpriority has to be high, because another one of the finalizers will
            # kill the worker thread that sends the shutdown message to the workers...
            multiprocessing.util.Finalize(None, pool.shutdown, exitpriority=sys.maxsize)

        _pool_set.add(pool)
        return pool

    @classmethod
    def warm_pool(cls) -> None:
        if get_compile_threads() <= 1:
            return
        _compile_start()
        # Pool is created on first access. Note for a SubprocPool, the sidecar process starts,
        # but its ProcessPoolExecutor does not initialize until a wakeup() call or the first
        # job is submitted.
        cls.process_pool()
        _compile_end()

    @classmethod
    def wait_pool_ready(cls, timeout=120) -> None:
        cls.use_process_pool()
        if cls._ready_future is not None:
            cls._ready_future.result(timeout=timeout)

    @classmethod
    def submit(cls, task: Callable[..., Any]) -> Any:
        if get_compile_threads() <= 1:
            return task()
        return cls.pool().submit(task)

    @classmethod
    def use_process_pool(cls):
        if get_compile_threads() <= 1:
            return False

        # Proton instrumentation backend requires compilation to happen in the main
        # process so it can instrument the Triton IR during JIT compilation.
        # Force synchronous compilation when proton profiling is enabled.
        if config.triton.proton_profiling:
            return False

        # Create a dummy job to check if the pool is ready. Submit it here instead of at
        # pool creation so we don't launch the full pool of worker subprocesses until
        # we're sure they're needed.
        if not cls._ready_future:
            cls._ready_future = cls.process_pool().submit(cls._get_ready)
        return cls._ready_future.done()

    @classmethod
    def wakeup(cls) -> None:
        """
        If using a SubprocPool, signal the sidecar process to start up its
        ProcessPoolExecutor.
        """
        if not cls.use_process_pool():
            return
        pool = cls.process_pool()
        if isinstance(pool, SubprocPool):
            pool.wakeup()

    def triton(self, kernel_name: str, source_code: str, device_str: str = "cuda"):
        """
        Async_compile.triton is more complicated than the other backends because
        we're trying to optimize compile time as much as possible for this hot callsite.

        First of all, the function is cached by CompiledTritonKernels; if there's a kernel
        already compiled, we grab it directly from the cache and return.

        Otherwise, if we have multiple compile threads, we kick off triton compilations on each
        worker process by giving it a kernel and source code to compile. The worker initializes
        a CachingAutotuner, runs triton compilation, and pickles the kernel back to us.
        We use TritonCompileResult to represent the objects being pickled back to us by each
        worker.

        Some maybe not obvious things that are pickled back to us:
        - Most of the time, we can avoid sending back CachingAutotuner.fn and other metadata
          and do not have to pay the cost of loading the triton kernel on the parent. But certain
          cases, like coordesc tuning and dynamic_scale_rblock, require us to reload the function
          in the parent lazily when we require it.
        - The AutotuneCache, if enabled, is constructed on each worker per triton config
          and pickled by to us via `CachingAutotuner.save_cache_hook`.
        """
        load_kernel = functools.partial(
            _load_triton_kernel_from_source, kernel_name, source_code
        )

        def reload_kernel_in_parent():
            # Benchmark how often this happens
            with dynamo_timed("reload_kernel_in_parent"):
                return load_kernel()

        counters["inductor"]["async_compile_cache_miss"] += 1

        kernel_code_log.info("Triton Kernel:\n%s", source_code)
        _compile_start()

        if os.environ.get("TRITON_INTERPRET", "0") == "1":
            return getattr(
                torch._inductor.codecache.PyCodeCache.load(source_code), kernel_name
            )

        is_parallel = self.use_process_pool()
        set_feature_use("parallel_compile_post_warmup", is_parallel)

        compile_id = torch._guards.CompileContext.current_compile_id()
        is_backward = getattr(V.graph, "is_backward", False)

        if (future := CompiledTritonKernels.get(source_code)) is not None:
            counters["inductor"]["async_compile_cache_hit"] += 1
            # Set reload_kernel_from_src properly based on source_code
            if isinstance(future, StaticAutotunerFuture):
                # Remove the future now that we've cache hit
                CompiledTritonKernels.remove_future(source_code)
                future.reload_kernel_from_src = reload_kernel_in_parent
            if is_parallel:
                return future
            else:
                return future.result()

        # Cache miss
        if is_parallel:
            # Ensure libdevice path is set in os.environ before passing to workers
            _set_triton_libdevice_path()
            # We want to support changing these env vars after (and while) the
            # process pool is running, so pass them to the subprocess to reset.
            env_vars = [
                "TORCHINDUCTOR_CACHE_DIR",
                "TRITON_CACHE_DIR",
                "TRITON_LIBDEVICE_PATH",
            ]
            extra_env = {v: os.environ[v] for v in env_vars if v in os.environ}
            extra_config = {
                "use_static_triton_launcher": torch._inductor.config.use_static_triton_launcher
            }

            if len(torch._inductor.config.autotune_lookup_table) > 0:
                m = size_hints_regex.search(source_code)
                if m:
                    size_hints_str = m.group(1)
                else:
                    size_hints_str = str(None)

                triton_src = source_code.split("@triton.jit\n")[1]
                from torch._inductor.runtime.triton_heuristics import (
                    generate_lookup_hash_from_source_code,
                )

                fn_hash = generate_lookup_hash_from_source_code(
                    size_hints_str, triton_src
                )

                if fn_hash in torch._inductor.config.autotune_lookup_table:
                    extra_config["autotune_lookup_table"] = {  # type: ignore[assignment]
                        fn_hash: torch._inductor.config.autotune_lookup_table[fn_hash]
                    }

            # Streaming compile is gated behind a config flag. When
            # enabled, the worker sends the (still-uncompiled) kernel
            # as the first message on an ``AF_UNIX`` socket, then
            # streams each ``CompileResult`` back as it finishes. The
            # parent's ``get_result`` returns the kernel as soon as it
            # arrives — without waiting for any config to compile —
            # so downstream consumers (notably the incremental autotune
            # plugin) can engage immediately and start dispatching as
            # soon as the first launcher arrives. When disabled, the
            # worker uses the original blocking flow (compile every
            # config, pickle the kernel back with ``compile_results``
            # populated as the future payload).
            if config.streaming_compile:
                (
                    sock_addr,
                    kernel_slot,
                    stream_q,
                    stream_sentinel,
                ) = _setup_streaming_compile(kernel_name)
            else:
                sock_addr = None
                kernel_slot = None
                stream_q = None
                stream_sentinel = None

            task = self.process_pool().submit(
                _worker_compile_triton,
                load_kernel,
                extra_env,
                extra_config,
                sock_addr,
            )

            def get_result() -> CachingAutotuner:
                elapsed_us: int | None = None
                if config.streaming_compile:
                    # Block on the kernel arriving via the streaming
                    # connection — NOT on ``task.result()``. The
                    # worker pushes the kernel back as soon as
                    # ``load_kernel()`` finishes; configs continue to
                    # compile in the background and stream into
                    # ``stream_q``.
                    assert (
                        kernel_slot is not None
                        and stream_q is not None
                        and stream_sentinel is not None
                    )
                    # If the worker subprocess dies before sending the
                    # kernel, no message arrives at the router for this
                    # connection and ``kernel_slot.get()`` would block
                    # forever. Catch that case via a task-done callback
                    # that pushes a failure into the slot if the task
                    # finishes with no kernel yet received.
                    sock_addr_with_kid = sock_addr  # type: ignore[assignment]
                    assert sock_addr_with_kid is not None
                    kernel_id_for_fail = int(sock_addr_with_kid.rsplit("#", 1)[1])

                    def _on_task_fail_before_kernel(future: Future) -> None:
                        exc = future.exception()
                        if exc is not None:
                            _get_streaming_router().fail_kernel(
                                kernel_id_for_fail, exc
                            )

                    task.add_done_callback(_on_task_fail_before_kernel)
                    item = kernel_slot.get()
                    if isinstance(item, _KernelLoadFailure):
                        exc = item.exc
                        if isinstance(exc, SubprocException):
                            raise exc.with_name(kernel_name) from exc
                        raise exc
                    kernel = cast("CachingAutotuner", item)

                    # ``_on_task_done`` is registered later (after the
                    # streaming handle is built and stashed on the
                    # holder), so the callback can reliably write the
                    # worker's elapsed_us into the handle even if the
                    # task completes synchronously at registration.
                    streaming_handle_holder: list[
                        StreamingCompileHandle | None
                    ] = [None]

                    def _on_task_done(future: Future) -> None:
                        h = streaming_handle_holder[0]
                        assert h is not None, (
                            "_on_task_done must run after handle is stashed"
                        )
                        try:
                            _, done_elapsed_us = future.result()
                        except Exception as cb_exc:
                            log.warning(
                                "streaming compile worker for %s failed: %s",
                                kernel_name,
                                cb_exc,
                            )
                            h.worker_done_event.set()
                            return
                        h.worker_elapsed_us[0] = done_elapsed_us
                        h.worker_done_event.set()
                else:
                    try:
                        result = task.result()
                    except SubprocException as e:
                        raise e.with_name(kernel_name) from e
                    kernel_or_none, elapsed_us = result
                    assert kernel_or_none is not None, (
                        "non-streaming worker must return a kernel"
                    )
                    kernel = kernel_or_none

                # Now that we've compiled, we should clear the future
                # so it can't be used again
                kernel.set_compile_info(compile_id, is_backward)
                CompiledTritonKernels.remove_future(source_code)

                kernel.restore_after_unpickle(old_values=None)

                if config.streaming_compile:
                    # Stash the streaming handle on the kernel —
                    # ``StreamingCompilePlugin`` consumes it at first
                    # ``run()``. The kernel itself was the first item on
                    # the wire and is not on this queue — only per-config
                    # results and failure markers come through.
                    #
                    # We deliberately SKIP ``kernel.precompile(...)`` here:
                    # it would block on the drain (defeating the streaming
                    # win) and force launcher creation without inputs.
                    # Instead we stash the ``_reload_kernel`` callback
                    # directly — normally set inside ``precompile`` — so
                    # coordesc / rblock scaling can still reload the
                    # JITFunction in the parent process.
                    #
                    # A per-kernel daemon ``_bg_drain_kernel`` thread
                    # starts immediately to drain the queue + make
                    # launchers in parallel with this process's other
                    # work (notably inductor codegen for *other* kernels).
                    # By the time first ``run()`` enters
                    # ``StreamingCompilePlugin.pre_dispatch``, the kernel
                    # is fully drained and the plugin just signals
                    # ``run()`` to proceed (where ``autotune_to_one_config``
                    # benches with the real input args).
                    assert stream_q is not None and stream_sentinel is not None
                    handle = StreamingCompileHandle(
                        queue=stream_q,
                        sentinel=stream_sentinel,
                        num_configs=len(kernel.configs or []),
                        static_triton_bundle_key=CompiledTritonKernels.key(
                            source_code
                        ),
                        drain_complete=threading.Event(),
                        launcher_q=_queue.Queue(),
                        launcher_sentinel=object(),
                        kernel_name=kernel_name,
                        worker_done_event=threading.Event(),
                        worker_elapsed_us=[None],
                        bg_drain_wait_ns=[0],
                    )
                    streaming_handle_holder[0] = handle
                    kernel._streaming_compile_handle = handle
                    kernel._reload_kernel = reload_kernel_in_parent
                    # Register ``_on_task_done`` only now that the
                    # handle is on ``streaming_handle_holder``: if the
                    # task already completed (worker fast, parent slow),
                    # ``add_done_callback`` would otherwise fire the
                    # callback synchronously with a ``None`` handle.
                    task.add_done_callback(_on_task_done)
                    threading.Thread(
                        target=_bg_drain_kernel,
                        args=(kernel, handle),
                        name=f"inductor-stream-bg-drain-{kernel_name}",
                        daemon=True,
                    ).start()
                else:
                    kernel.precompile(
                        warm_cache_only=False,
                        reload_kernel=reload_kernel_in_parent,
                        static_triton_bundle_key=CompiledTritonKernels.key(
                            source_code
                        ),
                    )
                    assert elapsed_us is not None
                    info = kernel.autotune_cache_info or {}
                    info["compile_time_us"] = elapsed_us
                    _add_triton_kernel_info(kernel_name, info)
                    get_metrics_context().add_top_n(
                        "triton_kernel_compile_times_us", kernel_name, elapsed_us
                    )
                return kernel

            future = LambdaFuture(get_result, future=task)
            CompiledTritonKernels.save(source_code, future)
            return future
        else:
            with dynamo_timed(
                "async_compile.precompile",
                log_pt2_compile_event=True,
                dynamo_compile_column_us="triton_compile_time_us",
                log_waitcounter=True,
                waitcounter_name_override="compile_triton",
            ):
                fail = None
                try:
                    start_ns = time_ns()
                    _set_triton_ptxas_path()
                    _set_triton_libdevice_path()
                    kernel = load_kernel()
                    kernel.set_compile_info(compile_id, is_backward)
                    kernel.precompile(
                        warm_cache_only=False,
                        static_triton_bundle_key=CompiledTritonKernels.key(source_code),
                    )
                    elapsed_us = (time_ns() - start_ns) // 1000
                    get_metrics_context().add_top_n(
                        "triton_kernel_compile_times_us", kernel_name, elapsed_us
                    )
                    info = kernel.autotune_cache_info or {}
                    info["compile_time_us"] = elapsed_us
                    _add_triton_kernel_info(kernel_name, info)
                    return kernel
                except Exception as e:
                    fail = str(e)
                    raise
                finally:
                    log_triton_builds(fail=fail)

    def multi_kernel(self, *args, **kwargs) -> Any:
        from torch._inductor.codegen.multi_kernel import MultiKernelCall

        # no need to call this in parallel since the sub-kernels are already parallel tasks
        return MultiKernelCall(*args, **kwargs)

    def size_hint_multi_kernel(self, *args, **kwargs) -> Any:
        from torch._inductor.codegen.multi_kernel import SizeHintMultiKernelCall

        return SizeHintMultiKernelCall(*args, **kwargs)

    def cpp(self, source_code: str):
        kernel_code_log.info("CPP Kernel:\n%s", source_code)
        if get_compile_threads() <= 1:
            return CppCodeCache.load(source_code).kernel
        else:
            get_result = CppCodeCache.load_async(source_code, submit_fn=self.submit)
            return LambdaFuture(lambda: get_result().kernel)

    def cpp_pybinding(self, argtypes: list[str], source_code: str):
        kernel_code_log.info("CPP+Bindings Kernel:\n%s", source_code)
        if get_compile_threads() <= 1:
            return CppPythonBindingsCodeCache.load_pybinding(argtypes, source_code)
        else:
            get_result = CppPythonBindingsCodeCache.load_pybinding_async(
                argtypes, source_code, submit_fn=self.submit
            )
            return LambdaFuture(get_result)

    def cutlass(self, cache_cls, source_code, dst_file_ext, aot_compile=False):
        def task():
            if aot_compile:
                # We rely on JITInductor to compile the CUDA code,
                # so that we can load it into AOTInductor.
                output_path, *_ = cache_cls.compile(source_code, "o")
                cache_cls.aot_kernels_o.append(output_path)
            return cache_cls.load(source_code, dst_file_ext)[0]

        return self.submit(task)

    def cuda(self, source_code, dst_file_ext, aot_compile=False):
        kernel_code_log.info("CUDA Kernel:\n%s", source_code)
        return self.cutlass(CUDACodeCache, source_code, dst_file_ext, aot_compile)

    def xpu(self, source_code, dst_file_ext, aot_compile=False):
        kernel_code_log.info("XPU Kernel:\n%s", source_code)
        return self.cutlass(XPUCodeCache, source_code, dst_file_ext, aot_compile)

    def rocm(
        self,
        source_code,
        dst_file_ext,
        aot_compile=False,
    ):
        kernel_code_log.info("ROCm Kernel:\n%s", source_code)

        def task():
            if aot_compile:
                output_path, *_ = ROCmCodeCache.compile(source_code, dst_file_ext="o")
                ROCmCodeCache.aot_kernels_o.append(output_path)
            if config.rocm.generate_test_runner:
                _ = ROCmCodeCache.compile(source_code, dst_file_ext="exe")
            return ROCmCodeCache.load(source_code, dst_file_ext)[0]

        return self.submit(task)

    def halide(self, meta: HalideMeta, source_code: str):
        kernel_code_log.info("Halide Kernel:\n%r\n%s", meta, source_code)
        if get_compile_threads() <= 1:
            return HalideCodeCache.generate_halide(meta, source_code)
        else:
            get_result = HalideCodeCache.generate_halide_async(
                meta, source_code, submit_fn=self.submit
            )
            return LambdaFuture(get_result)

    def cutedsl(self, kernel_name: str, source_code: str):
        """
        Compile CuteDSL (CUTLASS Python DSL) kernels.

        Args:
            kernel_name: Name of the kernel to be defined
            source_code: Source code of the CuteDSL kernel, as a string

        Note:
            CuteDSL currently requires source files to do its compilation, there we
            use the PyCodeCache to write the source code to a file and load it.
        """
        from torch._inductor.codegen.cutedsl.cutedsl_kernel import (
            CuteDSLKernelWrapper,
            MAIN_SUFFIX,
        )

        kernel_code_log.info("CuteDSL Kernel:\n%s", source_code)

        def task():
            key, path = torch._inductor.codecache.PyCodeCache.write(source_code)
            mod = torch._inductor.codecache.PyCodeCache.load_by_key_path(key, path)

            # Find our special entry point named function
            main_func_name = f"{kernel_name}_{MAIN_SUFFIX}"
            if not hasattr(mod, main_func_name):
                available = [name for name in dir(mod) if callable(getattr(mod, name))]
                raise RuntimeError(
                    f"Could not find CuteDSL main kernel function '{main_func_name}'. Available callables: {available}"
                )

            return CuteDSLKernelWrapper(getattr(mod, main_func_name), kernel_path=path)

        if get_compile_threads() <= 1:
            return task()
        else:
            future = self.submit(task)
            return LambdaFuture(lambda: future.result())

    def pallas(self, kernel_name: str, source_code: str):
        """
        Compile Pallas (JAX experimental) kernels.

        Args:
            kernel_name: Name of the kernel to be defined
            source_code: Source code of the Pallas kernel, as a string

        Note:
            Pallas kernels are Python code that uses JAX and Pallas APIs.
            We use the PyCodeCache to write the source code to a file and load it.
        """
        from torch._inductor.codegen.pallas import MAIN_SUFFIX, PallasKernelWrapper

        kernel_code_log.info("Pallas Kernel:\n%s", source_code)

        def task():
            key, path = torch._inductor.codecache.PyCodeCache.write(source_code)
            mod = torch._inductor.codecache.PyCodeCache.load_by_key_path(key, path)

            # Find our special entry point named function
            main_func_name = f"{kernel_name}_{MAIN_SUFFIX}"
            if not hasattr(mod, main_func_name):
                available = [name for name in dir(mod) if callable(getattr(mod, name))]
                raise RuntimeError(
                    f"Could not find Pallas main kernel function '{main_func_name}'. Available callables: {available}"
                )

            return PallasKernelWrapper(getattr(mod, main_func_name), kernel_path=path)

        if get_compile_threads() <= 1:
            return task()
        else:
            future = self.submit(task)
            return LambdaFuture(lambda: future.result())

    def nv_universal_gemm(self, kernel_name: str, source_code: str):
        """
        Compile NVIDIA Universal GEMM kernels.

        Args:
            kernel_name: Name of the kernel to be defined
            source_code: Source code of the kernel, as a string

        Note:
            NVIDIA Universal GEMM kernels are Python code that calls the cutlass_api library.
            We use the PyCodeCache to write the source code to a file and load it.
        """
        from torch._inductor.codegen.nv_universal_gemm.nv_universal_gemm_kernel import (
            NVUniversalGemmKernelWrapper,
        )
        from torch._inductor.codegen.nv_universal_gemm.nv_universal_gemm_scheduling import (
            MAIN_SUFFIX,
        )

        kernel_code_log.info("NVIDIA Universal GEMM Kernel:\n%s", source_code)

        def task():
            key, path = torch._inductor.codecache.PyCodeCache.write(source_code)
            mod = torch._inductor.codecache.PyCodeCache.load_by_key_path(key, path)

            # Find our special entry point named function
            main_func_name = f"{kernel_name}_{MAIN_SUFFIX}"
            if not hasattr(mod, main_func_name):
                available = [name for name in dir(mod) if callable(getattr(mod, name))]
                raise RuntimeError(
                    f"Could not find NVIDIA Universal GEMM main kernel function "
                    f"'{main_func_name}'. Available callables: {available}"
                )

            return NVUniversalGemmKernelWrapper(
                getattr(mod, main_func_name), kernel_path=path
            )

        if get_compile_threads() <= 1:
            return task()
        else:
            future = self.submit(task)
            return LambdaFuture(lambda: future.result())

    def metal(self, kernel_name: str, source: str, headers: list[str]) -> None:
        """Register a Metal kernel body; wait() compiles all registered kernels into one library."""
        if self._metal_sources is None:
            self._metal_sources = []
        self._metal_sources.append((kernel_name, source, headers))

    def wait(self, scope: dict[str, Any]) -> None:
        if get_compile_threads() > 1:
            with dynamo_timed(
                "async_compile.wait",
                log_pt2_compile_event=True,
                dynamo_compile_column_us="triton_compile_time_us",
                log_waitcounter=True,
                waitcounter_name_override="compile_triton",
            ):
                self._wait_futures(scope)

        if self._metal_sources:
            from torch._inductor.runtime.runtime_utils import compile_mps_shaders

            scope.update(compile_mps_shaders(self._metal_sources))
            self._metal_sources.clear()

        _compile_end()

    def _wait_futures(self, scope: dict[str, Any]) -> None:
        kernels = {
            key: value
            for key, value in scope.items()
            if isinstance(value, (Future, CodeCacheFuture))
        }
        pbar = tqdm(
            total=len(kernels),
            desc="Inductor Compilation",
            disable=config.disable_progress,
            delay=0,
        )
        # compile_worker_wait_timeout=0 (default) means "wait forever"; map
        # it to None so both Future.result() and CodeCacheFuture.result()
        # receive the same "no timeout" sentinel.
        wait_timeout = config.compile_worker_wait_timeout or None
        for key, result in kernels.items():
            if config.verbose_progress and not isinstance(pbar, _Faketqdm):
                pbar.set_postfix_str(key)
            try:
                kernel = result.result(timeout=wait_timeout)
                scope[key] = kernel
            except FuturesTimeoutError as e:
                # concurrent.futures.TimeoutError became an alias of the
                # builtin TimeoutError in Python 3.11; on 3.10 it is a
                # distinct class, so catch it explicitly.
                raise RuntimeError(
                    f"Inductor compile-worker future for {key!r} did not "
                    f"complete within {wait_timeout}s. Override with "
                    "TORCHINDUCTOR_COMPILE_WORKER_WAIT_TIMEOUT=<seconds>."
                ) from e
            except BrokenProcessPool as e:
                raise RuntimeError(
                    "A compilation subprocess exited unexpectedly. This "
                    "is likely due to a crash. To facilitate debugging, "
                    "you can re-run with TORCHINDUCTOR_COMPILE_THREADS=1 "
                    "to cause compilation to occur in the main process."
                ) from e
            pbar.update(1)


def maybe_warm_pool() -> None:
    if (
        os.environ.get("TORCH_TNT_IN_USE", "0") == "1"
        or os.environ.get("TORCH_WARM_POOL", "1") != "1"
        # The subprocess pool is only used for the Triton backend
        or not has_triton_package()
        # Skip for fbcode. We have internal reports of usages inside multiprocessing
        # pools that lead a multiplicative number of compile subprocesses.
        or config.is_fbcode()
    ):
        return

    AsyncCompile.warm_pool()
    # TODO: This starts the SubprocPool's internal process pool as early as possible at
    # the expense of creating a bunch of worker processes that might not be needed. We
    # could start them lazily if we're willing to lose a small amount of compile time.
    AsyncCompile.wakeup()


# On exit give the workers a chance to clean themselves up. Without this the
# resource_tracker can complain about leaked semaphores coming from the
# ProcessPoolExecutor:
#   UserWarning: resource_tracker: There appear to be 5 leaked semaphore objects
#   to clean up at shutdown
atexit.register(shutdown_compile_workers)
