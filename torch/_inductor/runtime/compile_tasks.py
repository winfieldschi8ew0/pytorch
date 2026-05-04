from __future__ import annotations

import functools
import io
import linecache
import os
import pickle
import sys
import threading
import time
import warnings
from pathlib import Path
from types import FunctionType, ModuleType
from collections.abc import Hashable
from typing import Any, TYPE_CHECKING

from torch._utils_internal import log_triton_builds


if TYPE_CHECKING:
    from collections.abc import Callable

    from torch._inductor.runtime.triton_heuristics import CachingAutotuner


def _reload_python_module(
    key: str, path: str, set_sys_modules: bool = True
) -> ModuleType:
    with open(path) as f:
        try:
            code = compile(f.read(), path, "exec", dont_inherit=True)
        except Exception as e:
            raise RuntimeError(
                f"Failed to import {path}\n{type(e).__name__}: {e}"
            ) from None
        mod = ModuleType(f"{__name__}.{key}")
        mod.__file__ = path
        mod.key = key  # type: ignore[attr-defined]
        exec(code, mod.__dict__, mod.__dict__)
        if set_sys_modules:
            sys.modules[mod.__name__] = mod
        return mod


@functools.cache
def _set_triton_ptxas_path() -> None:
    if os.environ.get("TRITON_PTXAS_PATH") is not None:
        return
    ptxas = Path(__file__).absolute().parents[1] / "bin" / "ptxas"
    if not ptxas.exists():
        return
    if ptxas.is_file() and os.access(ptxas, os.X_OK):
        os.environ["TRITON_PTXAS_PATH"] = str(ptxas)
    else:
        warnings.warn(f"{ptxas} exists but is not an executable")


def _set_triton_libdevice_path() -> None:
    """
    Use the CUDA toolkit's libdevice instead of Triton's bundled version.
    This ensures Triton's pow matches CUDA's powf for bitwise precision.
    Gated by config.eager_numerics.use_pytorch_libdevice.
    """
    from torch._inductor import config

    if not config.eager_numerics.use_pytorch_libdevice:
        return

    _set_triton_libdevice_path_impl()


def _set_triton_libdevice_path_impl() -> None:
    try:
        from triton import knobs
    except ImportError:
        return

    env_path = os.environ.get("TRITON_LIBDEVICE_PATH")
    if env_path is not None:
        knobs.nvidia.libdevice_path = env_path
        return

    if knobs.nvidia.libdevice_path is not None:
        return

    try:
        from torch.utils.cpp_extension import CUDA_HOME

        if CUDA_HOME is None:
            warnings.warn(
                "CUDA_HOME not set; using Triton's bundled libdevice which may "
                "cause minor precision differences in pow operations. "
                "To fix: set TRITON_LIBDEVICE_PATH to your CUDA toolkit's libdevice, "
                "e.g., export TRITON_LIBDEVICE_PATH=/usr/local/cuda/nvvm/libdevice/libdevice.10.bc",
                stacklevel=3,
            )
            return
        libdevice = Path(CUDA_HOME) / "nvvm" / "libdevice" / "libdevice.10.bc"
        if libdevice.is_file():
            knobs.nvidia.libdevice_path = str(libdevice)
            # Also set env var so subprocess compile workers inherit it
            os.environ["TRITON_LIBDEVICE_PATH"] = str(libdevice)
        else:
            warnings.warn(
                f"CUDA libdevice not found at {libdevice}; using Triton's bundled "
                "libdevice which may cause minor precision differences in pow operations. "
                "To fix: set TRITON_LIBDEVICE_PATH to your CUDA toolkit's libdevice, "
                "e.g., export TRITON_LIBDEVICE_PATH=/usr/local/cuda/nvvm/libdevice/libdevice.10.bc",
                stacklevel=3,
            )
    except ImportError:
        warnings.warn(
            "torch.utils.cpp_extension not available; using Triton's bundled "
            "libdevice which may cause minor precision differences in pow operations. "
            "To fix: set TRITON_LIBDEVICE_PATH to your CUDA toolkit's libdevice, "
            "e.g., export TRITON_LIBDEVICE_PATH=/usr/local/cuda/nvvm/libdevice/libdevice.10.bc",
            stacklevel=3,
        )


class _CompileFailureMarker:
    """Worker→parent sentinel marking a per-config compile failure on
    the streaming connection.

    The worker sends one instance per failed compile (``OutOfResources``,
    ``PTXASError``, ``IntelGPUError``). ``config_key`` is the failing
    config's ``triton_config_to_hashable`` key; the parent's drain side
    matches it against the still-pending set so failures for configs
    that were already satisfied (e.g., from the per-kernel pool) are
    silently dropped instead of double-counted.

    Distinct from the parent-side end-of-stream sentinel pushed by
    ``_setup_streaming_compile`` on connection close — that one is a
    bare ``object()`` compared by identity inside the parent process;
    this one is a class-typed message pickled across the wire.
    """

    def __init__(self, config_key: Hashable | None = None) -> None:
        self.config_key = config_key

    def __reduce__(self) -> tuple[object, ...]:
        # Pickle the config_key so the parent can match the failure
        # against its still-pending set. Identity is lost across pickle,
        # so the parent uses ``isinstance`` to detect markers.
        return (_CompileFailureMarker, (self.config_key,))


def _worker_compile_triton(
    load_kernel: Callable[[], CachingAutotuner],
    extra_env: dict[str, str],
    extra_config: dict[str, Any],
    streaming_address: str | None,
) -> tuple[CachingAutotuner | None, int]:
    """Worker-side entry point for ``AsyncCompile.triton``.

    When ``streaming_address`` is non-None (parent enabled
    ``config.streaming_compile``), the worker:
      1. Loads the kernel.
      2. Sends the (still-uncompiled) kernel back to the parent as the
         first message on the streaming connection — so the parent's
         ``get_result`` can return the kernel and let downstream
         consumers (notably the incremental autotune plugin) start
         dispatching before any config has finished compiling.
      3. Compiles every config in parallel and streams each
         ``CompileResult`` / ``_CompileFailureMarker`` back as it
         arrives. Connection close signals end-of-stream.
      4. Returns ``(None, elapsed_us)`` from the future — the kernel
         is no longer in the future's payload (it was streamed), so
         only the timing metric remains.

    When ``streaming_address`` is None, the worker uses the original
    blocking flow: ``kernel.precompile(warm_cache_only=True)`` compiles
    every config before the kernel is pickled back to the parent with
    ``compile_results`` populated. Returns ``(kernel, elapsed_us)``.
    """
    _set_triton_ptxas_path()
    os.environ.update(extra_env)
    # Set libdevice path if passed via env from main process
    libdevice_path = extra_env.get("TRITON_LIBDEVICE_PATH")
    if libdevice_path:
        try:
            from triton import knobs

            knobs.nvidia.libdevice_path = libdevice_path
        except ImportError:
            pass
    from torch._inductor import config

    with config.patch(extra_config):
        fail = None
        try:
            start_ns = time.time_ns()
            kernel = load_kernel()
            if streaming_address is not None:
                _stream_compile_triton(kernel, streaming_address)
                elapsed_ns = time.time_ns() - start_ns
                # Kernel was streamed back already; nothing left to
                # return via the future payload.
                linecache.clearcache()
                return None, elapsed_ns // 1000
            kernel.precompile(warm_cache_only=True)
            elapsed_ns = time.time_ns() - start_ns
            kernel.prepare_for_pickle()
            # We can release this memory in the compile subprocesses:
            linecache.clearcache()
            return kernel, elapsed_ns // 1000
        except Exception as e:
            fail = str(e)
            raise
        finally:
            log_triton_builds(fail=fail)


_DYN_KERNEL_MODULE_PREFIX = "torch._inductor.runtime.compile_tasks."

# ``threading.RLock`` returns instances of a private factory type that
# isn't directly importable, so we capture it once at module load by
# constructing a throwaway lock and using its ``type``. Cached because
# ``_streaming_persistent_id`` runs for every traversed pickle object.
_RLOCK_TYPE = type(threading.RLock())


def _streaming_persistent_id(obj: Any) -> object | None:
    """``persistent_id`` callback for the worker→parent streaming
    connection.

    Pickle invokes this for every object it traverses. Returning
    non-``None`` substitutes the object on the wire with the returned
    id; the parent's ``persistent_load`` resolves the id back to a
    safe value (``None`` for everything we substitute today).

    We substitute three classes of object that don't survive the trip
    to the parent process:

    1. **Threading primitives (``RLock``).** Pickle refuses these
       outright. Triton's ``JITFunction`` carries a per-instance
       ``_hash_lock`` that the parent doesn't need (it's recreated
       fresh on demand).
    2. **Module objects.** Most module references go unused on the
       parent post-unpickle. The dyn kernel module
       (``torch._inductor.runtime.compile_tasks.<hash>``) isn't
       importable on the parent at all; certain triton/template
       internals also aren't reachable by ``__name__`` and trip
       pickle's standard module reducer with ``cannot pickle 'module'
       object``. Substituting every module uniformly is the simplest
       correct policy.
    3. **Raw Python functions defined in the dyn kernel module.**
       ``@triton.jit``-decorated kernels and any helpers
       defined/imported into the dyn module pickle by name; lookup
       fails on the parent (no such module).

    All substitutions resolve to ``None`` on the parent. Downstream
    consumers either don't read the substituted attribute (the
    common case for ``make_launcher``, which reads ``arg_names``,
    ``params``, ``signature``, etc. — all of which survive) or
    handle ``None`` defensively (the rare case).
    """
    if isinstance(obj, _RLOCK_TYPE):
        return ("_streaming_skip",)
    # ALL module objects. Most module references end up unused on the
    # parent (the autotuner doesn't traverse them post-unpickle); the
    # ones that *are* needed (e.g., ``triton``, ``torch``) are
    # importable on the parent and would be re-fetched from
    # ``sys.modules`` by pickle's standard reducer. But some modules
    # pickle's standard reducer rejects (notably the dyn kernel
    # module, which isn't in the parent's ``sys.modules``, and
    # certain triton/template internals that aren't reachable by
    # ``__name__``). Substituting every module uniformly with
    # ``None`` is the simplest correct policy and matches what
    # ``__getstate__`` already does for fields like ``kernel.module``.
    if isinstance(obj, ModuleType):
        return ("_streaming_skip",)
    # Raw Python functions / lambdas defined in dyn modules. We
    # restrict to ``FunctionType`` (NOT arbitrary callables) because
    # ``triton.runtime.jit.JITFunction.__init__`` sets
    # ``self.__module__ = fn.__module__`` — so JITFunction instances
    # report the dyn module too even though the JITFunction class
    # itself lives in ``triton.runtime.jit`` and pickles fine. We want
    # to substitute the *underlying* ``fn`` (a raw FunctionType), not
    # the JITFunction wrapping it.
    if isinstance(obj, FunctionType):
        mod = obj.__module__
        if isinstance(mod, str) and mod.startswith(_DYN_KERNEL_MODULE_PREFIX):
            return ("_streaming_skip",)
    return None


def _streaming_persistent_load(pid: object) -> object:
    """``persistent_load`` callback for the parent's streaming reader.

    Resolves all ids produced by ``_streaming_persistent_id`` to
    ``None``. Anything else is an error (means the wire format
    diverged from what we sent).
    """
    if pid == ("_streaming_skip",):
        return None
    raise pickle.UnpicklingError(f"unsupported persistent id: {pid!r}")


class _StreamingPickler(pickle.Pickler):
    """Pickler for the worker→parent streaming connection. See
    ``_streaming_persistent_id`` for the substitution policy."""

    persistent_id = staticmethod(_streaming_persistent_id)


class _StreamingUnpickler(pickle.Unpickler):
    """Companion to ``_StreamingPickler`` on the parent side."""

    persistent_load = staticmethod(_streaming_persistent_load)


def _streaming_send(conn: Any, obj: Any) -> None:
    """Send ``obj`` over ``conn`` using ``_StreamingPickler``.

    Bypasses ``conn.send`` (which uses the default ForkingPickler) by
    pickling into a buffer and pushing the framed bytes via
    ``send_bytes``.
    """
    buf = io.BytesIO()
    _StreamingPickler(buf, protocol=pickle.HIGHEST_PROTOCOL).dump(obj)
    conn.send_bytes(buf.getvalue())


def _streaming_recv(conn: Any) -> Any:
    """Receive an object from ``conn`` using ``_StreamingUnpickler``.

    The reader thread on the parent uses this in place of
    ``conn.recv`` so persistent-id substitutions resolve correctly.
    """
    return _StreamingUnpickler(io.BytesIO(conn.recv_bytes())).load()


def _stream_compile_triton(
    kernel: CachingAutotuner, streaming_address: str
) -> None:
    """Send ``kernel_id``, then ``kernel``, then stream each
    ``CompileResult`` / ``_CompileFailureMarker`` over the parent's shared
    streaming router.

    ``streaming_address`` is in ``<unix-socket-path>#<kernel_id>`` format.
    The path is the parent's process-wide router socket; the suffix
    identifies which per-kernel queues the router should demux this
    connection's messages into.

    Sending the kernel first (after the kernel_id handshake) is what
    lets the parent's ``get_result`` return early: downstream consumers
    can engage with the kernel and start dispatching as soon as the
    first launcher arrives, instead of waiting for every config to
    finish.

    Mechanics:
      * ``prepare_for_pickle`` clears the autotuner-side
        ``self.launchers`` (required by ``__getstate__``'s assertion)
        and the JITFunction's unpicklable bits. We send the kernel,
        then immediately ``restore_after_unpickle`` so this
        worker-local copy can keep compiling.
      * Per-result sends go through ``_StreamingPickler``, whose
        ``persistent_id`` substitutes references to the worker's
        dynamically-loaded kernel module and other unpicklable bits
        (RLocks). The substitutions resolve to ``None`` on the parent.
        This means we don't have to mutate the live JITFunction (which
        other in-flight compiles in this same worker are using); each
        ``send`` just transparently swaps the problematic refs as it
        serializes.
      * Compile iteration delegates to
        ``CachingAutotuner._iter_compile_results(parallel=True,
        yield_failures=True)``: a ``ThreadPoolExecutor`` with one
        thread per config; results yield in completion order; failures
        come back as ``_CompileFailureMarker`` instances.
      * Connection close signals end-of-stream to the parent.
    """
    from multiprocessing.connection import Client

    addr, kid_str = streaming_address.rsplit("#", 1)
    kernel_id = int(kid_str)

    conn = Client(addr)
    try:
        # Handshake: tell the parent's router which per-kernel queues to
        # demux this connection's subsequent messages into.
        _streaming_send(conn, kernel_id)

        # Phase 1: ship the kernel back ASAP. The autotuner-side
        # mutation here is safe — no compiles are in flight yet.
        old_values = kernel.prepare_for_pickle()
        try:
            _streaming_send(conn, kernel)
        finally:
            kernel.restore_after_unpickle(old_values)

        # Phase 2: compile and stream. The JITFunction stays live for
        # parallel in-flight compiles; ``_StreamingPickler`` handles
        # the wire substitutions per-result without touching it.
        for item in kernel._iter_compile_results(
            parallel=True, yield_failures=True
        ):
            _streaming_send(conn, item)
    finally:
        conn.close()
