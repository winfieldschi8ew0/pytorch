from collections.abc import Callable
from typing import Any

_UNSAFE_EXPORT_CALLABLES: set[Callable[..., Any]] = set()


def register_unsafe_export_callable(*callables: Callable[..., Any]) -> None:
    """Register Python callables that are allowed to be serialized and
    deserialized in exported programs.

    By default, :func:`torch.export.save` and :func:`torch.export.load` only
    support serializing standard PyTorch operators (``torch.ops.*``) and
    higher-order operators. Exported programs that contain plain Python callable
    nodes -- such as predispatch wrapper functions from ``torch._functorch``
    (e.g., ``_jvp_increment_nesting``, ``_vmap_increment_nesting``) -- will raise
    :class:`SerializeError` during save or load.

    This function registers specific callables so they are permitted during
    serialization. It is intended for advanced use cases (e.g., models using
    ``torch.func.jvp`` or ``torch.vmap``) where the caller accepts the
    following tradeoff:

    .. warning::
        **No backwards compatibility guarantee.** Serialized artifacts
        containing registered callables may not be loadable across different
        PyTorch versions, because the callable targets are resolved by module
        path (e.g., ``torch._functorch.predispatch._jvp_increment_nesting``).
        If PyTorch renames, moves, or removes these internal functions, loading
        will fail.

    Args:
        *callables: One or more callable objects to allow for serialization.

    Example::

        from torch._export.serde import register_unsafe_export_callable
        from torch._functorch.predispatch import (
            _jvp_increment_nesting,
            _jvp_decrement_nesting,
        )

        register_unsafe_export_callable(
            _jvp_increment_nesting,
            _jvp_decrement_nesting,
        )
    """
    _UNSAFE_EXPORT_CALLABLES.update(callables)


def _is_unsafe_callable_registered(fn: Callable[..., Any]) -> bool:
    return fn in _UNSAFE_EXPORT_CALLABLES


def _reset_unsafe_export_callables() -> None:
    _UNSAFE_EXPORT_CALLABLES.clear()
