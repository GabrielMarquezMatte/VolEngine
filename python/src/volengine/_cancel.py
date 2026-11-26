"""The cancellation handle of the long calls."""

from __future__ import annotations

from ._native import Handle, check, create_handle, lib


class Cancel:
    """A signal that stops the calls it is passed to (`cancel=`), with `Cancelled`.

    `cancel()` may be called from any thread, also while a call runs (a `threading.Timer` gives a deadline), and more
    than once; a Cancel stays requested and cannot be reset, so make a new one for the next call. One Cancel can be
    passed to several calls: a request stops all of them. A request that arrives after the last check of a call
    leaves it to return its result.
    """

    def __init__(self) -> None:
        self._handle = create_handle(lib.ve_cancel_create, lib.ve_cancel_free)

    def cancel(self) -> None:
        check(lib.ve_cancel_request(self._handle.pointer))


def cancel_pointer(cancel: Cancel | None) -> int | None:
    """The native pointer of `cancel`, or None (a null pointer) for no cancellation."""
    if cancel is None:
        return None
    if not isinstance(cancel, Cancel):
        raise TypeError("cancel must be a volengine.Cancel or None")
    return cancel._handle.pointer


__all__ = ["Cancel", "cancel_pointer"]
