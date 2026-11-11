"""ctypes bindings that mirror include/volengine/volengine.h."""

from __future__ import annotations

import ctypes
import os
from collections.abc import Callable, Mapping
from pathlib import Path
from types import TracebackType
from typing import Any

VE_ABI_VERSION = 0
VE_INTERNAL_ERROR = 5
LIBRARY_ENV = "VOLENGINE_LIB"
LIBRARY_DIRECTORY = Path(__file__).resolve().parent / "_lib"
LIBRARY_NAMES = ("libvolengine.dll", "volengine.dll", "libvolengine.so", "libvolengine.dylib")

_HANDLE = ctypes.c_void_p
_OUT_HANDLE = ctypes.POINTER(ctypes.c_void_p)
_STATUS = ctypes.c_int32
_SIZE = ctypes.c_size_t
_I32 = ctypes.c_int32
_DOUBLE = ctypes.c_double
_OUT_DOUBLE = ctypes.POINTER(ctypes.c_double)
_OUT_SIZE = ctypes.POINTER(ctypes.c_size_t)
_OUT_I32 = ctypes.POINTER(ctypes.c_int32)
_OUT_VIEW = ctypes.POINTER(ctypes.POINTER(ctypes.c_double))
_OUT_TEXT = ctypes.POINTER(ctypes.c_char_p)

_SCALAR_CRITERION = [_HANDLE, _I32, _OUT_DOUBLE]
_VIEW = [_HANDLE, _OUT_VIEW, _OUT_SIZE]
_SE_VALUES = [_HANDLE, _I32, _HANDLE]

PROTOTYPES: dict[str, tuple[Any, list[Any]]] = {
    "ve_abi_version": (_I32, []),
    "ve_last_error": (ctypes.c_char_p, []),
    "ve_context_create": (_STATUS, [_OUT_HANDLE]),
    "ve_context_set_threads": (_STATUS, [_HANDLE, _I32]),
    "ve_context_threads": (_STATUS, [_HANDLE, _OUT_I32]),
    "ve_context_free": (None, [_HANDLE]),
    "ve_spec_create": (_STATUS, [_I32, _OUT_HANDLE]),
    "ve_spec_set_order": (_STATUS, [_HANDLE, _I32, _I32]),
    "ve_spec_set_dist": (_STATUS, [_HANDLE, _I32]),
    "ve_spec_set_include_mean": (_STATUS, [_HANDLE, _I32]),
    "ve_spec_free": (None, [_HANDLE]),
    "ve_fit_options_create": (_STATUS, [_OUT_HANDLE]),
    "ve_fit_options_set_max_iter": (_STATUS, [_HANDLE, _I32]),
    "ve_fit_options_set_tolerance": (_STATUS, [_HANDLE, _DOUBLE]),
    "ve_fit_options_set_var_init": (_STATUS, [_HANDLE, _I32]),
    "ve_fit_options_free": (None, [_HANDLE]),
    "ve_panel_create": (_STATUS, [_HANDLE, _SIZE, _SIZE, _OUT_HANDLE]),
    "ve_panel_set_strides": (_STATUS, [_HANDLE, ctypes.c_ssize_t, ctypes.c_ssize_t]),
    "ve_panel_set_ranges": (_STATUS, [_HANDLE, _HANDLE, _HANDLE]),
    "ve_panel_free": (None, [_HANDLE]),
    "ve_batch_create": (_STATUS, [_OUT_HANDLE]),
    "ve_batch_set_specs": (_STATUS, [_HANDLE, ctypes.POINTER(ctypes.c_void_p), _SIZE]),
    "ve_batch_set_panel": (_STATUS, [_HANDLE, _HANDLE]),
    "ve_batch_set_jobs": (_STATUS, [_HANDLE, _HANDLE, _HANDLE, _SIZE]),
    "ve_batch_set_options": (_STATUS, [_HANDLE, _HANDLE]),
    "ve_batch_job_count": (_STATUS, [_HANDLE, _OUT_SIZE]),
    "ve_batch_free": (None, [_HANDLE]),
    "ve_fit_batch": (_STATUS, [_HANDLE, _HANDLE, ctypes.POINTER(ctypes.c_void_p)]),
    "ve_fit_status": (_STATUS, [_HANDLE, _OUT_I32]),
    "ve_fit_nobs": (_STATUS, [_HANDLE, _OUT_SIZE]),
    "ve_fit_param_count": (_STATUS, [_HANDLE, _OUT_SIZE]),
    "ve_fit_param_name": (_STATUS, [_HANDLE, _SIZE, _OUT_TEXT]),
    "ve_fit_estimates": (_STATUS, [_HANDLE, _HANDLE]),
    "ve_fit_std_errors": (_STATUS, _SE_VALUES),
    "ve_fit_t_values": (_STATUS, _SE_VALUES),
    "ve_fit_p_values": (_STATUS, _SE_VALUES),
    "ve_loglik": (_STATUS, [_HANDLE, _OUT_DOUBLE]),
    "ve_aic": (_STATUS, _SCALAR_CRITERION),
    "ve_bic": (_STATUS, _SCALAR_CRITERION),
    "ve_hqic": (_STATUS, _SCALAR_CRITERION),
    "ve_sic": (_STATUS, _SCALAR_CRITERION),
    "ve_fit_sigma": (_STATUS, _VIEW),
    "ve_fit_residuals": (_STATUS, _VIEW),
    "ve_fit_std_residuals": (_STATUS, _VIEW),
    "ve_fit_fitted": (_STATUS, _VIEW),
    "ve_fit_free": (None, [_HANDLE]),
}


class VolEngineError(Exception):
    """A native call failed; `status` is the VE_* code and the message comes from ve_last_error()."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


def locate_library(override: str | None, directory: Path) -> Path:
    if override:
        path = Path(override)
        if not path.is_file():
            raise ImportError(f"{LIBRARY_ENV} points to {override}, which is not a file")
        return path
    for name in LIBRARY_NAMES:
        if (directory / name).is_file():
            return directory / name
    raise ImportError(
        f"VolEngine native library not found: set {LIBRARY_ENV} to the library file, "
        f"or place one of {', '.join(LIBRARY_NAMES)} in {directory}"
    )


def load_library(environ: Mapping[str, str], directory: Path) -> ctypes.CDLL:
    path = locate_library(environ.get(LIBRARY_ENV), directory)
    try:
        library = ctypes.CDLL(str(path))
    except OSError as error:
        raise ImportError(f"cannot load the VolEngine native library {path}: {error}") from error
    declare_prototypes(library)
    abi = library.ve_abi_version()
    if abi != VE_ABI_VERSION:
        raise ImportError(f"{path} has ABI version {abi}, this package needs {VE_ABI_VERSION}")
    return library


def declare_prototypes(library: ctypes.CDLL) -> None:
    for name, (restype, argtypes) in PROTOTYPES.items():
        function = getattr(library, name)
        function.restype = restype
        function.argtypes = argtypes


lib = load_library(os.environ, LIBRARY_DIRECTORY)


def last_error() -> str:
    message = lib.ve_last_error()
    if message is None:
        return ""
    return message.decode("utf-8", errors="replace")


def check(status: int) -> None:
    if status != 0:
        raise VolEngineError(status, last_error())


class Handle:
    """Owns one native handle and frees it exactly once."""

    def __init__(self, pointer: int | None, free: Callable[[int], None]) -> None:
        self._pointer: int | None = pointer
        self._free = free

    @property
    def pointer(self) -> int:
        if self._pointer is None:
            raise ValueError("the native handle is already closed")
        return self._pointer

    def close(self) -> None:
        pointer, self._pointer = self._pointer, None
        if pointer is not None:
            self._free(pointer)

    def __enter__(self) -> Handle:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()


def create_handle(create: Callable[..., int], free: Callable[[int], None], *args: Any) -> Handle:
    out = ctypes.c_void_p()
    check(create(*args, ctypes.byref(out)))
    if out.value is None:
        raise VolEngineError(VE_INTERNAL_ERROR, "the native create call returned a null handle")
    return Handle(out.value, free)


def read_scalar(ctype: Any, function: Callable[..., int], *args: Any) -> Any:
    out = ctype()
    check(function(*args, ctypes.byref(out)))
    return out.value
