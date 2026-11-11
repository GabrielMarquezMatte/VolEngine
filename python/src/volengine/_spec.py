"""Model specifications and fit options, validated eagerly through the native library."""

from __future__ import annotations

from dataclasses import KW_ONLY, dataclass

from ._native import Handle, check, create_handle, lib

MODELS = {"sgarch": 0}
DISTRIBUTIONS = {"norm": 0}
VAR_INITS = {"sample": 0, "backcast": 1}


def code_of(table: dict[str, int], kind: str, name: str) -> int:
    if name not in table:
        raise ValueError(f"unknown {kind} {name!r}; expected one of {sorted(table)}")
    return table[name]


@dataclass(frozen=True)
class Spec:
    """Model specification. Unknown names raise ValueError, native rejections VolEngineError."""

    model: str = "sgarch"
    _: KW_ONLY
    p: int = 1
    q: int = 1
    dist: str = "norm"
    include_mean: bool = True

    def __post_init__(self) -> None:
        self.native_handle().close()

    def native_handle(self) -> Handle:
        model = code_of(MODELS, "model", self.model)
        dist = code_of(DISTRIBUTIONS, "dist", self.dist)
        handle = create_handle(lib.ve_spec_create, lib.ve_spec_free, model)
        check(lib.ve_spec_set_order(handle.pointer, self.p, self.q))
        check(lib.ve_spec_set_dist(handle.pointer, dist))
        check(lib.ve_spec_set_include_mean(handle.pointer, int(self.include_mean)))
        return handle


@dataclass(frozen=True)
class FitOptions:
    """Optimizer settings for a fit."""

    _: KW_ONLY
    max_iter: int = 500
    tol: float = 1e-6
    var_init: str = "sample"

    def __post_init__(self) -> None:
        self.native_handle().close()

    def native_handle(self) -> Handle:
        var_init = code_of(VAR_INITS, "var_init", self.var_init)
        handle = create_handle(lib.ve_fit_options_create, lib.ve_fit_options_free)
        check(lib.ve_fit_options_set_max_iter(handle.pointer, self.max_iter))
        check(lib.ve_fit_options_set_tolerance(handle.pointer, self.tol))
        check(lib.ve_fit_options_set_var_init(handle.pointer, var_init))
        return handle
