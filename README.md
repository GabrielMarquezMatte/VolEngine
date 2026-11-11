# VolEngine

VolEngine estimates GARCH volatility models. The numerical core is C++23; its only stable contract is a C API
(`include/volengine/volengine.h`), so other languages can bind to it, and a Python package, `volengine`, is the
first consumer. Many series and many specifications are fitted in one call on a thread pool, with results that do
not depend on the thread count. Default conventions follow R's `rugarch`; those of Python's `arch` are an option.

## Status

Version 0.1.0 (pre-release). Only the sGARCH(1,1) model with normal innovations is implemented, with Hessian and
robust standard errors and the usual information criteria. More models and features come in later releases. Until
1.0 the C ABI may change between releases. See [CHANGELOG.md](CHANGELOG.md).

## Python example

```python
import numpy as np
import volengine as ve

returns = np.random.default_rng(1).standard_normal(3000)

ctx = ve.Context()
fit = ctx.fit(ve.Spec(), returns)

print(fit.status)                 # "converged"
print(fit.coef("robust"))         # CoefTable: names, estimate, std_error, t_value, p_value
print(ve.loglik(fit), ve.aic(fit))
print(fit.sigma[:5])              # conditional standard deviation, a read-only view
```

`ctx.fit` also takes a `(T, N)` array (one fit per column), a list of specs, `starts` and `lengths` for series of
different lengths, and `options=ve.FitOptions(var_init="backcast")` for the `arch` convention.

## Build

Needs CMake 3.28+ and a C++23 compiler (GCC 15+, a recent MSVC or Clang); the presets use Ninja, except
`msvc-release` (Visual Studio) and `native-release` (the default generator). GoogleTest and BS::thread_pool are
fetched by CMake.

```
cmake --preset gcc-release
cmake --build --preset gcc-release
ctest --preset gcc-release
```

Other presets: `gcc-debug`, `msvc-release` and `native-release` (the default compiler, taken from `CC` and `CXX`).
Install the Python package with `pip install -e "python[test]"` and point it to the built library with the
`VOLENGINE_LIB` environment variable. See [CONTRIBUTING.md](CONTRIBUTING.md) for details and
[ARCHITECTURE.md](ARCHITECTURE.md) for the code map.

## License

[MIT](LICENSE).
