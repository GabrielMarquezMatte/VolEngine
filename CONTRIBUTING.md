# Contributing

Small, focused pull requests are preferred over large rewrites. Read [ARCHITECTURE.md](ARCHITECTURE.md) for the
shape of the codebase and [STYLEGUIDE.md](STYLEGUIDE.md) for the code style.

## Ground rules

- The C API in `include/volengine/volengine.h` is the only stable contract. Everything under `src/` can change
  freely. The stability rules are in [ARCHITECTURE.md](ARCHITECTURE.md#stability-rules-of-the-c-api).
- Warnings are errors (`-Wall -Wextra -Wpedantic -Werror`, plus `-Wnrvo` on GCC 14+; `/W4 /WX` on MSVC).
- Commit messages carry no `Co-Authored-By` trailer.
- Do not copy or translate `rugarch` code (GPL-3). Reading it to understand a convention is fine.

## Build and test (C++)

Presets are in `CMakePresets.json`:

```
cmake --preset gcc-release
cmake --build --preset gcc-release
ctest --preset gcc-release
```

`gcc-debug` and `msvc-release` work the same way. `native-release` uses the default compiler (`CC` and `CXX`)
and is what CI uses on Linux and macOS. GCC 15 or newer is the supported GCC.

`ctest` runs the GoogleTest suite, a C11 compile of the public header and the `style` test.

### Windows and MinGW

The MinGW build links `libstdc++`, `libgcc` and `winpthread` statically, so a Windows Python can load
`libvolengine.dll` without the MSYS2 `bin` directory on `PATH`. Keep it that way when touching the link options.

## Benchmarks

The Google Benchmark suite in `tests/benchmarks/` is off by default (`VE_BUILD_BENCHMARKS`) and has its own preset:

```
cmake --preset gcc-bench
cmake --build --preset gcc-bench
build/gcc-bench/volengine_benchmarks --benchmark_repetitions=5 --benchmark_report_aggregates_only=true
```

CI builds it on Linux and never runs it. Results and the machine they come from are in
[docs/performance/benchmarks.md](docs/performance/benchmarks.md). Compare numbers only from the same machine.

The Python benchmark against `arch` runs the same fits on both libraries (`volengine` with `threads=1` and the
`arch` convention), checks that the log-likelihoods agree and writes a report with the machine and versions:

```
VOLENGINE_LIB=$PWD/build/gcc-release/libvolengine.dll python/.venv/Scripts/python python/benchmarks/bench_vs_arch.py --out docs/performance/python-vs-arch.md
```

It needs `arch` in the venv. The result is in [docs/performance/python-vs-arch.md](docs/performance/python-vs-arch.md).

## Python tests

Create the test environment from the Windows Python, not from the MSYS2 Python that comes first on `PATH`
(it has no numpy wheels):

```
py -3.14 -m venv python/.venv
python/.venv/Scripts/python -m pip install -e "python[test]"
```

Point the package at the library you built with `VOLENGINE_LIB`, then run pytest:

```
VOLENGINE_LIB=$PWD/build/gcc-release/libvolengine.dll python/.venv/Scripts/python -m pytest python/tests -q
```

The library is `libvolengine.dll` (MinGW), `Release/volengine.dll` (MSVC), `libvolengine.so` or
`libvolengine.dylib`. Without `VOLENGINE_LIB` the package looks in `python/src/volengine/_lib`. The oracle tests
read committed fixtures and need neither R nor `arch`.

## Regenerating the oracle fixtures

Only needed when the data or an oracle convention changes. The steps, requirements and fixture format are in
[tests/reference/README.md](tests/reference/README.md); the conventions and tolerances are in
[docs/guide/validation.md](docs/guide/validation.md). Record oracle versions and observed differences there
after regenerating.

## Style checks

- `python tests/style/check_style.py` runs on its own and as the `style` ctest. It fails on `else`, on folders
  with more than 8 modules and, for Python (including `python/benchmarks/`), on functions above 60 lines, more than 5 parameters or nesting above 3.
- C++ function size, parameter count and nesting are checked by clang-tidy with the repository `.clang-tidy`.
  CI runs it over `src/`; locally, after configuring with `-DCMAKE_EXPORT_COMPILE_COMMANDS=ON`:

  ```
  clang-tidy -p build/gcc-release src/fit/fit.cpp
  ```

## Documentation

Repository documents are in English. Measurements and justifications go in `docs/`, never in code comments
([STYLEGUIDE.md](STYLEGUIDE.md) says why). Add an entry under `## [Unreleased]` in [CHANGELOG.md](CHANGELOG.md)
for any user-visible change.
