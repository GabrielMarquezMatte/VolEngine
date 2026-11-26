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
VOLENGINE_LIB=$PWD/build/gcc-release/libvolengine.dll uv run --project python --group bench python python/benchmarks/bench_vs_arch.py --out docs/performance/python-vs-arch.md
```

It needs the `bench` group (`arch`). The result is in [docs/performance/python-vs-arch.md](docs/performance/python-vs-arch.md).

## C++ wrapper tests

The wrapper in `cpp/` is tested against a library you built, not against a release. Build the library (the
`gcc-release` preset, or `python python/scripts/build_native.py`, which also leaves it in `python/src/volengine/_lib`),
point `VOLENGINE_NATIVE_LIB` at it and run `cpp/scripts/ci_check.sh`, or the three commands in
[cpp/README.md](cpp/README.md#building-and-testing). On Windows, build and run from PowerShell with
`C:/msys64/ucrt64/bin` first in `PATH`: a Git Bash `ld` writes an empty `--output-def`, and Git's `mingw64` `libstdc++`
crashes the MinGW test executables. MSVC compiles the wrapper over the same MinGW DLL (Developer PowerShell,
`-G Ninja -DCMAKE_CXX_COMPILER=cl`).

## C# binding tests

The binding in `csharp/` needs the .NET SDK of `global.json` and is tested against a library you built. From the
repository root:

```
VOLENGINE_LIB=$PWD/python/src/volengine/_lib/libvolengine.dll dotnet test --solution csharp/VolEngine.slnx -c Release
```

The tests compare the declarations with `include/volengine/volengine.abi`, check that every contract function is
called, compare with the oracle fixtures and run the code of `docs/guide/csharp.md` (keep `GuideSnippetsTests` and the
guide in step). A new public member fails the build (RS0016) until it is listed in `PublicAPI.Unshipped.txt`.

To check a package as a user gets it, copy the library to a folder as `volengine-native-<rid>.<ext>` and run

```
dotnet pack csharp/src/VolEngine -c Release -p:Version=1.3.1-local -p:VolEngineNativeAssets=<folder> -p:VolEngineNativeRids=win-x64 -o <packages>
bash csharp/scripts/check_package.sh <packages> 1.3.1-local
```

which runs the clean consumer without `VOLENGINE_LIB` and again published with Native AOT (on Windows that needs the
Visual Studio C++ tools). A release packs all seven libraries; after a release, move `PublicAPI.Unshipped.txt` into
`PublicAPI.Shipped.txt`. `PackageValidationBaselineVersion` in `VolEngine.csproj` names the last package on nuget.org,
which package validation compares each new package with: move it to each release once that release is published.

## Python tests

The Python package is managed with [uv](https://docs.astral.sh/uv/). `uv sync --project python` creates
`python/.venv` with the `dev` group (pytest, mypy); the `bench` group adds `arch` for the benchmarks.

Point the package at the library you built with `VOLENGINE_LIB`, then run pytest:

```
VOLENGINE_LIB=$PWD/build/gcc-release/libvolengine.dll uv run --project python pytest python/tests -q
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

## The ABI contract

`include/volengine/volengine.abi` has one line per declaration of `volengine.h` (defines, typedefs, functions without
parameter names). It is generated: after changing the header, run

```
python scripts/abi_contract.py write
```

The `abi_contract` ctest and CI fail when the two differ, so a change to the interface always shows in the diff.
From the first stable release on, CI also fails when a line of the last release's contract is gone or changed: the
contract only grows. The Windows export list is made from the same file (`scripts/abi_contract.py def`).

## Documentation

Repository documents are in English. Measurements and justifications go in `docs/`, never in code comments
([STYLEGUIDE.md](STYLEGUIDE.md) says why). Add an entry under `## [Unreleased]` in [CHANGELOG.md](CHANGELOG.md)
for any user-visible change.
