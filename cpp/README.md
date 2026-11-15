# VolEngine C++ wrapper

A header-only C++23 wrapper over the VolEngine C API (`include/volengine/volengine.h`). Every C handle is a move-only
RAII type in namespace `ve`, every call that can fail returns `std::expected<T, ve::Error>` with the message of
`ve_last_error`, and no exception leaves the wrapper. Inputs are `std::span<const double>`; results are zero-copy
`std::span<const double>` views that are valid while the object that owns them lives.

```cpp
#include <volengine/volengine.hpp>

auto context = ve::Context::create().value();
auto fit = context.fit(ve::Spec{.model = ve::Model::gjrgarch, .dist = ve::Dist::student}, returns);
if (fit && fit->status() == ve::FitStatus::converged) {
    auto sigma = fit->sigma().value();               // a view, valid until `fit` is destroyed
    auto forecast = fit->forecast(10).value();       // independent of `fit`
}
```

The surface follows the Python package: `Context`, `Spec`, `FitOptions`, `Panel` and `Batch` in, `Fit` (estimates,
standard errors, criteria, series and diagnostics as members), `Forecast`, `Simulation`, and the DCC family (`DccSpec`,
`DccFit`, `DccForecast`, `DccSimulation`, `DccFuture`) including the Copula-DCC.

## Using it

The wrapper links the native library of a release. The library is downloaded from the GitHub Release of the tag you ask
for, so a project needs a network at configure time only once.

FetchContent:

```cmake
include(FetchContent)
FetchContent_Declare(volengine
    GIT_REPOSITORY https://github.com/GabrielMarquezMatte/VolEngine
    GIT_TAG v0.9.0
    SOURCE_SUBDIR cpp)
FetchContent_MakeAvailable(volengine)

add_executable(app main.cpp)
target_link_libraries(app PRIVATE volengine::volengine)
volengine_copy_native_library(app)   # needed on Windows: the DLL must sit next to the executable
```

`find_package`, after `cmake --install` of the same project (the install carries the headers and the library, so
`find_package` never touches the network):

```cmake
find_package(volengine 0.9 REQUIRED)
target_link_libraries(app PRIVATE volengine::volengine)
volengine_copy_native_library(app)
```

| Setting | Meaning |
|---|---|
| `VOLENGINE_VERSION` (cache) | The release tag to download. Taken from `git describe --tags --exact-match` when empty. Outside a tag and without `VOLENGINE_NATIVE_LIB` the configure fails and says so. |
| `VOLENGINE_NATIVE_LIB` (environment) | A library you built yourself. Nothing is downloaded. On Windows, `volengine.def` and `libvolengine.dll.a` must be next to the DLL (the build directory of a MinGW build has them; so does `python/src/volengine/_lib`). |
| `VOLENGINE_INSTALL` | On when `cpp/` is the top-level project: install rules and the `volengineConfig.cmake` package. |

The library asset is `volengine-native-<os>-<arch>.<so|dylib|dll>` for Windows x64, macOS arm64 and x64, and Linux glibc
and musl on x64 and arm64. On Windows the DLL is built by MinGW; the release carries a `.lib` for MSVC and a `.dll.a`
for MinGW, so both compilers link it.

## Building and testing

From the repository root, with the library built (`python python/scripts/build_native.py`, or the `gcc-release` preset):

```
VOLENGINE_NATIVE_LIB=<path to the library> cmake -S cpp -B build/cpp -G Ninja -DVOLENGINE_BUILD_TESTS=ON
cmake --build build/cpp
ctest --test-dir build/cpp --output-on-failure
```

`cpp/scripts/ci_check.sh` does that and also configures `cpp/tests/package` in both consumer modes. Every `VE_API`
function must be called by a header (`ctest` runs `check_abi_coverage.py`). `cpp/benchmarks` compares the wrapper with
direct C calls (`-DVOLENGINE_BUILD_BENCHMARKS=ON`).
