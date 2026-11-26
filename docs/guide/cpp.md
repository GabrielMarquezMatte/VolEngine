# C++ quick guide

A header-only C++23 wrapper over the C API. Handles are move-only RAII types in namespace `ve`, every call that can
fail returns `std::expected<T, ve::Error>` with the message of `ve_last_error`, inputs are `std::span<const double>`,
and results are zero-copy views that stay valid while the object that owns them lives.

## Use it from CMake

The wrapper links the native library of a release, which CMake downloads for the tag you name.

```cmake
include(FetchContent)
FetchContent_Declare(volengine
    GIT_REPOSITORY https://github.com/GabrielMarquezMatte/VolEngine
    GIT_TAG v1.2.0
    SOURCE_SUBDIR cpp)
FetchContent_MakeAvailable(volengine)

add_executable(app main.cpp)
target_link_libraries(app PRIVATE volengine::volengine)
volengine_copy_native_library(app)   # Windows: the DLL must sit next to the executable
```

Offline, install once and use `find_package(volengine 1.0 REQUIRED)` with the same two lines for the target
(`cmake --install` of a project that already fetched it carries the headers and the library). To use a library you
built yourself, set the environment variable `VOLENGINE_NATIVE_LIB` and nothing is downloaded. The settings are in
[cpp/README.md](../../cpp/README.md). The release carries one library per platform: Windows x64 (built with MinGW, with
a `.lib` for MSVC and a `.dll.a` for MinGW), macOS arm64 and x64, and Linux glibc and musl on x64 and arm64.

## Fit, forecast, simulate

```cpp
#include <cstdio>
#include <volengine/volengine.hpp>

int main()
{
    auto context = ve::Context::create().value();

    // Data to fit: a GARCH(1,1) with Student-t shocks, simulated by the library itself.
    const ve::Spec truth{.dist = ve::Dist::student,
                         .fixed = {{"mu", 0.0003}, {"omega", 2e-6}, {"alpha1", 0.08}, {"beta1", 0.9}, {"shape", 8.0}}};
    const auto data = context.simulate(truth, ve::SimOptions{.horizon = 3000, .paths = 1, .seed = 1});
    if (!data)
    {
        std::fprintf(stderr, "%s\n", data.error().message.c_str());
        return 1;
    }

    auto fit = context.fit(ve::Spec{.model = ve::Model::gjrgarch, .dist = ve::Dist::student}, data->returns().value());
    if (!fit || fit->status() != ve::FitStatus::converged)
    {
        std::fprintf(stderr, "the fit did not converge\n");
        return 1;
    }
    std::printf("loglik %.3f, BIC %.4f\n", fit->loglik().value(), fit->bic().value());
    const auto errors = fit->std_errors(ve::SeKind::robust).value();   // an error if the fit ran without them
    const auto forecast = fit->forecast(10).value();
    std::printf("sigma in 10 steps: %.5f (error of the first parameter %.5f)\n", forecast.sigma().value().back(), errors[0]);
}
```

A fit that does not converge is a status, not an error; an invalid argument (a bad spec, a NaN in the data, a wrong
shape) is an `ve::Error` with `status == ve::Status::invalid_argument` and the reason in `message`.

The fits are scale invariant: returns in fractions (0.012) or in percent (1.2) give the same model, with the
parameters, sigma and forecasts in the scale of the data, so there is no need to multiply by 100.

## Cumulative simulation

`SimOptions::checkpoints` keeps, per path, only the sum of the simulated series up to each checkpoint (strictly
increasing, from 1, the last equal to the horizon), read with `cumulative()`: paths x K values instead of
paths x horizon, and each sum is the left-to-right sum of the steps of the full simulation with the same seed. For a
DCC it is paths x K x m. `returns()` and `sigma()` fail in that mode, and correlations cannot be kept with it.

```cpp
// The 1, 5 and 10 day 1% quantile of the log return, from the same 100000 paths.
const auto paths = context.simulate(*fit, ve::SimOptions{.horizon = 10, .paths = 100000, .seed = 1,
                                                          .checkpoints = {1, 5, 10}}).value();
const auto sums = paths.cumulative().value();   // row p: the sums of path p at 1, 5 and 10 days
const std::size_t days[] = {1, 5, 10};
for (std::size_t j = 0; j < 3; ++j)
{
    std::vector<double> horizon_sums;
    for (std::size_t p = 0; p < 100000; ++p)
    {
        horizon_sums.push_back(sums[p * 3 + j]);
    }
    std::ranges::nth_element(horizon_sums, horizon_sums.begin() + 1000);
    std::printf("%zu days: %.4f\n", days[j], horizon_sums[1000]);
}
```

## DCC

```cpp
// `fits` is a std::vector<ve::Fit>, one per series, all of one length (at least 100).
auto dcc = context.fit_dcc(ve::DccSpec{.dist = ve::DccDist::mvt}, fits);
if (dcc && dcc->status() == ve::FitStatus::converged)
{
    auto correlation = dcc->correlation().value();     // T x m x m, row-major, a view
    auto forecast = dcc->forecast(5).value();           // R, H, sigma and mean for 5 steps
}
```

`std_errors`, `t_values` and `p_values` of a `DccFit` made with `DccOptions{.standard_errors = false}` are errors, as for
a `Fit`; `context.standard_errors(fit, univariate_fits)` returns a fit that has them. The Copula-DCC is
`DccSpec::copula`. Every function of the C API is called by the wrapper, so nothing in the interface is out of reach.

## Cancellation

`fit`, `fit_dcc`, `standard_errors` and `simulate` take an optional `std::stop_token` as their last argument. A stop
requested before or during the call makes it fail soon after with `ve::Status::cancelled`, and no result is made; a
call that finishes first succeeds. Without a token, or with one that can never be stopped, nothing changes. On a
`Context` that other calls keep busy, a stopped call waits for its turn in the context's queue before it fails: give
each caller that needs a deadline its own `Context`.

```cpp
std::stop_source source;
std::jthread deadline([&source] {
    std::this_thread::sleep_for(std::chrono::seconds(2));
    source.request_stop();
});
auto fits = context.fit(grid, panel, {}, source.get_token());
if (!fits && fits.error().status == ve::Status::cancelled)
{
    // stopped before the grid finished
}
```

`std::jthread` also stops on its own: a token from `std::jthread`'s `get_stop_token()` cancels the call when the
thread is asked to stop.
