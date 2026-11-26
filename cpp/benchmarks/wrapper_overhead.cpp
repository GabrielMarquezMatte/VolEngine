// The cost of the C++ wrapper against the same work through the C API: one univariate fit, and a batch of 12 specs over 21
// series. Each iteration builds its specs, panel, batch and options from scratch on both sides, so the difference is
// what the wrapper adds. Every iteration checks that both sides report the same log-likelihood.
#include <benchmark/benchmark.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <random>
#include <utility>
#include <vector>
#include <volengine/volengine.hpp>

namespace
{

struct Workload
{
    std::size_t rows;
    std::size_t cols;
    std::vector<double> data; // row-major
    std::vector<std::pair<ve::Model, ve::Dist>> specs;
    bool standard_errors;
};

// GARCH(1,1) series, in percent units, fixed seed.
std::vector<double> simulate(std::size_t rows, std::size_t cols)
{
    std::mt19937_64 engine(7);
    std::normal_distribution<double> normal;
    std::vector<double> data(rows * cols);
    for (std::size_t j = 0; j < cols; ++j)
    {
        double variance = 0.05 / (1.0 - 0.08 - 0.90);
        for (std::size_t t = 0; t < rows; ++t)
        {
            const double shock = std::sqrt(variance) * normal(engine);
            data[t * cols + j] = shock;
            variance = 0.05 + 0.08 * shock * shock + 0.90 * variance;
        }
    }
    return data;
}

Workload single_fit()
{
    return {3000, 1, simulate(3000, 1), {{ve::Model::sgarch, ve::Dist::norm}}, true};
}

Workload batch_fit()
{
    std::vector<std::pair<ve::Model, ve::Dist>> specs;
    for (const auto model : {ve::Model::sgarch, ve::Model::gjrgarch, ve::Model::egarch})
    {
        for (const auto dist : {ve::Dist::norm, ve::Dist::student, ve::Dist::ged, ve::Dist::sstd})
        {
            specs.emplace_back(model, dist);
        }
    }
    return {5000, 21, simulate(5000, 21), std::move(specs), false};
}

void check(ve_status status)
{
    if (status != VE_OK)
    {
        std::abort();
    }
}

// The same fits through the C API; returns the sum of the log-likelihoods.
double fit_with_c(ve_context* context, const Workload& workload)
{
    std::vector<ve_spec*> specs;
    for (const auto& [model, dist] : workload.specs)
    {
        ve_spec* spec = nullptr;
        check(ve_spec_create(static_cast<std::int32_t>(model), &spec));
        check(ve_spec_set_dist(spec, static_cast<std::int32_t>(dist)));
        specs.push_back(spec);
    }
    ve_panel* panel = nullptr;
    check(ve_panel_create(workload.data.data(), workload.rows, workload.cols, &panel));
    ve_fit_options* options = nullptr;
    check(ve_fit_options_create(&options));
    check(ve_fit_options_set_standard_errors(options, workload.standard_errors ? 1 : 0));
    std::vector<std::size_t> spec_index;
    std::vector<std::size_t> series_index;
    for (std::size_t s = 0; s < specs.size(); ++s)
    {
        for (std::size_t j = 0; j < workload.cols; ++j)
        {
            spec_index.push_back(s);
            series_index.push_back(j);
        }
    }
    ve_batch* batch = nullptr;
    check(ve_batch_create(&batch));
    std::vector<const ve_spec*> const_specs(specs.begin(), specs.end());
    check(ve_batch_set_specs(batch, const_specs.data(), const_specs.size()));
    check(ve_batch_set_panel(batch, panel));
    check(ve_batch_set_jobs(batch, spec_index.data(), series_index.data(), spec_index.size()));
    check(ve_batch_set_options(batch, options));
    std::vector<ve_fit*> fits(spec_index.size(), nullptr);
    check(ve_fit_batch_with_cancel(context, batch, nullptr, fits.data()));
    double total = 0.0;
    for (ve_fit* fit : fits)
    {
        double loglik = 0.0;
        check(ve_fit_loglik(fit, &loglik));
        total += loglik;
        ve_fit_free(fit);
    }
    ve_batch_free(batch);
    ve_fit_options_free(options);
    ve_panel_free(panel);
    for (ve_spec* spec : specs)
    {
        ve_spec_free(spec);
    }
    return total;
}

double fit_with_wrapper(const ve::Context& context, const Workload& workload)
{
    std::vector<ve::Spec> specs;
    for (const auto& [model, dist] : workload.specs)
    {
        specs.push_back(ve::Spec{.model = model, .dist = dist});
    }
    const auto panel = ve::Panel::create(workload.data, workload.rows, workload.cols).value();
    const auto fits =
        context.fit(specs, panel, ve::FitOptions{.standard_errors = workload.standard_errors}).value();
    double total = 0.0;
    for (const ve::Fit& fit : fits)
    {
        total += fit.loglik().value();
    }
    return total;
}

void run_c(benchmark::State& state, const Workload& workload)
{
    const auto context = ve::Context::create(0).value();
    const double reference = fit_with_wrapper(context, workload);
    for (auto _ : state)
    {
        const double total = fit_with_c(context.native(), workload);
        if (total != reference)
        {
            state.SkipWithError("the C and wrapper log-likelihoods differ");
            return;
        }
        benchmark::DoNotOptimize(total);
    }
}

void run_wrapper(benchmark::State& state, const Workload& workload)
{
    const auto context = ve::Context::create(0).value();
    const double reference = fit_with_c(context.native(), workload);
    for (auto _ : state)
    {
        const double total = fit_with_wrapper(context, workload);
        if (total != reference)
        {
            state.SkipWithError("the C and wrapper log-likelihoods differ");
            return;
        }
        benchmark::DoNotOptimize(total);
    }
}

void BM_FitDirectC(benchmark::State& state)
{
    static const Workload workload = single_fit();
    run_c(state, workload);
}

void BM_FitWrapper(benchmark::State& state)
{
    static const Workload workload = single_fit();
    run_wrapper(state, workload);
}

void BM_BatchDirectC(benchmark::State& state)
{
    static const Workload workload = batch_fit();
    run_c(state, workload);
}

void BM_BatchWrapper(benchmark::State& state)
{
    static const Workload workload = batch_fit();
    run_wrapper(state, workload);
}

} // namespace

BENCHMARK(BM_FitDirectC)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK(BM_FitWrapper)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK(BM_BatchDirectC)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK(BM_BatchWrapper)->Unit(benchmark::kMillisecond)->UseRealTime();
