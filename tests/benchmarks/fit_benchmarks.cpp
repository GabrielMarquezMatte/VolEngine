#include "fit/fit.hpp"
#include "model/layout.hpp"
#include "model/likelihood.hpp"
#include "simulated_panel.hpp"
#include "volengine/volengine.h"

#include <benchmark/benchmark.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace
{

constexpr std::size_t kObservations = 5000;
constexpr std::size_t kBatchSeries = 20;

struct NamedSpec
{
    std::string name;
    ve::Spec spec;
    std::optional<ve::testing::VariantProcess> process;
};

// p is the ARCH order and q the GARCH order: s21 is sGARCH(2, 1).
ve::Spec make_spec(int p, int q, int ar, int ma, ve::Dist dist)
{
    ve::Spec spec;
    spec.p = p;
    spec.q = q;
    spec.ar = ar;
    spec.ma = ma;
    spec.dist = dist;
    return spec;
}

ve::Spec variant_spec(ve::Model model, ve::Dist dist)
{
    ve::Spec spec = make_spec(1, 1, 0, 0, dist);
    spec.model = model;
    return spec;
}

ve::testing::VariantProcess oracle_process(const std::string& name)
{
    for (const ve::testing::VariantProcess& process : ve::testing::variant_processes())
    {
        if (name == process.name)
        {
            return process;
        }
    }
    throw std::logic_error("no simulated process named " + name);
}

// The variants are simulated from the processes of the oracle series; TGARCH is the ALLGARCH recursion with
// lambda = 1 and no eta2.
std::vector<NamedSpec> variant_specs()
{
    const ve::testing::VariantProcess tgarch{.name = "fgarch_tgarch11_norm",
                                             .model = ve::Model::FGarchAllGarch,
                                             .omega = 0.02,
                                             .alpha = 0.08,
                                             .beta = 0.88,
                                             .eta1 = 0.3,
                                             .power = 1.0};
    using ve::Dist;
    using ve::Model;
    return {
        {"egarch11_std", variant_spec(Model::EGarch, Dist::Std), oracle_process("egarch11_std")},
        {"gjrgarch11_norm", variant_spec(Model::GjrGarch, Dist::Norm), oracle_process("gjrgarch11_norm")},
        {"aparch11_ged", variant_spec(Model::ApArch, Dist::Ged), oracle_process("aparch11_ged")},
        {"igarch11_norm", variant_spec(Model::IGarch, Dist::Norm), oracle_process("igarch11_norm")},
        {"csgarch11_std", variant_spec(Model::CsGarch, Dist::Std), oracle_process("csgarch11_std")},
        {"fgarch_tgarch11_norm", variant_spec(Model::FGarchTGarch, Dist::Norm), tgarch},
        {"fgarch_allgarch11_norm", variant_spec(Model::FGarchAllGarch, Dist::Norm),
         oracle_process("fgarch_allgarch11_norm")},
    };
}

const std::vector<NamedSpec>& named_specs()
{
    static const std::vector<NamedSpec> specs = [] {
        std::vector<NamedSpec> all{
            {"s11_norm", make_spec(1, 1, 0, 0, ve::Dist::Norm), std::nullopt},
            {"s21_std", make_spec(2, 1, 0, 0, ve::Dist::Std), std::nullopt},
            {"arma11_s11_ged", make_spec(1, 1, 1, 1, ve::Dist::Ged), std::nullopt},
            {"arma22_s33_std", make_spec(3, 3, 2, 2, ve::Dist::Std), std::nullopt},
        };
        for (NamedSpec& variant : variant_specs())
        {
            all.push_back(std::move(variant));
        }
        return all;
    }();
    return specs;
}

std::vector<double> series_of(const NamedSpec& named)
{
    if (named.process)
    {
        return ve::testing::simulate_variant(*named.process, 1, kObservations);
    }
    return ve::bench::simulate_series(named.spec, 1, kObservations);
}

void fill_block(std::vector<double>& theta, std::size_t begin, int count, double value)
{
    for (int i = 0; i < count; ++i)
    {
        theta[begin + static_cast<std::size_t>(i)] = value;
    }
}

// The parameters of the process the series was simulated from; csGARCH keeps rho in eta1 and phi in eta2.
void apply_process(std::vector<double>& theta, const ve::model::Layout& layout,
                   const ve::testing::VariantProcess& process)
{
    const bool cs = layout.model == ve::Model::CsGarch;
    theta[layout.omega()] = process.omega;
    fill_block(theta, layout.alpha_begin(), layout.arch, process.alpha);
    fill_block(theta, layout.beta_begin(), layout.garch, process.beta);
    fill_block(theta, layout.gamma_begin(), layout.gamma, process.gamma);
    fill_block(theta, layout.eta1_begin(), layout.eta1, cs ? process.rho : process.eta1);
    fill_block(theta, layout.eta2_begin(), layout.eta2, cs ? process.phi : process.eta2);
    if (layout.free_power)
    {
        theta[layout.power()] = process.power;
    }
    if (layout.has_shape())
    {
        theta[layout.shape()] = process.shape;
    }
}

// A point inside the stationary region, in the natural space of the layout.
std::vector<double> reference_theta(const ve::model::Layout& layout)
{
    std::vector<double> theta(layout.count(), 0.0);
    theta[layout.mu()] = 0.05;
    for (int i = 0; i < layout.ar; ++i)
    {
        theta[layout.ar_begin() + static_cast<std::size_t>(i)] = i == 0 ? 0.3 : 0.05;
    }
    for (int i = 0; i < layout.ma; ++i)
    {
        theta[layout.ma_begin() + static_cast<std::size_t>(i)] = i == 0 ? 0.2 : 0.05;
    }
    theta[layout.omega()] = 0.01;
    for (int i = 0; i < layout.arch; ++i)
    {
        theta[layout.alpha_begin() + static_cast<std::size_t>(i)] = 0.08 / layout.arch;
    }
    for (int i = 0; i < layout.garch; ++i)
    {
        theta[layout.beta_begin() + static_cast<std::size_t>(i)] = 0.9 / layout.garch;
    }
    if (layout.has_shape())
    {
        theta[layout.shape()] = ve::bench::innovation_shape(layout.dist);
    }
    return theta;
}

std::vector<double> reference_theta(const NamedSpec& named, const ve::model::Layout& layout)
{
    std::vector<double> theta = reference_theta(layout);
    if (named.process)
    {
        apply_process(theta, layout, *named.process);
    }
    return theta;
}

template <std::size_t N>
std::vector<ve::ad::Dual<N>> dual_seeds(const std::vector<double>& theta)
{
    std::vector<ve::ad::Dual<N>> seeds;
    for (std::size_t i = 0; i < theta.size(); ++i)
    {
        seeds.push_back(ve::ad::variable<N>(theta[i], i));
    }
    return seeds;
}

template <std::size_t N>
double dual_loglik(std::span<const double> r, std::span<const ve::ad::Dual<N>> seeds,
                   const ve::model::Layout& layout)
{
    const ve::model::LoglikOutputs<ve::ad::Dual<N>> none{};
    return ve::model::loglik<ve::ad::Dual<N>>(r, seeds, layout, ve::VarInit::Sample, none).value;
}

double double_loglik(std::span<const double> r, const std::vector<double>& theta,
                     const ve::model::Layout& layout)
{
    const ve::model::LoglikOutputs<double> none{};
    return ve::model::loglik<double>(r, theta, layout, ve::VarInit::Sample, none);
}

template <class Evaluate>
void time_evaluations(benchmark::State& state, std::size_t parameters, Evaluate evaluate)
{
    if (!std::isfinite(evaluate()))
    {
        state.SkipWithError("the log-likelihood at the reference point is not finite");
        return;
    }
    for (auto _ : state)
    {
        benchmark::DoNotOptimize(evaluate());
    }
    state.counters["parameters"] = static_cast<double>(parameters);
    state.counters["evaluations_per_s"] =
        benchmark::Counter(static_cast<double>(state.iterations()), benchmark::Counter::kIsRate);
}

// The Dual seeds are built once, outside the timed loop.
void run_loglik(benchmark::State& state, const NamedSpec& named, bool with_gradient)
{
    const std::vector<double> r = series_of(named);
    const ve::model::Layout layout = ve::model::layout_of(named.spec);
    const std::vector<double> theta = reference_theta(named, layout);
    const std::size_t parameters = layout.count();
    if (!with_gradient)
    {
        time_evaluations(state, parameters, [&] { return double_loglik(r, theta, layout); });
        return;
    }
    if (parameters <= 8)
    {
        const auto seeds = dual_seeds<8>(theta);
        time_evaluations(state, parameters, [&] { return dual_loglik<8>(r, seeds, layout); });
        return;
    }
    const auto seeds = dual_seeds<16>(theta);
    time_evaluations(state, parameters, [&] { return dual_loglik<16>(r, seeds, layout); });
}

struct StatusCounts
{
    std::size_t converged = 0;
    std::size_t max_iter = 0;
    std::size_t other = 0;

    void add(int32_t status)
    {
        if (status == VE_FIT_CONVERGED)
        {
            ++converged;
            return;
        }
        if (status == VE_FIT_MAX_ITER)
        {
            ++max_iter;
            return;
        }
        ++other;
    }
};

// The counters of a benchmark are the status counts of its last iteration; every iteration repeats the same
// fits.
void report_status(benchmark::State& state, const StatusCounts& counts)
{
    state.counters["converged"] = static_cast<double>(counts.converged);
    state.counters["max_iter"] = static_cast<double>(counts.max_iter);
    state.counters["other"] = static_cast<double>(counts.other);
}

void run_fit_one(benchmark::State& state, const NamedSpec& named)
{
    const std::vector<double> r = series_of(named);
    const ve::FitConfig config;
    StatusCounts counts;
    for (auto _ : state)
    {
        ve::FitResult result = ve::fit_one(r, named.spec, config);
        counts = StatusCounts{};
        counts.add(static_cast<int32_t>(result.status));
        if (counts.other != 0)
        {
            state.SkipWithError("the fit ended neither CONVERGED nor MAX_ITER");
            return;
        }
        benchmark::DoNotOptimize(result.loglik);
    }
    report_status(state, counts);
    state.counters["fits_per_s"] =
        benchmark::Counter(static_cast<double>(state.iterations()), benchmark::Counter::kIsRate);
}

template <class T, void (*Free)(T*)>
struct CDeleter
{
    void operator()(T* handle) const { Free(handle); }
};

template <class T, void (*Free)(T*)>
using Owned = std::unique_ptr<T, CDeleter<T, Free>>;

void require_ok(ve_status status, const char* call)
{
    if (status != VE_OK)
    {
        throw std::runtime_error(std::string(call) + " failed: " + ve_last_error());
    }
}

Owned<ve_spec, ve_spec_free> create_spec(const ve::Spec& spec)
{
    ve_spec* raw = nullptr;
    require_ok(ve_spec_create(static_cast<int32_t>(spec.model), &raw), "ve_spec_create");
    Owned<ve_spec, ve_spec_free> handle(raw);
    require_ok(ve_spec_set_order(raw, spec.p, spec.q), "ve_spec_set_order");
    require_ok(ve_spec_set_arma(raw, spec.ar, spec.ma), "ve_spec_set_arma");
    require_ok(ve_spec_set_dist(raw, static_cast<int32_t>(spec.dist)), "ve_spec_set_dist");
    return handle;
}

// One ve_fit_batch workload: a context with the requested threads, a borrowed panel and one job per (spec,
// series).
class BatchWorkload
{
public:
  BatchWorkload(const std::vector<ve::Spec>& specs, std::vector<double> data, std::size_t series, int threads)
      : data_(std::move(data))
  {
      ve_context* ctx = nullptr;
      require_ok(ve_context_create(&ctx), "ve_context_create");
      context_.reset(ctx);
      require_ok(ve_context_set_threads(ctx, threads), "ve_context_set_threads");
      describe(specs, series);
  }

    std::size_t jobs() const { return jobs_; }

    // The status of every job, in job order: spec-major, then series.
    std::vector<int32_t> run_statuses()
    {
        std::vector<ve_fit*> fits(jobs_, nullptr);
        require_ok(ve_fit_batch(context_.get(), batch_.get(), fits.data()), "ve_fit_batch");
        std::vector<int32_t> statuses;
        for (ve_fit* fit : fits)
        {
            int32_t status = VE_FIT_FAILED;
            require_ok(ve_fit_status(fit, &status), "ve_fit_status");
            statuses.push_back(status);
            ve_fit_free(fit);
        }
        return statuses;
    }

    StatusCounts run()
    {
        StatusCounts counts;
        for (const int32_t status : run_statuses())
        {
            counts.add(status);
        }
        return counts;
    }

private:
    void describe(const std::vector<ve::Spec>& specs, std::size_t series)
    {
        std::vector<Owned<ve_spec, ve_spec_free>> handles;
        std::vector<const ve_spec*> raw_specs;
        std::vector<std::size_t> job_spec;
        std::vector<std::size_t> job_series;
        for (std::size_t s = 0; s < specs.size(); ++s)
        {
            handles.push_back(create_spec(specs[s]));
            raw_specs.push_back(handles.back().get());
            for (std::size_t j = 0; j < series; ++j)
            {
                job_spec.push_back(s);
                job_series.push_back(j);
            }
        }
        jobs_ = job_spec.size();
        ve_panel* panel = nullptr;
        require_ok(ve_panel_create(data_.data(), data_.size() / series, series, &panel), "ve_panel_create");
        panel_.reset(panel);
        ve_batch* batch = nullptr;
        require_ok(ve_batch_create(&batch), "ve_batch_create");
        batch_.reset(batch);
        require_ok(ve_batch_set_specs(batch, raw_specs.data(), raw_specs.size()), "ve_batch_set_specs");
        require_ok(ve_batch_set_panel(batch, panel), "ve_batch_set_panel");
        require_ok(ve_batch_set_jobs(batch, job_spec.data(), job_series.data(), jobs_), "ve_batch_set_jobs");
    }

    std::vector<double> data_;
    std::size_t jobs_ = 0;
    Owned<ve_context, ve_context_free> context_{nullptr};
    Owned<ve_panel, ve_panel_free> panel_{nullptr};
    Owned<ve_batch, ve_batch_free> batch_{nullptr};
};

void report_counts(benchmark::State& state, const StatusCounts& counts, std::size_t jobs)
{
    state.counters["jobs"] = static_cast<double>(jobs);
    report_status(state, counts);
    state.counters["fits_per_s"] =
        benchmark::Counter(static_cast<double>(jobs * state.iterations()), benchmark::Counter::kIsRate);
}

void run_batch(benchmark::State& state, const std::vector<ve::Spec>& specs, const ve::Spec& generator,
               int threads)
{
    try
    {
        const std::vector<double> data =
            ve::bench::simulate_panel(ve::bench::simulated_model_for(generator), kObservations, kBatchSeries);
        BatchWorkload workload(specs, data, kBatchSeries, threads);
        if (workload.jobs() == 0)
        {
            state.SkipWithError("the grid produced no jobs");
            return;
        }
        StatusCounts counts;
        for (auto _ : state)
        {
            counts = workload.run();
            if (counts.other != 0)
            {
                state.SkipWithError("a fit ended neither CONVERGED nor MAX_ITER");
                return;
            }
        }
        report_counts(state, counts, workload.jobs());
    }
    catch (const std::exception& error)
    {
        state.SkipWithError(error.what());
    }
}

void append_orders(std::vector<ve::Spec>& grid, ve::Dist dist, int arma)
{
    for (int p = 1; p <= 3; ++p)
    {
        for (int q = 1; q <= 3; ++q)
        {
            grid.push_back(make_spec(p, q, arma, arma, dist));
        }
    }
}

std::vector<ve::Spec> tcc_grid()
{
    std::vector<ve::Spec> grid;
    for (const ve::Dist dist : {ve::Dist::Norm, ve::Dist::Std, ve::Dist::Ged})
    {
        for (const int arma : {0, 1})
        {
            append_orders(grid, dist, arma);
        }
    }
    return grid;
}

constexpr std::size_t kModelCount = 14;
constexpr double kMaxFailureShare = 0.05;
constexpr std::array<const char*, kModelCount> kModelNames{
    "sgarch",       "egarch",        "gjrgarch",      "aparch",         "igarch",
    "csgarch",      "fgarch_garch",  "fgarch_tgarch", "fgarch_avgarch", "fgarch_ngarch",
    "fgarch_nagarch", "fgarch_aparch", "fgarch_gjrgarch", "fgarch_allgarch"};

using ModelCounts = std::array<StatusCounts, kModelCount>;

// Every model over the orders, ARMA and distributions of the TCC grid.
std::vector<ve::Spec> all_models_grid()
{
    std::vector<ve::Spec> grid;
    for (std::size_t model = 0; model < kModelCount; ++model)
    {
        for (ve::Spec spec : tcc_grid())
        {
            spec.model = static_cast<ve::Model>(model);
            grid.push_back(spec);
        }
    }
    return grid;
}

ModelCounts counts_by_model(const std::vector<ve::Spec>& specs, const std::vector<int32_t>& statuses)
{
    ModelCounts by_model{};
    const std::size_t per_spec = statuses.size() / specs.size();
    for (std::size_t job = 0; job < statuses.size(); ++job)
    {
        by_model[static_cast<std::size_t>(specs[job / per_spec].model)].add(statuses[job]);
    }
    return by_model;
}

// The names of the models over the failure limit; empty when every model is within it.
std::string models_over_limit(const ModelCounts& by_model)
{
    std::string over;
    for (std::size_t model = 0; model < kModelCount; ++model)
    {
        const StatusCounts& counts = by_model[model];
        const std::size_t fits = counts.converged + counts.max_iter + counts.other;
        if (static_cast<double>(counts.other) > kMaxFailureShare * static_cast<double>(fits))
        {
            over += std::string(over.empty() ? "" : ", ") + kModelNames[model];
        }
    }
    return over;
}

void report_models(benchmark::State& state, const ModelCounts& by_model)
{
    StatusCounts total;
    for (std::size_t model = 0; model < kModelCount; ++model)
    {
        const StatusCounts& counts = by_model[model];
        const std::string name = kModelNames[model];
        state.counters[name + "_ok"] = static_cast<double>(counts.converged + counts.max_iter);
        state.counters[name + "_other"] = static_cast<double>(counts.other);
        total.converged += counts.converged;
        total.max_iter += counts.max_iter;
        total.other += counts.other;
    }
    report_status(state, total);
}

// One call with every model of the grid; fails loudly if a model has more than 5% of its fits outside
// CONVERGED and MAX_ITER.
void run_all_models(benchmark::State& state, const ve::Spec& generator)
{
    try
    {
        const std::vector<ve::Spec> specs = all_models_grid();
        const std::vector<double> data =
            ve::bench::simulate_panel(ve::bench::simulated_model_for(generator), kObservations, kBatchSeries);
        BatchWorkload workload(specs, data, kBatchSeries, 0);
        if (workload.jobs() == 0)
        {
            state.SkipWithError("the grid produced no jobs");
            return;
        }
        ModelCounts by_model{};
        for (auto _ : state)
        {
            by_model = counts_by_model(specs, workload.run_statuses());
        }
        const std::string over = models_over_limit(by_model);
        if (!over.empty())
        {
            state.SkipWithError(("more than 5% of the fits ended neither CONVERGED nor MAX_ITER: " + over).c_str());
            return;
        }
        report_models(state, by_model);
        state.counters["jobs"] = static_cast<double>(workload.jobs());
        state.counters["fits_per_s"] =
            benchmark::Counter(static_cast<double>(workload.jobs() * state.iterations()), benchmark::Counter::kIsRate);
    }
    catch (const std::exception& error)
    {
        state.SkipWithError(error.what());
    }
}

int hardware_threads()
{
    return static_cast<int>(std::max(1U, std::thread::hardware_concurrency()));
}

// threads == 0 is the context's "hardware concurrency" setting.
std::string thread_label(int threads)
{
    if (threads == 0)
    {
        return "hw";
    }
    return std::to_string(threads);
}

void register_all()
{
    for (const NamedSpec& named : named_specs())
    {
        benchmark::RegisterBenchmark("BM_Loglik/" + named.name + "/double", run_loglik, named, false);
        benchmark::RegisterBenchmark("BM_Loglik/" + named.name + "/dual", run_loglik, named, true);
        benchmark::RegisterBenchmark("BM_FitOne/" + named.name, run_fit_one, named)
            ->Unit(benchmark::kMillisecond);
    }
    static const std::vector<ve::Spec> single{named_specs().front().spec};
    for (const int threads : {1, 2, 4, 8, 0})
    {
        const std::string name = "BM_Batch/threads:" + thread_label(threads);
        benchmark::RegisterBenchmark(name, run_batch, single, single.front(), threads)
            ->Unit(benchmark::kMillisecond)
            ->UseRealTime();
    }
    // Student-t innovations (nu = 8), like returns: on normal data the std fits run nu towards infinity,
    // which is not the cost being measured.
    static const std::vector<ve::Spec> grid = tcc_grid();
    static const ve::Spec heavy_tails = make_spec(1, 1, 0, 0, ve::Dist::Std);
    for (const int threads : {1, 0})
    {
        const std::string name = "BM_TccGrid/threads:" + thread_label(threads);
        benchmark::RegisterBenchmark(name, run_batch, grid, heavy_tails, threads)
            ->Unit(benchmark::kSecond)
            ->UseRealTime()
            ->Iterations(1);
    }
    benchmark::RegisterBenchmark("BM_TccGridAllModels/threads:hw", run_all_models, heavy_tails)
        ->Unit(benchmark::kSecond)
        ->UseRealTime()
        ->Iterations(1);
}

}

int main(int argc, char** argv)
{
    register_all();
    benchmark::AddCustomContext("hardware_threads", std::to_string(hardware_threads()));
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
