#include "dist/bessel.hpp"
#include "dcc/fit.hpp"
#include "dcc/standard_errors.hpp"
#include "diagnostics/gof.hpp"
#include "diagnostics/portmanteau.hpp"
#include "diagnostics/sign_bias.hpp"
#include "fit/fit.hpp"
#include "forecast/forecast.hpp"
#include "model/layout.hpp"
#include "model/likelihood.hpp"
#include "sim/philox.hpp"
#include "sim/samplers.hpp"
#include "sim/simulate.hpp"
#include "sim/start.hpp"
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

// The skewed series of the unit tests, each fitted under its own distribution; the ghyp case fits the nig
// series, since the C++ helper does not simulate ghyp.
std::vector<NamedSpec> skewed_specs()
{
    std::vector<NamedSpec> specs;
    for (const ve::testing::SkewedFit& fit : ve::testing::skewed_fits())
    {
        const ve::testing::VariantProcess process = ve::testing::skewed_processes()[fit.series];
        const std::string name = fit.dist == ve::Dist::Ghyp ? "sgarch11_ghyp" : fit.name;
        specs.push_back({name, variant_spec(process.model, fit.dist), process});
    }
    return specs;
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
        for (NamedSpec& skewed : skewed_specs())
        {
            all.push_back(std::move(skewed));
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
    if (layout.has_skew())
    {
        theta[layout.skew()] = process.skew;
    }
    if (layout.has_shape())
    {
        theta[layout.shape()] = process.shape;
    }
    if (layout.has_ghlambda())
    {
        theta[layout.ghlambda()] = -0.5;
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

// The regressor series of the unit tests, each fitted under the model that generated it.
struct RegressorCase
{
    std::string name;
    ve::Spec spec;
    ve::testing::RegressorProcess process;
};

const std::vector<RegressorCase>& regressor_cases()
{
    static const std::vector<RegressorCase> cases = [] {
        std::vector<RegressorCase> all;
        for (const ve::testing::RegressorProcess& process : ve::testing::regressor_processes())
        {
            ve::Spec spec = variant_spec(process.variance.model, process.variance.dist);
            spec.ar = process.ar != 0.0 ? 1 : 0;
            spec.mxreg = static_cast<int>(process.delta.size());
            spec.vxreg = process.has_vxreg ? 1 : 0;
            spec.archm = process.archm;
            all.push_back({process.name, spec, process});
        }
        return all;
    }();
    return cases;
}

ve::model::Data regressor_data(const ve::testing::RegressorSeries& series)
{
    const ve::model::RegressorMatrix mean{series.x, series.x.size() / kObservations};
    const ve::model::RegressorMatrix variance{series.v, series.v.size() / kObservations};
    return ve::model::Data{series.r, mean, variance};
}

// The parameters of the process the series was simulated from.
std::vector<double> regressor_theta(const RegressorCase& item, const ve::model::Layout& layout)
{
    std::vector<double> theta = reference_theta(layout);
    apply_process(theta, layout, item.process.variance);
    theta[layout.mu()] = item.process.mu;
    if (layout.ar > 0)
    {
        theta[layout.ar_begin()] = item.process.ar;
    }
    if (layout.has_archm())
    {
        theta[layout.archm_index()] = item.process.xi;
    }
    for (std::size_t k = 0; k < item.process.delta.size(); ++k)
    {
        theta[layout.mxreg_begin() + k] = item.process.delta[k];
    }
    if (layout.vxreg > 0)
    {
        theta[layout.vxreg_begin()] = item.process.zeta;
    }
    return theta;
}

void run_regressor_loglik(benchmark::State& state, const RegressorCase& item, bool with_gradient)
{
    const ve::testing::RegressorSeries series = ve::testing::simulate_regressor_process(item.process, 1, kObservations);
    const ve::model::Data data = regressor_data(series);
    const ve::model::Layout layout = ve::model::layout_of(item.spec);
    const std::vector<double> theta = regressor_theta(item, layout);
    if (!with_gradient)
    {
        const ve::model::LoglikOutputs<double> none{};
        time_evaluations(state, layout.count(),
                         [&] { return ve::model::loglik<double>(data, theta, layout, ve::VarInit::Sample, none); });
        return;
    }
    const auto seeds = dual_seeds<8>(theta);
    const ve::model::LoglikOutputs<ve::ad::Dual<8>> none{};
    time_evaluations(state, layout.count(), [&] {
        const std::span<const ve::ad::Dual<8>> parameters(seeds);
        return ve::model::loglik<ve::ad::Dual<8>>(data, parameters, layout, ve::VarInit::Sample, none).value;
    });
}

void run_regressor_fit_one(benchmark::State& state, const RegressorCase& item)
{
    const ve::testing::RegressorSeries series = ve::testing::simulate_regressor_process(item.process, 1, kObservations);
    const ve::model::Data data = regressor_data(series);
    const ve::FitConfig config;
    StatusCounts counts;
    for (auto _ : state)
    {
        ve::FitResult result = ve::fit_one(data, item.spec, config);
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

constexpr std::array<ve::Dist, 3> kSymmetricDists{ve::Dist::Norm, ve::Dist::Std, ve::Dist::Ged};
constexpr std::array<ve::Dist, 8> kAllDists{ve::Dist::Norm, ve::Dist::Std,  ve::Dist::Ged, ve::Dist::Sstd,
                                            ve::Dist::Sged, ve::Dist::Nig, ve::Dist::Ghyp, ve::Dist::Jsu};

std::vector<ve::Spec> tcc_grid(std::span<const ve::Dist> dists = kSymmetricDists)
{
    std::vector<ve::Spec> grid;
    for (const ve::Dist dist : dists)
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
std::vector<ve::Spec> all_models_grid(std::span<const ve::Dist> dists = kSymmetricDists)
{
    std::vector<ve::Spec> grid;
    for (std::size_t model = 0; model < kModelCount; ++model)
    {
        for (ve::Spec spec : tcc_grid(dists))
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

constexpr std::size_t kDistCount = 8;
constexpr std::array<const char*, kDistCount> kDistNames{"norm", "std", "ged", "sstd", "sged", "nig", "ghyp", "jsu"};

using DistCounts = std::array<StatusCounts, kDistCount>;

DistCounts counts_by_dist(const std::vector<ve::Spec>& specs, const std::vector<int32_t>& statuses)
{
    DistCounts by_dist{};
    const std::size_t per_spec = statuses.size() / specs.size();
    for (std::size_t job = 0; job < statuses.size(); ++job)
    {
        by_dist[static_cast<std::size_t>(specs[job / per_spec].dist)].add(statuses[job]);
    }
    return by_dist;
}

std::string dists_over_limit(const DistCounts& by_dist)
{
    std::string over;
    for (std::size_t dist = 0; dist < kDistCount; ++dist)
    {
        const StatusCounts& counts = by_dist[dist];
        const std::size_t fits = counts.converged + counts.max_iter + counts.other;
        if (static_cast<double>(counts.other) > kMaxFailureShare * static_cast<double>(fits))
        {
            over += std::string(over.empty() ? "" : ", ") + kDistNames[dist];
        }
    }
    return over;
}

void report_dists(benchmark::State& state, const DistCounts& by_dist)
{
    StatusCounts total;
    for (std::size_t dist = 0; dist < kDistCount; ++dist)
    {
        const StatusCounts& counts = by_dist[dist];
        const std::string name = kDistNames[dist];
        state.counters[name + "_ok"] = static_cast<double>(counts.converged + counts.max_iter);
        state.counters[name + "_other"] = static_cast<double>(counts.other);
        total.converged += counts.converged;
        total.max_iter += counts.max_iter;
        total.other += counts.other;
    }
    report_status(state, total);
}

// The whole univariate TCC workload in one call: 2016 specs x 20 series = 40320 fits. Fails loudly if a
// distribution has more than 5% of its fits outside CONVERGED and MAX_ITER.
void run_full_grid(benchmark::State& state, const ve::Spec& generator)
{
    try
    {
        const std::vector<ve::Spec> specs = all_models_grid(kAllDists);
        const std::vector<double> data =
            ve::bench::simulate_panel(ve::bench::simulated_model_for(generator), kObservations, kBatchSeries);
        BatchWorkload workload(specs, data, kBatchSeries, 0);
        if (workload.jobs() == 0)
        {
            state.SkipWithError("the grid produced no jobs");
            return;
        }
        DistCounts by_dist{};
        for (auto _ : state)
        {
            by_dist = counts_by_dist(specs, workload.run_statuses());
        }
        const std::string over = dists_over_limit(by_dist);
        if (!over.empty())
        {
            state.SkipWithError(("more than 5% of the fits ended neither CONVERGED nor MAX_ITER: " + over).c_str());
            return;
        }
        report_dists(state, by_dist);
        state.counters["jobs"] = static_cast<double>(workload.jobs());
        state.counters["fits_per_s"] =
            benchmark::Counter(static_cast<double>(workload.jobs() * state.iterations()), benchmark::Counter::kIsRate);
    }
    catch (const std::exception& error)
    {
        state.SkipWithError(error.what());
    }
}

// The arguments of the Bessel calls of a nig fit at the true parameters: alpha * sqrt(delta^2 + (z - mu)^2) for
// z the nig series standardised by its sample mean and standard deviation.
std::vector<double> nig_bessel_arguments()
{
    const ve::testing::VariantProcess process = ve::testing::skewed_processes()[3];
    const std::vector<double> r = ve::testing::simulate_variant(process, 1, kObservations);
    const double count = static_cast<double>(r.size());
    double mean = 0.0;
    for (const double x : r)
    {
        mean += x / count;
    }
    double variance = 0.0;
    for (const double x : r)
    {
        variance += (x - mean) * (x - mean) / count;
    }
    const double rho = process.skew;
    const double zeta = process.shape;
    const double lambda = -0.5;
    const double k0 = ve::dist::log_bessel_k(lambda, zeta);
    const double ratio = std::exp(ve::dist::log_bessel_k(lambda + 1.0, zeta) - k0);
    const double quotient = std::exp(ve::dist::log_bessel_k(lambda + 2.0, zeta) - k0);
    const double delta = 1.0 / std::sqrt(ratio / zeta + rho * rho / (1.0 - rho * rho) * (quotient - ratio * ratio));
    const double alpha = zeta / (delta * std::sqrt(1.0 - rho * rho));
    const double mu = -rho * alpha * delta * delta * ratio / zeta;
    std::vector<double> arguments;
    for (const double x : r)
    {
        const double z = (x - mean) / std::sqrt(variance);
        arguments.push_back(alpha * std::sqrt(delta * delta + (z - mu) * (z - mu)));
    }
    return arguments;
}

using BesselDual = ve::ad::Dual<8>;

// fixed_order: the double-order overload at order -1 (nig, K_1). free_order: the two-Dual overload at the ghyp
// order lambda - 1/2 with lambda = 1.
void run_log_bessel_k(benchmark::State& state, bool free_order)
{
    const std::vector<double> arguments = nig_bessel_arguments();
    const double order = free_order ? 0.5 : -1.0;
    const auto evaluate = [&] {
        double sum = 0.0;
        for (const double x : arguments)
        {
            const BesselDual argument = ve::ad::variable<8>(x, 1);
            const BesselDual value = free_order
                                         ? ve::dist::log_bessel_k(ve::ad::variable<8>(order, 0), argument)
                                         : ve::dist::log_bessel_k(order, argument);
            sum += ve::ad::value_of(value);
        }
        return sum;
    };
    if (!std::isfinite(evaluate()))
    {
        state.SkipWithError("log K is not finite at the reference arguments");
        return;
    }
    for (auto _ : state)
    {
        benchmark::DoNotOptimize(evaluate());
    }
    state.counters["arguments"] = static_cast<double>(arguments.size());
    state.counters["calls_per_s"] = benchmark::Counter(
        static_cast<double>(arguments.size() * state.iterations()), benchmark::Counter::kIsRate);
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

// The ARFIMA series of the unit tests under its own model, at the parameters of the process: the O(T^2) fractional
// difference dominates.
struct ArfimaCase
{
    ve::Spec spec;
    ve::testing::ArfimaProcess process;
    std::vector<double> r;
    std::vector<double> theta;
};

const ArfimaCase& arfima_case()
{
    static const ArfimaCase item = [] {
        const ve::testing::ArfimaProcess process = ve::testing::arfima_processes().front();
        ve::Spec spec = variant_spec(process.variance.model, process.variance.dist);
        spec.arfima = true;
        const ve::model::Layout layout = ve::model::layout_of(spec);
        std::vector<double> theta = reference_theta(layout);
        apply_process(theta, layout, process.variance);
        theta[layout.mu()] = process.mu;
        theta[layout.arfima_index()] = process.d;
        return ArfimaCase{spec, process, ve::testing::simulate_arfima(process, 1, kObservations), theta};
    }();
    return item;
}

void run_arfima_loglik(benchmark::State& state, bool with_gradient)
{
    const ArfimaCase& item = arfima_case();
    const ve::model::Layout layout = ve::model::layout_of(item.spec);
    if (!with_gradient)
    {
        const ve::model::LoglikOutputs<double> none{};
        time_evaluations(state, layout.count(), [&] {
            return ve::model::loglik<double>(item.r, item.theta, layout, ve::VarInit::Sample, none);
        });
        return;
    }
    const auto seeds = dual_seeds<8>(item.theta);
    const ve::model::LoglikOutputs<ve::ad::Dual<8>> none{};
    time_evaluations(state, layout.count(), [&] {
        const std::span<const ve::ad::Dual<8>> parameters(seeds);
        return ve::model::loglik<ve::ad::Dual<8>>(item.r, parameters, layout, ve::VarInit::Sample, none).value;
    });
}

void run_arfima_fit_one(benchmark::State& state)
{
    const ArfimaCase& item = arfima_case();
    const ve::FitConfig config;
    StatusCounts counts;
    for (auto _ : state)
    {
        ve::FitResult result = ve::fit_one(item.r, item.spec, config);
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

// One forecast of 60 steps from a fit made once, outside the timing: the cost of forecast::forecast alone.
constexpr std::size_t kForecastHorizon = 60;

void time_forecast(benchmark::State& state, const ve::FitResult& fit)
{
    if (fit.state.residuals.empty())
    {
        state.SkipWithError("the fit has no finite point to forecast from");
        return;
    }
    for (auto _ : state)
    {
        ve::forecast::Forecast forecast = ve::forecast::forecast(fit, kForecastHorizon);
        benchmark::DoNotOptimize(forecast.sigma.data());
        benchmark::DoNotOptimize(forecast.mean.data());
    }
}

void run_forecast(benchmark::State& state, const NamedSpec& named)
{
    time_forecast(state, ve::fit_one(series_of(named), named.spec, ve::FitConfig{}));
}

void run_arfima_forecast(benchmark::State& state)
{
    const ArfimaCase& item = arfima_case();
    time_forecast(state, ve::fit_one(item.r, item.spec, ve::FitConfig{}));
}

void register_forecasts()
{
    for (const NamedSpec& named : named_specs())
    {
        const bool measured = named.name == "s11_norm" || named.name == "egarch11_std" ||
                              named.name == "csgarch11_std" || named.name == "fgarch_allgarch11_norm";
        if (!measured)
        {
            continue;
        }
        benchmark::RegisterBenchmark("BM_Forecast/" + named.name + "/h:60", run_forecast, named);
    }
    benchmark::RegisterBenchmark("BM_Forecast/arfima_sgarch11_norm/h:60", run_arfima_forecast);
}

// 10 000 paths x 60 steps from a fit made once, outside the timing: the cost of sim::simulate alone.
constexpr std::size_t kSimulatedPaths = 10000;
constexpr std::size_t kSimulatedSteps = 60;

void run_simulate(benchmark::State& state, const NamedSpec& named, int threads)
{
    const ve::FitResult fit = ve::fit_one(series_of(named), named.spec, ve::FitConfig{});
    if (fit.state.residuals.empty())
    {
        state.SkipWithError("the fit has no finite point to simulate from");
        return;
    }
    ve::Context context;
    context.set_threads(threads);
    const ve::sim::Start start = ve::sim::start_of_fit(fit);
    const ve::sim::Request request{kSimulatedSteps, kSimulatedPaths, 42, {}};
    for (auto _ : state)
    {
        ve::sim::Simulation simulation = ve::sim::simulate(context, start, request);
        benchmark::DoNotOptimize(simulation.returns.data());
    }
}

// 1e5 draws of one distribution, at the parameters of Samplers.StandardizedMoments.
constexpr int kSamplerDraws = 100000;

void run_sampler(benchmark::State& state, ve::Dist dist, ve::dist::ShapeParameters<double> parameters)
{
    const ve::sim::Sampler sampler(dist, parameters);
    ve::sim::Stream stream(1, 0);
    for (auto _ : state)
    {
        double sum = 0.0;
        for (int i = 0; i < kSamplerDraws; ++i)
        {
            sum += sampler.draw(stream);
        }
        benchmark::DoNotOptimize(sum);
    }
}

void register_simulations()
{
    for (const NamedSpec& named : named_specs())
    {
        const bool measured =
            named.name == "s11_norm" || named.name == "egarch11_std" || named.name == "sgarch11_ghyp";
        if (!measured)
        {
            continue;
        }
        for (const int threads : {1, 0})
        {
            const std::string name =
                "BM_Simulate/" + named.name + "/paths:10000/h:60/threads:" + thread_label(threads);
            benchmark::RegisterBenchmark(name, run_simulate, named, threads)
                ->Unit(benchmark::kMillisecond)
                ->UseRealTime();
        }
    }
    struct SamplerCase
    {
        const char* name;
        ve::Dist dist;
        ve::dist::ShapeParameters<double> parameters;
    };
    const std::array<SamplerCase, 8> samplers{{{"norm", ve::Dist::Norm, {}},
                                               {"std", ve::Dist::Std, {0.0, 8.0, 0.0}},
                                               {"ged", ve::Dist::Ged, {0.0, 1.3, 0.0}},
                                               {"sstd", ve::Dist::Sstd, {1.4, 8.0, 0.0}},
                                               {"sged", ve::Dist::Sged, {0.7, 1.5, 0.0}},
                                               {"nig", ve::Dist::Nig, {-0.4, 1.5, 0.0}},
                                               {"ghyp", ve::Dist::Ghyp, {0.3, 2.0, -1.5}},
                                               {"jsu", ve::Dist::Jsu, {0.8, 1.5, 0.0}}}};
    for (const SamplerCase& item : samplers)
    {
        benchmark::RegisterBenchmark(std::string("BM_Sampler/") + item.name + "/draws:100000", run_sampler, item.dist,
                                     item.parameters)
            ->Unit(benchmark::kMillisecond);
    }
}

ve::diagnostics::TestResult diagnostic_of(const ve::FitResult& fit, const std::string& test)
{
    if (test == "ljung_box")
    {
        return ve::diagnostics::weighted_ljung_box(fit.std_residuals, 5, 0, false);
    }
    if (test == "arch_lm")
    {
        return ve::diagnostics::weighted_arch_lm(fit.residuals, fit.sigma, 5, 2);
    }
    if (test == "sign_bias")
    {
        return ve::diagnostics::sign_bias(fit.std_residuals, fit.residuals).joint;
    }
    const ve::model::Layout layout = ve::model::layout_of(fit.spec);
    const auto parameters = ve::model::shape_parameters_of<double>(fit.estimate, layout);
    return ve::diagnostics::pearson_gof(fit.std_residuals, fit.spec.dist, parameters, 50);
}

void run_diagnostic(benchmark::State& state, const NamedSpec& named, const std::string& test)
{
    const ve::FitResult fit = ve::fit_one(series_of(named), named.spec, ve::FitConfig{});
    if (fit.std_residuals.empty())
    {
        state.SkipWithError("the fit has no finite point to diagnose");
        return;
    }
    const auto reference = diagnostic_of(fit, test);
    if (!std::isfinite(reference.statistic) || !std::isfinite(reference.p_value))
    {
        state.SkipWithError("the diagnostic is not finite");
        return;
    }
    for (auto _ : state)
    {
        auto result = diagnostic_of(fit, test);
        benchmark::DoNotOptimize(result.statistic);
        benchmark::DoNotOptimize(result.p_value);
    }
}

void register_diagnostics()
{
    for (const NamedSpec& named : named_specs())
    {
        if (named.name == "s11_norm")
        {
            for (const std::string test : {"ljung_box", "arch_lm", "sign_bias", "gof"})
            {
                benchmark::RegisterBenchmark("BM_Diagnostics/" + named.name + "/" + test,
                                             run_diagnostic, named, test);
            }
        }
        if (named.name == "sgarch11_ghyp")
        {
            benchmark::RegisterBenchmark("BM_Diagnostics/" + named.name + "/gof", run_diagnostic,
                                         named, std::string("gof"));
        }
    }
}

constexpr std::size_t kDccSeries = 21;

ve::dcc::DccSpec dcc_spec(ve::dcc::DccModel model, ve::dcc::DccDist dist)
{
    ve::dcc::DccSpec spec;
    spec.model = model;
    spec.dist = dist;
    if (model == ve::dcc::DccModel::FDcc)
    {
        for (std::size_t i = 0; i < kDccSeries; ++i)
        {
            spec.groups.push_back(static_cast<int>(i / 7 + 1));
        }
    }
    return spec;
}

std::vector<ve::dcc::DccSpec> dcc_specs()
{
    std::vector<ve::dcc::DccSpec> specs;
    for (const auto model : {ve::dcc::DccModel::Dcc, ve::dcc::DccModel::ADcc, ve::dcc::DccModel::FDcc})
    {
        for (const auto dist : {ve::dcc::DccDist::MvNorm, ve::dcc::DccDist::MvT, ve::dcc::DccDist::MvLaplace})
        {
            specs.push_back(dcc_spec(model, dist));
        }
    }
    return specs;
}

// Simulation and all 21 univariate fits are outside every timed loop, including repetitions.
const std::vector<ve::FitResult>& dcc_univariates()
{
    static const std::vector<ve::FitResult> fits = [] {
        const ve::Spec spec = make_spec(1, 1, 0, 0, ve::Dist::Std);
        std::vector<ve::FitResult> results;
        for (std::size_t i = 0; i < kDccSeries; ++i)
        {
            results.push_back(ve::fit_one(ve::bench::simulate_series(spec, i + 1, kObservations), spec, {}));
        }
        return results;
    }();
    return fits;
}

std::vector<const ve::FitResult*> dcc_fit_pointers()
{
    std::vector<const ve::FitResult*> pointers;
    for (const auto& fit : dcc_univariates())
    {
        pointers.push_back(&fit);
    }
    return pointers;
}

void run_dcc(benchmark::State& state, const std::vector<ve::dcc::DccSpec>& specs, int threads)
{
    const auto fits = dcc_fit_pointers();
    ve::Context context;
    context.set_threads(threads);
    ve::dcc::DccOptions options;
    options.standard_errors = false;
    if (const auto problem = ve::dcc::check_dcc_batch(fits, specs, options))
    {
        state.SkipWithError(problem->c_str());
        return;
    }
    StatusCounts counts;
    for (auto _ : state)
    {
        auto results = ve::dcc::fit_dcc_batch(context, fits, specs, options);
        counts = {};
        for (auto& result : results)
        {
            counts.add(static_cast<int32_t>(result.status));
            if (!std::isfinite(result.loglik))
            {
                state.SkipWithError("the DCC fit has no finite point");
                return;
            }
            benchmark::DoNotOptimize(result.loglik);
        }
    }
    report_status(state, counts);
}

void run_dcc_standard_errors(benchmark::State& state)
{
    const auto fits = dcc_fit_pointers();
    ve::Context context;
    context.set_threads(1);
    ve::dcc::DccOptions options;
    options.standard_errors = false;
    const std::vector<ve::dcc::DccSpec> specs{dcc_specs().front()};
    const auto fitted = ve::dcc::fit_dcc_batch(context, fits, specs, options);
    for (auto _ : state)
    {
        const auto result = ve::dcc::dcc_standard_errors(fitted.front(), fits);
        if (!result)
        {
            state.SkipWithError(result.error().c_str());
            return;
        }
        benchmark::DoNotOptimize(result->robust.data());
    }
}

void register_dcc()
{
    const auto specs = dcc_specs();
    const std::array<std::string, 3> models{"dcc", "adcc", "fdcc"};
    const std::array<std::string, 3> distributions{"mvnorm", "mvt", "mvlaplace"};
    for (std::size_t i = 0; i < specs.size(); ++i)
    {
        const std::string name = "BM_DccFit/" + models[i / 3] + "_" + distributions[i % 3] + "/m:21";
        benchmark::RegisterBenchmark(name, run_dcc, std::vector<ve::dcc::DccSpec>{specs[i]}, 1)
            ->Unit(benchmark::kSecond)
            ->UseRealTime()
            ->Iterations(1);
    }
    for (const int threads : {1, 0})
    {
        const std::string name = "BM_DccGrid/m:21/T:5000/threads:" + thread_label(threads);
        benchmark::RegisterBenchmark(name, run_dcc, specs, threads)
            ->Unit(benchmark::kSecond)
            ->UseRealTime()
            ->Iterations(1);
    }
    benchmark::RegisterBenchmark("BM_DccStandardErrors/dcc_mvnorm/m:21", run_dcc_standard_errors)
        ->Unit(benchmark::kSecond)
        ->UseRealTime()
        ->Iterations(1);
}

ve::dcc::DccSpec copula_spec(ve::dcc::DccModel model, ve::dcc::DccDist dist, ve::dcc::StaticMethod method,
                             ve::dcc::DccCopula copula)
{
    ve::dcc::DccSpec spec = dcc_spec(model, dist);
    spec.copula = copula;
    spec.static_method = method;
    return spec;
}

// The seven copula forms: static mvnorm (Kendall), static mvt Kendall and ML, then dcc and adcc with mvnorm and mvt.
std::vector<ve::dcc::DccSpec> copula_forms(ve::dcc::DccCopula copula)
{
    using ve::dcc::DccDist;
    using ve::dcc::DccModel;
    using ve::dcc::StaticMethod;
    std::vector<ve::dcc::DccSpec> specs{
        copula_spec(DccModel::Static, DccDist::MvNorm, StaticMethod::Default, copula),
        copula_spec(DccModel::Static, DccDist::MvT, StaticMethod::Default, copula),
        copula_spec(DccModel::Static, DccDist::MvT, StaticMethod::Ml, copula),
    };
    for (const auto model : {DccModel::Dcc, DccModel::ADcc})
    {
        for (const auto dist : {DccDist::MvNorm, DccDist::MvT})
        {
            specs.push_back(copula_spec(model, dist, StaticMethod::Default, copula));
        }
    }
    return specs;
}

void run_copula_standard_errors(benchmark::State& state)
{
    const auto fits = dcc_fit_pointers();
    ve::Context context;
    context.set_threads(1);
    ve::dcc::DccOptions options;
    options.standard_errors = false;
    const std::vector<ve::dcc::DccSpec> specs{copula_spec(
        ve::dcc::DccModel::Dcc, ve::dcc::DccDist::MvNorm, ve::dcc::StaticMethod::Default, ve::dcc::DccCopula::Parametric)};
    const auto fitted = ve::dcc::fit_dcc_batch(context, fits, specs, options);
    for (auto _ : state)
    {
        const auto result = ve::dcc::dcc_standard_errors(fitted.front(), fits);
        if (!result)
        {
            state.SkipWithError(result.error().c_str());
            return;
        }
        benchmark::DoNotOptimize(result->robust.data());
    }
}

void register_copula()
{
    const std::vector<ve::dcc::DccSpec> forms = copula_forms(ve::dcc::DccCopula::Parametric);
    const std::array<std::string, 7> names{"static_mvnorm_kendall", "static_mvt_kendall", "static_mvt_ml",
                                           "dcc_mvnorm", "dcc_mvt", "adcc_mvnorm", "adcc_mvt"};
    for (std::size_t i = 0; i < forms.size(); ++i)
    {
        const std::string name = "BM_CopulaFit/" + names[i] + "_param/m:21";
        benchmark::RegisterBenchmark(name, run_dcc, std::vector<ve::dcc::DccSpec>{forms[i]}, 1)
            ->Unit(benchmark::kSecond)
            ->UseRealTime()
            ->Iterations(1);
    }
    std::vector<ve::dcc::DccSpec> grid = forms;
    for (const auto& spec : copula_forms(ve::dcc::DccCopula::Empirical))
    {
        grid.push_back(spec);
    }
    for (const int threads : {1, 0})
    {
        const std::string name = "BM_CopulaGrid/m:21/T:5000/threads:" + thread_label(threads);
        benchmark::RegisterBenchmark(name, run_dcc, grid, threads)
            ->Unit(benchmark::kSecond)
            ->UseRealTime()
            ->Iterations(1);
    }
    benchmark::RegisterBenchmark("BM_CopulaStandardErrors/dcc_mvnorm_param/m:21", run_copula_standard_errors)
        ->Unit(benchmark::kSecond)
        ->UseRealTime()
        ->Iterations(1);
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
    for (const RegressorCase& item : regressor_cases())
    {
        benchmark::RegisterBenchmark("BM_Loglik/" + item.name + "/double", run_regressor_loglik, item, false);
        benchmark::RegisterBenchmark("BM_Loglik/" + item.name + "/dual", run_regressor_loglik, item, true);
        benchmark::RegisterBenchmark("BM_FitOne/" + item.name, run_regressor_fit_one, item)
            ->Unit(benchmark::kMillisecond);
    }
    benchmark::RegisterBenchmark("BM_Loglik/arfima_sgarch11_norm/double", run_arfima_loglik, false);
    benchmark::RegisterBenchmark("BM_Loglik/arfima_sgarch11_norm/dual", run_arfima_loglik, true);
    benchmark::RegisterBenchmark("BM_FitOne/arfima_sgarch11_norm", run_arfima_fit_one)->Unit(benchmark::kMillisecond);
    register_forecasts();
    register_simulations();
    register_diagnostics();
    register_dcc();
    register_copula();
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
    benchmark::RegisterBenchmark("BM_TccGridFull/threads:hw", run_full_grid, heavy_tails)
        ->Unit(benchmark::kSecond)
        ->UseRealTime()
        ->Iterations(1);
    benchmark::RegisterBenchmark("BM_LogBesselK/fixed_order", run_log_bessel_k, false);
    benchmark::RegisterBenchmark("BM_LogBesselK/free_order", run_log_bessel_k, true);
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
