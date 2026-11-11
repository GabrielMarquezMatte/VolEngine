#include "capi_scope.hpp"
#include "fit/fit.hpp"
#include "simulated_garch.hpp"

#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <string>
#include <thread>
#include <vector>
#include <volengine/volengine.h>

namespace
{

using ve::testing::BatchInputs;
using ve::testing::BatchPtr;
using ve::testing::FitSet;
using ve::testing::PanelPtr;
using ve::testing::Snapshot;
using ve::testing::SpecPtr;

constexpr std::size_t kT = 3000;
constexpr std::size_t kSeries = 3;
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

const std::vector<double>& matrix()
{
    static const std::vector<double> data = ve::testing::simulate_garch_matrix(kT, kSeries);
    return data;
}

std::vector<double> column(
    const std::vector<double>& data, std::size_t series, std::size_t first, std::size_t length)
{
    std::vector<double> values(length);
    for (std::size_t row = 0; row < length; ++row)
    {
        values[row] = data[(first + row) * kSeries + series];
    }
    return values;
}

Snapshot direct_fit(const std::vector<double>& series, bool include_mean, const ve::FitConfig& config)
{
    ve::Spec spec;
    spec.include_mean = include_mean;
    const ve::FitResult result = ve::fit_one(series, spec, config);
    Snapshot snapshot;
    snapshot.status = static_cast<int32_t>(result.status);
    snapshot.nobs = result.nobs;
    snapshot.estimate = result.estimate;
    snapshot.loglik = result.loglik;
    snapshot.sigma = result.sigma;
    return snapshot;
}

std::vector<std::size_t> repeat_each(std::size_t value_count, std::size_t times)
{
    std::vector<std::size_t> values;
    for (std::size_t value = 0; value < value_count; ++value)
    {
        values.insert(values.end(), times, value);
    }
    return values;
}

std::vector<std::size_t> cycle(std::size_t value_count, std::size_t rounds)
{
    std::vector<std::size_t> values;
    for (std::size_t round = 0; round < rounds; ++round)
    {
        for (std::size_t value = 0; value < value_count; ++value)
        {
            values.push_back(value);
        }
    }
    return values;
}

// One sGARCH with mean fit per series of the panel.
std::vector<Snapshot> fit_each_series(const ve_panel* panel)
{
    ve::testing::ContextPtr context = ve::testing::make_context(2);
    SpecPtr spec = ve::testing::make_spec(1);
    BatchInputs inputs;
    inputs.specs = {spec.get()};
    inputs.panel = panel;
    inputs.job_spec = repeat_each(1, kSeries);
    inputs.job_series = cycle(kSeries, 1);
    BatchPtr batch = ve::testing::build_batch(inputs);
    FitSet fits = ve::testing::fit_all(context.get(), batch.get());
    std::vector<Snapshot> snapshots;
    for (std::size_t job = 0; job < fits.size(); ++job)
    {
        snapshots.push_back(ve::testing::read_snapshot(fits[job]));
    }
    return snapshots;
}

struct Rectangular
{
    ve::testing::ContextPtr context = ve::testing::make_context(2);
    SpecPtr spec = ve::testing::make_spec(1);
    PanelPtr panel;
    BatchPtr batch;

    explicit Rectangular(const std::vector<double>& data)
    {
        panel = ve::testing::make_panel(data.data(), kT, kSeries);
        BatchInputs inputs;
        inputs.specs = {spec.get()};
        inputs.panel = panel.get();
        inputs.job_spec = repeat_each(1, kSeries);
        inputs.job_series = cycle(kSeries, 1);
        batch = ve::testing::build_batch(inputs);
    }
};

std::vector<Snapshot> snapshots_of(const FitSet& fits)
{
    std::vector<Snapshot> snapshots;
    for (std::size_t job = 0; job < fits.size(); ++job)
    {
        snapshots.push_back(ve::testing::read_snapshot(fits[job]));
    }
    return snapshots;
}

struct SpecShape
{
    int32_t p;
    int32_t q;
    int32_t ar;
    int32_t ma;
    int32_t dist;
};

constexpr SpecShape kShapes[] = {{1, 1, 0, 0, VE_DIST_NORM}, {3, 3, 2, 2, VE_DIST_STD}, {2, 1, 1, 0, VE_DIST_GED}};

SpecPtr make_shaped_spec(const SpecShape& shape)
{
    ve_spec* raw = nullptr;
    EXPECT_EQ(ve_spec_create(VE_MODEL_SGARCH, &raw), VE_OK);
    SpecPtr spec(raw);
    EXPECT_EQ(ve_spec_set_order(raw, shape.p, shape.q), VE_OK);
    EXPECT_EQ(ve_spec_set_arma(raw, shape.ar, shape.ma), VE_OK);
    EXPECT_EQ(ve_spec_set_dist(raw, shape.dist), VE_OK);
    return spec;
}

Snapshot direct_shaped_fit(const std::vector<double>& series, const SpecShape& shape)
{
    ve::Spec spec;
    spec.p = shape.p;
    spec.q = shape.q;
    spec.ar = shape.ar;
    spec.ma = shape.ma;
    spec.dist = static_cast<ve::Dist>(shape.dist);
    const ve::FitResult result = ve::fit_one(series, spec, {});
    Snapshot snapshot;
    snapshot.status = static_cast<int32_t>(result.status);
    snapshot.nobs = result.nobs;
    snapshot.estimate = result.estimate;
    snapshot.loglik = result.loglik;
    snapshot.sigma = result.sigma;
    return snapshot;
}

FitSet mixed_fits(int32_t threads, const std::vector<SpecPtr>& specs, const ve_panel* panel)
{
    ve::testing::ContextPtr context = ve::testing::make_context(threads);
    BatchInputs inputs;
    for (const SpecPtr& spec : specs)
    {
        inputs.specs.push_back(spec.get());
    }
    inputs.panel = panel;
    inputs.job_spec = repeat_each(specs.size(), kSeries);
    inputs.job_series = cycle(kSeries, specs.size());
    BatchPtr batch = ve::testing::build_batch(inputs);
    return ve::testing::fit_all(context.get(), batch.get());
}

constexpr std::size_t kSpecCopies = 11;
constexpr std::size_t kThirtyThreeJobs = kSpecCopies * kSeries;

// More jobs than the blocks of four threads (8 per thread), so one and four threads split them differently.
struct ThirtyThreeJobs
{
    SpecPtr with_mean = ve::testing::make_spec(1);
    SpecPtr without_mean = ve::testing::make_spec(0);
    PanelPtr panel = ve::testing::make_panel(matrix().data(), kT, kSeries);
    BatchPtr batch;

    ThirtyThreeJobs()
    {
        BatchInputs inputs;
        for (std::size_t copy = 0; copy < kSpecCopies; ++copy)
        {
            const ve_spec* spec = with_mean.get();
            if (copy % 2 == 1)
            {
                spec = without_mean.get();
            }
            inputs.specs.push_back(spec);
        }
        inputs.panel = panel.get();
        inputs.job_spec = repeat_each(kSpecCopies, kSeries);
        inputs.job_series = cycle(kSeries, kSpecCopies);
        batch = ve::testing::build_batch(inputs);
    }
};

std::string last_error()
{
    return ve_last_error();
}

bool contains(const std::string& text, const std::string& part)
{
    return text.find(part) != std::string::npos;
}

}

TEST(CapiFit, CartesianBatchMatchesFitOne)
{
    ve::testing::ContextPtr context = ve::testing::make_context(3);
    SpecPtr with_mean = ve::testing::make_spec(1);
    SpecPtr without_mean = ve::testing::make_spec(0);
    PanelPtr panel = ve::testing::make_panel(matrix().data(), kT, kSeries);
    BatchInputs inputs;
    inputs.specs = {with_mean.get(), without_mean.get()};
    inputs.panel = panel.get();
    inputs.job_spec = repeat_each(2, kSeries);
    inputs.job_series = cycle(kSeries, 2);
    BatchPtr batch = ve::testing::build_batch(inputs);
    FitSet fits = ve::testing::fit_all(context.get(), batch.get());
    ASSERT_EQ(fits.size(), 6u);
    for (std::size_t job = 0; job < fits.size(); ++job)
    {
        const bool include_mean = inputs.job_spec[job] == 0;
        const std::vector<double> series = column(matrix(), inputs.job_series[job], 0, kT);
        const Snapshot expected = direct_fit(series, include_mean, {});
        EXPECT_EQ(expected.status, VE_FIT_CONVERGED) << "job " << job;
        EXPECT_EQ(ve::testing::read_snapshot(fits[job]), expected) << "job " << job;
    }
}

TEST(CapiFit, MixedBucketsInOneBatch)
{
    std::vector<SpecPtr> specs;
    for (const SpecShape& shape : kShapes)
    {
        specs.push_back(make_shaped_spec(shape));
    }
    PanelPtr panel = ve::testing::make_panel(matrix().data(), kT, kSeries);
    for (const int32_t threads : {1, 4})
    {
        FitSet fits = mixed_fits(threads, specs, panel.get());
        ASSERT_EQ(fits.size(), specs.size() * kSeries);
        for (std::size_t job = 0; job < fits.size(); ++job)
        {
            const std::vector<double> series = column(matrix(), job % kSeries, 0, kT);
            const Snapshot expected = direct_shaped_fit(series, kShapes[job / kSeries]);
            EXPECT_EQ(ve::testing::read_snapshot(fits[job]), expected) << "threads " << threads << " job " << job;
        }
    }
}

TEST(CapiFit, GeneratedParamNames)
{
    const SpecShape shape{2, 1, 1, 1, VE_DIST_STD};
    std::vector<SpecPtr> specs;
    specs.push_back(make_shaped_spec(shape));
    PanelPtr panel = ve::testing::make_panel(matrix().data(), kT, kSeries);
    FitSet fits = mixed_fits(1, specs, panel.get());
    const std::vector<std::string> expected = {
        "mu", "ar1", "ma1", "omega", "alpha1", "alpha2", "beta1", "shape"};
    std::size_t count = 0;
    ASSERT_EQ(ve_fit_param_count(fits[0], &count), VE_OK);
    ASSERT_EQ(count, expected.size());
    for (std::size_t i = 0; i < count; ++i)
    {
        const char* name = nullptr;
        ASSERT_EQ(ve_fit_param_name(fits[0], i, &name), VE_OK);
        EXPECT_EQ(std::string(name), expected[i]);
    }
}

TEST(CapiFit, ResultIndependentOfThreadCount)
{
    ThirtyThreeJobs scenario;
    ve::testing::ContextPtr single = ve::testing::make_context(1);
    ve::testing::ContextPtr many = ve::testing::make_context(4);
    const std::vector<Snapshot> one_thread =
        snapshots_of(ve::testing::fit_all(single.get(), scenario.batch.get()));
    const std::vector<Snapshot> four_threads =
        snapshots_of(ve::testing::fit_all(many.get(), scenario.batch.get()));
    ASSERT_EQ(one_thread.size(), kThirtyThreeJobs);
    EXPECT_EQ(one_thread, four_threads);
}

TEST(CapiFit, SetThreadsDuringFitBatchKeepsResults)
{
    ThirtyThreeJobs scenario;
    ve::testing::ContextPtr context = ve::testing::make_context(2);
    const std::vector<Snapshot> reference =
        snapshots_of(ve::testing::fit_all(context.get(), scenario.batch.get()));
    std::atomic<bool> fitting = true;
    std::atomic<int> toggles = 0;
    std::atomic<int> failed_set_threads = 0;
    std::thread toggler(
        [&]
        {
            for (int32_t round = 0; fitting; ++round)
            {
                const int32_t threads = round % 2 == 0 ? 1 : 4;
                failed_set_threads += ve_context_set_threads(context.get(), threads) != VE_OK;
                ++toggles;
            }
        });
    std::vector<std::vector<Snapshot>> runs;
    for (int run = 0; run < 6; ++run)
    {
        runs.push_back(snapshots_of(ve::testing::fit_all(context.get(), scenario.batch.get())));
    }
    fitting = false;
    toggler.join();
    EXPECT_GT(toggles, 0);
    EXPECT_EQ(failed_set_threads, 0);
    for (const std::vector<Snapshot>& run : runs)
    {
        EXPECT_EQ(run, reference);
    }
}

TEST(CapiFit, FortranOrderAndNegativeStrideMatchContiguous)
{
    PanelPtr contiguous_panel = ve::testing::make_panel(matrix().data(), kT, kSeries);
    const std::vector<Snapshot> contiguous = fit_each_series(contiguous_panel.get());
    std::vector<double> fortran(kT * kSeries);
    std::vector<double> reversed(kT * kSeries);
    for (std::size_t row = 0; row < kT; ++row)
    {
        for (std::size_t series = 0; series < kSeries; ++series)
        {
            const double value = matrix()[row * kSeries + series];
            fortran[series * kT + row] = value;
            reversed[(kT - 1 - row) * kSeries + (kSeries - 1 - series)] = value;
        }
    }
    PanelPtr fortran_panel = ve::testing::make_panel(fortran.data(), kT, kSeries);
    ASSERT_EQ(ve_panel_set_strides(fortran_panel.get(), 1, static_cast<ptrdiff_t>(kT)), VE_OK);
    PanelPtr reversed_panel = ve::testing::make_panel(reversed.data() + (kT * kSeries - 1), kT, kSeries);
    ASSERT_EQ(ve_panel_set_strides(reversed_panel.get(), -static_cast<ptrdiff_t>(kSeries), -1), VE_OK);
    EXPECT_EQ(fit_each_series(fortran_panel.get()), contiguous);
    EXPECT_EQ(fit_each_series(reversed_panel.get()), contiguous);
}

TEST(CapiFit, RaggedPanel)
{
    const size_t starts[] = {0, 1000, 2995};
    const size_t lengths[] = {3000, 2000, 5};
    PanelPtr panel = ve::testing::make_panel(matrix().data(), kT, kSeries);
    ASSERT_EQ(ve_panel_set_ranges(panel.get(), starts, lengths), VE_OK);
    const std::vector<Snapshot> fitted = fit_each_series(panel.get());
    ASSERT_EQ(fitted.size(), 3u);
    EXPECT_EQ(fitted[0].nobs, 3000u);
    EXPECT_EQ(fitted[1].nobs, 2000u);
    EXPECT_EQ(fitted[2].nobs, 5u);
    EXPECT_EQ(fitted[0].status, VE_FIT_CONVERGED);
    EXPECT_EQ(fitted[1].status, VE_FIT_CONVERGED);
    EXPECT_EQ(fitted[2].status, VE_FIT_INSUFFICIENT_DATA);
    EXPECT_EQ(fitted[1], direct_fit(column(matrix(), 1, 1000, 2000), true, {}));
    EXPECT_TRUE(fitted[2].sigma.empty());
    ASSERT_EQ(fitted[2].estimate.size(), 4u);
    EXPECT_TRUE(std::isnan(fitted[2].estimate[0]));
}

TEST(CapiFit, ZeroLengthSeries)
{
    const size_t starts[] = {0, 0, 0};
    const size_t lengths[] = {3000, 0, 3000};
    PanelPtr panel = ve::testing::make_panel(matrix().data(), kT, kSeries);
    ASSERT_EQ(ve_panel_set_ranges(panel.get(), starts, lengths), VE_OK);
    const std::vector<Snapshot> fitted = fit_each_series(panel.get());
    ASSERT_EQ(fitted.size(), 3u);
    EXPECT_EQ(fitted[1].status, VE_FIT_INSUFFICIENT_DATA);
    EXPECT_EQ(fitted[1].nobs, 0u);
    EXPECT_EQ(fitted[0].status, VE_FIT_CONVERGED);
    EXPECT_EQ(fitted[2].status, VE_FIT_CONVERGED);
}

TEST(CapiFit, JobIndexOutOfRange)
{
    Rectangular scenario(matrix());
    BatchInputs inputs;
    inputs.specs = {scenario.spec.get()};
    inputs.panel = scenario.panel.get();
    inputs.job_spec = {0, 5};
    inputs.job_series = {0, 1};
    BatchPtr bad_spec = ve::testing::build_batch(inputs);
    ve_fit* outputs[2] = {reinterpret_cast<ve_fit*>(0x1), reinterpret_cast<ve_fit*>(0x1)};
    EXPECT_EQ(ve_fit_batch(scenario.context.get(), bad_spec.get(), outputs), VE_INVALID_ARGUMENT);
    EXPECT_EQ(outputs[0], nullptr);
    EXPECT_EQ(outputs[1], nullptr);
    EXPECT_TRUE(contains(last_error(), "job 1")) << last_error();
    EXPECT_TRUE(contains(last_error(), "spec")) << last_error();
    inputs.job_spec = {0, 0};
    inputs.job_series = {0, 3};
    BatchPtr bad_series = ve::testing::build_batch(inputs);
    outputs[0] = outputs[1] = reinterpret_cast<ve_fit*>(0x1);
    EXPECT_EQ(ve_fit_batch(scenario.context.get(), bad_series.get(), outputs), VE_INVALID_ARGUMENT);
    EXPECT_EQ(outputs[0], nullptr);
    EXPECT_EQ(outputs[1], nullptr);
    EXPECT_TRUE(contains(last_error(), "job 1")) << last_error();
    EXPECT_TRUE(contains(last_error(), "series")) << last_error();
}

TEST(CapiFit, IncompleteBatchRejected)
{
    Rectangular scenario(matrix());
    ve_fit* outputs[3] = {reinterpret_cast<ve_fit*>(0x1), nullptr, nullptr};
    BatchInputs no_panel;
    no_panel.specs = {scenario.spec.get()};
    no_panel.job_spec = {0};
    no_panel.job_series = {0};
    BatchPtr missing_panel = ve::testing::build_batch(no_panel);
    EXPECT_EQ(ve_fit_batch(scenario.context.get(), missing_panel.get(), outputs), VE_INVALID_ARGUMENT);
    EXPECT_TRUE(contains(last_error(), "panel")) << last_error();
    BatchInputs no_jobs;
    no_jobs.specs = {scenario.spec.get()};
    no_jobs.panel = scenario.panel.get();
    BatchPtr missing_jobs = ve::testing::build_batch(no_jobs);
    EXPECT_EQ(ve_fit_batch(scenario.context.get(), missing_jobs.get(), outputs), VE_INVALID_ARGUMENT);
    EXPECT_TRUE(contains(last_error(), "jobs")) << last_error();
    BatchInputs no_specs;
    no_specs.panel = scenario.panel.get();
    no_specs.job_spec = {0};
    no_specs.job_series = {0};
    BatchPtr missing_specs = ve::testing::build_batch(no_specs);
    EXPECT_EQ(ve_fit_batch(scenario.context.get(), missing_specs.get(), outputs), VE_INVALID_ARGUMENT);
    EXPECT_TRUE(contains(last_error(), "specs")) << last_error();
}

TEST(CapiFit, FitBatchRejectsBadHandlesAndOutput)
{
    Rectangular scenario(matrix());
    ve_fit* outputs[kSeries] = {};
    EXPECT_EQ(ve_fit_batch(nullptr, scenario.batch.get(), outputs), VE_INVALID_HANDLE);
    EXPECT_EQ(ve_fit_batch(scenario.context.get(), nullptr, outputs), VE_INVALID_HANDLE);
    EXPECT_EQ(ve_fit_batch(scenario.context.get(), scenario.batch.get(), nullptr), VE_INVALID_ARGUMENT);
    EXPECT_TRUE(contains(last_error(), "fits_out")) << last_error();
}

TEST(CapiFit, BatchCopiesSpecsAndOptions)
{
    ve::testing::ContextPtr context = ve::testing::make_context(2);
    SpecPtr spec = ve::testing::make_spec(1);
    ve_fit_options* options = nullptr;
    ASSERT_EQ(ve_fit_options_create(&options), VE_OK);
    ASSERT_EQ(ve_fit_options_set_max_iter(options, 3), VE_OK);
    PanelPtr panel = ve::testing::make_panel(matrix().data(), kT, kSeries);
    BatchInputs inputs;
    inputs.specs = {spec.get()};
    inputs.panel = panel.get();
    inputs.job_spec = {0};
    inputs.job_series = {0};
    inputs.options = options;
    BatchPtr batch = ve::testing::build_batch(inputs);
    ASSERT_EQ(ve_spec_set_include_mean(spec.get(), 0), VE_OK);
    ASSERT_EQ(ve_fit_options_set_max_iter(options, 500), VE_OK);
    ve_fit_options_free(options);
    FitSet fits = ve::testing::fit_all(context.get(), batch.get());
    ve::FitConfig config;
    config.max_iter = 3;
    const Snapshot expected = direct_fit(column(matrix(), 0, 0, kT), true, config);
    EXPECT_EQ(expected.status, VE_FIT_MAX_ITER);
    EXPECT_EQ(ve::testing::read_snapshot(fits[0]), expected);
}

TEST(CapiFit, NaNInsideUsedRangeRejected)
{
    std::vector<double> data = matrix();
    data[100 * kSeries + 1] = kNaN;
    Rectangular scenario(data);
    FitSet fits(kSeries);
    EXPECT_EQ(ve_fit_batch(scenario.context.get(), scenario.batch.get(), fits.out()), VE_INVALID_ARGUMENT);
    EXPECT_TRUE(contains(last_error(), "NaN or infinite value")) << last_error();
    EXPECT_TRUE(contains(last_error(), "series 1")) << last_error();
    EXPECT_TRUE(contains(last_error(), "row 100")) << last_error();
    EXPECT_EQ(fits[0], nullptr);
    data[100 * kSeries + 1] = std::numeric_limits<double>::infinity();
    Rectangular infinite(data);
    EXPECT_EQ(ve_fit_batch(infinite.context.get(), infinite.batch.get(), fits.out()), VE_INVALID_ARGUMENT);
    EXPECT_TRUE(contains(last_error(), "row 100")) << last_error();
}

TEST(CapiFit, NaNOutsideSeriesRangeOrUnusedSeriesAccepted)
{
    std::vector<double> data = matrix();
    data[500 * kSeries + 1] = kNaN;
    const size_t starts[] = {0, 1000, 2995};
    const size_t lengths[] = {3000, 2000, 5};
    ve::testing::ContextPtr context = ve::testing::make_context(2);
    SpecPtr spec = ve::testing::make_spec(1);
    PanelPtr panel = ve::testing::make_panel(data.data(), kT, kSeries);
    ASSERT_EQ(ve_panel_set_ranges(panel.get(), starts, lengths), VE_OK);
    BatchInputs inputs;
    inputs.specs = {spec.get()};
    inputs.panel = panel.get();
    inputs.job_spec = {0, 0, 0};
    inputs.job_series = {0, 1, 2};
    BatchPtr ragged = ve::testing::build_batch(inputs);
    FitSet fits = ve::testing::fit_all(context.get(), ragged.get());
    EXPECT_EQ(ve::testing::read_snapshot(fits[1]).status, VE_FIT_CONVERGED);
    PanelPtr full = ve::testing::make_panel(data.data(), kT, kSeries);
    inputs.panel = full.get();
    inputs.job_spec = {0};
    inputs.job_series = {0};
    BatchPtr unused = ve::testing::build_batch(inputs);
    FitSet only_first = ve::testing::fit_all(context.get(), unused.get());
    EXPECT_EQ(ve::testing::read_snapshot(only_first[0]).status, VE_FIT_CONVERGED);
}

TEST(CapiBatch, SettersValidateArguments)
{
    ve_batch* batch = nullptr;
    ASSERT_EQ(ve_batch_create(&batch), VE_OK);
    BatchPtr owner(batch);
    SpecPtr spec = ve::testing::make_spec(1);
    const ve_spec* const entries[] = {spec.get()};
    const ve_spec* const with_null[] = {spec.get(), nullptr};
    const size_t indices[] = {0};
    EXPECT_EQ(ve_batch_set_specs(batch, entries, 0), VE_INVALID_ARGUMENT);
    EXPECT_EQ(ve_batch_set_specs(batch, nullptr, 1), VE_INVALID_ARGUMENT);
    EXPECT_EQ(ve_batch_set_specs(batch, with_null, 2), VE_INVALID_HANDLE);
    EXPECT_EQ(ve_batch_set_specs(nullptr, entries, 1), VE_INVALID_HANDLE);
    EXPECT_EQ(ve_batch_set_jobs(batch, indices, indices, 0), VE_INVALID_ARGUMENT);
    EXPECT_EQ(ve_batch_set_jobs(batch, nullptr, indices, 1), VE_INVALID_ARGUMENT);
    EXPECT_EQ(ve_batch_set_jobs(batch, indices, nullptr, 1), VE_INVALID_ARGUMENT);
    EXPECT_EQ(ve_batch_set_jobs(nullptr, indices, indices, 1), VE_INVALID_HANDLE);
    EXPECT_EQ(ve_batch_set_panel(batch, nullptr), VE_INVALID_HANDLE);
    EXPECT_EQ(ve_batch_set_options(batch, nullptr), VE_INVALID_HANDLE);
    size_t count = 7;
    EXPECT_EQ(ve_batch_job_count(batch, &count), VE_OK);
    EXPECT_EQ(count, 0u);
    EXPECT_EQ(ve_batch_job_count(batch, nullptr), VE_INVALID_ARGUMENT);
    EXPECT_EQ(ve_batch_job_count(nullptr, &count), VE_INVALID_HANDLE);
}

TEST(CapiBatch, CreateRejectsNullOutAndResetsOut)
{
    EXPECT_EQ(ve_batch_create(nullptr), VE_INVALID_ARGUMENT);
    ve_batch* batch = reinterpret_cast<ve_batch*>(0x1);
    ASSERT_EQ(ve_batch_create(&batch), VE_OK);
    EXPECT_NE(batch, nullptr);
    ve_batch_free(batch);
    ve_batch_free(nullptr);
}
