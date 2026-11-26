#include "capi_scope.hpp"
#include "dcc/dcc3_panel.hpp"
#include "simulated_garch.hpp"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <thread>
#include <vector>
#include <volengine/volengine.h>

namespace
{

using ve::testing::BatchInputs;
using ve::testing::BatchPtr;
using ve::testing::ContextPtr;
using ve::testing::FitSet;
using ve::testing::PanelPtr;
using ve::testing::SpecPtr;


constexpr std::size_t kT = 2000;
constexpr int32_t kThreads = 2;
// The long batch and the long simulation are many times the work that fits in kSignalDelay, so a signal at
// kSignalDelay lands inside the call.
constexpr std::size_t kLongJobs = 1000;
constexpr std::size_t kLongPaths = 2'000'000;
constexpr std::size_t kLongHorizon = 2;
constexpr std::chrono::milliseconds kSignalDelay{50};
// How soon after the signal a cancelled call returns, the start included.
constexpr std::chrono::milliseconds kPrompt{500};

struct CancelFree
{
    void operator()(ve_cancel* handle) const
    {
        ve_cancel_free(handle);
    }
};

struct OptionsFree
{
    void operator()(ve_sim_options* handle) const
    {
        ve_sim_options_free(handle);
    }
};

struct SimFree
{
    void operator()(ve_sim* handle) const
    {
        ve_sim_free(handle);
    }
};

using CancelPtr = std::unique_ptr<ve_cancel, CancelFree>;
using OptionsPtr = std::unique_ptr<ve_sim_options, OptionsFree>;
using SimPtr = std::unique_ptr<ve_sim, SimFree>;

CancelPtr make_cancel()
{
    ve_cancel* raw = nullptr;
    EXPECT_EQ(ve_cancel_create(&raw), VE_OK);
    return CancelPtr(raw);
}

CancelPtr requested_cancel()
{
    CancelPtr cancel = make_cancel();
    EXPECT_EQ(ve_cancel_request(cancel.get()), VE_OK);
    return cancel;
}

// A non-null address no call returns, to see that a call overwrites its output.
template <class T>
T* sentinel()
{
    static char byte = 0;
    return reinterpret_cast<T*>(&byte);
}

// Runs call() while another thread requests c after kSignalDelay, and expects it to return within kPrompt.
template <class F>
ve_status cancelled_during(ve_cancel* c, F&& call)
{
    const auto start = std::chrono::steady_clock::now();
    std::jthread signal(
        [c]
        {
            std::this_thread::sleep_for(kSignalDelay);
            ve_cancel_request(c);
        });
    const ve_status status = call();
    EXPECT_LT(std::chrono::steady_clock::now() - start, kPrompt);
    return status;
}

template <class T>
bool all_null(const std::vector<T*>& handles)
{
    return std::ranges::all_of(handles, [](const T* handle) { return handle == nullptr; });
}

const std::vector<double>& series()
{
    static const std::vector<double> data = ve::testing::simulate_garch_series(11, kT);
    return data;
}

// `jobs` sGARCH(1,1)-norm fits of one simulated series.
struct FitWork
{
    ContextPtr context = ve::testing::make_context(kThreads);
    SpecPtr spec = ve::testing::make_spec(1);
    PanelPtr panel = ve::testing::make_panel(series().data(), kT, 1);
    BatchPtr batch;
    std::size_t jobs;

    explicit FitWork(std::size_t count) : jobs(count)
    {
        BatchInputs inputs;
        inputs.specs = {spec.get()};
        inputs.panel = panel.get();
        inputs.job_spec.assign(jobs, 0);
        inputs.job_series.assign(jobs, 0);
        batch = ve::testing::build_batch(inputs);
    }

    ve_status run(const ve_cancel* c, FitSet& fits) const
    {
        return ve_fit_batch_with_cancel(context.get(), batch.get(), c, fits.out());
    }

    // For a call that must not publish anything: out starts as sentinels.
    ve_status run(const ve_cancel* c, std::vector<ve_fit*>& out) const
    {
        out.assign(jobs, sentinel<ve_fit>());
        return ve_fit_batch_with_cancel(context.get(), batch.get(), c, out.data());
    }
};

std::vector<ve::testing::Snapshot> snapshots(const FitSet& fits)
{
    std::vector<ve::testing::Snapshot> values;
    for (std::size_t job = 0; job < fits.size(); ++job)
    {
        values.push_back(ve::testing::read_snapshot(fits[job]));
    }
    return values;
}

OptionsPtr make_options(std::size_t horizon, std::size_t paths)
{
    ve_sim_options* raw = nullptr;
    EXPECT_EQ(ve_sim_options_create(horizon, paths, 7, &raw), VE_OK) << ve_last_error();
    return OptionsPtr(raw);
}

SpecPtr fixed_spec()
{
    SpecPtr spec = ve::testing::make_spec(1);
    EXPECT_EQ(ve_spec_set_fixed(spec.get(), "mu", 0.01), VE_OK);
    EXPECT_EQ(ve_spec_set_fixed(spec.get(), "omega", 0.05), VE_OK);
    EXPECT_EQ(ve_spec_set_fixed(spec.get(), "alpha1", 0.1), VE_OK);
    EXPECT_EQ(ve_spec_set_fixed(spec.get(), "beta1", 0.85), VE_OK);
    return spec;
}

struct SimValues
{
    std::vector<double> returns;
    std::vector<double> sigma;

    bool operator==(const SimValues&) const = default;
};

std::vector<double> view(ve_status (*read)(const ve_sim*, const double**, size_t*), const ve_sim* sim)
{
    const double* data = nullptr;
    size_t len = 0;
    EXPECT_EQ(read(sim, &data, &len), VE_OK);
    return {data, data + len};
}

// call(&out) runs a simulation that must succeed.
template <class F>
SimValues simulated(F&& call)
{
    ve_sim* raw = nullptr;
    const ve_status status = call(&raw);
    SimPtr sim(raw);
    EXPECT_EQ(status, VE_OK) << ve_last_error();
    return {view(ve_sim_returns, sim.get()), view(ve_sim_sigma, sim.get())};
}

}

TEST(CancelHandle, Lifecycle)
{
    ve_cancel* raw = sentinel<ve_cancel>();
    ASSERT_EQ(ve_cancel_create(&raw), VE_OK);
    CancelPtr cancel(raw);
    ASSERT_NE(raw, nullptr);
    EXPECT_EQ(ve_cancel_request(raw), VE_OK);
    EXPECT_EQ(ve_cancel_request(raw), VE_OK);
    EXPECT_EQ(ve_cancel_create(nullptr), VE_INVALID_ARGUMENT);
    EXPECT_EQ(ve_cancel_request(nullptr), VE_INVALID_HANDLE);
    ve_cancel_free(nullptr);
}

TEST(FitBatchWithCancel, RequestedBeforeCall)
{
    const FitWork work(3);
    CancelPtr cancel = requested_cancel();
    std::vector<ve_fit*> out;
    EXPECT_EQ(work.run(cancel.get(), out), VE_CANCELLED);
    EXPECT_EQ(ve_last_error(), std::string("cancelled"));
    EXPECT_TRUE(all_null(out));
}

TEST(FitBatchWithCancel, NullEqualsUnrequested)
{
    const FitWork work(3);
    CancelPtr cancel = make_cancel();
    FitSet without(work.jobs);
    FitSet with(work.jobs);
    ASSERT_EQ(work.run(nullptr, without), VE_OK) << ve_last_error();
    ASSERT_EQ(work.run(cancel.get(), with), VE_OK) << ve_last_error();
    EXPECT_EQ(snapshots(with), snapshots(without));
}

TEST(FitBatchWithCancel, InvalidArgumentWinsOverCancel)
{
    ContextPtr context = ve::testing::make_context(kThreads);
    SpecPtr spec = ve::testing::make_spec(1);
    PanelPtr panel = ve::testing::make_panel(series().data(), kT, 1);
    BatchInputs inputs;
    inputs.specs = {spec.get()};
    inputs.panel = panel.get();
    BatchPtr batch = ve::testing::build_batch(inputs);
    CancelPtr cancel = requested_cancel();
    FitSet fits(1);
    EXPECT_EQ(ve_fit_batch_with_cancel(context.get(), batch.get(), cancel.get(), fits.out()), VE_INVALID_ARGUMENT);
}

TEST(FitBatchWithCancel, RequestFromAnotherThreadStops)
{
    const FitWork work(kLongJobs);
    CancelPtr cancel = make_cancel();
    std::vector<ve_fit*> out;
    EXPECT_EQ(cancelled_during(cancel.get(), [&] { return work.run(cancel.get(), out); }), VE_CANCELLED);
    EXPECT_TRUE(all_null(out));
}

// Review Focus 1: one handle shared by two concurrent calls cancels both.
TEST(FitBatchWithCancel, OneHandleCancelsTwoConcurrentCalls)
{
    const FitWork first(kLongJobs);
    const FitWork second(kLongJobs);
    CancelPtr cancel = make_cancel();
    std::vector<ve_fit*> first_out;
    std::vector<ve_fit*> second_out;
    ve_status first_status = VE_OK;
    ve_status second_status = VE_OK;
    {
        std::jthread first_call([&] { first_status = first.run(cancel.get(), first_out); });
        std::jthread second_call([&] { second_status = second.run(cancel.get(), second_out); });
        std::this_thread::sleep_for(kSignalDelay);
        EXPECT_EQ(ve_cancel_request(cancel.get()), VE_OK);
    }
    EXPECT_EQ(first_status, VE_CANCELLED);
    EXPECT_EQ(second_status, VE_CANCELLED);
    EXPECT_TRUE(all_null(first_out));
    EXPECT_TRUE(all_null(second_out));
}

// A cancelled call returns without waiting for another call that keeps the same context busy.
TEST(FitBatchWithCancel, SharedContextDoesNotWaitForTheOtherCall)
{
    const FitWork cancelled(kLongJobs);
    const FitWork other(kLongJobs);
    ve_context* shared = cancelled.context.get();
    CancelPtr cancel = make_cancel();
    std::vector<ve_fit*> out(cancelled.jobs, sentinel<ve_fit>());
    FitSet other_fits(other.jobs);
    ve_status status = VE_OK;
    ve_status other_status = VE_INTERNAL_ERROR;
    {
        std::jthread first(
            [&]
            {
                status = cancelled_during(cancel.get(),
                                          [&] {
                                              return ve_fit_batch_with_cancel(shared, cancelled.batch.get(),
                                                                              cancel.get(), out.data());
                                          });
            });
        std::this_thread::sleep_for(kSignalDelay / 5);
        other_status = ve_fit_batch_with_cancel(shared, other.batch.get(), nullptr, other_fits.out());
    }
    EXPECT_EQ(status, VE_CANCELLED);
    EXPECT_TRUE(all_null(out));
    EXPECT_EQ(other_status, VE_OK) << ve_last_error();
}

// Review Focus 2: a requested handle stays requested for the next call.
TEST(FitBatchWithCancel, RequestedHandleStaysRequested)
{
    const FitWork work(kLongJobs);
    CancelPtr cancel = make_cancel();
    std::vector<ve_fit*> out;
    ASSERT_EQ(cancelled_during(cancel.get(), [&] { return work.run(cancel.get(), out); }), VE_CANCELLED);
    const FitWork small(1);
    EXPECT_EQ(small.run(cancel.get(), out), VE_CANCELLED);
    EXPECT_TRUE(all_null(out));
}

// Review Focus 3: the pool is free again after a cancelled call.
TEST(FitBatchWithCancel, ContextUsableAfterCancel)
{
    const FitWork work(kLongJobs);
    CancelPtr cancel = make_cancel();
    std::vector<ve_fit*> out;
    ASSERT_EQ(cancelled_during(cancel.get(), [&] { return work.run(cancel.get(), out); }), VE_CANCELLED);
    EXPECT_EQ(ve_context_set_threads(work.context.get(), 2), VE_OK);
    const FitWork small(1);
    FitSet again(small.jobs);
    EXPECT_EQ(ve_fit_batch_with_cancel(work.context.get(), small.batch.get(), nullptr, again.out()), VE_OK)
        << ve_last_error();
}

TEST(SimulateWithCancel, RequestedBeforeCall)
{
    const FitWork work(1);
    FitSet fits(1);
    ASSERT_EQ(work.run(nullptr, fits), VE_OK);
    SpecPtr spec = fixed_spec();
    OptionsPtr options = make_options(10, 300);
    CancelPtr cancel = requested_cancel();
    ve_sim* from_fit = sentinel<ve_sim>();
    ve_sim* from_spec = sentinel<ve_sim>();
    EXPECT_EQ(ve_simulate_fit_with_cancel(work.context.get(), fits[0], options.get(), cancel.get(), &from_fit),
              VE_CANCELLED);
    EXPECT_EQ(from_fit, nullptr);
    EXPECT_EQ(ve_last_error(), std::string("cancelled"));
    EXPECT_EQ(ve_simulate_spec_with_cancel(work.context.get(), spec.get(), options.get(), cancel.get(), &from_spec),
              VE_CANCELLED);
    EXPECT_EQ(from_spec, nullptr);
}

TEST(SimulateWithCancel, InvalidArgumentWinsOverCancel)
{
    ContextPtr context = ve::testing::make_context(kThreads);
    SpecPtr spec = fixed_spec();
    OptionsPtr options = make_options(10, 300);
    ASSERT_EQ(ve_sim_options_set_correlations(options.get(), 1), VE_OK);
    CancelPtr cancel = requested_cancel();
    ve_sim* out = nullptr;
    EXPECT_EQ(ve_simulate_spec_with_cancel(context.get(), spec.get(), options.get(), cancel.get(), &out),
              VE_INVALID_ARGUMENT);
}

TEST(SimulateWithCancel, NullEqualsUnrequested)
{
    const FitWork work(1);
    FitSet fits(1);
    ASSERT_EQ(work.run(nullptr, fits), VE_OK);
    SpecPtr spec = fixed_spec();
    OptionsPtr options = make_options(10, 300);
    CancelPtr cancel = make_cancel();
    ve_context* context = work.context.get();
    const ve_fit* fit = fits[0];
    const SimValues fit_without = simulated(
        [&](ve_sim** out) { return ve_simulate_fit_with_cancel(context, fit, options.get(), nullptr, out); });
    const SimValues fit_with = simulated(
        [&](ve_sim** out) { return ve_simulate_fit_with_cancel(context, fit, options.get(), cancel.get(), out); });
    const SimValues spec_without = simulated(
        [&](ve_sim** out) { return ve_simulate_spec_with_cancel(context, spec.get(), options.get(), nullptr, out); });
    const SimValues spec_with = simulated(
        [&](ve_sim** out)
        { return ve_simulate_spec_with_cancel(context, spec.get(), options.get(), cancel.get(), out); });
    EXPECT_EQ(fit_with, fit_without);
    EXPECT_EQ(spec_with, spec_without);
}

TEST(SimulateWithCancel, RequestFromAnotherThreadStops)
{
    ContextPtr context = ve::testing::make_context(kThreads);
    SpecPtr spec = fixed_spec();
    OptionsPtr options = make_options(kLongHorizon, kLongPaths);
    CancelPtr cancel = make_cancel();
    ve_sim* out = sentinel<ve_sim>();
    EXPECT_EQ(cancelled_during(cancel.get(),
                               [&] {
                                   return ve_simulate_spec_with_cancel(context.get(), spec.get(), options.get(),
                                                                       cancel.get(), &out);
                               }),
              VE_CANCELLED);
    EXPECT_EQ(out, nullptr);
}

namespace
{

template <class Handle, void (*Free)(Handle*)>
struct Release
{
    void operator()(Handle* handle) const
    {
        Free(handle);
    }
};

using DccSpecPtr = std::unique_ptr<ve_dcc_spec, Release<ve_dcc_spec, ve_dcc_spec_free>>;
using DccOptionsPtr = std::unique_ptr<ve_dcc_options, Release<ve_dcc_options, ve_dcc_options_free>>;
using DccBatchPtr = std::unique_ptr<ve_dcc_batch, Release<ve_dcc_batch, ve_dcc_batch_free>>;
using DccFitPtr = std::unique_ptr<ve_dcc_fit, Release<ve_dcc_fit, ve_dcc_fit_free>>;
using DccSimPtr = std::unique_ptr<ve_dcc_sim, Release<ve_dcc_sim, ve_dcc_sim_free>>;

// The long DCC batch is this many specs, many times the work that fits in kSignalDelay.
constexpr std::size_t kLongSpecs = 300;

// Owns the ve_dcc_fit handles a batch writes. They start as sentinels, so a call that must not publish shows it.
class DccFitSet
{
public:
    explicit DccFitSet(std::size_t count) : fits_(count, sentinel<ve_dcc_fit>())
    {
    }

    DccFitSet(const DccFitSet&) = delete;
    DccFitSet& operator=(const DccFitSet&) = delete;

    ~DccFitSet()
    {
        for (ve_dcc_fit* fit : fits_)
        {
            if (fit != sentinel<ve_dcc_fit>())
            {
                ve_dcc_fit_free(fit);
            }
        }
    }

    ve_dcc_fit** out()
    {
        return fits_.data();
    }

    ve_dcc_fit* operator[](std::size_t index) const
    {
        return fits_.at(index);
    }

    bool all_null() const
    {
        return std::ranges::all_of(fits_, [](const ve_dcc_fit* fit) { return fit == nullptr; });
    }

private:
    std::vector<ve_dcc_fit*> fits_;
};

// The T x 3 rows of dcc3 in C order, kept alive for the borrowed panel.
const std::vector<double>& dcc3_rows()
{
    static const std::vector<double> rows = []
    {
        const std::vector<std::vector<double>> columns = ve::testing::read_dcc3();
        std::vector<double> flat(ve::testing::kDcc3Rows * ve::testing::kDcc3Series);
        for (std::size_t t = 0; t < ve::testing::kDcc3Rows; ++t)
        {
            for (std::size_t i = 0; i < ve::testing::kDcc3Series; ++i)
            {
                flat[t * ve::testing::kDcc3Series + i] = columns[i][t];
            }
        }
        return flat;
    }();
    return rows;
}

DccSpecPtr dcc_spec(int32_t model, int32_t dist)
{
    ve_dcc_spec* raw = nullptr;
    EXPECT_EQ(ve_dcc_spec_create(model, &raw), VE_OK) << ve_last_error();
    DccSpecPtr spec(raw);
    EXPECT_EQ(ve_dcc_spec_set_dist(raw, dist), VE_OK);
    return spec;
}

// A static copula with Kendall's tau: nothing for the optimizer to estimate.
DccSpecPtr kendall_spec()
{
    DccSpecPtr spec = dcc_spec(VE_DCC_MODEL_STATIC, VE_DCC_DIST_MVNORM);
    EXPECT_EQ(ve_dcc_spec_set_copula(spec.get(), VE_DCC_COPULA_PARAMETRIC), VE_OK);
    EXPECT_EQ(ve_dcc_spec_set_static_method(spec.get(), VE_DCC_STATIC_KENDALL), VE_OK);
    return spec;
}

DccOptionsPtr dcc_options(bool standard_errors)
{
    ve_dcc_options* raw = nullptr;
    EXPECT_EQ(ve_dcc_options_create(&raw), VE_OK);
    DccOptionsPtr options(raw);
    EXPECT_EQ(ve_dcc_options_set_standard_errors(raw, standard_errors ? 1 : 0), VE_OK);
    return options;
}

FitSet fit_dcc3_univariates()
{
    ContextPtr context = ve::testing::make_context(1);
    SpecPtr spec = ve::testing::make_spec(1);
    PanelPtr panel = ve::testing::make_panel(dcc3_rows().data(), ve::testing::kDcc3Rows, ve::testing::kDcc3Series);
    BatchInputs inputs;
    inputs.specs = {spec.get()};
    inputs.panel = panel.get();
    inputs.job_spec = {0, 0, 0};
    inputs.job_series = {0, 1, 2};
    BatchPtr batch = ve::testing::build_batch(inputs);
    return ve::testing::fit_all(context.get(), batch.get());
}

// The sGARCH(1,1) fits of the three series of dcc3, and a context to run DCC calls on.
struct DccWork
{
    ContextPtr context = ve::testing::make_context(kThreads);
    FitSet univariate = fit_dcc3_univariates();
    std::vector<const ve_fit*> fits;

    DccWork()
    {
        for (std::size_t i = 0; i < univariate.size(); ++i)
        {
            fits.push_back(univariate[i]);
        }
    }

    // `count` specs cycling through `specs`, fitted with or without the standard errors.
    DccBatchPtr batch(const std::vector<const ve_dcc_spec*>& specs, std::size_t count, bool standard_errors) const
    {
        std::vector<const ve_dcc_spec*> cycled;
        for (std::size_t i = 0; i < count; ++i)
        {
            cycled.push_back(specs[i % specs.size()]);
        }
        ve_dcc_batch* raw = nullptr;
        EXPECT_EQ(ve_dcc_batch_create(&raw), VE_OK);
        DccBatchPtr owner(raw);
        EXPECT_EQ(ve_dcc_batch_set_specs(raw, cycled.data(), cycled.size()), VE_OK) << ve_last_error();
        EXPECT_EQ(ve_dcc_batch_set_fits(raw, fits.data(), fits.size()), VE_OK) << ve_last_error();
        DccOptionsPtr options = dcc_options(standard_errors);
        EXPECT_EQ(ve_dcc_batch_set_options(raw, options.get()), VE_OK);
        return owner;
    }

    // A dcc/mvnorm fit without standard errors, to hand to ve_dcc_standard_errors* and ve_simulate_dcc*.
    DccFitPtr fitted() const
    {
        DccSpecPtr spec = dcc_spec(VE_DCC_MODEL_DCC, VE_DCC_DIST_MVNORM);
        DccBatchPtr one = batch({spec.get()}, 1, false);
        ve_dcc_fit* raw = nullptr;
        EXPECT_EQ(ve_dcc_fit_batch_with_cancel(context.get(), one.get(), nullptr, &raw), VE_OK) << ve_last_error();
        return DccFitPtr(raw);
    }
};

// What a DCC fit reports (estimates, loglik, both standard errors), as bit patterns so that NaN compares.
std::vector<std::uint64_t> report_of(const ve_dcc_fit* fit)
{
    size_t count = 0;
    EXPECT_EQ(ve_dcc_fit_param_count(fit, &count), VE_OK);
    std::vector<double> numbers(count);
    EXPECT_EQ(ve_dcc_fit_estimates(fit, numbers.data()), VE_OK);
    double loglik = 0.0;
    EXPECT_EQ(ve_dcc_fit_loglik(fit, &loglik), VE_OK);
    numbers.push_back(loglik);
    for (int32_t kind : {VE_SE_HESSIAN, VE_SE_ROBUST})
    {
        std::vector<double> errors(count);
        EXPECT_EQ(ve_dcc_fit_std_errors(fit, kind, errors.data()), VE_OK) << ve_last_error();
        numbers.insert(numbers.end(), errors.begin(), errors.end());
    }
    std::vector<std::uint64_t> bits;
    for (double number : numbers)
    {
        bits.push_back(std::bit_cast<std::uint64_t>(number));
    }
    return bits;
}

std::vector<double> dcc_view(ve_status (*read)(const ve_dcc_sim*, const double**, size_t*), const ve_dcc_sim* sim)
{
    const double* data = nullptr;
    size_t len = 0;
    EXPECT_EQ(read(sim, &data, &len), VE_OK);
    return {data, data + len};
}

// The returns, sigma, correlation and covariance of a simulation with correlations that must succeed.
std::vector<std::vector<double>> simulated_dcc(const DccWork& work, const ve_dcc_fit* fit, const ve_sim_options* options,
                                               const ve_cancel* c)
{
    ve_dcc_sim* raw = nullptr;
    const ve_status status =
        ve_simulate_dcc_with_cancel(work.context.get(), fit, options, nullptr, c, &raw);
    DccSimPtr sim(raw);
    EXPECT_EQ(status, VE_OK) << ve_last_error();
    return {dcc_view(ve_dcc_sim_returns, sim.get()), dcc_view(ve_dcc_sim_sigma, sim.get()),
            dcc_view(ve_dcc_sim_correlation, sim.get()), dcc_view(ve_dcc_sim_covariance, sim.get())};
}

}

TEST(DccFitBatchWithCancel, RequestedBeforeCall)
{
    const DccWork work;
    DccSpecPtr spec = dcc_spec(VE_DCC_MODEL_DCC, VE_DCC_DIST_MVNORM);
    DccBatchPtr batch = work.batch({spec.get()}, 3, true);
    CancelPtr cancel = requested_cancel();
    DccFitSet out(3);
    EXPECT_EQ(ve_dcc_fit_batch_with_cancel(work.context.get(), batch.get(), cancel.get(), out.out()), VE_CANCELLED);
    EXPECT_EQ(ve_last_error(), std::string("cancelled"));
    EXPECT_TRUE(out.all_null());
}

TEST(DccFitBatchWithCancel, NullEqualsUnrequested)
{
    const DccWork work;
    DccSpecPtr dcc = dcc_spec(VE_DCC_MODEL_DCC, VE_DCC_DIST_MVNORM);
    DccSpecPtr adcc = dcc_spec(VE_DCC_MODEL_ADCC, VE_DCC_DIST_MVT);
    DccBatchPtr batch = work.batch({dcc.get(), adcc.get()}, 2, true);
    CancelPtr cancel = make_cancel();
    DccFitSet without(2);
    DccFitSet with(2);
    ASSERT_EQ(ve_dcc_fit_batch_with_cancel(work.context.get(), batch.get(), nullptr, without.out()), VE_OK)
        << ve_last_error();
    ASSERT_EQ(ve_dcc_fit_batch_with_cancel(work.context.get(), batch.get(), cancel.get(), with.out()), VE_OK)
        << ve_last_error();
    for (std::size_t i = 0; i < 2; ++i)
    {
        SCOPED_TRACE(i);
        EXPECT_EQ(report_of(with[i]), report_of(without[i]));
    }
}

// Review Focus 4: nothing to optimize, the check at the start of the spec still cancels.
TEST(DccFitBatchWithCancel, NothingToEstimateStillCancels)
{
    const DccWork work;
    DccSpecPtr spec = kendall_spec();
    DccBatchPtr batch = work.batch({spec.get()}, 1, false);
    DccFitSet unrequested(1);
    ASSERT_EQ(ve_dcc_fit_batch_with_cancel(work.context.get(), batch.get(), nullptr, unrequested.out()), VE_OK)
        << ve_last_error();
    CancelPtr cancel = requested_cancel();
    DccFitSet out(1);
    EXPECT_EQ(ve_dcc_fit_batch_with_cancel(work.context.get(), batch.get(), cancel.get(), out.out()), VE_CANCELLED);
    EXPECT_TRUE(out.all_null());
}

TEST(DccFitBatchWithCancel, InvalidArgumentWinsOverCancel)
{
    const DccWork work;
    ve_dcc_batch* raw = nullptr;
    ASSERT_EQ(ve_dcc_batch_create(&raw), VE_OK);
    DccBatchPtr empty(raw);
    CancelPtr cancel = requested_cancel();
    DccFitSet out(1);
    EXPECT_EQ(ve_dcc_fit_batch_with_cancel(work.context.get(), empty.get(), cancel.get(), out.out()),
              VE_INVALID_ARGUMENT);
}

TEST(DccFitBatchWithCancel, RequestFromAnotherThreadStops)
{
    const DccWork work;
    DccSpecPtr dcc = dcc_spec(VE_DCC_MODEL_DCC, VE_DCC_DIST_MVNORM);
    DccSpecPtr adcc = dcc_spec(VE_DCC_MODEL_ADCC, VE_DCC_DIST_MVNORM);
    DccSpecPtr mvt = dcc_spec(VE_DCC_MODEL_DCC, VE_DCC_DIST_MVT);
    DccBatchPtr batch = work.batch({dcc.get(), adcc.get(), mvt.get()}, kLongSpecs, true);
    CancelPtr cancel = make_cancel();
    DccFitSet out(kLongSpecs);
    EXPECT_EQ(cancelled_during(cancel.get(),
                               [&]
                               {
                                   return ve_dcc_fit_batch_with_cancel(work.context.get(), batch.get(), cancel.get(),
                                                                       out.out());
                               }),
              VE_CANCELLED);
    EXPECT_TRUE(out.all_null());
}

TEST(DccStandardErrorsWithCancel, RequestedBeforeCall)
{
    const DccWork work;
    DccFitPtr fit = work.fitted();
    CancelPtr cancel = requested_cancel();
    ve_dcc_fit* out = sentinel<ve_dcc_fit>();
    EXPECT_EQ(ve_dcc_standard_errors_with_cancel(work.context.get(), fit.get(), work.fits.data(), work.fits.size(),
                                                 cancel.get(), &out),
              VE_CANCELLED);
    EXPECT_EQ(out, nullptr);
    EXPECT_EQ(ve_last_error(), std::string("cancelled"));
}

TEST(DccStandardErrorsWithCancel, NullEqualsUnrequested)
{
    const DccWork work;
    DccFitPtr fit = work.fitted();
    CancelPtr cancel = make_cancel();
    ve_dcc_fit* without = nullptr;
    ve_dcc_fit* with = nullptr;
    ASSERT_EQ(ve_dcc_standard_errors_with_cancel(work.context.get(), fit.get(), work.fits.data(), work.fits.size(),
                                                 nullptr, &without),
              VE_OK)
        << ve_last_error();
    DccFitPtr without_owner(without);
    ASSERT_EQ(ve_dcc_standard_errors_with_cancel(work.context.get(), fit.get(), work.fits.data(), work.fits.size(),
                                                 cancel.get(), &with),
              VE_OK)
        << ve_last_error();
    DccFitPtr with_owner(with);
    EXPECT_EQ(report_of(with), report_of(without));
}

TEST(DccStandardErrorsWithCancel, InvalidArgumentWinsOverCancel)
{
    const DccWork work;
    DccFitPtr fit = work.fitted();
    CancelPtr cancel = requested_cancel();
    ve_dcc_fit* out = sentinel<ve_dcc_fit>();
    EXPECT_EQ(ve_dcc_standard_errors_with_cancel(work.context.get(), fit.get(), work.fits.data(), 2, cancel.get(),
                                                 &out),
              VE_INVALID_ARGUMENT);
}

TEST(SimulateDccWithCancel, RequestedBeforeCall)
{
    const DccWork work;
    DccFitPtr fit = work.fitted();
    OptionsPtr options = make_options(10, 300);
    CancelPtr cancel = requested_cancel();
    ve_dcc_sim* out = sentinel<ve_dcc_sim>();
    EXPECT_EQ(ve_simulate_dcc_with_cancel(work.context.get(), fit.get(), options.get(), nullptr, cancel.get(), &out),
              VE_CANCELLED);
    EXPECT_EQ(out, nullptr);
    EXPECT_EQ(ve_last_error(), std::string("cancelled"));
}

TEST(SimulateDccWithCancel, NullEqualsUnrequested)
{
    const DccWork work;
    DccFitPtr fit = work.fitted();
    OptionsPtr options = make_options(10, 300);
    ASSERT_EQ(ve_sim_options_set_correlations(options.get(), 1), VE_OK);
    CancelPtr cancel = make_cancel();
    EXPECT_EQ(simulated_dcc(work, fit.get(), options.get(), cancel.get()),
              simulated_dcc(work, fit.get(), options.get(), nullptr));
}

TEST(SimulateDccWithCancel, InvalidArgumentWinsOverCancel)
{
    const DccWork work;
    DccFitPtr fit = work.fitted();
    CancelPtr cancel = requested_cancel();
    ve_dcc_sim* out = sentinel<ve_dcc_sim>();
    EXPECT_EQ(ve_simulate_dcc_with_cancel(work.context.get(), fit.get(), nullptr, nullptr, cancel.get(), &out),
              VE_INVALID_HANDLE);
    EXPECT_EQ(out, nullptr);
}

TEST(SimulateDccWithCancel, RequestFromAnotherThreadStops)
{
    const DccWork work;
    DccFitPtr fit = work.fitted();
    OptionsPtr options = make_options(kLongHorizon, kLongPaths);
    CancelPtr cancel = make_cancel();
    ve_dcc_sim* out = sentinel<ve_dcc_sim>();
    EXPECT_EQ(cancelled_during(cancel.get(),
                               [&]
                               {
                                   return ve_simulate_dcc_with_cancel(work.context.get(), fit.get(), options.get(),
                                                                      nullptr, cancel.get(), &out);
                               }),
              VE_CANCELLED);
    EXPECT_EQ(out, nullptr);
}
