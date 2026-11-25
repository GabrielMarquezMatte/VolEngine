#include "capi_scope.hpp"
#include "simulated_garch.hpp"

#include <algorithm>
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

// Runs call() while another thread requests c after kSignalDelay.
template <class F>
ve_status cancelled_during(ve_cancel* c, F&& call)
{
    std::jthread signal(
        [c]
        {
            std::this_thread::sleep_for(kSignalDelay);
            ve_cancel_request(c);
        });
    return call();
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
