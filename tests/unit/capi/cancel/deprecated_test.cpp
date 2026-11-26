// The calls without a cancel handle are deprecated; this file is where they are still called.
#define VE_NO_DEPRECATED_WARNINGS

#include "capi_scope.hpp"
#include "dcc/dcc3_panel.hpp"

#include <bit>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <memory>
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

template <class Handle, void (*Free)(Handle*)>
struct Release
{
    void operator()(Handle* handle) const
    {
        Free(handle);
    }
};

using OptionsPtr = std::unique_ptr<ve_sim_options, Release<ve_sim_options, ve_sim_options_free>>;
using SimPtr = std::unique_ptr<ve_sim, Release<ve_sim, ve_sim_free>>;
using DccSpecPtr = std::unique_ptr<ve_dcc_spec, Release<ve_dcc_spec, ve_dcc_spec_free>>;
using DccOptionsPtr = std::unique_ptr<ve_dcc_options, Release<ve_dcc_options, ve_dcc_options_free>>;
using DccBatchPtr = std::unique_ptr<ve_dcc_batch, Release<ve_dcc_batch, ve_dcc_batch_free>>;
using DccFitPtr = std::unique_ptr<ve_dcc_fit, Release<ve_dcc_fit, ve_dcc_fit_free>>;
using DccSimPtr = std::unique_ptr<ve_dcc_sim, Release<ve_dcc_sim, ve_dcc_sim_free>>;

std::vector<std::uint64_t> bits_of(const std::vector<double>& numbers)
{
    std::vector<std::uint64_t> bits;
    for (double number : numbers)
    {
        bits.push_back(std::bit_cast<std::uint64_t>(number));
    }
    return bits;
}

// The estimates and loglik of each fit.
std::vector<std::uint64_t> report_of(const FitSet& fits)
{
    std::vector<double> numbers;
    for (std::size_t i = 0; i < fits.size(); ++i)
    {
        const ve::testing::Snapshot snapshot = ve::testing::read_snapshot(fits[i]);
        numbers.insert(numbers.end(), snapshot.estimate.begin(), snapshot.estimate.end());
        numbers.push_back(snapshot.loglik);
    }
    return bits_of(numbers);
}

// The estimates, loglik and both standard errors of a DCC fit.
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
    return bits_of(numbers);
}

std::vector<std::uint64_t> returns_of(const ve_sim* sim)
{
    const double* data = nullptr;
    size_t len = 0;
    EXPECT_EQ(ve_sim_returns(sim, &data, &len), VE_OK);
    return bits_of({data, data + len});
}

std::vector<std::uint64_t> returns_of(const ve_dcc_sim* sim)
{
    const double* data = nullptr;
    size_t len = 0;
    EXPECT_EQ(ve_dcc_sim_returns(sim, &data, &len), VE_OK);
    return bits_of({data, data + len});
}

// The T x 3 rows of dcc3 in C order, kept alive for the borrowed panel.
std::vector<double> dcc3_rows()
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
}

// sGARCH(1,1) fits of the three dcc3 series, a dcc/mvnorm batch on them and the options to simulate.
struct Work
{
    ContextPtr context = ve::testing::make_context(2);
    std::vector<double> rows = dcc3_rows();
    SpecPtr spec = ve::testing::make_spec(1);
    PanelPtr panel = ve::testing::make_panel(rows.data(), ve::testing::kDcc3Rows, ve::testing::kDcc3Series);
    BatchPtr batch;
    DccSpecPtr dcc_spec;
    DccBatchPtr dcc_batch;
    OptionsPtr options;

    Work()
    {
        BatchInputs inputs;
        inputs.specs = {spec.get()};
        inputs.panel = panel.get();
        inputs.job_spec = {0, 0, 0};
        inputs.job_series = {0, 1, 2};
        batch = ve::testing::build_batch(inputs);

        ve_sim_options* raw_options = nullptr;
        EXPECT_EQ(ve_sim_options_create(10, 50, 7, &raw_options), VE_OK);
        options.reset(raw_options);

        ve_dcc_spec* raw_spec = nullptr;
        EXPECT_EQ(ve_dcc_spec_create(VE_DCC_MODEL_DCC, &raw_spec), VE_OK);
        dcc_spec.reset(raw_spec);
    }

    // The DCC batch on `fits`, with or without the standard errors.
    DccBatchPtr dcc(const std::vector<const ve_fit*>& fits, bool standard_errors) const
    {
        ve_dcc_batch* raw = nullptr;
        EXPECT_EQ(ve_dcc_batch_create(&raw), VE_OK);
        DccBatchPtr owner(raw);
        const ve_dcc_spec* specs[] = {dcc_spec.get()};
        EXPECT_EQ(ve_dcc_batch_set_specs(raw, specs, 1), VE_OK);
        EXPECT_EQ(ve_dcc_batch_set_fits(raw, fits.data(), fits.size()), VE_OK);
        ve_dcc_options* raw_options = nullptr;
        EXPECT_EQ(ve_dcc_options_create(&raw_options), VE_OK);
        DccOptionsPtr options_owner(raw_options);
        EXPECT_EQ(ve_dcc_options_set_standard_errors(raw_options, standard_errors ? 1 : 0), VE_OK);
        EXPECT_EQ(ve_dcc_batch_set_options(raw, raw_options), VE_OK);
        return owner;
    }
};

std::vector<const ve_fit*> pointers(const FitSet& fits)
{
    std::vector<const ve_fit*> out;
    for (std::size_t i = 0; i < fits.size(); ++i)
    {
        out.push_back(fits[i]);
    }
    return out;
}

}

TEST(Deprecated, EachDelegatesToItsVariant)
{
    const Work work;
    ve_context* ctx = work.context.get();

    FitSet old_fits(3);
    FitSet new_fits(3);
    ASSERT_EQ(ve_fit_batch(ctx, work.batch.get(), old_fits.out()), VE_OK) << ve_last_error();
    ASSERT_EQ(ve_fit_batch_with_cancel(ctx, work.batch.get(), nullptr, new_fits.out()), VE_OK) << ve_last_error();
    EXPECT_EQ(report_of(old_fits), report_of(new_fits));

    ve_sim* raw = nullptr;
    ASSERT_EQ(ve_simulate_fit(ctx, new_fits[0], work.options.get(), &raw), VE_OK) << ve_last_error();
    const SimPtr old_fit_sim(raw);
    ASSERT_EQ(ve_simulate_fit_with_cancel(ctx, new_fits[0], work.options.get(), nullptr, &raw), VE_OK);
    const SimPtr new_fit_sim(raw);
    EXPECT_EQ(returns_of(old_fit_sim.get()), returns_of(new_fit_sim.get()));

    SpecPtr fixed = ve::testing::make_spec(1);
    ASSERT_EQ(ve_spec_set_fixed(fixed.get(), "mu", 0.01), VE_OK);
    ASSERT_EQ(ve_spec_set_fixed(fixed.get(), "omega", 0.05), VE_OK);
    ASSERT_EQ(ve_spec_set_fixed(fixed.get(), "alpha1", 0.1), VE_OK);
    ASSERT_EQ(ve_spec_set_fixed(fixed.get(), "beta1", 0.85), VE_OK);
    ASSERT_EQ(ve_simulate_spec(ctx, fixed.get(), work.options.get(), &raw), VE_OK) << ve_last_error();
    const SimPtr old_spec_sim(raw);
    ASSERT_EQ(ve_simulate_spec_with_cancel(ctx, fixed.get(), work.options.get(), nullptr, &raw), VE_OK);
    const SimPtr new_spec_sim(raw);
    EXPECT_EQ(returns_of(old_spec_sim.get()), returns_of(new_spec_sim.get()));

    const std::vector<const ve_fit*> fits = pointers(new_fits);
    const DccBatchPtr with_errors = work.dcc(fits, true);
    ve_dcc_fit* dcc_raw = nullptr;
    ASSERT_EQ(ve_dcc_fit_batch(ctx, with_errors.get(), &dcc_raw), VE_OK) << ve_last_error();
    const DccFitPtr old_dcc(dcc_raw);
    ASSERT_EQ(ve_dcc_fit_batch_with_cancel(ctx, with_errors.get(), nullptr, &dcc_raw), VE_OK) << ve_last_error();
    const DccFitPtr new_dcc(dcc_raw);
    EXPECT_EQ(report_of(old_dcc.get()), report_of(new_dcc.get()));

    const DccBatchPtr without_errors = work.dcc(fits, false);
    ASSERT_EQ(ve_dcc_fit_batch_with_cancel(ctx, without_errors.get(), nullptr, &dcc_raw), VE_OK) << ve_last_error();
    const DccFitPtr plain(dcc_raw);
    ASSERT_EQ(ve_dcc_standard_errors(ctx, plain.get(), fits.data(), fits.size(), &dcc_raw), VE_OK) << ve_last_error();
    const DccFitPtr old_errors(dcc_raw);
    ASSERT_EQ(ve_dcc_standard_errors_with_cancel(ctx, plain.get(), fits.data(), fits.size(), nullptr, &dcc_raw), VE_OK)
        << ve_last_error();
    const DccFitPtr new_errors(dcc_raw);
    EXPECT_EQ(report_of(old_errors.get()), report_of(new_errors.get()));

    ve_dcc_sim* dcc_sim = nullptr;
    ASSERT_EQ(ve_simulate_dcc(ctx, new_dcc.get(), work.options.get(), nullptr, &dcc_sim), VE_OK) << ve_last_error();
    const DccSimPtr old_dcc_sim(dcc_sim);
    ASSERT_EQ(ve_simulate_dcc_with_cancel(ctx, new_dcc.get(), work.options.get(), nullptr, nullptr, &dcc_sim), VE_OK);
    const DccSimPtr new_dcc_sim(dcc_sim);
    EXPECT_EQ(returns_of(old_dcc_sim.get()), returns_of(new_dcc_sim.get()));
}
