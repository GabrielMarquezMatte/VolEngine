#include "capi_scope.hpp"
#include "fit/criteria.hpp"
#include "fit/fit.hpp"
#include "simulated_garch.hpp"

#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <thread>
#include <vector>
#include <volengine/volengine.h>

namespace
{

using ve::testing::FitSet;
using ve::testing::Snapshot;

constexpr std::size_t kT = 3000;

using ArrayReader = ve_status (*)(const ve_fit*, int32_t, double*);
using SeriesView = ve_status (*)(const ve_fit*, const double**, std::size_t*);
using CoreCriterion = double (*)(const ve::FitResult&, ve::IcScale);

// A single sGARCH(1,1) fit of `series` through the C API.
FitSet fit_series(const std::vector<double>& series, int32_t include_mean)
{
    ve::testing::ContextPtr context = ve::testing::make_context(1);
    ve::testing::SpecPtr spec = ve::testing::make_spec(include_mean);
    ve::testing::PanelPtr panel = ve::testing::make_panel(series.data(), series.size(), 1);
    ve::testing::BatchInputs inputs;
    inputs.specs = {spec.get()};
    inputs.panel = panel.get();
    inputs.job_spec = {0};
    inputs.job_series = {0};
    ve::testing::BatchPtr batch = ve::testing::build_batch(inputs);
    return ve::testing::fit_all(context.get(), batch.get());
}

std::vector<double> simulated()
{
    return ve::testing::simulate_garch_series(1, kT);
}

double normal_cdf(double x)
{
    return 0.5 * std::erfc(-x / std::sqrt(2.0));
}

std::vector<double> read_array(const ve_fit* fit, ArrayReader reader, int32_t kind)
{
    std::size_t count = 0;
    EXPECT_EQ(ve_fit_param_count(fit, &count), VE_OK);
    std::vector<double> values(count);
    EXPECT_EQ(reader(fit, kind, values.data()), VE_OK);
    return values;
}

}

TEST(CapiFit, AccessorsConsistent)
{
    const std::vector<double> series = simulated();
    FitSet fits = fit_series(series, 1);
    const ve_fit* fit = fits[0];
    ASSERT_NE(fit, nullptr);
    const ve::FitResult expected = ve::fit_one(series, ve::Spec{}, ve::FitConfig{});
    std::size_t count = 0;
    ASSERT_EQ(ve_fit_param_count(fit, &count), VE_OK);
    ASSERT_EQ(count, 4u);
    const char* names[] = {"mu", "omega", "alpha1", "beta1"};
    for (std::size_t i = 0; i < count; ++i)
    {
        const char* name = nullptr;
        ASSERT_EQ(ve_fit_param_name(fit, i, &name), VE_OK);
        EXPECT_STREQ(name, names[i]);
    }
    std::vector<double> estimates(count);
    ASSERT_EQ(ve_fit_estimates(fit, estimates.data()), VE_OK);
    EXPECT_EQ(estimates, expected.estimate);
    const std::vector<double> errors = read_array(fit, ve_fit_std_errors, VE_SE_HESSIAN);
    EXPECT_EQ(errors, expected.se_hessian);
    EXPECT_EQ(read_array(fit, ve_fit_std_errors, VE_SE_ROBUST), expected.se_robust);
    const std::vector<double> t_values = read_array(fit, ve_fit_t_values, VE_SE_HESSIAN);
    const std::vector<double> p_values = read_array(fit, ve_fit_p_values, VE_SE_HESSIAN);
    for (std::size_t i = 0; i < count; ++i)
    {
        EXPECT_NEAR(t_values[i], estimates[i] / errors[i], 1e-12 * std::fabs(t_values[i]));
        EXPECT_NEAR(p_values[i], 2.0 * (1.0 - normal_cdf(std::fabs(t_values[i]))), 1e-12);
    }
    const std::vector<double> robust_t = read_array(fit, ve_fit_t_values, VE_SE_ROBUST);
    EXPECT_EQ(robust_t[0], estimates[0] / expected.se_robust[0]);
}

TEST(CapiFit, CriteriaMatchCoreAndRescale)
{
    const std::vector<double> series = simulated();
    FitSet fits = fit_series(series, 1);
    const ve_fit* fit = fits[0];
    const ve::FitResult expected = ve::fit_one(series, ve::Spec{}, ve::FitConfig{});
    const ArrayReader readers[] = {ve_aic, ve_bic, ve_hqic, ve_sic};
    const CoreCriterion cores[] = {ve::aic, ve::bic, ve::hqic, ve::sic};
    for (std::size_t i = 0; i < 4; ++i)
    {
        double per_obs = 0.0;
        double total = 0.0;
        ASSERT_EQ(readers[i](fit, VE_IC_PER_OBS, &per_obs), VE_OK);
        ASSERT_EQ(readers[i](fit, VE_IC_TOTAL, &total), VE_OK);
        EXPECT_EQ(per_obs, cores[i](expected, ve::IcScale::PerObs));
        EXPECT_EQ(total, cores[i](expected, ve::IcScale::Total));
        EXPECT_NEAR(per_obs * static_cast<double>(kT), total, 1e-9);
    }
    double loglik = 0.0;
    ASSERT_EQ(ve_loglik(fit, &loglik), VE_OK);
    EXPECT_EQ(loglik, expected.loglik);
}

TEST(CapiFit, NoMeanSpecHasThreeParameters)
{
    FitSet fits = fit_series(simulated(), 0);
    const ve_fit* fit = fits[0];
    std::size_t count = 0;
    ASSERT_EQ(ve_fit_param_count(fit, &count), VE_OK);
    EXPECT_EQ(count, 3u);
    const char* name = nullptr;
    ASSERT_EQ(ve_fit_param_name(fit, 0, &name), VE_OK);
    EXPECT_STREQ(name, "omega");
}

TEST(CapiFit, SeriesViewsHaveNobsElements)
{
    FitSet fits = fit_series(simulated(), 1);
    const ve_fit* fit = fits[0];
    std::size_t nobs = 0;
    ASSERT_EQ(ve_fit_nobs(fit, &nobs), VE_OK);
    EXPECT_EQ(nobs, kT);
    const SeriesView views[] = {ve_fit_sigma, ve_fit_residuals, ve_fit_std_residuals, ve_fit_fitted};
    for (SeriesView view : views)
    {
        const double* data = nullptr;
        std::size_t length = 0;
        ASSERT_EQ(view(fit, &data, &length), VE_OK);
        EXPECT_EQ(length, kT);
        ASSERT_NE(data, nullptr);
        EXPECT_TRUE(std::isfinite(data[length - 1]));
    }
}

TEST(CapiFit, AccessorsRejectBadArguments)
{
    FitSet fits = fit_series(simulated(), 1);
    const ve_fit* fit = fits[0];
    double values[4] = {};
    int32_t status = 0;
    const double* data = nullptr;
    std::size_t length = 0;
    const char* name = nullptr;
    EXPECT_EQ(ve_fit_param_name(fit, 4, &name), VE_INVALID_ARGUMENT);
    EXPECT_EQ(ve_fit_param_name(fit, 0, nullptr), VE_INVALID_ARGUMENT);
    EXPECT_EQ(ve_fit_std_errors(fit, 2, values), VE_INVALID_ARGUMENT);
    EXPECT_EQ(ve_fit_std_errors(fit, -1, values), VE_INVALID_ARGUMENT);
    EXPECT_EQ(ve_fit_t_values(fit, 7, values), VE_INVALID_ARGUMENT);
    EXPECT_EQ(ve_fit_p_values(fit, 7, values), VE_INVALID_ARGUMENT);
    EXPECT_EQ(ve_aic(fit, 2, values), VE_INVALID_ARGUMENT);
    EXPECT_EQ(ve_bic(fit, -1, values), VE_INVALID_ARGUMENT);
    EXPECT_EQ(ve_hqic(fit, 2, values), VE_INVALID_ARGUMENT);
    EXPECT_EQ(ve_sic(fit, 2, values), VE_INVALID_ARGUMENT);
    EXPECT_EQ(ve_fit_estimates(fit, nullptr), VE_INVALID_ARGUMENT);
    EXPECT_EQ(ve_fit_std_errors(fit, VE_SE_HESSIAN, nullptr), VE_INVALID_ARGUMENT);
    EXPECT_EQ(ve_loglik(fit, nullptr), VE_INVALID_ARGUMENT);
    EXPECT_EQ(ve_fit_sigma(fit, nullptr, &length), VE_INVALID_ARGUMENT);
    EXPECT_EQ(ve_fit_sigma(fit, &data, nullptr), VE_INVALID_ARGUMENT);
    EXPECT_EQ(ve_fit_status(fit, nullptr), VE_INVALID_ARGUMENT);
    EXPECT_EQ(ve_fit_nobs(fit, nullptr), VE_INVALID_ARGUMENT);
    EXPECT_EQ(ve_fit_param_count(fit, nullptr), VE_INVALID_ARGUMENT);
    EXPECT_EQ(ve_fit_status(nullptr, &status), VE_INVALID_HANDLE);
    EXPECT_EQ(ve_fit_estimates(nullptr, values), VE_INVALID_HANDLE);
    EXPECT_EQ(ve_fit_residuals(nullptr, &data, &length), VE_INVALID_HANDLE);
    EXPECT_EQ(ve_aic(nullptr, VE_IC_TOTAL, values), VE_INVALID_HANDLE);
    EXPECT_EQ(ve_fit_param_name(nullptr, 0, &name), VE_INVALID_HANDLE);
    ve_fit_free(nullptr);
}

TEST(CapiFit, HandlesOutliveInputs)
{
    std::vector<double> data = simulated();
    const ve::FitResult expected = ve::fit_one(data, ve::Spec{}, ve::FitConfig{});
    FitSet fits(0);
    {
        ve::testing::ContextPtr context = ve::testing::make_context(2);
        ve::testing::SpecPtr spec = ve::testing::make_spec(1);
        ve_fit_options* options = nullptr;
        ASSERT_EQ(ve_fit_options_create(&options), VE_OK);
        ve::testing::PanelPtr panel = ve::testing::make_panel(data.data(), data.size(), 1);
        ve::testing::BatchInputs inputs;
        inputs.specs = {spec.get()};
        inputs.panel = panel.get();
        inputs.job_spec = {0};
        inputs.job_series = {0};
        inputs.options = options;
        ve::testing::BatchPtr batch = ve::testing::build_batch(inputs);
        fits = ve::testing::fit_all(context.get(), batch.get());
        ve_fit_options_free(options);
    }
    std::vector<double>().swap(data);
    ASSERT_EQ(fits.size(), 1u);
    const Snapshot snapshot = ve::testing::read_snapshot(fits[0]);
    EXPECT_EQ(snapshot.estimate, expected.estimate);
    EXPECT_EQ(snapshot.sigma, expected.sigma);
}

TEST(CapiFit, ConcurrentReads)
{
    FitSet fits = fit_series(simulated(), 1);
    const ve_fit* fit = fits[0];
    const Snapshot baseline = ve::testing::read_snapshot(fit);
    std::atomic<int> mismatches{0};
    std::vector<std::thread> readers;
    for (int worker = 0; worker < 8; ++worker)
    {
        readers.emplace_back(
            [&]
            {
                for (int round = 0; round < 1000; ++round)
                {
                    if (ve::testing::read_snapshot(fit) != baseline)
                    {
                        ++mismatches;
                    }
                }
            });
    }
    for (std::thread& reader : readers)
    {
        reader.join();
    }
    EXPECT_EQ(mismatches.load(), 0);
}

TEST(CapiFit, FailedFitAccessorsReturnNaN)
{
    const std::vector<double> constant(kT, 0.25);
    FitSet fits = fit_series(constant, 1);
    const ve_fit* fit = fits[0];
    ASSERT_NE(fit, nullptr);
    int32_t status = -1;
    ASSERT_EQ(ve_fit_status(fit, &status), VE_OK);
    EXPECT_EQ(status, VE_FIT_FAILED);
    double loglik = 0.0;
    EXPECT_EQ(ve_loglik(fit, &loglik), VE_OK);
    EXPECT_TRUE(std::isnan(loglik));
    double criterion = 0.0;
    EXPECT_EQ(ve_aic(fit, VE_IC_PER_OBS, &criterion), VE_OK);
    EXPECT_TRUE(std::isnan(criterion));
    EXPECT_EQ(ve_sic(fit, VE_IC_TOTAL, &criterion), VE_OK);
    EXPECT_TRUE(std::isnan(criterion));
    const std::vector<double> errors = read_array(fit, ve_fit_std_errors, VE_SE_ROBUST);
    EXPECT_TRUE(std::isnan(errors[0]));
    const std::vector<double> p_values = read_array(fit, ve_fit_p_values, VE_SE_HESSIAN);
    EXPECT_TRUE(std::isnan(p_values[0]));
    const double* data = reinterpret_cast<const double*>(0x1);
    std::size_t length = 5;
    EXPECT_EQ(ve_fit_sigma(fit, &data, &length), VE_OK);
    EXPECT_EQ(length, 0u);
}
