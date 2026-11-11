#include "fit/criteria.hpp"
#include "fit/fit.hpp"
#include "model/layout.hpp"
#include "model/likelihood.hpp"
#include "model/start_values.hpp"
#include "simulated_garch.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <initializer_list>
#include <limits>
#include <optional>
#include <random>
#include <span>
#include <string>
#include <vector>

namespace
{

using ve::FitResult;
using ve::FitStatus;
using ve::Spec;

constexpr std::size_t kObservations = 5000;

std::vector<double> simulate_garch_returns(std::size_t count)
{
    std::mt19937_64 engine(7);
    std::normal_distribution<double> shock(0.0, 1.0);
    double mu = 0.05;
    double omega = 0.01;
    double alpha = 0.08;
    double beta = 0.9;
    double sigma2 = omega / (1.0 - alpha - beta);
    std::vector<double> r(count);
    for (double& value : r)
    {
        double residual = std::sqrt(sigma2) * shock(engine);
        value = mu + residual;
        sigma2 = omega + alpha * residual * residual + beta * sigma2;
    }
    return r;
}

const std::vector<double>& simulated()
{
    static const std::vector<double> series = simulate_garch_returns(kObservations);
    return series;
}

FitResult fit_simulated(const Spec& spec)
{
    return ve::fit_one(simulated(), spec, ve::FitConfig{});
}

bool all_nan(const std::vector<double>& values)
{
    return std::ranges::all_of(values, [](double value) { return std::isnan(value); });
}

std::uint64_t bits(double value)
{
    return std::bit_cast<std::uint64_t>(value);
}

TEST(Fit, RecoversSimulatedParameters)
{
    FitResult result = fit_simulated(Spec{});
    ASSERT_EQ(result.status, FitStatus::Converged);
    ASSERT_EQ(result.estimate.size(), 4u);
    EXPECT_NEAR(result.estimate[0], 0.05, 0.03);
    EXPECT_NEAR(result.estimate[2], 0.08, 0.02);
    EXPECT_NEAR(result.estimate[3], 0.9, 0.03);
    EXPECT_TRUE(std::isfinite(result.loglik));
    EXPECT_EQ(result.nobs, kObservations);
}

TEST(Fit, NamesFollowRugarch)
{
    FitResult with_mean = fit_simulated(Spec{});
    std::vector<std::string> expected{"mu", "omega", "alpha1", "beta1"};
    EXPECT_EQ(with_mean.names, expected);
    Spec no_mean;
    no_mean.include_mean = false;
    FitResult without_mean = fit_simulated(no_mean);
    std::vector<std::string> expected_no_mean{"omega", "alpha1", "beta1"};
    EXPECT_EQ(without_mean.names, expected_no_mean);
    EXPECT_EQ(without_mean.estimate.size(), 3u);
}

TEST(Fit, StandardErrorsPositiveAndRobustDiffers)
{
    FitResult result = fit_simulated(Spec{});
    ASSERT_EQ(result.se_hessian.size(), 4u);
    ASSERT_EQ(result.se_robust.size(), 4u);
    bool any_different = false;
    for (std::size_t i = 0; i < 4; ++i)
    {
        EXPECT_GT(result.se_hessian[i], 0.0);
        EXPECT_GT(result.se_robust[i], 0.0);
        any_different = any_different || result.se_hessian[i] != result.se_robust[i];
    }
    EXPECT_TRUE(any_different);
}

TEST(Fit, HessianAndRobustStandardErrorsAgreeWhenModelIsCorrect)
{
    FitResult result = fit_simulated(Spec{});
    ASSERT_EQ(result.status, FitStatus::Converged);
    for (std::size_t i = 0; i < 4; ++i)
    {
        double ratio = result.se_robust[i] / result.se_hessian[i];
        EXPECT_GT(ratio, 0.5) << result.names[i];
        EXPECT_LT(ratio, 2.0) << result.names[i];
    }
}

TEST(Fit, FittedSeriesConsistent)
{
    FitResult result = fit_simulated(Spec{});
    ASSERT_EQ(result.sigma.size(), kObservations);
    ASSERT_EQ(result.residuals.size(), kObservations);
    ASSERT_EQ(result.std_residuals.size(), kObservations);
    ASSERT_EQ(result.fitted.size(), kObservations);
    double mu = result.estimate[0];
    const std::vector<double>& r = simulated();
    for (std::size_t t = 0; t < kObservations; ++t)
    {
        EXPECT_EQ(result.residuals[t], r[t] - mu);
        EXPECT_NEAR(result.std_residuals[t] * result.sigma[t], result.residuals[t], 1e-12);
        EXPECT_EQ(result.fitted[t], r[t] - result.residuals[t]);
        EXPECT_NEAR(result.fitted[t], mu, 1e-15);
        EXPECT_GT(result.sigma[t], 0.0);
    }
}

TEST(Fit, FittedIsZeroWithoutMean)
{
    Spec no_mean;
    no_mean.include_mean = false;
    FitResult result = fit_simulated(no_mean);
    ASSERT_EQ(result.fitted.size(), kObservations);
    EXPECT_TRUE(std::ranges::all_of(result.fitted, [](double value) { return value == 0.0; }));
    EXPECT_EQ(result.residuals[10], simulated()[10]);
}

TEST(Fit, InsufficientData)
{
    std::vector<double> r{0.1, -0.2, 0.3, -0.1, 0.2};
    FitResult result = ve::fit_one(r, Spec{}, ve::FitConfig{});
    EXPECT_EQ(result.status, FitStatus::InsufficientData);
    EXPECT_EQ(result.nobs, 5u);
    std::vector<std::string> expected{"mu", "omega", "alpha1", "beta1"};
    EXPECT_EQ(result.names, expected);
    EXPECT_EQ(result.estimate.size(), 4u);
    EXPECT_TRUE(all_nan(result.estimate));
    EXPECT_TRUE(all_nan(result.se_hessian));
    EXPECT_TRUE(all_nan(result.se_robust));
    EXPECT_TRUE(std::isnan(result.loglik));
    EXPECT_TRUE(result.sigma.empty());
    EXPECT_TRUE(result.residuals.empty());
}

TEST(Fit, EmptySeriesIsInsufficientData)
{
    FitResult result = ve::fit_one(std::vector<double>{}, Spec{}, ve::FitConfig{});
    EXPECT_EQ(result.status, FitStatus::InsufficientData);
    EXPECT_EQ(result.nobs, 0u);
}

TEST(Fit, ConstantSeriesFails)
{
    std::vector<double> r(1000, 0.3);
    FitResult result = ve::fit_one(r, Spec{}, ve::FitConfig{});
    EXPECT_EQ(result.status, FitStatus::Failed);
    EXPECT_EQ(result.nobs, 1000u);
    EXPECT_TRUE(all_nan(result.estimate));
    EXPECT_TRUE(std::isnan(result.loglik));
    EXPECT_TRUE(result.sigma.empty());
}

TEST(Fit, NonFiniteObservationFails)
{
    std::vector<double> r = simulate_garch_returns(200);
    r[50] = std::numeric_limits<double>::quiet_NaN();
    FitResult result = ve::fit_one(r, Spec{}, ve::FitConfig{});
    EXPECT_EQ(result.status, FitStatus::Failed);
    EXPECT_TRUE(all_nan(result.estimate));
    EXPECT_TRUE(result.sigma.empty());
}

TEST(Fit, StandardErrorsAreNaNOrPositiveWhenArchEffectIsAbsent)
{
    std::mt19937_64 engine(3);
    std::normal_distribution<double> shock(0.0, 1.0);
    std::vector<double> r(3000);
    std::ranges::generate(r, [&] { return shock(engine); });
    FitResult result = ve::fit_one(r, Spec{}, ve::FitConfig{});
    ASSERT_TRUE(result.status == FitStatus::Converged || result.status == FitStatus::MaxIter);
    for (std::size_t i = 0; i < result.estimate.size(); ++i)
    {
        EXPECT_TRUE(std::isfinite(result.estimate[i]));
        EXPECT_TRUE(std::isnan(result.se_hessian[i]) || result.se_hessian[i] > 0.0);
        EXPECT_TRUE(std::isnan(result.se_robust[i]) || result.se_robust[i] > 0.0);
    }
}

TEST(Fit, IterationLimitReportsMaxIter)
{
    ve::FitConfig config;
    config.max_iter = 2;
    FitResult result = ve::fit_one(simulated(), Spec{}, config);
    EXPECT_EQ(result.status, FitStatus::MaxIter);
    EXPECT_TRUE(std::isfinite(result.loglik));
    EXPECT_EQ(result.sigma.size(), kObservations);
}

struct ScaledFit
{
    double scale;
    FitResult result;
};

// mu scales like the data, omega like its square; alpha and beta do not scale.
double natural_power(const FitResult& result, std::size_t index)
{
    const std::string& name = result.names[index];
    if (name == "mu")
    {
        return 1.0;
    }
    if (name == "omega")
    {
        return 2.0;
    }
    return 0.0;
}

ScaledFit fit_scaled(std::span<const double> series, double scale, const Spec& spec,
                     const ve::FitConfig& config)
{
    std::vector<double> r(series.begin(), series.end());
    std::ranges::transform(r, r.begin(), [scale](double value) { return value * scale; });
    return {scale, ve::fit_one(r, spec, config)};
}

void expect_same_up_to_scale(const FitResult& reference, const ScaledFit& scaled)
{
    const FitResult& result = scaled.result;
    ASSERT_EQ(result.status, FitStatus::Converged) << "scale " << scaled.scale;
    double count = static_cast<double>(result.nobs);
    EXPECT_NEAR(result.loglik + count * std::log(scaled.scale), reference.loglik, 1e-6 * count)
        << "scale " << scaled.scale;
    for (std::size_t i = 0; i < result.estimate.size(); ++i)
    {
        double unit = std::pow(scaled.scale, natural_power(result, i));
        EXPECT_NEAR(result.estimate[i] / unit / reference.estimate[i], 1.0, 1e-5)
            << result.names[i] << " at scale " << scaled.scale;
        EXPECT_NEAR(result.se_hessian[i] / unit / reference.se_hessian[i], 1.0, 1e-3)
            << result.names[i] << " at scale " << scaled.scale;
        EXPECT_NEAR(result.se_robust[i] / unit / reference.se_robust[i], 1.0, 1e-3)
            << result.names[i] << " at scale " << scaled.scale;
    }
}

void expect_scale_invariant(std::span<const double> series, const Spec& spec, ve::VarInit var_init)
{
    ve::FitConfig config;
    config.var_init = var_init;
    FitResult reference = fit_scaled(series, 1.0, spec, config).result;
    ASSERT_EQ(reference.status, FitStatus::Converged);
    for (double scale : {1e-2, 1e-4})
    {
        expect_same_up_to_scale(reference, fit_scaled(series, scale, spec, config));
    }
}

TEST(Fit, ScaleInvariantWithMean)
{
    expect_scale_invariant(simulated(), Spec{}, ve::VarInit::Sample);
    expect_scale_invariant(simulated(), Spec{}, ve::VarInit::Backcast);
}

TEST(Fit, ScaleInvariantWithoutMean)
{
    Spec no_mean;
    no_mean.include_mean = false;
    expect_scale_invariant(simulated(), no_mean, ve::VarInit::Sample);
    expect_scale_invariant(simulated(), no_mean, ve::VarInit::Backcast);
}

TEST(Fit, Deterministic)
{
    FitResult first = fit_simulated(Spec{});
    FitResult second = fit_simulated(Spec{});
    ASSERT_EQ(first.estimate.size(), second.estimate.size());
    for (std::size_t i = 0; i < first.estimate.size(); ++i)
    {
        EXPECT_EQ(bits(first.estimate[i]), bits(second.estimate[i]));
        EXPECT_EQ(bits(first.se_hessian[i]), bits(second.se_hessian[i]));
        EXPECT_EQ(bits(first.se_robust[i]), bits(second.se_robust[i]));
    }
}

using ve::testing::SimulatedModel;

std::size_t index_of(const FitResult& result, const std::string& name)
{
    auto found = std::ranges::find(result.names, name);
    EXPECT_NE(found, result.names.end()) << name;
    return static_cast<std::size_t>(found - result.names.begin());
}

double estimate_of(const FitResult& result, const std::string& name)
{
    return result.estimate[index_of(result, name)];
}

bool all_finite(const std::vector<double>& values)
{
    return std::ranges::all_of(values, [](double value) { return std::isfinite(value); });
}

Spec generic_spec(int p, int q, ve::Dist dist)
{
    Spec spec;
    spec.p = p;
    spec.q = q;
    spec.dist = dist;
    return spec;
}

SimulatedModel arma11_sgarch21_std()
{
    SimulatedModel model;
    model.ar = {0.5};
    model.ma = {-0.2};
    model.alpha = {0.05, 0.04};
    model.beta = {0.88};
    model.dist = ve::Dist::Std;
    model.shape = 6.0;
    return model;
}

TEST(FitGeneric, RecoversSGarch21Std)
{
    SimulatedModel model;
    model.alpha = {0.05, 0.04};
    model.beta = {0.88};
    model.dist = ve::Dist::Std;
    model.shape = 6.0;
    std::vector<double> r = ve::testing::simulate_arma_garch(model, 21, kObservations);
    FitResult result = ve::fit_one(r, generic_spec(2, 1, ve::Dist::Std), ve::FitConfig{});
    ASSERT_EQ(result.status, FitStatus::Converged);
    EXPECT_NEAR(estimate_of(result, "alpha1"), 0.05, 0.03);
    EXPECT_NEAR(estimate_of(result, "alpha2"), 0.04, 0.03);
    EXPECT_NEAR(estimate_of(result, "beta1"), 0.88, 0.04);
    EXPECT_NEAR(estimate_of(result, "shape"), 6.0, 1.5);
}

TEST(FitGeneric, RecoversArma11SGarch11Ged)
{
    SimulatedModel model;
    model.ar = {0.5};
    model.ma = {-0.2};
    model.dist = ve::Dist::Ged;
    model.shape = 1.5;
    std::vector<double> r = ve::testing::simulate_arma_garch(model, 22, kObservations);
    Spec spec = generic_spec(1, 1, ve::Dist::Ged);
    spec.ar = 1;
    spec.ma = 1;
    FitResult result = ve::fit_one(r, spec, ve::FitConfig{});
    ASSERT_EQ(result.status, FitStatus::Converged);
    EXPECT_NEAR(estimate_of(result, "ar1"), 0.5, 0.05);
    EXPECT_NEAR(estimate_of(result, "ma1"), -0.2, 0.06);
    EXPECT_NEAR(estimate_of(result, "shape"), 1.5, 0.15);
}

TEST(FitGeneric, RecoversSGarch12Norm)
{
    SimulatedModel model;
    model.beta = {0.5, 0.4};
    std::vector<double> r = ve::testing::simulate_arma_garch(model, 23, kObservations);
    FitResult result = ve::fit_one(r, generic_spec(1, 2, ve::Dist::Norm), ve::FitConfig{});
    ASSERT_EQ(result.status, FitStatus::Converged);
    EXPECT_NEAR(estimate_of(result, "beta1") + estimate_of(result, "beta2"), 0.9, 0.03);
}

TEST(FitGeneric, NamesAreGenerated)
{
    Spec spec = generic_spec(2, 1, ve::Dist::Std);
    spec.ar = 1;
    FitResult result =
        ve::fit_one(ve::testing::simulate_portable_garch_series(5, 1000), spec, ve::FitConfig{});
    std::vector<std::string> expected{"mu", "ar1", "omega", "alpha1", "alpha2", "beta1", "shape"};
    EXPECT_EQ(result.names, expected);
    EXPECT_EQ(result.estimate.size(), expected.size());
    EXPECT_EQ(result.se_hessian.size(), expected.size());
    EXPECT_EQ(result.se_robust.size(), expected.size());
}

TEST(FitGeneric, ScaleInvarianceStillHolds)
{
    std::vector<double> r = ve::testing::simulate_arma_garch(arma11_sgarch21_std(), 25, kObservations);
    Spec spec = generic_spec(2, 1, ve::Dist::Std);
    spec.ar = 1;
    spec.ma = 1;
    expect_scale_invariant(r, spec, ve::VarInit::Sample);
    expect_scale_invariant(r, spec, ve::VarInit::Backcast);
}

void expect_insufficient(const Spec& spec, std::size_t length, std::size_t parameter_count)
{
    std::vector<double> r = ve::testing::simulate_portable_garch_series(9, length);
    FitResult result = ve::fit_one(r, spec, ve::FitConfig{});
    EXPECT_EQ(result.status, FitStatus::InsufficientData) << "length " << length;
    EXPECT_EQ(result.nobs, length);
    EXPECT_EQ(result.names.size(), parameter_count);
    ASSERT_EQ(result.estimate.size(), parameter_count);
    EXPECT_TRUE(all_nan(result.estimate));
    EXPECT_TRUE(std::isnan(result.loglik));
    EXPECT_TRUE(result.sigma.empty());
}

TEST(FitGeneric, ShortSeriesInsufficientData)
{
    expect_insufficient(generic_spec(3, 3, ve::Dist::Norm), 3, 8);
    Spec ar6;
    ar6.ar = 6;
    expect_insufficient(ar6, 7, 10);
    expect_insufficient(Spec{}, 5, 4);
}

TEST(FitGeneric, ExplosiveArmaEndsFinite)
{
    SimulatedModel model;
    model.ar = {0.999};
    std::vector<double> r = ve::testing::simulate_arma_garch(model, 26, kObservations);
    Spec spec;
    spec.ar = 1;
    spec.ma = 1;
    FitResult result = ve::fit_one(r, spec, ve::FitConfig{});
    ASSERT_TRUE(result.status == FitStatus::Converged || result.status == FitStatus::MaxIter ||
                result.status == FitStatus::Failed);
    EXPECT_TRUE(all_finite(result.estimate) || all_nan(result.estimate));
    if (result.status == FitStatus::Converged)
    {
        EXPECT_TRUE(all_finite(result.estimate));
        EXPECT_TRUE(std::isfinite(result.loglik));
    }
}

// Normal innovations put the Student-t optimum at nu -> infinity: the fit must stop at a large finite nu.
void expect_near_normal_std_fit(const SimulatedModel& model, const Spec& spec, std::uint64_t seed)
{
    std::vector<double> r = ve::testing::simulate_arma_garch(model, seed, kObservations);
    FitResult result = ve::fit_one(r, spec, ve::FitConfig{});
    ASSERT_TRUE(result.status == FitStatus::Converged || result.status == FitStatus::MaxIter)
        << "seed " << seed;
    EXPECT_GT(estimate_of(result, "shape"), 20.0) << "seed " << seed;
    EXPECT_TRUE(std::isfinite(estimate_of(result, "shape"))) << "seed " << seed;
    std::vector<std::string> names{"mu", "omega", "alpha1"};
    for (int j = 1; j <= spec.q; ++j)
    {
        names.push_back("beta" + std::to_string(j));
    }
    for (const std::string& name : names)
    {
        double se = result.se_hessian[index_of(result, name)];
        EXPECT_TRUE(std::isfinite(se)) << name << ", seed " << seed;
        EXPECT_GT(se, 0.0) << name << ", seed " << seed;
    }
}

TEST(FitGeneric, NearNormalDataWithStd)
{
    for (std::uint64_t seed = 1; seed <= 24; ++seed)
    {
        expect_near_normal_std_fit(SimulatedModel{}, generic_spec(1, 1, ve::Dist::Std), seed);
    }
}

TEST(FitGeneric, NearNormalDataWithArmaSGarch13Std)
{
    SimulatedModel model;
    model.ar = {0.3};
    model.ma = {0.2};
    model.beta = {0.3, 0.3, 0.3};
    Spec spec = generic_spec(1, 3, ve::Dist::Std);
    spec.ar = 1;
    spec.ma = 1;
    for (std::uint64_t seed = 1; seed <= 20; ++seed)
    {
        expect_near_normal_std_fit(model, spec, seed);
    }
}

void expect_bit_identical(const std::vector<double>& first, const std::vector<double>& second)
{
    ASSERT_EQ(first.size(), second.size());
    for (std::size_t i = 0; i < first.size(); ++i)
    {
        EXPECT_EQ(bits(first[i]), bits(second[i])) << "entry " << i;
    }
}

Spec full_spec(int order, int arma_order)
{
    Spec spec = generic_spec(order, order, ve::Dist::Std);
    spec.ar = arma_order;
    spec.ma = arma_order;
    return spec;
}

TEST(FitGeneric, DeterministicAcrossBuckets)
{
    std::vector<double> r = ve::testing::simulate_arma_garch(arma11_sgarch21_std(), 27, 2000);
    ve::FitConfig config;
    config.max_iter = 40;
    for (const Spec& spec : {Spec{}, full_spec(2, 2), full_spec(6, 6)})
    {
        FitResult first = ve::fit_one(r, spec, config);
        FitResult second = ve::fit_one(r, spec, config);
        EXPECT_EQ(first.names.size(), first.estimate.size());
        EXPECT_EQ(first.status, second.status);
        EXPECT_EQ(bits(first.loglik), bits(second.loglik));
        expect_bit_identical(first.estimate, second.estimate);
        expect_bit_identical(first.se_hessian, second.se_hessian);
        expect_bit_identical(first.se_robust, second.se_robust);
        expect_bit_identical(first.sigma, second.sigma);
    }
}

TEST(FitGeneric, BucketSpecsHaveFourElevenAndTwentySevenParameters)
{
    std::vector<double> r = ve::testing::simulate_portable_garch_series(9, 3);
    EXPECT_EQ(ve::fit_one(r, Spec{}, ve::FitConfig{}).names.size(), 4u);
    EXPECT_EQ(ve::fit_one(r, full_spec(2, 2), ve::FitConfig{}).names.size(), 11u);
    EXPECT_EQ(ve::fit_one(r, full_spec(6, 6), ve::FitConfig{}).names.size(), 27u);
}

using ve::testing::VariantProcess;

constexpr std::uint64_t kFirstVariantSeed = 31;

const VariantProcess& variant_process(std::size_t index)
{
    static const std::vector<VariantProcess> processes = ve::testing::variant_processes();
    return processes[index];
}

const std::vector<double>& variant_series(std::size_t index)
{
    static const std::vector<std::vector<double>> series = []
    {
        std::vector<std::vector<double>> all;
        for (std::size_t i = 0; i < 6; ++i)
        {
            all.push_back(ve::testing::simulate_variant(variant_process(i), kFirstVariantSeed + i, kObservations));
        }
        return all;
    }();
    return series[index];
}

Spec variant_spec(const VariantProcess& process)
{
    Spec spec;
    spec.model = process.model;
    spec.dist = process.dist;
    return spec;
}

double true_value(const VariantProcess& process, const std::string& name)
{
    bool cs = process.model == ve::Model::CsGarch;
    if (name == "mu")
    {
        return 0.05;
    }
    if (name == "omega")
    {
        return process.omega;
    }
    if (name == "alpha1")
    {
        return process.alpha;
    }
    if (name == "beta1")
    {
        return process.beta;
    }
    if (name == "gamma1")
    {
        return process.gamma;
    }
    if (name == "eta11")
    {
        return cs ? process.rho : process.eta1;
    }
    if (name == "eta21")
    {
        return cs ? process.phi : process.eta2;
    }
    if (name == "shape")
    {
        return process.shape;
    }
    return process.power;
}

std::vector<double> true_theta(const VariantProcess& process, const std::vector<std::string>& names)
{
    std::vector<double> theta;
    for (const std::string& name : names)
    {
        theta.push_back(true_value(process, name));
    }
    return theta;
}

double loglik_at(std::span<const double> r, const Spec& spec, std::span<const double> theta)
{
    return ve::model::loglik<double>(r, theta, ve::model::layout_of(spec), ve::VarInit::Sample, {});
}

class FitModels : public ::testing::TestWithParam<std::size_t>
{
};

TEST_P(FitModels, BeatsTrueParameters)
{
    const VariantProcess& process = variant_process(GetParam());
    const std::vector<double>& r = variant_series(GetParam());
    Spec spec = variant_spec(process);
    FitResult result = ve::fit_one(r, spec, ve::FitConfig{});
    ASSERT_EQ(result.status, FitStatus::Converged);
    double truth = loglik_at(r, spec, true_theta(process, result.names));
    ASSERT_TRUE(std::isfinite(truth));
    EXPECT_GE(result.loglik, truth - 1e-6 * std::abs(truth));
}

INSTANTIATE_TEST_SUITE_P(Processes, FitModels, ::testing::Range<std::size_t>(0, 6),
                         [](const ::testing::TestParamInfo<std::size_t>& info)
                         { return std::string(variant_process(info.param).name); });

// About two standard errors at T = 5000 for the apARCH gamma; no tolerance for mu and omega.
std::optional<double> recovery_tolerance(const FitResult& result, std::size_t index)
{
    ve::model::Layout layout = ve::model::layout_of(result.spec);
    const std::string& name = result.names[index];
    bool cs = layout.model == ve::Model::CsGarch;
    if (name == "mu" || name == "omega")
    {
        return std::nullopt;
    }
    if (cs && name == "eta11")
    {
        return 0.02;
    }
    if (layout.model == ve::Model::ApArch && name == "gamma1")
    {
        return 0.15;
    }
    if (name == "delta")
    {
        return 0.3;
    }
    if (name == "shape")
    {
        return layout.dist == ve::Dist::Std ? 1.5 : 0.15;
    }
    if (layout.kind(index) == ve::model::ParameterKind::Weight)
    {
        return 0.03;
    }
    return 0.06;
}

TEST(FitModels, RecoversIdentifiedParameters)
{
    for (std::size_t i = 0; i < 5; ++i)
    {
        const VariantProcess& process = variant_process(i);
        FitResult result = ve::fit_one(variant_series(i), variant_spec(process), ve::FitConfig{});
        ASSERT_EQ(result.status, FitStatus::Converged) << process.name;
        for (std::size_t j = 0; j < result.names.size(); ++j)
        {
            std::optional<double> tolerance = recovery_tolerance(result, j);
            double truth = true_value(process, result.names[j]);
            EXPECT_TRUE(!tolerance || std::abs(result.estimate[j] - truth) <= *tolerance)
                << process.name << " " << result.names[j] << " = " << result.estimate[j] << ", true " << truth;
        }
    }
}

TEST(FitModels, GarchSubmodelFitsLikeSGarch)
{
    Spec submodel;
    submodel.model = ve::Model::FGarchGarch;
    FitResult sgarch = fit_simulated(Spec{});
    FitResult fgarch = fit_simulated(submodel);
    ASSERT_EQ(sgarch.status, FitStatus::Converged);
    ASSERT_EQ(fgarch.status, FitStatus::Converged);
    EXPECT_LE(std::abs(fgarch.loglik - sgarch.loglik), 1e-6 * std::abs(sgarch.loglik));
}

// omega scales like sigma^power; the eGARCH omega shifts by (1 - beta) log(scale^2) instead.
double expected_scaled_estimate(const FitResult& reference, std::size_t index, double scale)
{
    ve::model::Layout layout = ve::model::layout_of(reference.spec);
    double value = reference.estimate[index];
    if (layout.include_mean && index == layout.mu())
    {
        return value * scale;
    }
    if (index != layout.omega())
    {
        return value;
    }
    if (layout.model == ve::Model::EGarch)
    {
        return value + (1.0 - reference.estimate[layout.beta_begin()]) * std::log(scale * scale);
    }
    double power = ve::model::power_of<double>(reference.estimate, layout);
    return value * std::pow(scale, power);
}

// The scaled omega depends on beta (eGARCH) or on a free power as well, so its standard error does not
// scale alone.
std::optional<double> se_unit(const FitResult& reference, std::size_t index, double scale)
{
    ve::model::Layout layout = ve::model::layout_of(reference.spec);
    if (layout.include_mean && index == layout.mu())
    {
        return scale;
    }
    if (index != layout.omega())
    {
        return 1.0;
    }
    if (layout.model == ve::Model::EGarch || layout.free_power)
    {
        return std::nullopt;
    }
    return std::pow(scale, layout.fixed_power);
}

void expect_scaled_se(double expected, double actual, const std::string& name)
{
    if (std::isnan(expected))
    {
        EXPECT_TRUE(std::isnan(actual)) << name;
        return;
    }
    EXPECT_NEAR(actual / expected, 1.0, 1e-3) << name;
}

void expect_se_scaled(const FitResult& reference, const FitResult& result, std::size_t index, double scale)
{
    std::optional<double> unit = se_unit(reference, index, scale);
    if (!unit)
    {
        return;
    }
    expect_scaled_se(reference.se_hessian[index] * *unit, result.se_hessian[index], result.names[index]);
    expect_scaled_se(reference.se_robust[index] * *unit, result.se_robust[index], result.names[index]);
}

void expect_model_scale_invariant(std::span<const double> series, const Spec& spec, ve::VarInit var_init)
{
    constexpr double kScale = 100.0;
    ve::FitConfig config;
    config.var_init = var_init;
    // A free power maps omega nonlinearly, so its standard errors scale only at a stationary point.
    config.tol = 1e-9;
    FitResult reference = fit_scaled(series, 1.0, spec, config).result;
    FitResult result = fit_scaled(series, kScale, spec, config).result;
    ASSERT_EQ(reference.status, FitStatus::Converged);
    ASSERT_EQ(result.status, FitStatus::Converged);
    double count = static_cast<double>(result.nobs);
    EXPECT_NEAR(result.loglik + count * std::log(kScale), reference.loglik, 1e-6 * count);
    for (std::size_t i = 0; i < result.estimate.size(); ++i)
    {
        EXPECT_NEAR(result.estimate[i] / expected_scaled_estimate(reference, i, kScale), 1.0, 1e-5)
            << result.names[i];
        expect_se_scaled(reference, result, i, kScale);
    }
}

// csGARCH fits its own series; the others fit the fGARCH-ALLGARCH series, which nests every power submodel.
TEST(FitModels, ScaleInvarianceAllModels)
{
    constexpr std::size_t kCsGarchSeries = 4;
    constexpr std::size_t kAllGarchSeries = 5;
    for (int model = 0; model < 14; ++model)
    {
        Spec spec;
        spec.model = static_cast<ve::Model>(model);
        SCOPED_TRACE(model);
        std::span<const double> r = variant_series(kAllGarchSeries);
        if (spec.model == ve::Model::CsGarch)
        {
            r = variant_series(kCsGarchSeries);
        }
        expect_model_scale_invariant(r, spec, ve::VarInit::Sample);
        expect_model_scale_invariant(r, spec, ve::VarInit::Backcast);
    }
}

TEST(FitModels, DeterministicAcrossBuckets)
{
    std::vector<double> r = ve::testing::simulate_variant(variant_process(5), kFirstVariantSeed + 5, 600);
    Spec spec = full_spec(6, 6);
    spec.model = ve::Model::FGarchAllGarch;
    ASSERT_EQ(ve::model::layout_of(spec).count(), 40u);
    ve::FitConfig config;
    config.max_iter = 30;
    FitResult first = ve::fit_one(r, spec, config);
    FitResult second = ve::fit_one(r, spec, config);
    EXPECT_EQ(first.status, second.status);
    EXPECT_EQ(bits(first.loglik), bits(second.loglik));
    expect_bit_identical(first.estimate, second.estimate);
    expect_bit_identical(first.se_hessian, second.se_hessian);
    expect_bit_identical(first.se_robust, second.se_robust);
    expect_bit_identical(first.sigma, second.sigma);
}

TEST(FitModels, IGarchDerivedParameter)
{
    FitResult result = ve::fit_one(variant_series(3), variant_spec(variant_process(3)), ve::FitConfig{});
    ASSERT_EQ(result.status, FitStatus::Converged);
    std::size_t derived = index_of(result, "beta1");
    EXPECT_NEAR(result.estimate[derived], 1.0 - estimate_of(result, "alpha1"), 1e-15);
    for (std::size_t i = 0; i < result.estimate.size(); ++i)
    {
        bool expect_nan = i == derived;
        EXPECT_EQ(std::isnan(result.se_hessian[i]), expect_nan) << result.names[i];
        EXPECT_EQ(std::isnan(result.se_robust[i]), expect_nan) << result.names[i];
        EXPECT_TRUE(expect_nan || (result.se_hessian[i] > 0.0 && std::isfinite(result.se_hessian[i])));
        EXPECT_TRUE(expect_nan || (result.se_robust[i] > 0.0 && std::isfinite(result.se_robust[i])));
    }
    double estimated = static_cast<double>(result.estimate.size() - 1);
    double n = static_cast<double>(result.nobs);
    EXPECT_DOUBLE_EQ(ve::aic(result, ve::IcScale::Total), -2.0 * result.loglik + 2.0 * estimated);
    EXPECT_DOUBLE_EQ(ve::bic(result, ve::IcScale::Total), -2.0 * result.loglik + std::log(n) * estimated);
}

void expect_ends_finite(const FitResult& result)
{
    ASSERT_TRUE(result.status == FitStatus::Converged || result.status == FitStatus::MaxIter ||
                result.status == FitStatus::Failed);
    EXPECT_TRUE(all_finite(result.estimate) || all_nan(result.estimate));
    if (result.status == FitStatus::Converged)
    {
        EXPECT_TRUE(all_finite(result.estimate));
        EXPECT_TRUE(std::isfinite(result.loglik));
    }
}

TEST(FitModels, PowerStdLowShapeEndsFinite)
{
    VariantProcess process = variant_process(2);
    process.dist = ve::Dist::Std;
    process.shape = 2.5;
    std::vector<double> r = ve::testing::simulate_variant(process, kFirstVariantSeed + 2, kObservations);
    FitResult result = ve::fit_one(r, variant_spec(process), ve::FitConfig{});
    expect_ends_finite(result);
    if (result.status == FitStatus::Converged)
    {
        EXPECT_GT(estimate_of(result, "shape"), estimate_of(result, "delta"));
    }
}

TEST(FitModels, EGarchOutlierEndsFinite)
{
    std::vector<double> r = variant_series(0);
    r[2500] = 1e3 * ve::model::standardization_of(r).scale;
    expect_ends_finite(ve::fit_one(r, variant_spec(variant_process(0)), ve::FitConfig{}));
}

TEST(FitModels, ShortSeriesUsesEstimatedCount)
{
    Spec spec;
    spec.model = ve::Model::IGarch;
    spec.include_mean = false;
    ASSERT_EQ(ve::model::layout_of(spec).estimated_count(), 2u);
    std::vector<double> r = ve::testing::simulate_portable_garch_series(9, 4);
    EXPECT_NE(ve::fit_one(r, spec, ve::FitConfig{}).status, FitStatus::InsufficientData);
    r.pop_back();
    EXPECT_EQ(ve::fit_one(r, spec, ve::FitConfig{}).status, FitStatus::InsufficientData);
}

}
