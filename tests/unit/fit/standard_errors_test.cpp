#include "fit/standard_errors.hpp"

#include "fit/fit.hpp"
#include "model/layout.hpp"
#include "model/data.hpp"
#include "model/start_values.hpp"
#include "model/parameter_map.hpp"
#include "simulated_garch.hpp"

#include <cmath>
#include <cstddef>
#include <string>
#include <gtest/gtest.h>
#include <span>
#include <vector>

namespace
{

ve::fit::StandardErrors standard_errors_at(const ve::model::Data& data, const ve::Spec& spec,
                                           std::span<const double> theta)
{
    ve::fit::LikelihoodProblem problem{data, ve::model::layout_of(spec), ve::VarInit::Sample};
    std::size_t k = theta.size();
    std::vector<double> residuals(data.r.size());
    std::vector<double> sigma2(data.r.size());
    std::vector<double> scores(data.r.size() * k);
    double loglik = ve::fit::loglik_with_scores(problem, theta, {residuals, sigma2, scores});
    EXPECT_TRUE(std::isfinite(loglik));
    ve::fit::GradientFn gradient = [&problem](std::span<const double> point, std::span<double> grad)
    { return ve::fit::loglik_gradient(problem, point, grad); };
    return ve::fit::standard_errors(gradient, theta, scores, problem.layout, ve::model::standardization_of(data));
}

ve::fit::StandardErrors standard_errors_at(std::span<const double> r, const ve::Spec& spec,
                                           std::span<const double> theta)
{
    return standard_errors_at(ve::model::Data{r}, spec, theta);
}

void expect_nan_only_at(const ve::fit::StandardErrors& errors, std::size_t boundary, std::size_t k)
{
    ASSERT_EQ(errors.hessian.size(), k);
    ASSERT_EQ(errors.robust.size(), k);
    for (std::size_t i = 0; i < k; ++i)
    {
        if (i == boundary)
        {
            EXPECT_TRUE(std::isnan(errors.hessian[i]));
            EXPECT_TRUE(std::isnan(errors.robust[i]));
            continue;
        }
        EXPECT_TRUE(std::isfinite(errors.hessian[i])) << i;
        EXPECT_GT(errors.hessian[i], 0.0) << i;
        EXPECT_TRUE(std::isfinite(errors.robust[i])) << i;
        EXPECT_GT(errors.robust[i], 0.0) << i;
    }
}

TEST(StandardErrors, BoundaryParameterGetsNaNOthersFinite)
{
    std::vector<double> r = ve::testing::simulate_arma_garch(ve::testing::SimulatedModel{}, 31, 5000);
    ve::Spec spec;
    spec.p = 2;
    ve::FitResult fitted = ve::fit_one(r, spec, ve::FitConfig{});
    ASSERT_TRUE(fitted.status == ve::FitStatus::Converged || fitted.status == ve::FitStatus::MaxIter);
    std::size_t alpha2 = ve::model::layout_of(spec).alpha_begin() + 1;
    std::vector<double> theta = fitted.estimate;
    theta[alpha2] = 0.0;
    expect_nan_only_at(standard_errors_at(r, spec, theta), alpha2, theta.size());
}

TEST(StandardErrors, StdShapeNearTheNormalLimitGetsNaNOthersFinite)
{
    std::vector<double> r = ve::testing::simulate_arma_garch(ve::testing::SimulatedModel{}, 31, 5000);
    ve::Spec spec;
    spec.dist = ve::Dist::Std;
    ve::FitResult fitted = ve::fit_one(r, spec, ve::FitConfig{});
    ASSERT_TRUE(fitted.status == ve::FitStatus::Converged || fitted.status == ve::FitStatus::MaxIter);
    std::size_t shape = ve::model::layout_of(spec).shape();
    std::vector<double> theta = fitted.estimate;
    for (double nu : {1e9, 1e22})
    {
        theta[shape] = nu;
        expect_nan_only_at(standard_errors_at(r, spec, theta), shape, theta.size());
    }
}

// Each standard error is sqrt(v_i) = 0.1 (i + 1): the log-likelihood is -sum (theta_i - center_i)^2 / (2 v_i), and
// every observation's score is +-1 in each entry.
void expect_exact_quadratic_errors(const ve::model::Layout& layout, std::span<const double> center,
                                   std::size_t boundary)
{
    std::size_t k = center.size();
    std::vector<double> center_copy(center.begin(), center.end());
    ve::fit::GradientFn gradient = [center_copy](std::span<const double> theta, std::span<double> grad)
    {
        for (std::size_t i = 0; i < theta.size(); ++i)
        {
            double sd = 0.1 * static_cast<double>(i + 1);
            grad[i] = -(theta[i] - center_copy[i]) / (sd * sd);
        }
        return true;
    };
    constexpr std::size_t kCount = 10;
    std::vector<double> scores(kCount * k);
    for (std::size_t t = 0; t < scores.size(); ++t)
    {
        scores[t] = (t / k) % 2 == 0 ? 1.0 : -1.0;
    }
    ve::fit::StandardErrors errors = ve::fit::standard_errors(gradient, center, scores, layout, ve::model::Standardization{});
    for (std::size_t i = 0; i < k; ++i)
    {
        double sd = 0.1 * static_cast<double>(i + 1);
        if (i == boundary)
        {
            EXPECT_TRUE(std::isnan(errors.hessian[i]));
            EXPECT_TRUE(std::isnan(errors.robust[i]));
            continue;
        }
        EXPECT_NEAR(errors.hessian[i], sd, 1e-9 * sd) << i;
        EXPECT_NEAR(errors.robust[i], sd * sd * std::sqrt(static_cast<double>(kCount)), 1e-9 * sd) << i;
    }
}

// A point forced onto these boundaries is far from the optimum of a real likelihood, whose remaining block is then
// not negative definite (and with phi = 0 the csGARCH omega and rho are not separately identified).
TEST(StandardErrors, ModelBoundaries)
{
    ve::Spec cs;
    cs.model = ve::Model::CsGarch;
    cs.dist = ve::Dist::Std;
    ve::model::Layout cs_layout = ve::model::layout_of(cs);
    std::vector<double> cs_theta{0.05, 0.01, 0.06, 0.85, 0.99, 0.0, 7.0};
    expect_exact_quadratic_errors(cs_layout, cs_theta, cs_layout.eta2_begin());

    ve::Spec all;
    all.model = ve::Model::FGarchAllGarch;
    ve::model::Layout all_layout = ve::model::layout_of(all);
    std::vector<double> all_theta{0.05, 0.02, 0.08, 0.88, 1.0 - 1e-9, 0.2, 1.5};
    expect_exact_quadratic_errors(all_layout, all_theta, all_layout.eta1_begin());
}

TEST(StandardErrors, InvalidEverywhereGivesNaN)
{
    ve::model::Layout layout = ve::model::layout_of(ve::Spec{});
    std::vector<double> theta{0.05, 0.01, 0.08, 0.9};
    std::vector<double> scores(10 * theta.size(), 0.1);
    ve::fit::GradientFn never = [](std::span<const double>, std::span<double>) { return false; };
    ve::fit::StandardErrors errors = ve::fit::standard_errors(never, theta, scores, layout, ve::model::Standardization{});
    for (std::size_t i = 0; i < theta.size(); ++i)
    {
        EXPECT_TRUE(std::isnan(errors.hessian[i]));
        EXPECT_TRUE(std::isnan(errors.robust[i]));
    }
}


TEST(StandardErrors, SkewedFinite)
{
    for (const ve::testing::SkewedFit& fit : ve::testing::skewed_fits())
    {
        SCOPED_TRACE(fit.name);
        ve::Spec spec;
        spec.model = ve::testing::skewed_processes()[fit.series].model;
        spec.dist = fit.dist;
        std::vector<double> r = ve::testing::simulate_skewed(fit.series, 5000);
        ve::FitResult result = ve::fit_one(r, spec, ve::FitConfig{});
        ASSERT_EQ(result.status, ve::FitStatus::Converged);
        for (std::size_t i = 0; i < result.estimate.size(); ++i)
        {
            EXPECT_TRUE(std::isfinite(result.se_hessian[i])) << result.names[i];
            EXPECT_GT(result.se_hessian[i], 0.0) << result.names[i];
            EXPECT_TRUE(std::isfinite(result.se_robust[i])) << result.names[i];
            EXPECT_GT(result.se_robust[i], 0.0) << result.names[i];
        }
    }
}

TEST(StandardErrors, LeavesOutNonEstimated)
{
    std::vector<double> r = ve::testing::simulate_arma_garch(ve::testing::SimulatedModel{}, 42, 3000);
    ve::Spec reduced;
    reduced.include_mean = false;
    ve::FitResult fitted = ve::fit_one(r, reduced, ve::FitConfig{});
    ASSERT_EQ(fitted.status, ve::FitStatus::Converged);
    ve::fit::StandardErrors expected = standard_errors_at(r, reduced, fitted.estimate);

    ve::Spec fixed;
    fixed.fixed = {{"mu", 0.0}};
    std::vector<double> theta{0.0};
    theta.insert(theta.end(), fitted.estimate.begin(), fitted.estimate.end());
    ve::fit::StandardErrors errors = standard_errors_at(r, fixed, theta);
    ASSERT_EQ(errors.hessian.size(), theta.size());
    EXPECT_TRUE(std::isnan(errors.hessian[0]));
    EXPECT_TRUE(std::isnan(errors.robust[0]));
    for (std::size_t i = 0; i < expected.hessian.size(); ++i)
    {
        EXPECT_NEAR(errors.hessian[i + 1], expected.hessian[i], 1e-9 * expected.hessian[i]) << i;
        EXPECT_NEAR(errors.robust[i + 1], expected.robust[i], 1e-9 * expected.robust[i]) << i;
    }
}

TEST(StandardErrors, DerivedOmegaGetsNaNOthersFinite)
{
    std::vector<double> r = ve::testing::simulate_arma_garch(ve::testing::SimulatedModel{}, 42, 3000);
    ve::Spec spec;
    spec.variance_targeting = true;
    ve::FitResult fitted = ve::fit_one(r, spec, ve::FitConfig{});
    ASSERT_EQ(fitted.status, ve::FitStatus::Converged);
    expect_nan_only_at(standard_errors_at(r, spec, fitted.estimate), ve::model::layout_of(spec).omega(),
                       fitted.estimate.size());
}

// The step of a regressor coefficient is relative to s / s_x, so a column in other units gives the standard error
// of the same fit in those units: delta / 1000 for X * 1000.
TEST(StandardErrors, RegressorStepsFollowNaturalScale)
{
    const ve::testing::RegressorProcess process = ve::testing::regressor_processes()[0];
    ve::testing::RegressorSeries series = ve::testing::simulate_regressor_process(process, 61, 3000);
    ve::Spec spec;
    spec.ar = 1;
    spec.mxreg = 2;
    ve::model::Data data{series.r, {series.x, 2}};
    ve::FitResult fitted = ve::fit_one(data, spec, ve::FitConfig{});
    ASSERT_EQ(fitted.status, ve::FitStatus::Converged);
    ve::fit::StandardErrors base = standard_errors_at(data, spec, fitted.estimate);

    std::vector<double> scaled_x = series.x;
    for (double& value : scaled_x)
    {
        value *= 1000.0;
    }
    std::vector<double> scaled_theta = fitted.estimate;
    ve::model::Layout layout = ve::model::layout_of(spec);
    for (std::size_t k = 0; k < 2; ++k)
    {
        scaled_theta[layout.mxreg_begin() + k] /= 1000.0;
    }
    ve::fit::StandardErrors scaled = standard_errors_at(ve::model::Data{series.r, {scaled_x, 2}}, spec, scaled_theta);
    for (std::size_t i = 0; i < fitted.estimate.size(); ++i)
    {
        bool regressor = i >= layout.mxreg_begin() && i < layout.mxreg_begin() + 2;
        double factor = regressor ? 1e-3 : 1.0;
        EXPECT_NEAR(scaled.hessian[i], base.hessian[i] * factor, 1e-4 * base.hessian[i] * factor) << fitted.names[i];
        EXPECT_NEAR(scaled.robust[i], base.robust[i] * factor, 1e-4 * base.robust[i] * factor) << fitted.names[i];
    }
}

TEST(StandardErrors, NothingEstimatedGivesAllNaN)
{
    std::vector<double> r = ve::testing::simulate_arma_garch(ve::testing::SimulatedModel{}, 42, 3000);
    ve::Spec spec;
    spec.fixed = {{"mu", 0.05}, {"omega", 0.01}, {"alpha1", 0.08}, {"beta1", 0.9}};
    ve::fit::StandardErrors errors = standard_errors_at(r, spec, std::vector<double>{0.05, 0.01, 0.08, 0.9});
    ASSERT_EQ(errors.hessian.size(), 4u);
    for (std::size_t i = 0; i < 4; ++i)
    {
        EXPECT_TRUE(std::isnan(errors.hessian[i]));
        EXPECT_TRUE(std::isnan(errors.robust[i]));
    }
}

}
