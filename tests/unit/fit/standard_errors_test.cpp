#include "fit/standard_errors.hpp"

#include "fit/fit.hpp"
#include "model/layout.hpp"
#include "model/start_values.hpp"
#include "simulated_garch.hpp"

#include <cmath>
#include <cstddef>
#include <gtest/gtest.h>
#include <span>
#include <vector>

namespace
{

ve::fit::StandardErrors standard_errors_at(std::span<const double> r, const ve::Spec& spec,
                                           std::span<const double> theta)
{
    ve::fit::LikelihoodProblem problem{r, ve::model::layout_of(spec), ve::VarInit::Sample};
    std::size_t k = theta.size();
    std::vector<double> residuals(r.size());
    std::vector<double> sigma2(r.size());
    std::vector<double> scores(r.size() * k);
    double loglik = ve::fit::loglik_with_scores(problem, theta, {residuals, sigma2, scores});
    EXPECT_TRUE(std::isfinite(loglik));
    ve::fit::GradientFn gradient = [&problem](std::span<const double> point, std::span<double> grad)
    { return ve::fit::loglik_gradient(problem, point, grad); };
    double series_scale = ve::model::standardization_of(r).scale;
    return ve::fit::standard_errors(gradient, theta, scores, problem.layout, series_scale);
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

TEST(StandardErrors, InvalidEverywhereGivesNaN)
{
    ve::model::Layout layout = ve::model::layout_of(ve::Spec{});
    std::vector<double> theta{0.05, 0.01, 0.08, 0.9};
    std::vector<double> scores(10 * theta.size(), 0.1);
    ve::fit::GradientFn never = [](std::span<const double>, std::span<double>) { return false; };
    ve::fit::StandardErrors errors = ve::fit::standard_errors(never, theta, scores, layout, 1.0);
    for (std::size_t i = 0; i < theta.size(); ++i)
    {
        EXPECT_TRUE(std::isnan(errors.hessian[i]));
        EXPECT_TRUE(std::isnan(errors.robust[i]));
    }
}

}
