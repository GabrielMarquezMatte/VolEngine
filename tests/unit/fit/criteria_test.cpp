#include "fit/criteria.hpp"
#include "fit/fit_result.hpp"

#include <cmath>
#include <cstddef>
#include <gtest/gtest.h>
#include <limits>

namespace
{

using ve::IcScale;

ve::FitResult synthetic_result(double loglik, std::size_t parameters, std::size_t observations)
{
    ve::FitResult result;
    result.status = ve::FitStatus::Converged;
    result.estimate.assign(parameters, 0.1);
    result.loglik = loglik;
    result.nobs = observations;
    return result;
}

TEST(Criteria, MatchFormulas)
{
    ve::FitResult result = synthetic_result(-100.0, 4, 200);
    double n = 200.0;
    EXPECT_NEAR(ve::aic(result, IcScale::PerObs), (200.0 + 8.0) / n, 1e-14);
    EXPECT_NEAR(ve::bic(result, IcScale::PerObs), (200.0 + 4.0 * std::log(200.0)) / n, 1e-14);
    EXPECT_NEAR(ve::hqic(result, IcScale::PerObs), (200.0 + 8.0 * std::log(std::log(200.0))) / n, 1e-14);
    EXPECT_NEAR(ve::sic(result, IcScale::PerObs), 200.0 / n + std::log(208.0 / 200.0), 1e-14);
}

TEST(Criteria, TotalScaleIsPerObservationTimesCount)
{
    ve::FitResult result = synthetic_result(-100.0, 4, 200);
    EXPECT_NEAR(ve::aic(result, IcScale::Total), 208.0, 1e-12);
    EXPECT_NEAR(ve::bic(result, IcScale::Total), 200.0 + 4.0 * std::log(200.0), 1e-12);
    EXPECT_NEAR(ve::hqic(result, IcScale::Total), 200.0 + 8.0 * std::log(std::log(200.0)), 1e-12);
    EXPECT_NEAR(ve::sic(result, IcScale::Total), 200.0 + 200.0 * std::log(208.0 / 200.0), 1e-12);
}

TEST(Criteria, NonFiniteLoglikGivesNaN)
{
    ve::FitResult result = synthetic_result(std::numeric_limits<double>::quiet_NaN(), 4, 200);
    EXPECT_TRUE(std::isnan(ve::aic(result, IcScale::PerObs)));
    EXPECT_TRUE(std::isnan(ve::bic(result, IcScale::Total)));
    EXPECT_TRUE(std::isnan(ve::hqic(result, IcScale::PerObs)));
    EXPECT_TRUE(std::isnan(ve::sic(result, IcScale::Total)));
}

}
