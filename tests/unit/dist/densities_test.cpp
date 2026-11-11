#include "ad/dual.hpp"
#include "dist/densities.hpp"
#include "dist/reference_quadrature.hpp"

#include <cmath>
#include <gtest/gtest.h>
#include <initializer_list>
#include <numbers>
#include <string>

namespace
{

using ve::Dist;

struct DensityCase
{
    Dist dist;
    double shape;
};

std::string case_name(const ::testing::TestParamInfo<DensityCase>& info)
{
    std::string names[] = {"Norm", "Std", "Ged"};
    std::string shape = std::to_string(static_cast<int>(std::lround(info.param.shape * 10.0)));
    return names[static_cast<int>(info.param.dist)] + shape;
}

class Densities : public ::testing::TestWithParam<DensityCase>
{
};

TEST_P(Densities, IntegrateToOneWithUnitVariance)
{
    DensityCase c = GetParam();
    ve::dist::LogDensity<double> density = ve::dist::make_log_density(c.dist, c.shape);
    auto f = [&](double z) { return std::exp(ve::dist::log_density(density, z)); };
    double mass = ve::testing::trapezoid(f);
    double variance = ve::testing::trapezoid([&](double z) { return z * z * f(z); });
    if (c.dist == Dist::Std)
    {
        mass += ve::testing::student_tails(density, c.shape, 0.0);
        variance += ve::testing::student_tails(density, c.shape, 2.0);
    }
    EXPECT_NEAR(mass, 1.0, 1e-6);
    EXPECT_NEAR(variance, 1.0, 1e-6);
}

INSTANTIATE_TEST_SUITE_P(Densities, Densities,
                         ::testing::Values(DensityCase{Dist::Norm, 0.0}, DensityCase{Dist::Std, 5.0},
                                           DensityCase{Dist::Std, 2.5}, DensityCase{Dist::Ged, 0.8},
                                           DensityCase{Dist::Ged, 1.5}, DensityCase{Dist::Ged, 2.0}),
                         case_name);

TEST(Densities, GedWithShapeTwoIsNormal)
{
    ve::dist::LogDensity<double> ged = ve::dist::make_log_density(Dist::Ged, 2.0);
    ve::dist::LogDensity<double> norm = ve::dist::make_log_density(Dist::Norm, 0.0);
    for (double z : {-3.0, -0.5, 0.0, 2.0})
    {
        EXPECT_NEAR(ve::dist::log_density(ged, z), ve::dist::log_density(norm, z), 1e-14) << "z = " << z;
    }
}

TEST(Densities, NormalMatchesClosedForm)
{
    ve::dist::LogDensity<double> norm = ve::dist::make_log_density(Dist::Norm, 0.0);
    double expected = -0.5 * std::log(2.0 * std::numbers::pi) - 0.5 * 1.7 * 1.7;
    EXPECT_NEAR(ve::dist::log_density(norm, 1.7), expected, 1e-15);
}

// As nu -> infinity, log f_std(z) - log f_norm(z) = (z^4 - 6 z^2 + 3) / (4 nu) + O(nu^-2), the fourth Hermite
// polynomial over 4 nu. The fit reaches nu far past 1e6 on normal data, where this difference is far below
// the size of lgamma(nu / 2), and its derivative in x = log(nu - 2) is about -(z^4 - 6 z^2 + 3) / (4 nu).
TEST(Densities, StudentApproachesNormalForLargeShape)
{
    using Number = ve::ad::Dual<1>;
    ve::dist::LogDensity<double> norm = ve::dist::make_log_density(Dist::Norm, 0.0);
    for (double nu : {1e6, 1e9, 1e12, 1e15, 1e200, 1e300})
    {
        Number shape = ve::ad::variable<1>(nu, 0);
        shape.grad[0] = nu - 2.0;
        ve::dist::LogDensity<Number> student = ve::dist::make_log_density(Dist::Std, shape);
        for (double z : {0.0, 0.5, 2.0, 4.0})
        {
            double hermite = (z * z * z * z - 6.0 * z * z + 3.0) / 4.0;
            Number log_f = ve::dist::log_density(student, Number{z});
            double tolerance = 1e-4 * std::abs(hermite) / nu + 1e-14;
            EXPECT_NEAR(log_f.value - ve::dist::log_density(norm, z), hermite / nu, tolerance)
                << "nu = " << nu << ", z = " << z;
            EXPECT_NEAR(log_f.grad[0], -hermite / nu, tolerance) << "nu = " << nu << ", z = " << z;
        }
    }
}

TEST(Densities, ShapeDomain)
{
    EXPECT_TRUE(ve::dist::in_domain(Dist::Norm, 0.0));
    EXPECT_FALSE(ve::dist::in_domain(Dist::Std, 2.0));
    EXPECT_TRUE(ve::dist::in_domain(Dist::Std, 2.0001));
    EXPECT_FALSE(ve::dist::in_domain(Dist::Ged, 0.0));
    EXPECT_TRUE(ve::dist::in_domain(Dist::Ged, 0.01));
    EXPECT_FALSE(ve::dist::in_domain(Dist::Ged, std::nan("")));
}

}
