#include "ad/dual.hpp"
#include "dist/densities.hpp"
#include "dist/reference_quadrature.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <gtest/gtest.h>
#include <initializer_list>
#include <limits>
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

struct SkewedCase
{
    Dist dist;
    double skew;
    double shape;
    double ghlambda;
    bool negative_skew;
};

std::string skewed_case_name(const ::testing::TestParamInfo<SkewedCase>& info)
{
    std::string names[] = {"Norm", "Std", "Ged", "Sstd", "Sged", "Nig", "Ghyp", "Jsu"};
    return names[static_cast<int>(info.param.dist)] + std::to_string(info.index);
}

ve::dist::ShapeParameters<double> parameters_of(const SkewedCase& c)
{
    return {c.skew, c.shape, c.ghlambda};
}

double density_at(Dist dist, const ve::dist::ShapeParameters<double>& parameters, double z)
{
    return std::exp(ve::dist::log_density(ve::dist::make_density(dist, parameters), z));
}

class SkewedDensities : public ::testing::TestWithParam<SkewedCase>
{
};

TEST_P(SkewedDensities, StandardizedMoments)
{
    SkewedCase c = GetParam();
    ve::dist::Density<double> density = ve::dist::make_density(c.dist, parameters_of(c));
    auto f = [&](double z) { return std::exp(ve::dist::log_density(density, z)); };
    ve::testing::RawMoments moments = ve::testing::reference_moments(f, ve::dist::density_kink(density));
    EXPECT_NEAR(moments[0], 1.0, 1e-9);
    EXPECT_NEAR(moments[1], 0.0, 1e-9);
    EXPECT_NEAR(moments[2], 1.0, 1e-9);
}

// A negative skew (xi < 1, rho < 0, jsu nu < 0) stretches the left tail, which pulls the mean below the
// median: less than half the mass lies below the mean 0.
TEST_P(SkewedDensities, SkewSign)
{
    SkewedCase c = GetParam();
    ve::dist::Density<double> density = ve::dist::make_density(c.dist, parameters_of(c));
    auto below = [&](double z)
    {
        if (z > 0.0)
        {
            return 0.0;
        }
        return std::exp(ve::dist::log_density(density, z));
    };
    double mass_below = ve::testing::trapezoid(below);
    if (c.negative_skew)
    {
        EXPECT_LT(mass_below, 0.5 - 1e-3);
        return;
    }
    EXPECT_GT(mass_below, 0.5 + 1e-3);
}

double log_density_at(Dist dist, const std::array<double, 4>& point)
{
    ve::dist::ShapeParameters<double> parameters{point[1], point[2], point[3]};
    return ve::dist::log_density(ve::dist::make_density(dist, parameters), point[0]);
}

TEST_P(SkewedDensities, DerivativesMatchFiniteDifferences)
{
    using Number = ve::ad::Dual<8>;
    SkewedCase c = GetParam();
    ve::dist::ShapeParameters<Number> parameters{ve::ad::variable<8>(c.skew, 1),
                                                 ve::ad::variable<8>(c.shape, 2),
                                                 ve::ad::variable<8>(c.ghlambda, 3)};
    ve::dist::Density<Number> density = ve::dist::make_density(c.dist, parameters);
    for (double z : {-3.0, -0.2, 0.4, 5.0})
    {
        Number log_f = ve::dist::log_density(density, ve::ad::variable<8>(z, 0));
        std::array<double, 4> point = {z, c.skew, c.shape, c.ghlambda};
        for (std::size_t k = 0; k < point.size(); ++k)
        {
            double step = 1e-5 * std::max(1.0, std::abs(point[k]));
            std::array<double, 4> up = point;
            std::array<double, 4> down = point;
            up[k] += step;
            down[k] -= step;
            double central = (log_density_at(c.dist, up) - log_density_at(c.dist, down)) / (2.0 * step);
            EXPECT_NEAR(log_f.grad[k], central, 1e-6 * std::max(1.0, std::abs(log_f.grad[k])))
                << "z = " << z << ", k = " << k;
        }
    }
}

INSTANTIATE_TEST_SUITE_P(SkewedDensities, SkewedDensities,
                         ::testing::Values(SkewedCase{Dist::Sstd, 0.7, 5.0, 0.0, true},
                                           SkewedCase{Dist::Sstd, 1.4, 12.0, 0.0, false},
                                           SkewedCase{Dist::Sged, 0.8, 1.2, 0.0, true},
                                           SkewedCase{Dist::Sged, 1.3, 2.5, 0.0, false},
                                           SkewedCase{Dist::Nig, -0.5, 0.8, 0.0, true},
                                           SkewedCase{Dist::Nig, 0.4, 3.0, 0.0, false},
                                           SkewedCase{Dist::Ghyp, -0.3, 1.5, 1.0, true},
                                           SkewedCase{Dist::Ghyp, 0.6, 0.7, -2.5, false},
                                           SkewedCase{Dist::Jsu, -0.8, 1.5, 0.0, true},
                                           SkewedCase{Dist::Jsu, 0.6, 3.0, 0.0, false}),
                         skewed_case_name);

constexpr int kFormPoints = 50;

double form_point(int i)
{
    return -8.0 + 16.0 * static_cast<double>(i) / static_cast<double>(kFormPoints - 1);
}

TEST(SkewedDensities, ReduceToKnownForms)
{
    using Parameters = ve::dist::ShapeParameters<double>;
    for (int i = 0; i < kFormPoints; ++i)
    {
        double z = form_point(i);
        for (double nu : {5.0, 12.0})
        {
            double student = std::exp(ve::dist::log_density(ve::dist::make_log_density(Dist::Std, nu), z));
            EXPECT_NEAR(density_at(Dist::Sstd, {1.0, nu, 0.0}, z), student, 1e-14 * student) << "z = " << z;
        }
        for (double nu : {1.2, 2.5})
        {
            double ged = std::exp(ve::dist::log_density(ve::dist::make_log_density(Dist::Ged, nu), z));
            EXPECT_NEAR(density_at(Dist::Sged, {1.0, nu, 0.0}, z), ged, 1e-14 * ged) << "z = " << z;
        }
        for (Parameters nig_case : {Parameters{-0.5, 0.8, 0.0}, Parameters{0.4, 3.0, 0.0}})
        {
            double nig = density_at(Dist::Nig, nig_case, z);
            Parameters ghyp_case{nig_case.skew, nig_case.shape, -0.5};
            EXPECT_NEAR(density_at(Dist::Ghyp, ghyp_case, z), nig, 1e-12 * nig) << "z = " << z;
        }
    }
}

TEST(SkewedDensities, SymmetricWithoutSkew)
{
    for (int i = 0; i < kFormPoints; ++i)
    {
        double z = form_point(i);
        double nig = density_at(Dist::Nig, {0.0, 1.5, 0.0}, z);
        double ghyp = density_at(Dist::Ghyp, {0.0, 0.7, 1.0}, z);
        double jsu = density_at(Dist::Jsu, {0.0, 1.5, 0.0}, z);
        EXPECT_NEAR(density_at(Dist::Nig, {0.0, 1.5, 0.0}, -z), nig, 1e-14 * nig) << "z = " << z;
        EXPECT_NEAR(density_at(Dist::Ghyp, {0.0, 0.7, 1.0}, -z), ghyp, 1e-14 * ghyp) << "z = " << z;
        EXPECT_NEAR(density_at(Dist::Jsu, {0.0, 1.5, 0.0}, -z), jsu, 1e-14 * jsu) << "z = " << z;
    }
}

TEST(SkewedDensities, Domains)
{
    using ve::dist::in_domain;
    using Parameters = ve::dist::ShapeParameters<double>;
    double infinity = std::numeric_limits<double>::infinity();
    EXPECT_TRUE(in_domain(Dist::Norm, Parameters{0.0, 0.0, 0.0}));
    EXPECT_FALSE(in_domain(Dist::Std, Parameters{0.0, 2.0, 0.0}));
    EXPECT_TRUE(in_domain(Dist::Std, Parameters{0.0, 2.01, 0.0}));
    EXPECT_FALSE(in_domain(Dist::Ged, Parameters{0.0, 0.0, 0.0}));
    EXPECT_FALSE(in_domain(Dist::Sstd, Parameters{1.0, 2.0, 0.0}));
    EXPECT_TRUE(in_domain(Dist::Sstd, Parameters{1.0, 2.01, 0.0}));
    EXPECT_FALSE(in_domain(Dist::Sstd, Parameters{0.0, 5.0, 0.0}));
    EXPECT_FALSE(in_domain(Dist::Sged, Parameters{-0.5, 1.5, 0.0}));
    EXPECT_FALSE(in_domain(Dist::Sged, Parameters{0.9, 0.0, 0.0}));
    EXPECT_TRUE(in_domain(Dist::Sged, Parameters{0.9, 0.1, 0.0}));
    EXPECT_FALSE(in_domain(Dist::Nig, Parameters{1.0, 1.0, 0.0}));
    EXPECT_FALSE(in_domain(Dist::Nig, Parameters{-1.0, 1.0, 0.0}));
    EXPECT_FALSE(in_domain(Dist::Nig, Parameters{0.5, 0.0, 0.0}));
    EXPECT_TRUE(in_domain(Dist::Nig, Parameters{-0.99, 1.0, 0.0}));
    EXPECT_FALSE(in_domain(Dist::Ghyp, Parameters{1.0, 1.0, 0.5}));
    EXPECT_TRUE(in_domain(Dist::Ghyp, Parameters{0.5, 1.0, -50.0}));
    EXPECT_TRUE(in_domain(Dist::Ghyp, Parameters{0.5, 1.0, 50.0}));
    EXPECT_FALSE(in_domain(Dist::Ghyp, Parameters{0.5, 1.0, std::nan("")}));
    EXPECT_FALSE(in_domain(Dist::Ghyp, Parameters{0.5, 1.0, infinity}));
    EXPECT_FALSE(in_domain(Dist::Jsu, Parameters{0.5, 0.0, 0.0}));
    EXPECT_FALSE(in_domain(Dist::Jsu, Parameters{0.5, -1.0, 0.0}));
    EXPECT_TRUE(in_domain(Dist::Jsu, Parameters{-3.0, 0.5, 0.0}));
    EXPECT_FALSE(in_domain(Dist::Jsu, Parameters{std::nan(""), 0.5, 0.0}));
}

// Review Focus 1: near-normal data drives zeta up, and the Bessel arguments past 700, onto the asymptotic
// path.
TEST(SkewedDensities, FarTailsStayFinite)
{
    using Parameters = ve::dist::ShapeParameters<double>;
    double normal_at_zero = -0.5 * std::log(2.0 * std::numbers::pi);
    ve::dist::Density<double> nig = ve::dist::make_density(Dist::Nig, Parameters{0.0, 2000.0, 0.0});
    ve::dist::Density<double> ghyp = ve::dist::make_density(Dist::Ghyp, Parameters{0.2, 5000.0, 3.0});
    for (double z : {-40.0, 0.0, 40.0})
    {
        EXPECT_TRUE(std::isfinite(ve::dist::log_density(nig, z))) << "z = " << z;
        EXPECT_TRUE(std::isfinite(ve::dist::log_density(ghyp, z))) << "z = " << z;
    }
    EXPECT_NEAR(ve::dist::log_density(nig, 0.0), normal_at_zero, 1e-2);
    EXPECT_NEAR(ve::dist::log_density(ghyp, 0.0), normal_at_zero, 1e-2);
}

}
