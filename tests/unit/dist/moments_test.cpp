#include "ad/dual.hpp"
#include "dist/densities.hpp"
#include "dist/moments.hpp"
#include "dist/reference_quadrature.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <gtest/gtest.h>
#include <initializer_list>
#include <limits>
#include <numbers>

namespace
{

using ve::Dist;
using ve::dist::abs_moment;
using ve::dist::asymmetric_moment;
using ve::dist::shifted_moment;

struct DistCase
{
    Dist dist;
    double shape;
};

void expect_relative(double actual, double expected, double tolerance)
{
    EXPECT_NEAR(actual, expected, tolerance * std::abs(expected));
}

double normal_cdf(double x)
{
    return 0.5 * std::erfc(-x / std::numbers::sqrt2);
}

double normal_pdf(double x)
{
    return std::exp(-0.5 * x * x) / std::sqrt(2.0 * std::numbers::pi);
}

double reference_asymmetric_moment(DistCase c, double eta1, double power)
{
    ve::dist::LogDensity<double> density = ve::dist::make_log_density(c.dist, c.shape);
    auto integrand = [&](double z)
    { return std::pow(std::abs(z) - eta1 * z, power) * std::exp(ve::dist::log_density(density, z)); };
    double moment = ve::testing::trapezoid(integrand);
    if (c.dist == Dist::Std)
    {
        double asymmetry = 0.5 * (std::pow(1.0 - eta1, power) + std::pow(1.0 + eta1, power));
        moment += asymmetry * ve::testing::student_tails(density, c.shape, power);
    }
    return moment;
}

TEST(Moments, AbsMomentClosedForms)
{
    EXPECT_NEAR(abs_moment(Dist::Norm, 0.0, 1.0), std::sqrt(2.0 / std::numbers::pi), 1e-15);
    EXPECT_NEAR(abs_moment(Dist::Norm, 0.0, 2.0), 1.0, 1e-14);
    EXPECT_NEAR(abs_moment(Dist::Norm, 0.0, 4.0), 3.0, 1e-13);
    EXPECT_NEAR(abs_moment(Dist::Std, 6.0, 2.0), 1.0, 1e-12);
    EXPECT_NEAR(abs_moment(Dist::Std, 6.0, 4.0), 6.0, 1e-12);
    for (double power : {0.5, 1.0, 1.7})
    {
        EXPECT_NEAR(abs_moment(Dist::Ged, 2.0, power), abs_moment(Dist::Norm, 0.0, power), 1e-14) << power;
    }
    for (double nu : {0.8, 1.5})
    {
        EXPECT_NEAR(abs_moment(Dist::Ged, nu, 2.0), 1.0, 1e-12) << nu;
    }
}

// The fit reaches nu far past 1e6 on normal data, where lgamma(nu / 2) is far larger than the moment.
TEST(Moments, StdMomentApproachesNormalForLargeShape)
{
    expect_relative(abs_moment(Dist::Std, 30.0, 4.0), 3.0 * 28.0 / 26.0, 1e-12);
    for (double nu : {30.0, 1e6, 1e15, 1e300})
    {
        expect_relative(abs_moment(Dist::Std, nu, 2.0), 1.0, 1e-12);
    }
    expect_relative(abs_moment(Dist::Std, 1e15, 1.5), abs_moment(Dist::Norm, 0.0, 1.5), 1e-12);
}

TEST(Moments, StdMomentInfiniteAtOrAbovePower)
{
    double infinity = std::numeric_limits<double>::infinity();
    EXPECT_EQ(abs_moment(Dist::Std, 3.0, 3.0), infinity);
    EXPECT_EQ(abs_moment(Dist::Std, 2.5, 2.7), infinity);
    EXPECT_EQ(asymmetric_moment(Dist::Std, 2.5, 0.3, 2.7), infinity);
    EXPECT_EQ(shifted_moment(Dist::Std, 2.5, 0.3, 0.2, 2.7), infinity);
    EXPECT_TRUE(std::isfinite(abs_moment(Dist::Std, 3.1, 3.0)));
}

TEST(Moments, AsymmetricMatchesQuadrature)
{
    for (DistCase c : {DistCase{Dist::Norm, 0.0}, DistCase{Dist::Std, 6.0}, DistCase{Dist::Ged, 1.5}})
    {
        for (double eta1 : {-0.5, 0.0, 0.7})
        {
            for (double power : {0.8, 1.5, 2.0})
            {
                double expected = reference_asymmetric_moment(c, eta1, power);
                expect_relative(asymmetric_moment(c.dist, c.shape, eta1, power), expected, 1e-6);
            }
        }
    }
}

// With eta2 = 0 the quadrature meets the closed form of the asymmetric moment; with eta1 = 0 and power 2 it is
// the NAGARCH E(z - c)^2 = 1 + c^2 of any standardized density; and for the normal E|z - c| has a closed form.
TEST(Moments, ShiftedReducesToKnownForms)
{
    for (DistCase c : {DistCase{Dist::Norm, 0.0}, DistCase{Dist::Ged, 1.5}, DistCase{Dist::Std, 8.0}})
    {
        for (double eta1 : {-0.4, 0.3})
        {
            for (double power : {1.0, 1.5, 2.0})
            {
                double expected = asymmetric_moment(c.dist, c.shape, eta1, power);
                expect_relative(shifted_moment(c.dist, c.shape, eta1, 0.0, power), expected, 1e-10);
            }
        }
    }
    for (DistCase c : {DistCase{Dist::Norm, 0.0}, DistCase{Dist::Std, 4.5}, DistCase{Dist::Ged, 1.0}})
    {
        for (double shift : {-0.5, 0.3, 1.2})
        {
            expect_relative(shifted_moment(c.dist, c.shape, 0.0, shift, 2.0), 1.0 + shift * shift, 1e-8);
        }
    }
    for (double shift : {-0.7, 0.2})
    {
        double expected = shift * (2.0 * normal_cdf(shift) - 1.0) + 2.0 * normal_pdf(shift);
        expect_relative(shifted_moment(Dist::Norm, 0.0, 0.0, shift, 1.0), expected, 1e-10);
    }
}

template <class T>
std::array<T, 3> all_moments(Dist dist, const std::array<T, 4>& x)
{
    return {abs_moment(dist, x[0], x[3]), asymmetric_moment(dist, x[0], x[1], x[3]),
            shifted_moment(dist, x[0], x[1], x[2], x[3])};
}

double central_difference(DistCase c, std::array<double, 4> x, std::size_t i, std::size_t moment)
{
    double step = 1e-6 * std::max(1.0, std::abs(x[i]));
    double center = x[i];
    x[i] = center + step;
    double plus = all_moments(c.dist, x)[moment];
    x[i] = center - step;
    double minus = all_moments(c.dist, x)[moment];
    return (plus - minus) / (2.0 * step);
}

TEST(Moments, DerivativesMatchFiniteDifferences)
{
    using Number = ve::ad::Dual<8>;
    for (DistCase c : {DistCase{Dist::Std, 6.0}, DistCase{Dist::Std, 40.0}, DistCase{Dist::Ged, 1.5}})
    {
        std::array<double, 4> x{c.shape, 0.3, 0.4, 1.5};
        std::array<Number, 4> seeded{};
        for (std::size_t i = 0; i < x.size(); ++i)
        {
            seeded[i] = ve::ad::variable<8>(x[i], i);
        }
        std::array<Number, 3> moments = all_moments(c.dist, seeded);
        for (std::size_t moment = 0; moment < moments.size(); ++moment)
        {
            for (std::size_t i = 0; i < x.size(); ++i)
            {
                double gradient = moments[moment].grad[i];
                double difference = central_difference(c, x, i, moment);
                EXPECT_NEAR(gradient, difference, 1e-6 * std::max(1.0, std::abs(gradient)))
                    << "shape " << c.shape << ", moment " << moment << ", variable " << i;
            }
        }
    }
}

}
