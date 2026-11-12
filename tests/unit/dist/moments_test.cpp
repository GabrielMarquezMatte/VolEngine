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
#include <string>
#include <vector>

namespace
{

using ve::Dist;
using ve::dist::abs_moment;
using ve::dist::asymmetric_moment;
using ve::dist::probability_below_zero;
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
    EXPECT_NEAR(abs_moment(Dist::Norm, {}, 1.0), std::sqrt(2.0 / std::numbers::pi), 1e-15);
    EXPECT_NEAR(abs_moment(Dist::Norm, {}, 2.0), 1.0, 1e-14);
    EXPECT_NEAR(abs_moment(Dist::Norm, {}, 4.0), 3.0, 1e-13);
    EXPECT_NEAR(abs_moment(Dist::Std, {.shape = 6.0}, 2.0), 1.0, 1e-12);
    EXPECT_NEAR(abs_moment(Dist::Std, {.shape = 6.0}, 4.0), 6.0, 1e-12);
    for (double power : {0.5, 1.0, 1.7})
    {
        EXPECT_NEAR(abs_moment(Dist::Ged, {.shape = 2.0}, power), abs_moment(Dist::Norm, {}, power), 1e-14)
            << power;
    }
    for (double nu : {0.8, 1.5})
    {
        EXPECT_NEAR(abs_moment(Dist::Ged, {.shape = nu}, 2.0), 1.0, 1e-12) << nu;
    }
}

// The fit reaches nu far past 1e6 on normal data, where lgamma(nu / 2) is far larger than the moment.
TEST(Moments, StdMomentApproachesNormalForLargeShape)
{
    expect_relative(abs_moment(Dist::Std, {.shape = 30.0}, 4.0), 3.0 * 28.0 / 26.0, 1e-12);
    for (double nu : {30.0, 1e6, 1e15, 1e300})
    {
        expect_relative(abs_moment(Dist::Std, {.shape = nu}, 2.0), 1.0, 1e-12);
    }
    expect_relative(abs_moment(Dist::Std, {.shape = 1e15}, 1.5), abs_moment(Dist::Norm, {}, 1.5), 1e-12);
}

TEST(Moments, StdMomentInfiniteAtOrAbovePower)
{
    double infinity = std::numeric_limits<double>::infinity();
    EXPECT_EQ(abs_moment(Dist::Std, {.shape = 3.0}, 3.0), infinity);
    EXPECT_EQ(abs_moment(Dist::Std, {.shape = 2.5}, 2.7), infinity);
    EXPECT_EQ(asymmetric_moment(Dist::Std, {.shape = 2.5}, 0.3, 2.7), infinity);
    EXPECT_EQ(shifted_moment(Dist::Std, {.shape = 2.5}, 0.3, 0.2, 2.7), infinity);
    EXPECT_TRUE(std::isfinite(abs_moment(Dist::Std, {.shape = 3.1}, 3.0)));
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
                expect_relative(asymmetric_moment(c.dist, {.shape = c.shape}, eta1, power), expected, 1e-6);
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
                double expected = asymmetric_moment(c.dist, {.shape = c.shape}, eta1, power);
                double shifted = shifted_moment(c.dist, {.shape = c.shape}, eta1, 0.0, power);
                expect_relative(shifted, expected, 1e-10);
            }
        }
    }
    for (DistCase c : {DistCase{Dist::Norm, 0.0}, DistCase{Dist::Std, 4.5}, DistCase{Dist::Ged, 1.0}})
    {
        for (double shift : {-0.5, 0.3, 1.2})
        {
            double shifted = shifted_moment(c.dist, {.shape = c.shape}, 0.0, shift, 2.0);
            expect_relative(shifted, 1.0 + shift * shift, 1e-8);
        }
    }
    for (double shift : {-0.7, 0.2})
    {
        double expected = shift * (2.0 * normal_cdf(shift) - 1.0) + 2.0 * normal_pdf(shift);
        expect_relative(shifted_moment(Dist::Norm, {}, 0.0, shift, 1.0), expected, 1e-10);
    }
}

// With nu - power near 0, a std tail under z = edge + t / (1 - t) is (1 - t)^(nu - power - 1), which Gauss-Legendre
// integrates poorly.
TEST(Moments, StdHeavyTailMomentMatchesClosedForm)
{
    for (std::array<double, 2> nu_power : {std::array{3.0, 2.5}, std::array{2.5, 2.2}, std::array{2.2, 1.9},
                                           std::array{4.0, 2.5}, std::array{3.0, 1.5}})
    {
        auto [nu, power] = nu_power;
        for (double eta1 : {0.0, 0.2})
        {
            double expected = asymmetric_moment(Dist::Std, {.shape = nu}, eta1, power);
            EXPECT_NEAR(shifted_moment(Dist::Std, {.shape = nu}, eta1, 0.0, power), expected, 1e-8 * expected)
                << "nu " << nu << ", power " << power << ", eta1 " << eta1;
        }
    }
}

template <class T>
std::array<T, 3> all_moments(Dist dist, const std::array<T, 4>& x)
{
    return {abs_moment(dist, {.shape = x[0]}, x[3]), asymmetric_moment(dist, {.shape = x[0]}, x[1], x[3]),
            shifted_moment(dist, {.shape = x[0]}, x[1], x[2], x[3])};
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

// At eta2 = 0 the quadrature changes which split point follows eta2, so a central difference across it mixes two
// rules. There d/d eta2 of the shifted moment is (power / 2) ((1 + eta1)^power - (1 - eta1)^power) E|z|^(power - 1).
double expected_derivative(DistCase c, const std::array<double, 4>& x, std::size_t i, std::size_t moment)
{
    bool shift_at_zero = moment == 2 && i == 2 && x[2] == 0.0;
    if (!shift_at_zero)
    {
        return central_difference(c, x, i, moment);
    }
    double asymmetry = std::pow(1.0 + x[1], x[3]) - std::pow(1.0 - x[1], x[3]);
    return 0.5 * x[3] * asymmetry * abs_moment(c.dist, {.shape = c.shape}, x[3] - 1.0);
}

void expect_gradients_match(DistCase c, const std::array<double, 4>& x)
{
    using Number = ve::ad::Dual<8>;
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
            double expected = expected_derivative(c, x, i, moment);
            EXPECT_NEAR(gradient, expected, 1e-6 * std::max(1.0, std::abs(gradient)))
                << "shape " << c.shape << ", eta2 " << x[2] << ", moment " << moment << ", variable " << i;
        }
    }
}

TEST(Moments, StdShiftedMomentSplitsAtTheCusp)
{
    expect_relative(shifted_moment(Dist::Std, {.shape = 4.5}, 0.3, 0.5, 1.5), 1.26984189752, 1e-9);
}

TEST(Moments, DerivativesMatchFiniteDifferences)
{
    for (DistCase c : {DistCase{Dist::Std, 6.0}, DistCase{Dist::Std, 40.0}, DistCase{Dist::Ged, 1.5},
                       DistCase{Dist::Ged, 0.8}, DistCase{Dist::Norm, 0.0}})
    {
        expect_gradients_match(c, {c.shape, 0.3, 0.4, 1.5});
        expect_gradients_match(c, {c.shape, 0.3, 0.0, 0.8});
    }
}

// norm, std and ged keep the values recorded with GCC before the Density variant and the kink split. Other
// standard libraries round lgamma, pow and exp differently in the last bits, so the check allows 4 ulps.
TEST(Moments, SymmetricUnchanged)
{
    EXPECT_DOUBLE_EQ(shifted_moment(Dist::Std, {.shape = 4.5}, 0.3, 0.5, 1.5), 0x1.45145bccd1599p+0);
    EXPECT_DOUBLE_EQ(shifted_moment(Dist::Ged, {.shape = 1.5}, -0.4, -0.3, 0.8), 0x1.c345836c0c961p-1);
    EXPECT_DOUBLE_EQ(shifted_moment(Dist::Norm, {}, 0.2, 0.0, 1.5), 0x1.bef6589a7740bp-1);
    EXPECT_DOUBLE_EQ(abs_moment(Dist::Std, {.shape = 6.0}, 1.3), 0x1.8f33133cba259p-1);
    EXPECT_DOUBLE_EQ(asymmetric_moment(Dist::Ged, {.shape = 1.2}, 0.3, 1.7), 0x1.d73cc4872552fp-1);
    using Number = ve::ad::Dual<8>;
    Number moment = shifted_moment(Dist::Std, {.shape = ve::ad::variable<8>(6.0, 0)},
                                   ve::ad::variable<8>(0.3, 1), ve::ad::variable<8>(0.4, 2),
                                   ve::ad::variable<8>(1.5, 3));
    EXPECT_DOUBLE_EQ(moment.value, 0x1.2cf79e0107bcp+0);
    std::array<double, 4> gradient = {0x1.6497c15daa294p-7, 0x1.d8cce45356801p-1, 0x1.1de1d413c16bdp+0,
                                      0x1.4ffa5edf97d1fp-1};
    for (std::size_t i = 0; i < gradient.size(); ++i)
    {
        EXPECT_DOUBLE_EQ(moment.grad[i], gradient[i]) << i;
    }
    for (Dist dist : {Dist::Norm, Dist::Std, Dist::Ged})
    {
        EXPECT_EQ(probability_below_zero(dist, ve::dist::ShapeParameters<double>{.shape = 5.0}), 0.5);
    }
}

struct SkewedCase
{
    Dist dist;
    ve::dist::ShapeParameters<double> parameters;
    bool negative_skew;
};

// The cases of the Densities tests.
const std::array<SkewedCase, 10> kSkewedCases = {
    SkewedCase{Dist::Sstd, {0.7, 5.0, 0.0}, true},  SkewedCase{Dist::Sstd, {1.4, 12.0, 0.0}, false},
    SkewedCase{Dist::Sged, {0.8, 1.2, 0.0}, true},  SkewedCase{Dist::Sged, {1.3, 2.5, 0.0}, false},
    SkewedCase{Dist::Nig, {-0.5, 0.8, 0.0}, true},  SkewedCase{Dist::Nig, {0.4, 3.0, 0.0}, false},
    SkewedCase{Dist::Ghyp, {-0.3, 1.5, 1.0}, true}, SkewedCase{Dist::Ghyp, {0.6, 0.7, -2.5}, false},
    SkewedCase{Dist::Jsu, {-0.8, 1.5, 0.0}, true},  SkewedCase{Dist::Jsu, {0.6, 3.0, 0.0}, false}};

std::string case_label(const SkewedCase& c)
{
    return "dist " + std::to_string(static_cast<int>(c.dist)) + " (" + std::to_string(c.parameters.skew) +
           ", " + std::to_string(c.parameters.shape) + ", " + std::to_string(c.parameters.ghlambda) + ")";
}

double reference_shock_moment(const SkewedCase& c, double eta1, double power)
{
    ve::dist::Density<double> density = ve::dist::make_density(c.dist, c.parameters);
    auto integrand = [&](double z)
    { return std::pow(std::abs(z) - eta1 * z, power) * std::exp(ve::dist::log_density(density, z)); };
    return ve::testing::reference_integral(integrand, ve::dist::density_kink(density), 0.0);
}

double reference_probability_below_zero(const SkewedCase& c)
{
    ve::dist::Density<double> density = ve::dist::make_density(c.dist, c.parameters);
    auto below = [&](double z)
    {
        if (z >= 0.0)
        {
            return 0.0;
        }
        return std::exp(ve::dist::log_density(density, z));
    };
    return ve::testing::reference_integral(below, ve::dist::density_kink(density), 0.0);
}

// E(z - c)^2 = 1 + c^2 holds for any standardized density.
void expect_moments_match_reference(const SkewedCase& c)
{
    expect_relative(abs_moment(c.dist, c.parameters, 2.0), 1.0, 1e-10);
    expect_relative(shifted_moment(c.dist, c.parameters, 0.0, 0.3, 2.0), 1.09, 1e-8);
    for (double power : {0.8, 1.0, 1.5, 2.0})
    {
        for (double eta1 : {-0.5, 0.0, 0.7})
        {
            double expected = reference_shock_moment(c, eta1, power);
            EXPECT_NEAR(asymmetric_moment(c.dist, c.parameters, eta1, power) / expected, 1.0, 1e-6)
                << case_label(c) << ", power " << power << ", eta1 " << eta1;
            if (eta1 == 0.0)
            {
                EXPECT_NEAR(abs_moment(c.dist, c.parameters, power) / expected, 1.0, 1e-6)
                    << case_label(c) << ", power " << power;
            }
        }
    }
}

TEST(Moments, SkewedMatchReferenceQuadrature)
{
    for (const SkewedCase& c : kSkewedCases)
    {
        expect_moments_match_reference(c);
    }
}

// A negative skew (xi < 1, rho < 0, jsu nu < 0) puts less than half the mass below the mean 0.
TEST(Moments, ProbabilityBelowZero)
{
    for (const SkewedCase& c : kSkewedCases)
    {
        double probability = probability_below_zero(c.dist, c.parameters);
        EXPECT_NEAR(probability, reference_probability_below_zero(c), 1e-9) << case_label(c);
        EXPECT_EQ(probability < 0.5, c.negative_skew) << case_label(c);
        EXPECT_GT(probability, 0.0) << case_label(c);
        EXPECT_LT(probability, 1.0) << case_label(c);
    }
    for (Dist dist : {Dist::Sstd, Dist::Sged})
    {
        for (double skew : {0.05, 20.0})
        {
            ve::dist::ShapeParameters<double> parameters{skew, 5.0, 0.0};
            double probability = probability_below_zero(dist, parameters);
            double abs_mean = abs_moment(dist, parameters, 1.0);
            double kappa = asymmetric_moment(dist, parameters, 0.3, 1.5);
            EXPECT_GT(probability, 0.0) << skew;
            EXPECT_LT(probability, 1.0) << skew;
            EXPECT_TRUE(std::isfinite(abs_mean) && abs_mean > 0.0) << skew;
            EXPECT_TRUE(std::isfinite(kappa) && kappa > 0.0) << skew;
        }
    }
}

// The narrow side of a Fernandez-Steel density has the z-scale 1 / (xi sigma) or xi / sigma, ~1e-3 here.
TEST(Moments, VerySkewedMatchReferenceQuadrature)
{
    for (Dist dist : {Dist::Sstd, Dist::Sged})
    {
        for (double skew : {0.05, 20.0, 30.0})
        {
            SkewedCase c{dist, {skew, 5.0, 0.0}, skew < 1.0};
            EXPECT_NEAR(abs_moment(c.dist, c.parameters, 2.0), 1.0, 1e-6) << case_label(c);
            double kappa = reference_shock_moment(c, 0.3, 1.5);
            EXPECT_NEAR(asymmetric_moment(c.dist, c.parameters, 0.3, 1.5) / kappa, 1.0, 1e-6)
                << case_label(c);
            double probability = reference_probability_below_zero(c);
            EXPECT_NEAR(probability_below_zero(c.dist, c.parameters), probability, 1e-6) << case_label(c);
        }
    }
}

// The tail of the sstd is the tail of the t.
TEST(Moments, SkewedStudentInfiniteAtOrAbovePower)
{
    double infinity = std::numeric_limits<double>::infinity();
    ve::dist::ShapeParameters<double> parameters{0.8, 2.5, 0.0};
    EXPECT_EQ(abs_moment(Dist::Sstd, parameters, 2.7), infinity);
    EXPECT_EQ(asymmetric_moment(Dist::Sstd, parameters, 0.3, 2.7), infinity);
    EXPECT_EQ(shifted_moment(Dist::Sstd, parameters, 0.3, 0.2, 2.7), infinity);
    EXPECT_EQ(abs_moment(Dist::Sstd, parameters, 2.5), infinity);
    EXPECT_TRUE(std::isfinite(abs_moment(Dist::Sstd, parameters, 2.4)));
}

// x = (skew, shape, ghlambda, eta1, eta2, power).
template <class T>
std::array<T, 4> skewed_functions(Dist dist, const std::array<T, 6>& x)
{
    ve::dist::ShapeParameters<T> parameters{x[0], x[1], x[2]};
    return {abs_moment(dist, parameters, x[5]), asymmetric_moment(dist, parameters, x[3], x[5]),
            shifted_moment(dist, parameters, x[3], x[4], x[5]), probability_below_zero(dist, parameters)};
}

double skewed_central_difference(Dist dist, std::array<double, 6> x, std::size_t i, std::size_t function)
{
    double step = 1e-6 * std::max(1.0, std::abs(x[i]));
    double center = x[i];
    x[i] = center + step;
    double plus = skewed_functions(dist, x)[function];
    x[i] = center - step;
    double minus = skewed_functions(dist, x)[function];
    return (plus - minus) / (2.0 * step);
}

void expect_skewed_gradients_match(Dist dist, const std::array<double, 6>& x)
{
    using Number = ve::ad::Dual<8>;
    std::array<Number, 6> seeded{};
    for (std::size_t i = 0; i < x.size(); ++i)
    {
        seeded[i] = ve::ad::variable<8>(x[i], i);
    }
    std::array<Number, 4> values = skewed_functions(dist, seeded);
    for (std::size_t function = 0; function < values.size(); ++function)
    {
        for (std::size_t i = 0; i < x.size(); ++i)
        {
            double gradient = values[function].grad[i];
            double expected = skewed_central_difference(dist, x, i, function);
            EXPECT_NEAR(gradient, expected, 1e-6 * std::max(1.0, std::abs(gradient)))
                << "dist " << static_cast<int>(dist) << ", eta2 " << x[4] << ", function " << function
                << ", variable " << i;
        }
    }
}

struct DerivativeCase
{
    Dist dist;
    ve::dist::ShapeParameters<double> parameters;
};

TEST(Moments, SkewedDerivativesMatchFiniteDifferences)
{
    for (DerivativeCase c :
         {DerivativeCase{Dist::Sstd, {0.8, 6.0, 0.0}}, DerivativeCase{Dist::Nig, {-0.4, 1.2, 0.0}},
          DerivativeCase{Dist::Ghyp, {0.3, 2.0, 1.5}}, DerivativeCase{Dist::Jsu, {0.5, 2.0, 0.0}}})
    {
        const ve::dist::ShapeParameters<double>& p = c.parameters;
        expect_skewed_gradients_match(c.dist, {p.skew, p.shape, p.ghlambda, 0.3, 0.4, 1.5});
        expect_skewed_gradients_match(c.dist, {p.skew, p.shape, p.ghlambda, -0.2, -0.3, 0.8});
    }
}

// jsu is z = c (sqrt(w) sinh(omega) + sinh((r + nu) / tau)) of a standard normal r, with w = exp(1 / tau^2),
// omega = -nu / tau and c = [(w - 1) (w cosh(2 omega) + 1) / 2]^(-1/2).
struct JohnsonTransform
{
    double nu;
    double tau;
    double center;
    double c;

    double z_of(double r) const
    {
        return c * (center + std::sinh((r + nu) / tau));
    }

    double r_of(double z) const
    {
        return -nu + tau * std::asinh(z / c - center);
    }
};

JohnsonTransform johnson_transform(double nu, double tau)
{
    double w = std::exp(1.0 / (tau * tau));
    double omega = -nu / tau;
    double c = 1.0 / std::sqrt(0.5 * (w - 1.0) * (w * std::cosh(2.0 * omega) + 1.0));
    return {nu, tau, std::sqrt(w) * std::sinh(omega), c};
}

// E(|z| - eta1 z)^power as an integral in r, split at the cusp r(0). Past |r| = 60 the normal density
// outweighs any power of z for the tau here, and the integrand is taken as 0 there.
double johnson_reference_moment(const JohnsonTransform& transform, double eta1, double power)
{
    auto integrand = [&](double r)
    {
        if (std::abs(r) > 60.0)
        {
            return 0.0;
        }
        double z = transform.z_of(r);
        return std::pow(std::abs(z) - eta1 * z, power) * normal_pdf(r);
    };
    double cusp = transform.r_of(0.0);
    return ve::testing::reference_integral(integrand, cusp, cusp);
}

// A jsu with a small tau holds its mass in a peak of width ~c, far below the unit scale of the z quadrature.
TEST(Moments, JohnsonMatchesNormalTransform)
{
    for (double tau : {0.2, 0.3, 0.5, 0.7, 1.0, 3.0})
    {
        for (double nu : {-2.5, 0.0, 2.5})
        {
            SCOPED_TRACE(::testing::Message() << "nu " << nu << ", tau " << tau);
            ve::dist::ShapeParameters<double> parameters{nu, tau, 0.0};
            JohnsonTransform transform = johnson_transform(nu, tau);
            EXPECT_NEAR(abs_moment(Dist::Jsu, parameters, 2.0), 1.0, 1e-6);
            EXPECT_NEAR(probability_below_zero(Dist::Jsu, parameters), normal_cdf(transform.r_of(0.0)), 1e-8);
            double kappa = johnson_reference_moment(transform, 0.3, 1.5);
            EXPECT_NEAR(asymmetric_moment(Dist::Jsu, parameters, 0.3, 1.5) / kappa, 1.0, 1e-6);
        }
    }
}

// sged with a small shape has a kernel of scale lambda << 1 and a stretched-exponential tail; nig and ghyp
// with a small zeta have a peak of width delta at their location. sstd is left out: the reference loses its
// tail as the shape nears 2.
std::vector<SkewedCase> sharp_cases()
{
    std::vector<SkewedCase> cases;
    for (double skew : {0.3, 3.0})
    {
        for (double shape : {0.3, 0.5, 0.8})
        {
            cases.push_back({Dist::Sged, {skew, shape, 0.0}, skew < 1.0});
        }
    }
    for (double zeta : {0.05, 0.3})
    {
        for (double rho : {-0.9, 0.9})
        {
            cases.push_back({Dist::Nig, {rho, zeta, 0.0}, rho < 0.0});
            cases.push_back({Dist::Ghyp, {rho, zeta, -2.5}, rho < 0.0});
            cases.push_back({Dist::Ghyp, {rho, zeta, 1.5}, rho < 0.0});
        }
    }
    return cases;
}

TEST(Moments, SharpCoresMatchReferenceQuadrature)
{
    for (const SkewedCase& c : sharp_cases())
    {
        SCOPED_TRACE(case_label(c));
        EXPECT_NEAR(abs_moment(c.dist, c.parameters, 2.0), 1.0, 1e-6);
        EXPECT_NEAR(probability_below_zero(c.dist, c.parameters), reference_probability_below_zero(c), 1e-6);
        double kappa = reference_shock_moment(c, 0.3, 1.5);
        EXPECT_NEAR(asymmetric_moment(c.dist, c.parameters, 0.3, 1.5) / kappa, 1.0, 1e-6);
    }
}

// Review Focus 1 at the far edge of the domain: a nig or ghyp that double precision cannot standardize is not
// computable, and its moments are NaN; one that is computable integrates to one with mean 0 and variance 1.
void expect_standardized_or_not_computable(const SkewedCase& c)
{
    SCOPED_TRACE(case_label(c) + " zeta " + std::to_string(c.parameters.shape));
    if (!ve::dist::in_domain(c.dist, c.parameters))
    {
        EXPECT_TRUE(std::isnan(abs_moment(c.dist, c.parameters, 2.0)));
        EXPECT_TRUE(std::isnan(asymmetric_moment(c.dist, c.parameters, 0.3, 1.5)));
        EXPECT_TRUE(std::isnan(shifted_moment(c.dist, c.parameters, 0.3, 0.2, 1.5)));
        EXPECT_TRUE(std::isnan(probability_below_zero(c.dist, c.parameters)));
        return;
    }
    ve::dist::Density<double> density = ve::dist::make_density(c.dist, c.parameters);
    auto f = [&](double z) { return std::exp(ve::dist::log_density(density, z)); };
    ve::testing::RawMoments moments = ve::testing::reference_moments(f, 0.0);
    EXPECT_NEAR(moments[0], 1.0, 1e-6);
    EXPECT_NEAR(moments[1], 0.0, 1e-6);
    EXPECT_NEAR(moments[2], 1.0, 1e-6);
    EXPECT_NEAR(abs_moment(c.dist, c.parameters, 2.0), 1.0, 1e-6);
    double probability = probability_below_zero(c.dist, c.parameters);
    EXPECT_NEAR(probability, reference_probability_below_zero(c), 1e-6);
}

std::vector<SkewedCase> extreme_hyperbolic_cases(std::initializer_list<double> zetas,
                                                 std::initializer_list<double> rhos)
{
    std::vector<SkewedCase> cases;
    for (double zeta : zetas)
    {
        for (double rho : rhos)
        {
            for (double sign : {-1.0, 1.0})
            {
                cases.push_back({Dist::Nig, {sign * rho, zeta, 0.0}, sign < 0.0});
                cases.push_back({Dist::Ghyp, {sign * rho, zeta, -2.5}, sign < 0.0});
                cases.push_back({Dist::Ghyp, {sign * rho, zeta, 1.5}, sign < 0.0});
            }
        }
    }
    return cases;
}

TEST(Moments, ExtremeHyperbolicStandardizedOrNotComputable)
{
    for (const SkewedCase& c : extreme_hyperbolic_cases({1e9, 1e12}, {0.99, 1.0 - 1e-12}))
    {
        expect_standardized_or_not_computable(c);
    }
}

// The computable edge: zeta at kMaxHyperbolicShape, and rho up to the last double below 1.
TEST(Moments, HyperbolicEdgeIsStandardized)
{
    double last_below_one = std::nextafter(1.0, 0.0);
    for (const SkewedCase& c : extreme_hyperbolic_cases({ve::dist::kMaxHyperbolicShape, 1e3, 1.0},
                                                        {0.99, 1.0 - 1e-12, last_below_one}))
    {
        EXPECT_TRUE(ve::dist::in_domain(c.dist, c.parameters)) << case_label(c);
        expect_standardized_or_not_computable(c);
    }
    double past_cap = std::nextafter(ve::dist::kMaxHyperbolicShape, 2e6);
    for (const SkewedCase& c : extreme_hyperbolic_cases({past_cap}, {0.0}))
    {
        EXPECT_FALSE(ve::dist::in_domain(c.dist, c.parameters)) << case_label(c);
        expect_standardized_or_not_computable(c);
    }
}

// Below tau ~ 0.053 the jsu constant c overflows to 0, and the moments are NaN rather than 0.
TEST(Moments, JohnsonPastOverflowIsNaN)
{
    for (double tau : {0.05, 0.03, 0.02})
    {
        SCOPED_TRACE(::testing::Message() << "tau " << tau);
        ve::dist::ShapeParameters<double> parameters{0.5, tau, 0.0};
        EXPECT_TRUE(std::isnan(abs_moment(Dist::Jsu, parameters, 2.0)));
        EXPECT_TRUE(std::isnan(abs_moment(Dist::Jsu, parameters, 1.0)));
        EXPECT_TRUE(std::isnan(shifted_moment(Dist::Jsu, parameters, 0.3, 0.2, 1.5)));
        EXPECT_TRUE(std::isnan(probability_below_zero(Dist::Jsu, parameters)));
    }
}

TEST(Moments, JohnsonDerivativesMatchFiniteDifferences)
{
    for (double tau : {0.3, 1.0})
    {
        for (double nu : {-2.5, 2.5})
        {
            SCOPED_TRACE(::testing::Message() << "nu " << nu << ", tau " << tau);
            expect_skewed_gradients_match(Dist::Jsu, {nu, tau, 0.0, 0.3, 0.4, 1.5});
            expect_skewed_gradients_match(Dist::Jsu, {nu, tau, 0.0, -0.2, -0.3, 0.8});
        }
    }
}

}
