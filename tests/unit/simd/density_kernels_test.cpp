#include "simd/density_kernels.hpp"

#include "dist/symmetric.hpp"
#include "model/likelihood/terms.hpp"
#include "simd/fixtures.hpp"
#include "simd/targets.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <gtest/gtest.h>
#include <limits>
#include <string>
#include <vector>

namespace
{

using ve::Dist;
using ve::dist::LogDensity;

constexpr double kEpsilon = 2.220446049250313e-16;
// The vector Log and Log1p are within 4 ulps and Exp within 1: a term is within 8 ulps of the largest magnitude
// that goes into it (the constant, the density part, the log of the variance), and the ged adds the amplification of
// y^p = exp(p log y), a relative error of p |log y| ulps.
constexpr double kTermUlps = 8.0;
// The partials have no cancellation except the one of the variance, taken against 1 / sigma2.
constexpr double kPartialTolerance = 1e-13;
// The sums add up to 5001 terms in the lanes and in the scalar loop in different orders.
constexpr double kSumTolerance = 1e-13;

const std::size_t kLengths[] = {1, 3, 7, 600, 5001};

struct Case
{
    std::string name;
    Dist dist;
    double shape;
};

const Case kCases[] = {{"norm", Dist::Norm, 0.0}, {"std 2.5", Dist::Std, 2.5}, {"std 8", Dist::Std, 8.0},
                       {"std 30", Dist::Std, 30.0}, {"ged 0.8", Dist::Ged, 0.8}, {"ged 1.5", Dist::Ged, 1.5},
                       {"ged 3", Dist::Ged, 3.0}};

struct Series
{
    std::vector<double> e;
    std::vector<double> sigma2;
};

Series make_series(ve::test::Stream& stream, std::size_t n)
{
    Series series{stream.vector(n), std::vector<double>(n)};
    for (std::size_t t = 0; t < n; ++t)
    {
        series.e[t] *= 2.5;
        series.sigma2[t] = std::exp(0.6 * stream.next());
    }
    return series;
}

LogDensity<double> density_of(const Case& c)
{
    return ve::dist::make_log_density(c.dist, c.shape);
}

// log f(z) - log(sigma2) / 2 as the scalar likelihood forms it (z^2 for norm and std, |z| for ged).
double scalar_term(const LogDensity<double>& density, double e, double sigma2)
{
    double log_density = 0.0;
    if (density.dist == Dist::Norm)
    {
        log_density = ve::dist::normal_log_density_at(density, e * e / sigma2);
    }
    if (density.dist == Dist::Std)
    {
        log_density = ve::dist::student_log_density_at(density, e * e / sigma2);
    }
    if (density.dist == Dist::Ged)
    {
        log_density = ve::dist::ged_log_density_at(density, std::fabs(e) / std::sqrt(sigma2));
    }
    return log_density - 0.5 * std::log(sigma2);
}

// The size every term is compared against: the parts that add up to it.
double magnitude(const LogDensity<double>& density, double e, double sigma2)
{
    const double term = scalar_term(density, e, sigma2);
    const double half_log = 0.5 * std::fabs(std::log(sigma2));
    const double part = std::fabs(term - density.log_constant + 0.5 * std::log(sigma2));
    double amplification = 1.0;
    if (density.dist == Dist::Ged)
    {
        const double y = std::fabs(e) / std::sqrt(sigma2) / density.scale;
        amplification += y > 0.0 ? density.power * std::fabs(std::log(y)) : 0.0;
    }
    return amplification * (std::fabs(density.log_constant) + part + half_log + 1.0);
}

TEST(DensityKernels, TermsMatchTheScalarFormsOnEveryTarget)
{
    for (const auto target : ve::test::runnable_targets())
    {
        const ve::test::ForcedTarget forced(target);
        ve::test::Stream stream(31);
        for (const Case& c : kCases)
        {
            const LogDensity<double> density = density_of(c);
            for (const std::size_t n : kLengths)
            {
                const Series series = make_series(stream, n);
                std::vector<double> terms(n);
                const double sum = ve::simd::symmetric_terms(density, series.e, series.sigma2, terms);
                double expected_sum = 0.0;
                double absolute_sum = 0.0;
                for (std::size_t t = 0; t < n; ++t)
                {
                    const double expected = scalar_term(density, series.e[t], series.sigma2[t]);
                    const double bound = kTermUlps * kEpsilon * magnitude(density, series.e[t], series.sigma2[t]);
                    EXPECT_NEAR(terms[t], expected, bound) << c.name << " n = " << n << " t = " << t;
                    expected_sum += expected;
                    absolute_sum += std::fabs(expected);
                }
                EXPECT_NEAR(sum, expected_sum, kSumTolerance * absolute_sum) << c.name << " n = " << n;
            }
        }
    }
}

TEST(DensityKernels, PartialsMatchTheScalarAdjointOnEveryTarget)
{
    for (const auto target : ve::test::runnable_targets())
    {
        const ve::test::ForcedTarget forced(target);
        ve::test::Stream stream(37);
        for (const Case& c : kCases)
        {
            const LogDensity<double> density = density_of(c);
            for (const std::size_t n : kLengths)
            {
                const Series series = make_series(stream, n);
                std::vector<double> terms(n);
                std::vector<double> d_e(n);
                std::vector<double> d_sigma2(n);
                ve::simd::SymmetricPartials partials{d_e, d_sigma2};
                const double sum =
                    ve::simd::symmetric_terms_with_partials(density, {series.e, series.sigma2, terms}, partials);
                double scale_sum = 0.0;
                double power_sum = 0.0;
                double scale_magnitude = 0.0;
                double power_magnitude = 0.0;
                // The ged partial in the power is v log(y) / 2: its log is taken of a y that differs from the scalar
                // one by an ulp, which near y = 1 is a large relative error of a small log, bounded by ulps of v.
                double power_floor = 0.0;
                for (std::size_t t = 0; t < n; ++t)
                {
                    const ve::dist::TermAdjoint expected =
                        ve::dist::term_adjoint(density, series.e[t], series.sigma2[t]);
                    EXPECT_NEAR(d_e[t], expected.e, kPartialTolerance * std::fabs(expected.e))
                        << c.name << " n = " << n << " t = " << t;
                    EXPECT_NEAR(d_sigma2[t], expected.sigma2,
                                kPartialTolerance * (std::fabs(expected.sigma2) + 1.0 / series.sigma2[t]))
                        << c.name << " n = " << n << " t = " << t;
                    scale_sum += expected.scale;
                    power_sum += expected.power;
                    scale_magnitude += std::fabs(expected.scale);
                    power_magnitude += std::fabs(expected.power);
                    if (density.dist == Dist::Ged)
                    {
                        const double y = std::fabs(series.e[t]) / std::sqrt(series.sigma2[t]) / density.scale;
                        power_floor += 0.5 * std::pow(y, density.power);
                    }
                }
                EXPECT_NEAR(partials.d_scale, scale_sum, kPartialTolerance * scale_magnitude) << c.name << " n = " << n;
                EXPECT_NEAR(partials.d_power, power_sum, kPartialTolerance * power_magnitude + 16.0 * kEpsilon * power_floor)
                    << c.name << " n = " << n;
                EXPECT_TRUE(std::isfinite(sum));
            }
        }
    }
}

// The fused kernel is the value kernel with the partials added: same terms, same sum, to the bit.
TEST(DensityKernels, TheFusedKernelHasTheValueOfTheValueKernel)
{
    for (const auto target : ve::test::runnable_targets())
    {
        const ve::test::ForcedTarget forced(target);
        ve::test::Stream stream(41);
        for (const Case& c : kCases)
        {
            const LogDensity<double> density = density_of(c);
            for (const std::size_t n : kLengths)
            {
                const Series series = make_series(stream, n);
                std::vector<double> value_terms(n);
                std::vector<double> fused_terms(n);
                std::vector<double> d_e(n);
                std::vector<double> d_sigma2(n);
                ve::simd::SymmetricPartials partials{d_e, d_sigma2};
                const double value = ve::simd::symmetric_terms(density, series.e, series.sigma2, value_terms);
                const double fused =
                    ve::simd::symmetric_terms_with_partials(density, {series.e, series.sigma2, fused_terms}, partials);
                EXPECT_EQ(value, fused) << c.name << " n = " << n;
                EXPECT_EQ(value_terms, fused_terms) << c.name << " n = " << n;
            }
        }
    }
}

// A variance that is not positive and finite, anywhere in the vector or in its tail, gives the invalid value exactly.
TEST(DensityKernels, AnInvalidVarianceGivesTheInvalidLoglik)
{
    const double bad[] = {0.0, -1.0, std::numeric_limits<double>::quiet_NaN(),
                          std::numeric_limits<double>::infinity()};
    for (const auto target : ve::test::runnable_targets())
    {
        const ve::test::ForcedTarget forced(target);
        ve::test::Stream stream(43);
        for (const Case& c : kCases)
        {
            const LogDensity<double> density = density_of(c);
            for (const std::size_t n : {1U, 7U, 600U})
            {
                for (const std::size_t where : {std::size_t{0}, n / 2, n - 1})
                {
                    for (const double value : bad)
                    {
                        Series series = make_series(stream, n);
                        series.sigma2[where] = value;
                        std::vector<double> d_e(n);
                        std::vector<double> d_sigma2(n);
                        ve::simd::SymmetricPartials partials{d_e, d_sigma2};
                        EXPECT_EQ(ve::simd::symmetric_terms(density, series.e, series.sigma2, {}),
                                  ve::model::kInvalidLoglik)
                            << c.name << " n = " << n << " at " << where << " value " << value;
                        EXPECT_EQ(ve::simd::symmetric_terms_with_partials(density, {series.e, series.sigma2, {}},
                                                                           partials),
                                  ve::model::kInvalidLoglik)
                            << c.name << " n = " << n << " at " << where << " value " << value;
                    }
                }
            }
        }
    }
}

// A term that is not finite makes the sum not finite: the invalid value, as model::sum_of_terms.
TEST(DensityKernels, ATermThatOverflowsGivesTheInvalidLoglik)
{
    for (const auto target : ve::test::runnable_targets())
    {
        const ve::test::ForcedTarget forced(target);
        ve::test::Stream stream(47);
        for (const Case& c : kCases)
        {
            const LogDensity<double> density = density_of(c);
            Series series = make_series(stream, 11);
            series.e[5] = 1e300;
            series.sigma2[5] = 1e-300;
            const double expected = ve::model::sum_of_terms<double>(
                series.e, series.sigma2,
                [&](double e, double sigma2) { return scalar_term(density, e, sigma2) + 0.5 * std::log(sigma2); }, {});
            EXPECT_EQ(expected, ve::model::kInvalidLoglik) << c.name << ": the scalar sum is invalid too";
            EXPECT_EQ(ve::simd::symmetric_terms(density, series.e, series.sigma2, {}), ve::model::kInvalidLoglik)
                << c.name;
        }
    }
}

// A residual of exactly zero: the ged term is its constant, and its partials follow the Dual convention (slope 0 at
// the kink), as ve::dist::term_adjoint.
TEST(DensityKernels, AZeroResidualFollowsTheDualConvention)
{
    for (const auto target : ve::test::runnable_targets())
    {
        const ve::test::ForcedTarget forced(target);
        ve::test::Stream stream(53);
        for (const Case& c : kCases)
        {
            const LogDensity<double> density = density_of(c);
            Series series = make_series(stream, 9);
            series.e[0] = 0.0;
            series.e[8] = 0.0;
            std::vector<double> terms(9);
            std::vector<double> d_e(9);
            std::vector<double> d_sigma2(9);
            ve::simd::SymmetricPartials partials{d_e, d_sigma2};
            const double sum = ve::simd::symmetric_terms_with_partials(density, {series.e, series.sigma2, terms}, partials);
            ASSERT_TRUE(std::isfinite(sum)) << c.name;
            for (const std::size_t t : {std::size_t{0}, std::size_t{8}})
            {
                const ve::dist::TermAdjoint expected = ve::dist::term_adjoint(density, 0.0, series.sigma2[t]);
                EXPECT_EQ(d_e[t], expected.e) << c.name;
                EXPECT_NEAR(d_sigma2[t], expected.sigma2, kPartialTolerance * std::fabs(expected.sigma2)) << c.name;
                EXPECT_NEAR(terms[t], scalar_term(density, 0.0, series.sigma2[t]),
                            kTermUlps * kEpsilon * magnitude(density, 0.0, series.sigma2[t]))
                    << c.name;
            }
        }
    }
}

// The vector exp is only defined up to about 709.78 (above it, it wraps): y^power must still overflow to +infinity
// like std::pow, so the term is -infinity and the sum invalid, for a huge y with a small power and for a y a little
// above 1 with a huge power.
TEST(DensityKernels, AGedPowerThatOverflowsGivesTheInvalidLoglik)
{
    struct Overflow
    {
        double shape;
        double e;
        double sigma2;
    };
    for (const auto target : ve::test::runnable_targets())
    {
        const ve::test::ForcedTarget forced(target);
        for (const double shape : {3.0, 22026.0})
        {
            const LogDensity<double> density = ve::dist::make_log_density(Dist::Ged, shape);
            Series series{std::vector<double>(7, 0.5), std::vector<double>(7, 1.0)};
            // y = 1e250 / scale for the small power, y = 1.07 for the huge one.
            series.e[3] = shape < 100.0 ? 1e100 : 1.07 * density.scale;
            series.sigma2[3] = shape < 100.0 ? 1e-300 : 1.0;
            std::vector<double> d_e(7);
            std::vector<double> d_sigma2(7);
            ve::simd::SymmetricPartials partials{d_e, d_sigma2};
            EXPECT_EQ(std::pow(std::fabs(series.e[3]) / std::sqrt(series.sigma2[3]) / density.scale, density.power),
                      std::numeric_limits<double>::infinity())
                << "the scalar power overflows too, shape " << shape;
            EXPECT_EQ(ve::simd::symmetric_terms(density, series.e, series.sigma2, {}), ve::model::kInvalidLoglik)
                << "shape " << shape;
            EXPECT_EQ(ve::simd::symmetric_terms_with_partials(density, {series.e, series.sigma2, {}}, partials),
                      ve::model::kInvalidLoglik)
                << "shape " << shape;
        }
    }
}

}
