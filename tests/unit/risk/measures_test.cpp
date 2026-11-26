#include "dist/cdf.hpp"
#include "risk/measures.hpp"

#include <boost/math/distributions/normal.hpp>
#include <boost/math/distributions/students_t.hpp>
#include <boost/math/quadrature/exp_sinh.hpp>
#include <boost/math/quadrature/tanh_sinh.hpp>
#include <cmath>
#include <cstddef>
#include <gtest/gtest.h>
#include <limits>
#include <numbers>
#include <vector>

namespace
{

using ve::Dist;
using ve::risk::Location;
using ve::risk::Measure;
using ve::risk::ReturnKind;

ve::PanelView row_major(const std::vector<double>& data, std::size_t t, std::size_t n)
{
    ve::PanelView view;
    view.data = data.data();
    view.t = t;
    view.n = n;
    view.stride_t = static_cast<std::ptrdiff_t>(n);
    view.stride_n = 1;
    return view;
}

struct DistCase
{
    Dist dist;
    ve::dist::ShapeParameters<double> parameters;
};

// The densities of tests/unit/dist/cdf_test.cpp's quantile cases, one or more per distribution.
const std::vector<DistCase> kDists{
    {Dist::Norm, {}},
    {Dist::Std, {.shape = 8.0}},
    {Dist::Ged, {.shape = 0.6}},
    {Dist::Ged, {.shape = 1.4}},
    {Dist::Sstd, {.skew = 1.4, .shape = 8.0}},
    {Dist::Sged, {.skew = 0.7, .shape = 1.5}},
    {Dist::Nig, {.skew = -0.4, .shape = 1.5}},
    {Dist::Ghyp, {.skew = 0.3, .shape = 2.0, .ghlambda = -1.5}},
    {Dist::Ghyp, {.skew = 0.3, .shape = 2.0, .ghlambda = 2.0}},
    {Dist::Jsu, {.skew = 0.8, .shape = 1.5}},
};

// The standardized Student-t tail mean of the spec: -s f(t_p) (nu + t_p^2) / ((nu - 1) p).
double student_tail_mean(double nu, double p)
{
    const boost::math::students_t_distribution<double> student(nu);
    const double t = boost::math::quantile(student, p);
    const double scale = std::sqrt((nu - 2.0) / nu);
    return -scale * boost::math::pdf(student, t) * (nu + t * t) / ((nu - 1.0) * p);
}

TEST(RiskEmpirical, HandValues)
{
    const std::vector<double> returns{-0.05, 0.01, 0.02, -0.02, 0.03};
    const std::vector<double> weights{1.0};
    const Measure measure = ve::risk::empirical(row_major(returns, 5, 1), weights, ReturnKind::simple, 0.8);
    // p = 0.2: position 0.8 between -0.05 and -0.02; only -0.05 is at or below the quantile -0.026.
    EXPECT_NEAR(measure.var, 0.026, 1e-15);
    EXPECT_DOUBLE_EQ(measure.es, 0.05);
}

TEST(RiskEmpirical, SingleRow)
{
    const std::vector<double> returns{0.01, -0.03};
    const std::vector<double> weights{0.5, 0.5};
    const Measure measure = ve::risk::empirical(row_major(returns, 1, 2), weights, ReturnKind::simple, 0.99);
    EXPECT_DOUBLE_EQ(measure.var, 0.01);
    EXPECT_DOUBLE_EQ(measure.es, 0.01);
}

TEST(RiskEmpirical, LogEqualsSimple)
{
    std::vector<double> log_returns(300);
    for (std::size_t k = 0; k < log_returns.size(); ++k)
    {
        log_returns[k] = 0.03 * std::sin(1.3 * static_cast<double>(k));
    }
    std::vector<double> simple(log_returns.size());
    for (std::size_t k = 0; k < simple.size(); ++k)
    {
        simple[k] = std::expm1(log_returns[k]);
    }
    const std::vector<double> weights{0.6, 0.4};
    const Measure from_log = ve::risk::empirical(row_major(log_returns, 150, 2), weights, ReturnKind::log, 0.95);
    const Measure from_simple = ve::risk::empirical(row_major(simple, 150, 2), weights, ReturnKind::simple, 0.95);
    EXPECT_NEAR(from_log.var, from_simple.var, 1e-15);
    EXPECT_NEAR(from_log.es, from_simple.es, 1e-15);
}

TEST(RiskEmpirical, WeightsNotNormalized)
{
    // Rows (s1, s2): the portfolio 2 s1 - s2 is {0.03, -0.07, 0.0}; at confidence 0.5 the quantile is the median 0.
    const std::vector<double> returns{0.01, -0.01, -0.03, 0.01, 0.0, 0.0};
    const std::vector<double> weights{2.0, -1.0};
    const Measure measure = ve::risk::empirical(row_major(returns, 3, 2), weights, ReturnKind::simple, 0.5);
    EXPECT_DOUBLE_EQ(measure.var, 0.0);
    EXPECT_DOUBLE_EQ(measure.es, 0.035);
}

TEST(RiskEmpirical, StridedEqualsContiguous)
{
    constexpr std::size_t paths = 200;
    constexpr std::size_t checkpoints = 3;
    constexpr std::size_t series = 4;
    std::vector<double> cumulative(paths * checkpoints * series);
    for (std::size_t k = 0; k < cumulative.size(); ++k)
    {
        cumulative[k] = 0.05 * std::cos(0.37 * static_cast<double>(k));
    }
    constexpr std::size_t checkpoint = 1;
    ve::PanelView slice;
    slice.data = cumulative.data() + checkpoint * series;
    slice.t = paths;
    slice.n = series;
    slice.stride_t = static_cast<std::ptrdiff_t>(checkpoints * series);
    slice.stride_n = 1;
    std::vector<double> copy;
    for (std::size_t path = 0; path < paths; ++path)
    {
        for (std::size_t j = 0; j < series; ++j)
        {
            copy.push_back(cumulative[(path * checkpoints + checkpoint) * series + j]);
        }
    }
    const std::vector<double> weights{0.1, 0.2, 0.3, 0.4};
    const Measure strided = ve::risk::empirical(slice, weights, ReturnKind::log, 0.99);
    const Measure contiguous = ve::risk::empirical(row_major(copy, paths, series), weights, ReturnKind::log, 0.99);
    EXPECT_EQ(strided.var, contiguous.var);
    EXPECT_EQ(strided.es, contiguous.es);
}

TEST(RiskParametric, NormalClosedForm)
{
    const Measure measure = ve::risk::parametric(Dist::Norm, {}, Location{.mean = 0.001, .sigma = 0.02}, 0.975);
    constexpr double z = 1.959963984540054;
    const double density = std::exp(-0.5 * z * z) / std::sqrt(2.0 * 3.141592653589793);
    EXPECT_NEAR(measure.var, -(0.001 - 0.02 * z), 1e-14);
    EXPECT_NEAR(measure.es, -(0.001 - 0.02 * density / 0.025), 1e-14);
}

TEST(RiskParametric, StudentClosedForm)
{
    constexpr double nu = 5.0;
    const Measure measure = ve::risk::parametric(Dist::Std, {.shape = nu}, Location{.mean = 0.0, .sigma = 1.0}, 0.99);
    const boost::math::students_t_distribution<double> student(nu);
    const double var = -std::sqrt((nu - 2.0) / nu) * boost::math::quantile(student, 0.01);
    EXPECT_NEAR(measure.var, var, 1e-13 * var);
    const double es = -student_tail_mean(nu, 0.01);
    EXPECT_NEAR(measure.es, es, 1e-13 * es);
}

TEST(RiskParametric, QuadratureMatchesStudent)
{
    for (double nu : {3.0, 8.0, 30.0})
    {
        for (double p : {0.05, 0.01, 0.001})
        {
            const double expected = student_tail_mean(nu, p);
            const double integrated = ve::risk::tail_mean_by_quadrature(Dist::Std, {.shape = nu}, p);
            EXPECT_NEAR(integrated, expected, 1e-10 * std::abs(expected)) << "nu " << nu << " p " << p;
        }
    }
}

TEST(RiskParametric, QuadratureMatchesNormal)
{
    const boost::math::normal_distribution<double> normal;
    for (double p : {0.05, 0.01, 0.0001})
    {
        const double expected = -boost::math::pdf(normal, boost::math::quantile(normal, p)) / p;
        EXPECT_NEAR(ve::risk::tail_mean_by_quadrature(Dist::Norm, {}, p), expected, 1e-10 * std::abs(expected)) << p;
    }
}

// E[z] = 0, so p m_p = -(the integral of z f(z) above q_p). With p past one half the lower integral crosses the kink
// of the ged and the skewed densities; the upper one, integrated here on its own, does not.
TEST(RiskParametric, QuadratureAcrossTheKink)
{
    for (const DistCase& sample : kDists)
    {
        const ve::dist::Density<double> density = ve::dist::make_density(sample.dist, sample.parameters);
        const auto integrand = [&density](double z) { return z * std::exp(ve::dist::log_density(density, z)); };
        for (double p : {0.55, 0.8})
        {
            const double q = ve::dist::quantile(sample.dist, sample.parameters, p);
            const double upper = boost::math::quadrature::exp_sinh<double>().integrate(
                integrand, q, std::numeric_limits<double>::infinity(), 1e-13);
            EXPECT_NEAR(p * ve::risk::tail_mean_by_quadrature(sample.dist, sample.parameters, p), -upper, 1e-10)
                << static_cast<int>(sample.dist) << " shape " << sample.parameters.shape << " p " << p;
        }
    }
}

// Strong skew puts the Fernandez-Steel kink, which is not at 0, below the quantile at usual confidences: p m_p must
// still equal minus the integral above q (which does not cross the kink here).
TEST(RiskParametric, QuadratureAcrossAStrongSkewKink)
{
    struct KinkCase
    {
        DistCase sample;
        double p;
    };
    for (const KinkCase& item : {KinkCase{{Dist::Sstd, {.skew = 10.0, .shape = 3.0}}, 0.05},
                                 KinkCase{{Dist::Sstd, {.skew = 10.0, .shape = 3.0}}, 0.1},
                                 KinkCase{{Dist::Sstd, {.skew = 5.0, .shape = 5.0}}, 0.1},
                                 KinkCase{{Dist::Sged, {.skew = 4.0, .shape = 0.2}}, 0.1},
                                 KinkCase{{Dist::Sged, {.skew = 4.0, .shape = 0.2}}, 0.3},
                                 KinkCase{{Dist::Sged, {.skew = 1.5, .shape = 0.6}}, 0.5}})
    {
        const auto& [dist, parameters] = item.sample;
        const ve::dist::Density<double> density = ve::dist::make_density(dist, parameters);
        const auto integrand = [&density](double z) { return z * std::exp(ve::dist::log_density(density, z)); };
        const double q = ve::dist::quantile(dist, parameters, item.p);
        ASSERT_LT(ve::dist::density_kink(density), q);
        const double upper = boost::math::quadrature::exp_sinh<double>().integrate(
            integrand, q, std::numeric_limits<double>::infinity(), 1e-13);
        EXPECT_NEAR(item.p * ve::risk::tail_mean(dist, parameters, item.p), -upper, 1e-11)
            << static_cast<int>(dist) << " skew " << parameters.skew << " p " << item.p;
    }
}

// The Johnson SU's tail mean equals (1 / p) times the integral of its quantile over (0, p), which tanh-sinh resolves
// even where the density is a narrow spike (large |skew|, small tau).
TEST(RiskParametric, JsuMatchesTheQuantileIntegral)
{
    struct JsuCase
    {
        double skew;
        double shape;
        double p;
    };
    for (const JsuCase& item : {JsuCase{5.0, 0.3, 0.05}, JsuCase{3.0, 0.5, 0.01}, JsuCase{0.8, 1.5, 0.01},
                                JsuCase{-2.0, 0.6, 0.05}, JsuCase{0.0, 4.0, 0.001}})
    {
        const ve::dist::ShapeParameters<double> parameters{.skew = item.skew, .shape = item.shape};
        const auto quantile = [&parameters](double u) { return ve::dist::quantile(Dist::Jsu, parameters, u); };
        const double expected =
            boost::math::quadrature::tanh_sinh<double>().integrate(quantile, 0.0, item.p, 1e-14) / item.p;
        const double tail = ve::risk::tail_mean(Dist::Jsu, parameters, item.p);
        EXPECT_NEAR(tail, expected, 1e-9 * std::abs(expected)) << item.skew << " " << item.shape << " " << item.p;
        EXPECT_LT(tail, quantile(item.p)) << item.skew << " " << item.shape << " " << item.p;
    }
}

// exp(710) - 1 overflows, and 0 * inf is NaN: the core reports a NaN measure instead of sorting NaN.
TEST(RiskEmpirical, NonFinitePortfolioIsNan)
{
    const std::vector<double> returns{710.0, 0.0, 0.0, 0.0, 0.01, 0.0};
    const std::vector<double> weights{0.0, 1.0};
    const Measure measure = ve::risk::empirical(row_major(returns, 3, 2), weights, ReturnKind::log, 0.9);
    EXPECT_TRUE(std::isnan(measure.var));
    EXPECT_TRUE(std::isnan(measure.es));
}

TEST(RiskParametric, EsAtLeastVar)
{
    for (const DistCase& sample : kDists)
    {
        for (double confidence : {0.95, 0.99, 0.999, 0.9999})
        {
            const Measure measure =
                ve::risk::parametric(sample.dist, sample.parameters, Location{.mean = 0.0, .sigma = 1.0}, confidence);
            EXPECT_TRUE(std::isfinite(measure.var) && std::isfinite(measure.es))
                << static_cast<int>(sample.dist) << " " << confidence;
            EXPECT_GT(measure.es, measure.var) << static_cast<int>(sample.dist) << " " << confidence;
        }
    }
}

TEST(RiskParametric, HeavyTails)
{
    for (const DistCase& sample : {DistCase{Dist::Std, {.shape = 2.1}}, DistCase{Dist::Ged, {.shape = 0.6}}})
    {
        const Measure measure =
            ve::risk::parametric(sample.dist, sample.parameters, Location{.mean = 0.0, .sigma = 1.0}, 0.999);
        EXPECT_TRUE(std::isfinite(measure.var) && std::isfinite(measure.es)) << static_cast<int>(sample.dist);
        EXPECT_GT(measure.es, measure.var) << static_cast<int>(sample.dist);
    }
}

// Past |skew| / shape ~ 354 the Johnson SU is the standardized lognormal of log-sd 1 / shape to rounding: with
// h = 1 / (2 shape^2), a = Phi^-1(p), d = sqrt(e^(2h) - 1) and s the sign of the skew, VaR = s (e^h - e^(s a / shape))
// / (e^h d) and ES = s (1 - Phi(a - s / shape) / p) / d (a negative skew mirrors the lognormal).
TEST(RiskParametric, JsuLargeSkewIsTheLognormalLimit)
{
    const boost::math::normal_distribution<double> normal;
    constexpr double shape = 1.0;
    constexpr double h = 0.5;
    const double d = std::sqrt(std::expm1(2.0 * h));
    for (double skew : {-700.0, -400.0, 400.0, 700.0})
    {
        for (double confidence : {0.95, 0.99})
        {
            const double p = 1.0 - confidence;
            const double a = boost::math::quantile(normal, p);
            const double sign = skew > 0.0 ? 1.0 : -1.0;
            const double var = sign * (std::exp(h) - std::exp(sign * a / shape)) / (std::exp(h) * d);
            const double es = sign * (1.0 - boost::math::cdf(normal, a - sign / shape) / p) / d;
            const Measure measure = ve::risk::parametric(Dist::Jsu, {.skew = skew, .shape = shape},
                                                         Location{.mean = 0.0, .sigma = 1.0}, confidence);
            EXPECT_NEAR(measure.var, var, 1e-11 * std::abs(var)) << skew << " " << confidence;
            EXPECT_NEAR(measure.es, es, 1e-11 * std::abs(es)) << skew << " " << confidence;
        }
    }
}

// Below shape ~ 0.053 the Johnson SU's variance factor overflows a double although its quantile does not: with
// skew 0 the center is 0 and log V = 4h - log 2, h = 1 / (2 shape^2), so VaR = -c sinh(a / shape), c = V^(-1/2),
// and ES = c e^h Phi(a + 1 / shape) / (2p) (the c e^h Phi(a - 1 / shape) / (2p) term is below rounding).
TEST(RiskParametric, JsuSmallShapeStaysFinite)
{
    const boost::math::normal_distribution<double> normal;
    for (double shape : {0.05, 0.04})
    {
        constexpr double p = 0.01;
        const double h = 0.5 / (shape * shape);
        const double log_scale = -0.5 * (4.0 * h - std::numbers::ln2);
        const double a = boost::math::quantile(normal, p);
        const double var = -std::exp(log_scale) * std::sinh(a / shape);
        const double es = std::exp(log_scale + h) * boost::math::cdf(normal, a + 1.0 / shape) / (2.0 * p);
        const Measure measure = ve::risk::parametric(Dist::Jsu, {.skew = 0.0, .shape = shape},
                                                     Location{.mean = 0.0, .sigma = 1.0}, 1.0 - p);
        EXPECT_GT(var, 0.0) << shape;
        EXPECT_NEAR(measure.var, var, 1e-11 * var) << shape;
        EXPECT_NEAR(measure.es, es, 1e-11 * es) << shape;
    }
}

// A scale below the smallest normal double cannot be represented: the measure is NaN, never the center.
TEST(RiskParametric, JsuUnrepresentableIsNan)
{
    for (const DistCase& sample :
         {DistCase{Dist::Jsu, {.skew = 0.5, .shape = 0.02}}, DistCase{Dist::Jsu, {.skew = 1000.0, .shape = 1.0}}})
    {
        const Measure measure =
            ve::risk::parametric(sample.dist, sample.parameters, Location{.mean = 0.0, .sigma = 1.0}, 0.99);
        EXPECT_TRUE(std::isnan(measure.var)) << sample.parameters.skew << " " << sample.parameters.shape;
        EXPECT_TRUE(std::isnan(measure.es)) << sample.parameters.skew << " " << sample.parameters.shape;
    }
}

// Every return is finite but their sum overflows: ES is still the mean of the tail, 1.5e308. p = 0.4 puts the
// quantile at position 1.6, between -1.5e308 and 0: -6e307.
TEST(RiskEmpirical, TailSumOverflowStaysFinite)
{
    const std::vector<double> returns{-1.5e308, -1.5e308, 0.0, 0.0, 0.0};
    const std::vector<double> weights{1.0};
    const Measure measure = ve::risk::empirical(row_major(returns, 5, 1), weights, ReturnKind::simple, 0.6);
    EXPECT_DOUBLE_EQ(measure.var, 6e307);
    EXPECT_EQ(measure.es, 1.5e308);
}

// 1e308 - (-1e308) overflows; the interpolated quantile between them does not.
TEST(RiskEmpirical, InterpolationAcrossTheRangeStaysFinite)
{
    const std::vector<double> returns{1e308, -1e308};
    const std::vector<double> weights{1.0};
    const Measure median = ve::risk::empirical(row_major(returns, 2, 1), weights, ReturnKind::simple, 0.5);
    EXPECT_EQ(median.var, 0.0);
    EXPECT_EQ(median.es, 1e308);
    const Measure quarter = ve::risk::empirical(row_major(returns, 2, 1), weights, ReturnKind::simple, 0.75);
    EXPECT_DOUBLE_EQ(quarter.var, 5e307);
    EXPECT_EQ(quarter.es, 1e308);
}

// numpy.quantile's linear method takes a + (b - a) t below t = 0.5 and b - (b - a) (1 - t) from it on, which here
// differ in the last digit: np.quantile(returns, 0.4) is 0.00027963251408746025 (numpy 2.5.4).
TEST(RiskEmpirical, InterpolationIsNumpyLerp)
{
    const std::vector<double> returns{0.015044876543591856, -0.0162162916647514, 0.017917661415551206,
                                      -0.0069043142010255945, 0.0050689303241628295};
    const std::vector<double> weights{1.0};
    const Measure measure = ve::risk::empirical(row_major(returns, 5, 1), weights, ReturnKind::simple, 0.6);
    EXPECT_EQ(measure.var, -0.00027963251408746025);
}

// Past a skew of ~1.3e154 (or below its inverse) xi^2 overflows the sstd standardisation, though the density is then
// its half-kernel limit: with X the unit-variance Student-t, m1 = E|X| and s = sqrt(1 - m1^2), z = (|X| - m1) / s for
// a large skew and -(|X| - m1) / s for a small one. With T = X / r, r = sqrt((nu - 2) / nu), and f its density,
// E[|T|; |T| <= t] = 2 (nu f(0) - (nu + t^2) f(t)) / (nu - 1) and E[|T|; |T| >= t] = 2 (nu + t^2) f(t) / (nu - 1).
TEST(RiskParametric, SstdExtremeSkewIsTheHalfKernelLimit)
{
    constexpr double nu = 5.0;
    const boost::math::students_t_distribution<double> student(nu);
    const double r = std::sqrt((nu - 2.0) / nu);
    const double m1 = r * 2.0 * std::sqrt(nu) * std::tgamma(0.5 * (nu + 1.0))
                      / (std::sqrt(std::numbers::pi) * (nu - 1.0) * std::tgamma(0.5 * nu));
    const double s = std::sqrt(1.0 - m1 * m1);
    for (double confidence : {0.95, 0.99})
    {
        const double p = 1.0 - confidence;
        const double t_low = boost::math::quantile(student, 0.5 * (1.0 + p));
        const double below = 2.0 * (nu * boost::math::pdf(student, 0.0) - (nu + t_low * t_low) * boost::math::pdf(student, t_low))
                             / (nu - 1.0);
        const Measure large{.var = -(r * t_low - m1) / s, .es = -(r * below / p - m1) / s};
        const double t_high = boost::math::quantile(student, 1.0 - 0.5 * p);
        const double above = 2.0 * (nu + t_high * t_high) * boost::math::pdf(student, t_high) / (nu - 1.0);
        const Measure small{.var = (r * t_high - m1) / s, .es = (r * above / p - m1) / s};
        for (double skew : {1e155, 1e200, 1e300, 1e-155, 1e-200, 1e-300})
        {
            const Measure expected = skew > 1.0 ? large : small;
            const Measure measure = ve::risk::parametric(Dist::Sstd, {.skew = skew, .shape = nu},
                                                         Location{.mean = 0.0, .sigma = 1.0}, confidence);
            EXPECT_NEAR(measure.var, expected.var, 1e-10 * std::abs(expected.var)) << skew << " " << confidence;
            EXPECT_NEAR(measure.es, expected.es, 1e-10 * std::abs(expected.es)) << skew << " " << confidence;
        }
    }
}

// The sged has no closed form here; past the overflow its measures equal those just inside it (the other half holds
// a mass below 1e-300), and stay ordered.
TEST(RiskParametric, SgedExtremeSkewStaysFinite)
{
    for (double shape : {0.5, 1.5, 10.0})
    {
        for (double skew : {1e155, 1e300, 1e-155, 1e-300})
        {
            const Measure measure =
                ve::risk::parametric(Dist::Sged, {.skew = skew, .shape = shape}, Location{.mean = 0.0, .sigma = 1.0}, 0.99);
            const double inside = skew > 1.0 ? 1e140 : 1e-140;
            const Measure reference =
                ve::risk::parametric(Dist::Sged, {.skew = inside, .shape = shape}, Location{.mean = 0.0, .sigma = 1.0}, 0.99);
            EXPECT_NEAR(measure.var, reference.var, 1e-12 * std::abs(reference.var)) << skew << " " << shape;
            EXPECT_NEAR(measure.es, reference.es, 1e-12 * std::abs(reference.es)) << skew << " " << shape;
            EXPECT_GT(measure.es, measure.var) << skew << " " << shape;
        }
    }
}

TEST(RiskParametric, ZeroSigma)
{
    const Measure measure = ve::risk::parametric(Dist::Sstd, {.skew = 1.2, .shape = 6.0},
                                                 Location{.mean = 0.004, .sigma = 0.0}, 0.99);
    EXPECT_EQ(measure.var, -0.004);
    EXPECT_EQ(measure.es, -0.004);
}

}
