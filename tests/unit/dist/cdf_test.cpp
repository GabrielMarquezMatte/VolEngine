#include "dist/cdf.hpp"
#include "dist/moments.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <limits>
#include <vector>

namespace
{

using ve::Dist;

struct CdfCase
{
    Dist dist;
    ve::dist::ShapeParameters<double> parameters;
    std::array<double, 2> expected;
};

constexpr std::array<CdfCase, 8> kCases{{
    {Dist::Norm, {}, {0.066807201268858085, 0.61791142218895256}},
    {Dist::Std, {.shape = 6.0}, {0.057919999999999978, 0.63704250791223438}},
    {Dist::Ged, {.shape = 1.4}, {0.064408960501297186, 0.64087619258778505}},
    {Dist::Sstd, {.skew = 1.3, .shape = 6.0}, {0.039613691296787761, 0.67012366619190511}},
    {Dist::Sged, {.skew = 0.8, .shape = 1.6}, {0.076324648554635791, 0.59341768109990278}},
    {Dist::Nig, {.skew = 0.3, .shape = 1.2}, {0.043694526826888819, 0.67937614224277576}},
    {Dist::Ghyp, {.skew = -0.2, .shape = 2.0, .ghlambda = -1.5}, {0.065568350139092985, 0.61898170922993212}},
    {Dist::Jsu, {.skew = 0.5, .shape = 1.8}, {0.047633763880890301, 0.66242721987116615}},
}};

TEST(Cdf, MatchesRugarch)
{
    constexpr std::array<double, 2> points{-1.5, 0.3};
    for (const CdfCase& sample : kCases)
    {
        for (std::size_t i = 0; i < points.size(); ++i)
        {
            double tolerance = sample.expected[i] * 1e-9;
            if (sample.dist == Dist::Nig || sample.dist == Dist::Ghyp)
            {
                tolerance = 1e-8;
            }
            double actual = ve::dist::cdf(sample.dist, sample.parameters, points[i]);
            EXPECT_NEAR(actual, sample.expected[i], tolerance);
        }
    }
}

void check_derivative(Dist dist, const ve::dist::ShapeParameters<double>& parameters)
{
    auto density = ve::dist::make_density(dist, parameters);
    constexpr double h = 1e-4;
    for (double z : {-2.0, -0.3, 0.7, 2.5})
    {
        double upper = ve::dist::cdf(dist, parameters, z + h);
        double lower = ve::dist::cdf(dist, parameters, z - h);
        double derivative = (upper - lower) / (2 * h);
        double expected = std::exp(ve::dist::log_density(density, z));
        EXPECT_NEAR(derivative, expected, expected * 1e-6) << static_cast<int>(dist) << " " << z;
    }
    double below_zero = ve::dist::probability_below_zero(dist, parameters);
    EXPECT_NEAR(ve::dist::cdf(dist, parameters, 0.0), below_zero, 1e-10);
}

TEST(Cdf, DerivativeIsTheDensity)
{
    for (const CdfCase& sample : kCases)
    {
        check_derivative(sample.dist, sample.parameters);
        auto parameters = sample.parameters;
        parameters.shape += 0.8;
        if (ve::dist::is_fernandez_steel(sample.dist))
        {
            parameters.skew = 1.0 / parameters.skew;
        }
        check_derivative(sample.dist, parameters);
    }
}

TEST(Cdf, SortedMatchesPointwise)
{
    std::vector<double> points;
    for (std::size_t i = 0; i < 200; ++i)
    {
        points.push_back(-6.0 + 12.0 * static_cast<double>(i) / 199.0);
    }
    for (const CdfCase& sample : kCases)
    {
        auto values = ve::dist::sorted_cdf(sample.dist, sample.parameters, points);
        for (std::size_t i = 0; i < points.size(); ++i)
        {
            EXPECT_NEAR(values[i], ve::dist::cdf(sample.dist, sample.parameters, points[i]), 1e-10);
        }
        EXPECT_TRUE(ve::dist::sorted_cdf(sample.dist, sample.parameters, {}).empty());
    }
}

void check_tails(Dist dist, const ve::dist::ShapeParameters<double>& parameters)
{
    double previous = 0.0;
    for (double z : {-40.0, -8.0, 0.0, 8.0, 40.0})
    {
        double value = ve::dist::cdf(dist, parameters, z);
        EXPECT_TRUE(std::isfinite(value));
        EXPECT_GE(value, previous);
        EXPECT_LE(value, 1.0);
        previous = value;
    }
}

TEST(Cdf, TailsStayInRange)
{
    for (const CdfCase& sample : kCases)
    {
        check_tails(sample.dist, sample.parameters);
        double infinity = std::numeric_limits<double>::infinity();
        EXPECT_EQ(ve::dist::cdf(sample.dist, sample.parameters, -infinity), 0.0);
        EXPECT_EQ(ve::dist::cdf(sample.dist, sample.parameters, infinity), 1.0);
    }
    check_tails(Dist::Std, {.shape = 2.05});
    check_tails(Dist::Ghyp, {.skew = -0.2, .shape = 0.3, .ghlambda = -1.5});
}

// The pointwise nig and ghyp cdf switches from the tail integral to the integral from 0 at |z| = 1: it stays monotone
// across both switches to rounding, even for a peaked density (zeta 0.01); the switch used to jump by 8.6e-9.
TEST(Cdf, ContinuousAcrossTheTailSwitch)
{
    for (ve::dist::ShapeParameters<double> parameters :
         {ve::dist::ShapeParameters<double>{.skew = 0.9, .shape = 0.01}, ve::dist::ShapeParameters<double>{.skew = -0.5, .shape = 0.05},
          ve::dist::ShapeParameters<double>{.skew = 0.3, .shape = 1.2}})
    {
        for (double edge : {-1.0, 1.0})
        {
            double below = ve::dist::cdf(Dist::Nig, parameters, std::nextafter(edge, -2.0));
            double at = ve::dist::cdf(Dist::Nig, parameters, edge);
            double above = ve::dist::cdf(Dist::Nig, parameters, std::nextafter(edge, 2.0));
            EXPECT_LE(below, at + 1e-14) << parameters.skew << " " << edge;
            EXPECT_LE(at, above + 1e-14) << parameters.skew << " " << edge;
            EXPECT_NEAR(below, above, 1e-12) << parameters.skew << " " << edge;
        }
    }
}

// sorted_cdf on the sample size of a fit, with the narrow gaps of 5000 points: the same values as pointwise.
TEST(Cdf, SortedMatchesPointwiseOnDenseSamples)
{
    std::vector<double> points;
    for (std::size_t i = 0; i < 5000; ++i)
    {
        double u = (static_cast<double>(i) + 0.5) / 5000.0;
        points.push_back(4.0 * std::tan(3.0 * (u - 0.5)) / std::tan(1.5));
    }
    for (ve::dist::ShapeParameters<double> parameters :
         {ve::dist::ShapeParameters<double>{.skew = -0.2, .shape = 2.0, .ghlambda = -1.5},
          ve::dist::ShapeParameters<double>{.skew = 0.6, .shape = 0.05, .ghlambda = 1.0}})
    {
        auto values = ve::dist::sorted_cdf(Dist::Ghyp, parameters, points);
        for (std::size_t i = 0; i < points.size(); i += 97)
        {
            EXPECT_NEAR(values[i], ve::dist::cdf(Dist::Ghyp, parameters, points[i]), 1e-10) << points[i];
        }
    }
}

}
