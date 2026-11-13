#include "dist/moments.hpp"
#include "sim/philox.hpp"
#include "sim/samplers.hpp"

#include <boost/math/special_functions/bessel.hpp>
#include <cmath>
#include <cstddef>
#include <functional>
#include <gtest/gtest.h>
#include <string>
#include <vector>

namespace
{

constexpr std::size_t kDraws = 400000;
constexpr double kStandardErrors = 5.0;

// The sample mean of f over the draws lies within kStandardErrors of target.
void expect_mean(const std::vector<double>& draws, const std::function<double(double)>& f, double target,
                 const std::string& what)
{
    double sum = 0.0;
    double squares = 0.0;
    for (double x : draws)
    {
        double value = f(x);
        sum += value;
        squares += value * value;
    }
    double n = static_cast<double>(draws.size());
    double mean = sum / n;
    double variance = squares / n - mean * mean;
    double error = std::sqrt(std::max(variance, 0.0) / n);
    EXPECT_NEAR(mean, target, kStandardErrors * error + 1e-12) << what;
}

std::vector<double> draw_all(const std::function<double(ve::sim::Stream&)>& draw, std::size_t count)
{
    ve::sim::Stream stream(2026, 11);
    std::vector<double> draws(count);
    for (double& x : draws)
    {
        x = draw(stream);
    }
    return draws;
}

TEST(Samplers, GammaMoments)
{
    for (double shape : {0.3, 1.0, 7.5})
    {
        std::vector<double> draws =
            draw_all([shape](ve::sim::Stream& s) { return ve::sim::gamma(s, shape); }, kDraws);
        std::string what = "shape " + std::to_string(shape);
        expect_mean(draws, [](double x) { return x; }, shape, what + " mean");
        expect_mean(draws, [shape](double x) { return (x - shape) * (x - shape); }, shape, what + " variance");
    }
}

struct GigCase
{
    double lambda;
    double chi;
    double psi;
};

TEST(Samplers, GigMoments)
{
    // The three regions of Hormann and Leydold: without shift, with shift, and the two non-T-concave cases.
    for (GigCase c : {GigCase{-0.5, 1.0, 2.0}, GigCase{1.5, 0.3, 4.0}, GigCase{-3.0, 2.0, 0.5},
                      GigCase{0.4, 0.1, 0.1}, GigCase{0.0, 0.05, 0.2}})
    {
        std::vector<double> draws =
            draw_all([c](ve::sim::Stream& s) { return ve::sim::gig(s, c.lambda, c.chi, c.psi); }, kDraws);
        double w = std::sqrt(c.chi * c.psi);
        double ratio = boost::math::cyl_bessel_k(c.lambda + 1.0, w) / boost::math::cyl_bessel_k(c.lambda, w);
        std::string what = "lambda " + std::to_string(c.lambda);
        expect_mean(draws, [](double x) { return x; }, std::sqrt(c.chi / c.psi) * ratio, what + " E W");
        expect_mean(draws, [](double x) { return 1.0 / x; },
                    std::sqrt(c.psi / c.chi) * ratio - 2.0 * c.lambda / c.chi, what + " E 1/W");
    }
}

struct DistCase
{
    ve::Dist dist;
    ve::dist::ShapeParameters<double> parameters;
    std::string name;
};

std::vector<DistCase> standardized_cases()
{
    using ve::Dist;
    return {{Dist::Norm, {}, "norm"},
            {Dist::Std, {0.0, 8.0, 0.0}, "std 8"},
            {Dist::Ged, {0.0, 1.3, 0.0}, "ged 1.3"},
            {Dist::Ged, {0.0, 0.6, 0.0}, "ged 0.6"},
            {Dist::Sstd, {1.4, 8.0, 0.0}, "sstd"},
            {Dist::Sged, {0.7, 1.5, 0.0}, "sged"},
            {Dist::Nig, {-0.4, 1.5, 0.0}, "nig"},
            {Dist::Ghyp, {0.3, 2.0, -1.5}, "ghyp -1.5"},
            {Dist::Ghyp, {0.3, 2.0, 2.0}, "ghyp 2"},
            {Dist::Jsu, {0.8, 1.5, 0.0}, "jsu"}};
}

TEST(Samplers, StandardizedMoments)
{
    for (const DistCase& c : standardized_cases())
    {
        ve::sim::Sampler sampler(c.dist, c.parameters);
        std::vector<double> draws = draw_all([&sampler](ve::sim::Stream& s) { return sampler.draw(s); }, kDraws);
        expect_mean(draws, [](double z) { return z; }, 0.0, c.name + " mean");
        expect_mean(draws, [](double z) { return z * z; }, 1.0, c.name + " variance");
        expect_mean(draws, [](double z) { return z < 0.0 ? 1.0 : 0.0; },
                    ve::dist::probability_below_zero(c.dist, c.parameters), c.name + " P(z < 0)");
        expect_mean(draws, [](double z) { return std::abs(z); }, ve::dist::abs_moment(c.dist, c.parameters, 1.0),
                    c.name + " E|z|");
        expect_mean(draws,
                    [](double z) { return std::pow(std::abs(z - 0.2) - 0.3 * (z - 0.2), 1.5); },
                    ve::dist::shifted_moment(c.dist, c.parameters, 0.3, 0.2, 1.5), c.name + " shifted moment");
    }
}

TEST(Samplers, FiniteAtDomainEdges)
{
    using ve::Dist;
    std::vector<DistCase> edges{{Dist::Std, {0.0, 2.05, 0.0}, "std 2.05"},
                                {Dist::Ged, {0.0, 0.25, 0.0}, "ged 0.25"},
                                {Dist::Nig, {0.2, 1e4, 0.0}, "nig zeta 1e4"},
                                {Dist::Ghyp, {0.2, 0.3, -3.0}, "ghyp lambda -3 zeta 0.3"},
                                {Dist::Jsu, {0.5, 0.2, 0.0}, "jsu shape 0.2"}};
    for (const DistCase& c : edges)
    {
        ve::sim::Sampler sampler(c.dist, c.parameters);
        std::vector<double> draws = draw_all([&sampler](ve::sim::Stream& s) { return sampler.draw(s); }, 100000);
        for (double z : draws)
        {
            ASSERT_TRUE(std::isfinite(z)) << c.name;
        }
    }
}

}
