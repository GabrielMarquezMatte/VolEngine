#include "garch/csgarch.hpp"
#include "garch/egarch.hpp"
#include "garch/gjrgarch.hpp"
#include "garch/power_garch.hpp"
#include "garch/sgarch.hpp"
#include "mean/arma.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <gtest/gtest.h>
#include <numbers>
#include <span>
#include <vector>

namespace
{

using ve::garch::Presample;

constexpr std::size_t kLength = 40;

std::vector<double> residual_series()
{
    std::vector<double> e(kLength);
    for (std::size_t t = 0; t < kLength; ++t)
    {
        e[t] = 0.3 * std::sin(0.9 * static_cast<double>(t)) - 0.1 * std::cos(0.37 * static_cast<double>(t));
    }
    return e;
}

std::vector<double> shift_series()
{
    std::vector<double> c(kLength);
    for (std::size_t t = 0; t < kLength; ++t)
    {
        c[t] = 0.01 + 0.005 * std::sin(0.2 * static_cast<double>(t));
    }
    return c;
}

const std::vector<double> kAlpha{0.06, 0.03};
const std::vector<double> kBeta{0.7, 0.1};
const std::vector<double> kGamma{0.04, -0.02};

// The held values, then the step at every later t, as a caller of the step writes it.
template <class Step>
std::vector<double> stepped(const Presample<double>& presample, Step step)
{
    std::vector<double> state(kLength);
    std::fill(state.begin(), state.begin() + static_cast<std::ptrdiff_t>(presample.held), presample.value);
    for (std::size_t t = presample.held; t < kLength; ++t)
    {
        state[t] = step(std::span<const double>(state), t);
    }
    return state;
}

void expect_same(const std::vector<double>& looped, const std::vector<double>& whole)
{
    for (std::size_t t = 0; t < kLength; ++t)
    {
        ASSERT_EQ(looped[t], whole[t]) << "t = " << t;
    }
}

void expect_sgarch_and_gjr_steps(const Presample<double>& presample)
{
    std::vector<double> e = residual_series();
    std::vector<double> c = shift_series();
    std::vector<double> whole(kLength);
    ve::garch::SGarchParameters<double> s{0.02, kAlpha, kBeta, c};
    ve::garch::sgarch_variance<double>(e, s, presample, whole);
    expect_same(stepped(presample,
                        [&](std::span<const double> sigma2, std::size_t t)
                        { return ve::garch::sgarch_step<double>(e, s, presample, sigma2, t); }),
                whole);
    ve::garch::GjrGarchParameters<double> g{0.02, kAlpha, kBeta, kGamma, c};
    ve::garch::gjrgarch_variance<double>(e, g, presample, whole);
    expect_same(stepped(presample,
                        [&](std::span<const double> sigma2, std::size_t t)
                        { return ve::garch::gjrgarch_step<double>(e, g, presample, sigma2, t); }),
                whole);
}

// The series function holds sigma^power and takes the power of it at the end; the step works in sigma^power.
void expect_power_steps(const Presample<double>& presample)
{
    std::vector<double> e = residual_series();
    std::vector<double> c = shift_series();
    std::vector<double> moments{1.0, 1.1};
    ve::garch::PowerGarchParameters<double> p{0.02, kAlpha, kBeta, kGamma, kGamma, 1.4, moments, 1e-6, true, c};
    std::vector<double> whole(kLength);
    ve::garch::power_garch_variance<double>(e, p, presample, whole);
    double presample_power = ve::garch::raise(presample.value, 0.5 * p.power, p.free_power);
    std::vector<double> looped = stepped(
        {presample_power, presample.held}, [&](std::span<const double> power, std::size_t t)
        { return ve::garch::power_garch_step<double>(e, p, presample_power, power, t); });
    for (std::size_t t = 0; t < kLength; ++t)
    {
        ASSERT_EQ(ve::garch::raise(looped[t], 2.0 / p.power, p.free_power), whole[t]) << "t = " << t;
    }
}

void expect_egarch_steps(const Presample<double>& presample)
{
    std::vector<double> e = residual_series();
    std::vector<double> c = shift_series();
    ve::garch::EGarchParameters<double> p{-0.1, kAlpha, kBeta, kGamma, std::sqrt(2.0 / std::numbers::pi), c};
    std::vector<double> whole(kLength);
    ve::garch::egarch_variance<double>(e, p, presample, whole);
    double log_presample = std::log(presample.value);
    std::vector<double> looped = stepped(
        {log_presample, presample.held}, [&](std::span<const double> log_sigma2, std::size_t t)
        { return ve::garch::egarch_step<double>(e, p, log_presample, log_sigma2, t); });
    for (std::size_t t = 0; t < kLength; ++t)
    {
        ASSERT_EQ(std::exp(looped[t]), whole[t]) << "t = " << t;
    }
}

void expect_csgarch_steps(const Presample<double>& presample)
{
    std::vector<double> squared(kLength);
    std::ranges::transform(residual_series(), squared.begin(), [](double v) { return v * v; });
    std::vector<double> c = shift_series();
    ve::garch::CsGarchParameters<double> p{0.02, kAlpha, kBeta, 0.95, 0.03, c};
    std::vector<double> whole(kLength);
    std::vector<double> whole_permanent(kLength);
    ve::garch::csgarch_variance<double>(squared, p, presample, whole, whole_permanent);
    double long_run = p.omega / (1.0 - p.rho);
    std::vector<double> sigma2(kLength, presample.value + long_run);
    std::vector<double> permanent(kLength, long_run);
    for (std::size_t t = presample.held; t < kLength; ++t)
    {
        auto step = ve::garch::csgarch_step<double>(squared, p, presample, sigma2, permanent, t);
        permanent[t] = step.permanent;
        sigma2[t] = step.variance;
    }
    expect_same(sigma2, whole);
    expect_same(permanent, whole_permanent);
}

TEST(Garch, StepLoopsEqualSeriesFunctions)
{
    for (std::size_t held : {std::size_t{0}, std::size_t{2}})
    {
        Presample<double> presample{0.05, held};
        expect_sgarch_and_gjr_steps(presample);
        expect_power_steps(presample);
        expect_egarch_steps(presample);
        expect_csgarch_steps(presample);
    }
}

TEST(Mean, StepLoopEqualsRecursion)
{
    std::vector<double> d = residual_series();
    std::vector<double> ar{0.3, -0.1};
    std::vector<double> ma{0.2, 0.05};
    ve::mean::ArmaParameters<double> p{0.0, ar, ma};
    std::vector<double> whole(kLength);
    ve::mean::arma_residuals_of_deviations<double>(d, p, whole);
    std::vector<double> looped(kLength);
    for (std::size_t t = 0; t < kLength; ++t)
    {
        looped[t] = ve::mean::arma_step<double>(d, looped, p, t);
    }
    expect_same(looped, whole);
}

}
