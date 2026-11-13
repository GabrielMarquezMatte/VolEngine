#include "ad/dual.hpp"

#include <array>
#include <cmath>
#include <functional>
#include <gtest/gtest.h>

namespace
{

constexpr double kStep = 1e-6;

double central_difference(const std::function<double(double)>& f, double x, double step)
{
    return (f(x + step) - f(x - step)) / (2.0 * step);
}

TEST(Dual, ArithmeticAndChainRule)
{
    auto x = ve::ad::variable<4>(1.5, 0);
    auto y = ve::ad::variable<4>(-0.7, 1);
    auto f = ve::ad::log(x * x + ve::ad::exp(y)) / (y - 2.0);
    auto plain = [](double a, double b) { return std::log(a * a + std::exp(b)) / (b - 2.0); };
    EXPECT_NEAR(f.value, plain(1.5, -0.7), 1e-14);
    double dx = central_difference([&](double a) { return plain(a, -0.7); }, 1.5, kStep);
    double dy = central_difference([&](double b) { return plain(1.5, b); }, -0.7, kStep);
    EXPECT_NEAR(f.grad[0], dx, 1e-8);
    EXPECT_NEAR(f.grad[1], dy, 1e-8);
    EXPECT_EQ(f.grad[2], 0.0);
    EXPECT_EQ(f.grad[3], 0.0);
}

TEST(Dual, MixedOperandsAndNegation)
{
    auto x = ve::ad::variable<1>(2.0, 0);
    auto f = 3.0 - 4.0 / x + (-x) * 0.5 + x / 8.0 + 1.0 * x + x - 1.0;
    double expected = 3.0 - 2.0 - 1.0 + 0.25 + 2.0 + 2.0 - 1.0;
    EXPECT_NEAR(f.value, expected, 1e-14);
    EXPECT_NEAR(f.grad[0], 4.0 / 4.0 - 0.5 + 0.125 + 1.0 + 1.0, 1e-14);
    EXPECT_EQ(ve::ad::sqrt(x).grad[0], 0.5 / std::sqrt(2.0));
}

TEST(Dual, PowAndAbs)
{
    auto x = ve::ad::variable<2>(1.3, 0);
    auto y = ve::ad::variable<2>(0.8, 1);
    auto p = ve::ad::pow(x, y);
    EXPECT_NEAR(p.grad[0], central_difference([](double a) { return std::pow(a, 0.8); }, 1.3, 1e-7), 1e-7);
    EXPECT_NEAR(p.grad[1], central_difference([](double b) { return std::pow(1.3, b); }, 0.8, 1e-7), 1e-7);
    auto q = ve::ad::pow(x, 2.5);
    EXPECT_NEAR(q.grad[0], central_difference([](double a) { return std::pow(a, 2.5); }, 1.3, 1e-7), 1e-7);
    EXPECT_EQ(q.grad[1], 0.0);
    EXPECT_EQ(ve::ad::abs(ve::ad::variable<1>(-2.0, 0)).grad[0], -1.0);
    EXPECT_EQ(ve::ad::abs(ve::ad::variable<1>(2.0, 0)).grad[0], 1.0);
    auto zero = ve::ad::abs(ve::ad::variable<1>(0.0, 0));
    EXPECT_EQ(zero.value, 0.0);
    EXPECT_EQ(zero.grad[0], 0.0);
}

// x = e^u with u = 460: x and the seed x' = x sit near 1e200, where 1 / x^2 underflows.
TEST(Dual, QuotientsOfLargeOperandsKeepTheirDerivative)
{
    auto u = ve::ad::variable<1>(460.0, 0);
    auto x = ve::ad::exp(u);
    EXPECT_NEAR((1.0 / x).grad[0] / (1.0 / x).value, -1.0, 1e-15);
    EXPECT_NEAR((u / x).grad[0] / (u / x).value, 1.0 / 460.0 - 1.0, 1e-15);
    EXPECT_NEAR((x / x).grad[0], 0.0, 1e-15);
}

TEST(Dual, Log1pMatchesItsDerivative)
{
    auto x = ve::ad::variable<1>(1e-12, 0);
    auto f = ve::ad::log1p(x);
    EXPECT_EQ(f.value, std::log1p(1e-12));
    EXPECT_EQ(f.grad[0], 1.0 / (1.0 + 1e-12));
}

TEST(Dual, ComparisonsUseValue)
{
    auto a = ve::ad::variable<2>(1.0, 0);
    auto b = ve::ad::constant<2>(2.0);
    EXPECT_TRUE(a < b);
    EXPECT_TRUE(b > a);
    EXPECT_TRUE(a <= b);
    EXPECT_TRUE(b >= a);
    EXPECT_FALSE(b < a);
    EXPECT_EQ(ve::ad::value_of(a), 1.0);
    EXPECT_EQ(ve::ad::value_of(2.5), 2.5);
}

TEST(Dual, PowWithZeroBaseAndVariableExponentHasFiniteGradient)
{
    auto base = ve::ad::abs(ve::ad::variable<2>(0.0, 0));
    auto exponent = ve::ad::variable<2>(0.7, 1);
    auto p = ve::ad::pow(base, exponent);
    EXPECT_EQ(p.value, 0.0);
    EXPECT_EQ(p.grad[0], 0.0);
    EXPECT_EQ(p.grad[1], 0.0);
    auto q = ve::ad::pow(ve::ad::variable<2>(0.0, 0), ve::ad::variable<2>(2.5, 1));
    EXPECT_EQ(q.value, 0.0);
    EXPECT_EQ(q.grad[0], 0.0);
    EXPECT_EQ(q.grad[1], 0.0);
}

TEST(Dual, HyperbolicFunctions)
{
    for (double v : {-2.0, 0.3, 1.7})
    {
        auto x = ve::ad::variable<2>(v, 1);
        auto a = ve::ad::asinh(x);
        auto s = ve::ad::sinh(x);
        auto c = ve::ad::cosh(x);
        EXPECT_EQ(a.value, std::asinh(v));
        EXPECT_EQ(s.value, std::sinh(v));
        EXPECT_EQ(c.value, std::cosh(v));
        EXPECT_NEAR(a.grad[1], 1.0 / std::sqrt(1.0 + v * v), 1e-15);
        EXPECT_NEAR(s.grad[1], std::cosh(v), 1e-15);
        EXPECT_NEAR(c.grad[1], std::sinh(v), 1e-15);
        EXPECT_EQ(a.grad[0], 0.0);
    }
}

TEST(Dual, TrigonometricFunctions)
{
    for (double v : {-2.0, 0.3, 1.7})
    {
        auto x = ve::ad::variable<2>(v, 1);
        auto s = ve::ad::sin(x);
        auto c = ve::ad::cos(x);
        EXPECT_EQ(s.value, std::sin(v));
        EXPECT_EQ(c.value, std::cos(v));
        EXPECT_EQ(s.grad[1], std::cos(v));
        EXPECT_EQ(c.grad[1], -std::sin(v));
        EXPECT_EQ(s.grad[0], 0.0);
    }
}

TEST(Dual, ErfcMatchesItsDerivative)
{
    for (double v : {-2.0, 0.0, 0.3, 4.0})
    {
        auto x = ve::ad::variable<2>(v, 1);
        auto e = ve::ad::erfc(x);
        EXPECT_EQ(e.value, std::erfc(v));
        double step = 1e-6;
        double difference = (std::erfc(v + step) - std::erfc(v - step)) / (2.0 * step);
        EXPECT_NEAR(e.grad[1], difference, 1e-9);
        EXPECT_EQ(e.grad[0], 0.0);
    }
}

}
