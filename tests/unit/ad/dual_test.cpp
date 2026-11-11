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

}
