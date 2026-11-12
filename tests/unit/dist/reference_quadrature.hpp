#pragma once

#include "dist/densities.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>

namespace ve::testing
{

constexpr double kTrapezoidBound = 60.0;
constexpr std::size_t kTrapezoidPoints = 200001;

// The trapezoid rule over [-kTrapezoidBound, kTrapezoidBound].
template <class Function>
double trapezoid(const Function& integrand)
{
    double step = 2.0 * kTrapezoidBound / static_cast<double>(kTrapezoidPoints - 1);
    double sum = 0.0;
    for (std::size_t i = 0; i < kTrapezoidPoints; ++i)
    {
        double z = -kTrapezoidBound + step * static_cast<double>(i);
        double weight = step;
        if (i == 0 || i == kTrapezoidPoints - 1)
        {
            weight = 0.5 * step;
        }
        sum += weight * integrand(z);
    }
    return sum;
}

// The Student-t tails beyond the trapezoid bound decay only as a power of z and do not vanish at 60, so
// 2 int_B^inf z^moment f(z) dz is added from the series of C a^m z^-2m (1 + a/z^2)^-m, with a = nu - 2 and
// m = (nu + 1) / 2.
inline double student_tails(const ve::dist::LogDensity<double>& density, double nu, double moment)
{
    constexpr int kTailTerms = 8;
    double a = nu - 2.0;
    double m = 0.5 * (nu + 1.0);
    double both_sides = 2.0 * std::exp(density.log_constant + m * std::log(a));
    double coefficient = 1.0;
    double tails = 0.0;
    for (int j = 0; j < kTailTerms; ++j)
    {
        double power = -2.0 * m - 2.0 * j + moment;
        tails += coefficient * std::pow(a, j) * std::pow(kTrapezoidBound, power + 1.0) / -(power + 1.0);
        coefficient *= -(m + j) / (j + 1.0);
    }
    return both_sides * tails;
}

constexpr std::size_t kGradedPoints = 40001;
constexpr std::size_t kTailPoints = 4001;

// int z^k f(z) dz for k = 0, 1, 2.
using RawMoments = std::array<double, 3>;

template <class Function>
void accumulate_moments(const Function& density, double z, double weight, RawMoments& moments)
{
    double term = weight * density(z);
    moments[0] += term;
    moments[1] += term * z;
    moments[2] += term * z * z;
}

// The trapezoid rule in t over z = kink + (bound - kink) t^3: a Fernandez-Steel density has a power
// singularity in a derivative at its kink, which a uniform grid resolves only to O(h^(1 + shape)), and the
// cube makes the integrand smooth at t = 0.
template <class Function>
void graded_moments(const Function& density, double kink, double bound, RawMoments& moments)
{
    double span = bound - kink;
    double step = 1.0 / static_cast<double>(kGradedPoints - 1);
    for (std::size_t i = 1; i < kGradedPoints; ++i)
    {
        double t = step * static_cast<double>(i);
        double weight = 3.0 * std::fabs(span) * t * t * step;
        if (i == kGradedPoints - 1)
        {
            weight *= 0.5;
        }
        accumulate_moments(density, kink + span * t * t * t, weight, moments);
    }
}

// The part beyond |z| = kTrapezoidBound, by the trapezoid rule in u = 1 / |z|: a tail z^-p becomes
// u^(p - 2), which vanishes at u = 0 for the moments taken here, and an exponential tail vanishes with all
// its derivatives.
template <class Function>
void tail_moments(const Function& density, RawMoments& moments)
{
    double step = 1.0 / (kTrapezoidBound * static_cast<double>(kTailPoints - 1));
    for (std::size_t i = 1; i < kTailPoints; ++i)
    {
        double u = step * static_cast<double>(i);
        double weight = step / (u * u);
        if (i == kTailPoints - 1)
        {
            weight *= 0.5;
        }
        accumulate_moments(density, 1.0 / u, weight, moments);
        accumulate_moments(density, -1.0 / u, weight, moments);
    }
}

// The raw moments 0 to 2 over the real line, split at the kink of the density.
template <class Function>
RawMoments reference_moments(const Function& density, double kink)
{
    RawMoments moments{};
    graded_moments(density, kink, -kTrapezoidBound, moments);
    graded_moments(density, kink, kTrapezoidBound, moments);
    tail_moments(density, moments);
    return moments;
}

// int integrand(z) dz over the real line, split at two points where it is not smooth: the graded rule runs
// out of each point, and the two meet halfway between them.
template <class Function>
double reference_integral(const Function& integrand, double first, double second)
{
    double low = std::min(first, second);
    double high = std::max(first, second);
    RawMoments moments{};
    graded_moments(integrand, low, -kTrapezoidBound, moments);
    graded_moments(integrand, high, kTrapezoidBound, moments);
    tail_moments(integrand, moments);
    if (high > low)
    {
        double middle = 0.5 * (low + high);
        graded_moments(integrand, low, middle, moments);
        graded_moments(integrand, high, middle, moments);
    }
    return moments[0];
}

}
