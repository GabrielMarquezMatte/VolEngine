#pragma once

#include "dist/densities.hpp"

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

}
