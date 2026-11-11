#include "ad/special.hpp"

#include <cmath>

namespace ve::ad
{

namespace
{

// The asymptotic series through x^-12 is accurate to ~1e-15 only from x = 10.
constexpr double kAsymptoticThreshold = 10.0;

double digamma_asymptotic(double x)
{
    double inverse = 1.0 / x;
    double inverse2 = inverse * inverse;
    double series = inverse2 * (-1.0 / 12.0 + inverse2 * (1.0 / 120.0 + inverse2 * (-1.0 / 252.0 +
                    inverse2 * (1.0 / 240.0 + inverse2 * (-1.0 / 132.0 + inverse2 * (691.0 / 32760.0))))));
    return std::log(x) - 0.5 * inverse + series;
}

}

double digamma(double x)
{
    double shift = 0.0;
    while (x < kAsymptoticThreshold)
    {
        shift -= 1.0 / x;
        x += 1.0;
    }
    return shift + digamma_asymptotic(x);
}

}
