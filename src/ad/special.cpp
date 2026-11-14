#include "ad/special.hpp"

#include <boost/math/special_functions/gamma.hpp>
#include <cmath>

namespace ve::ad
{

namespace
{

// promote_double<false>: on MinGW the pool threads run the x87 unit at another precision than the main thread, so a
// long double evaluation would differ between threads.
using LogGammaPolicy = boost::math::policies::policy<
    boost::math::policies::promote_double<false>, boost::math::policies::domain_error<boost::math::policies::ignore_error>,
    boost::math::policies::pole_error<boost::math::policies::ignore_error>,
    boost::math::policies::overflow_error<boost::math::policies::ignore_error>,
    boost::math::policies::evaluation_error<boost::math::policies::ignore_error>>;

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

double log_gamma(double x)
{
    return boost::math::lgamma(x, LogGammaPolicy{});
}

}
