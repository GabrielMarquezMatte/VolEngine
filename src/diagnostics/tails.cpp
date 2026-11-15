#include "diagnostics/tails.hpp"

#include <boost/math/special_functions/beta.hpp>
#include <boost/math/policies/policy.hpp>
#include <boost/math/special_functions/gamma.hpp>

#include <cmath>

namespace ve::diagnostics
{

namespace
{

// Computing in double, not promoted to long double, keeps the p-values to the bit whatever the thread's x87
// precision (on MinGW the Python main thread and the pool threads differ).
using DoublePolicy = boost::math::policies::policy<boost::math::policies::promote_double<false>>;

}

double gamma_upper(double x, double shape, double scale)
{
    if (std::isnan(x))
    {
        return x;
    }
    return boost::math::gamma_q(shape, x / scale, DoublePolicy{});
}

double chi_squared_upper(double x, double degrees)
{
    return gamma_upper(x, degrees / 2.0, 2.0);
}

double student_two_sided(double t, double degrees)
{
    if (std::isnan(t))
    {
        return t;
    }
    return boost::math::ibeta(degrees / 2.0, 0.5, degrees / (degrees + t * t), DoublePolicy{});
}

}
