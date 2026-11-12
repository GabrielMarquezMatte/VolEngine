#include "dist/bessel.hpp"

#include <boost/math/special_functions/bessel.hpp>
#include <boost/math/policies/policy.hpp>

#include <cmath>
#include <limits>
#include <numbers>

namespace ve::dist
{

namespace
{

// Boost's K underflows to zero from about x = 708; the asymptotic series is accurate to rounding well before that.
constexpr double kAsymptoticFrom = 700.0;
// Boost rounds the order to an int through its default policy, which throws for NaN and for |order| beyond int
// range; K_order(x) is +inf for any x below the asymptotic start long before that.
constexpr double kMaxBoostOrder = 1e9;
constexpr int kMaxTerms = 30;
constexpr double kTermTolerance = 1e-17;

namespace policies = boost::math::policies;

using IgnoreErrors = policies::policy<policies::overflow_error<policies::ignore_error>,
                                      policies::underflow_error<policies::ignore_error>,
                                      policies::domain_error<policies::ignore_error>,
                                      policies::evaluation_error<policies::ignore_error>>;

double asymptotic_log_k(double order, double x)
{
    double four_order_squared = 4.0 * order * order;
    double term = 1.0;
    double sum = 0.0;
    for (int k = 1; k <= kMaxTerms; ++k)
    {
        double odd = 2.0 * k - 1.0;
        term *= (four_order_squared - odd * odd) / (8.0 * k * x);
        sum += term;
        if (std::fabs(term) < kTermTolerance)
        {
            break;
        }
    }
    return 0.5 * std::log(std::numbers::pi / (2.0 * x)) - x + std::log1p(sum);
}

}

double log_bessel_k(double order, double x)
{
    if (x >= kAsymptoticFrom)
    {
        return asymptotic_log_k(order, x);
    }
    if (std::isnan(order))
    {
        return order;
    }
    if (std::fabs(order) >= kMaxBoostOrder)
    {
        return std::numeric_limits<double>::infinity();
    }
    return std::log(boost::math::cyl_bessel_k(std::fabs(order), x, IgnoreErrors()));
}

}
