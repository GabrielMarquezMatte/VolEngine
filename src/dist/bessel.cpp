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
// range, and both its recurrence and ours run once per unit of order; K_order(x < 700) overflows long before 1e5.
constexpr double kMaxBoostOrder = 1e5;
constexpr double kRescaleAbove = 1e100;
constexpr int kMaxTerms = 30;
constexpr double kTermTolerance = 1e-17;
// Boost takes K_v with |4 v^2 - 25| < 8 x eps^(1/4) from the first four terms of the Hankel series
// (A&S 9.7.2), exact at v = 5/2 but neither accurate nor smooth in v beside it. Its eps is that of the type
// it computes in, long double on GCC: the band uses the eps^(1/4) of double, the wider of the two.
constexpr double kFiveHalvesBand = 0x1p-13;

namespace policies = boost::math::policies;

using IgnoreErrors = policies::policy<policies::overflow_error<policies::ignore_error>,
                                      policies::underflow_error<policies::ignore_error>,
                                      policies::domain_error<policies::ignore_error>,
                                      policies::evaluation_error<policies::ignore_error>>;

// log(e^x K_order(x)) by the Hankel series, for 0 <= order < 2 and x >= kAsymptoticFrom, where it converges to
// rounding within a handful of terms; beyond |order| ~ 130 it diverges at x = 700.
double hankel_scaled_log_k(double order, double x)
{
    double four_order_squared = 4.0 * order * order;
    double term = 1.0;
    double sum = 0.0;
    for (int k = 1; k <= kMaxTerms && std::fabs(term) >= kTermTolerance; ++k)
    {
        double odd = 2.0 * k - 1.0;
        term *= (four_order_squared - odd * odd) / (8.0 * k * x);
        sum += term;
    }
    return 0.5 * std::log(std::numbers::pi / (2.0 * x)) + std::log1p(sum);
}

// log(e^x K_order(x)) for order >= 0 and x >= kAsymptoticFrom: the series gives K_frac and K_(frac+1), and
// the forward recurrence K_(v+1) = K_(v-1) + 2v/x K_v, stable for K, climbs to the order. Boost cannot serve
// here: its K underflows for log K < -745, which large orders reach already at x = 900.
double scaled_recurrence_log_k(double order, double x)
{
    double fraction = order - std::floor(order);
    int steps = static_cast<int>(order - fraction);
    if (steps == 0)
    {
        return hankel_scaled_log_k(fraction, x);
    }
    double below = std::exp(hankel_scaled_log_k(fraction, x));
    double above = std::exp(hankel_scaled_log_k(fraction + 1.0, x));
    double log_scale = 0.0;
    for (int n = 1; n < steps; ++n)
    {
        double next = below + 2.0 * (fraction + n) / x * above;
        below = above;
        above = next;
        if (above > kRescaleAbove)
        {
            below /= above;
            log_scale += std::log(above);
            above = 1.0;
        }
    }
    return log_scale + std::log(above);
}

double boost_log_k(double order, double x)
{
    return std::log(boost::math::cyl_bessel_k(order, x, IgnoreErrors()));
}

// order >= 0 and x < kAsymptoticFrom. Beside 5/2, K_v = K_(v-2) + 2 (v - 1) / x K_(v-1), from orders Boost
// takes by its general method.
double log_k_avoiding_five_halves_shortcut(double order, double x)
{
    if (std::fabs(4.0 * order * order - 25.0) >= 8.0 * x * kFiveHalvesBand)
    {
        return boost_log_k(order, x);
    }
    double below = boost_log_k(order - 2.0, x);
    double above = boost_log_k(order - 1.0, x);
    return above + std::log(2.0 * (order - 1.0) / x + std::exp(below - above));
}

}

double log_bessel_k(double order, double x)
{
    if (std::isnan(x) || x < 0.0 || std::isnan(order))
    {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (std::fabs(order) >= kMaxBoostOrder)
    {
        return std::numeric_limits<double>::infinity();
    }
    if (x >= kAsymptoticFrom)
    {
        return scaled_recurrence_log_k(std::fabs(order), x) - x;
    }
    return log_k_avoiding_five_halves_shortcut(std::fabs(order), x);
}

double log_bessel_k_scaled(double order, double x)
{
    if (x >= kAsymptoticFrom && std::fabs(order) < kMaxBoostOrder)
    {
        return scaled_recurrence_log_k(std::fabs(order), x);
    }
    return log_bessel_k(order, x) + x;
}

}
