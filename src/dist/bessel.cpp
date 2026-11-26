#include "dist/bessel.hpp"

// MSVC's code generator reports C4701 (local possibly used uninitialized) for n and u in Boost's
// bessel_ik.hpp, which set them on every path that reads them; /external:W0 does not reach code-generation
// warnings, so /WX would fail on Boost's code.
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4701)
#endif
#include <boost/math/special_functions/bessel.hpp>
#include <boost/math/policies/policy.hpp>
#if defined(_MSC_VER)
#pragma warning(pop)
#endif

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

// promote_double<false>: Boost computes in double, not long double. On MinGW a pool thread runs the x87 unit at
// another precision than the Python main thread, so a long double evaluation would differ between threads.
using IgnoreErrors = policies::policy<policies::promote_double<false>,
                                      policies::overflow_error<policies::ignore_error>,
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

// K_2(x) from K_0 and K_1 as Boost's bessel_kn takes it (detail/bessel_kn.hpp), its rescaling and overflow
// included, so that the result has the bits of cyl_bessel_k(2, x) for 0 < x.
double k2_from(double k0, double k1, double x)
{
    constexpr double kMax = std::numeric_limits<double>::max();
    double prev = k0;
    double current = k1;
    double scale = 1.0;
    double fact = 2.0 / x;
    if ((kMax - std::fabs(prev)) / fact < std::fabs(current))
    {
        scale /= current;
        prev /= current;
        current = 1.0;
    }
    double value = fact * current + prev;
    if (kMax * scale < std::fabs(value))
    {
        return std::numeric_limits<double>::infinity();
    }
    return value / scale;
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

// log K_v(x) = log(1/2 int exp(-x cosh t + v t) dt), the trapezoid rule on a grid of step h centred on the peak of
// the integrand, t* = asinh(v / x). The integrand is analytic and decays double exponentially on the upper side and
// exponentially (e^(v t)) on the lower one, so the rule converges to rounding for h <= 0.4 sigma, sigma the width of
// the peak (x^2 + v^2)^(-1/4) (measured at 1e-14 on log K, 4e-15 on the slope in x), and the same sum divided by its
// total gives both slopes, E[-cosh t] in x and E[t] in the order. The sum stops where the integrand falls below
// exp(-kSlopeCutoff) of its peak. Validated for |order| <= kSlopeMaxOrder and x >= kSlopeMinX: for a smaller x the
// lower tail needs more than ~2e3 nodes at an order of 0.1, and past kSlopeMaxOrder it was not tested.
constexpr double kSlopeMaxOrder = 200.0;
constexpr double kSlopeMinX = 1e-8;
// The step 0.4 (x^2 + v^2)^(-1/4) falls below the resolution of exp at x ~ 1e25 and the sum would not end; the order
// slope (~ v / x) is also lost in rounding long before. 1e6 is where the tests stop.
constexpr double kSlopeMaxX = 1e6;
constexpr int kSlopeMaxNodes = 100000;
constexpr double kSlopeCutoff = 40.0;
constexpr double kSlopeMaxStep = 0.2;
constexpr double kSlopeStepOverSigma = 0.4;

// The integrand exp(-x cosh t + order t) and the grid peak + k step.
struct TrapezoidGrid
{
    double order;
    double x;
    double peak;
    double step;
    double cosh_peak;
    double exp_peak;
};

struct TrapezoidSums
{
    double total;
    double cosh_moment;
    double t_moment;
};

// One side of the grid, t = peak + direction k h, k >= 1, to the cutoff; e^t by recurrence.
void add_trapezoid_side(const TrapezoidGrid& grid, double direction, TrapezoidSums& sums)
{
    double cosh_peak = grid.cosh_peak;
    double ratio = std::exp(direction * grid.step);
    double up = grid.exp_peak;
    double down = 1.0 / up;
    for (int k = 1; k <= kSlopeMaxNodes; ++k)
    {
        up *= ratio;
        down /= ratio;
        double cosh_t = 0.5 * (up + down);
        double t = grid.peak + direction * grid.step * k;
        double exponent = -grid.x * (cosh_t - cosh_peak) + grid.order * (t - grid.peak);
        if (!(exponent >= -kSlopeCutoff))
        {
            return;
        }
        double weight = std::exp(exponent);
        sums.total += weight;
        sums.cosh_moment += cosh_t * weight;
        sums.t_moment += t * weight;
    }
}

detail::LogBesselSlopes trapezoid_slopes(double order, double x)
{
    double v = std::fabs(order);
    double peak = std::asinh(v / x);
    double sigma = std::pow(x * x + v * v, -0.25);
    double step = std::min(kSlopeMaxStep, kSlopeStepOverSigma * sigma);
    double cosh_peak = std::cosh(peak);
    TrapezoidSums sums{1.0, cosh_peak, peak};
    TrapezoidGrid grid{v, x, peak, step, cosh_peak, std::exp(peak)};
    add_trapezoid_side(grid, 1.0, sums);
    add_trapezoid_side(grid, -1.0, sums);
    double sign = std::copysign(1.0, order);
    double order_slope = order == 0.0 ? 0.0 : sign * sums.t_moment / sums.total;
    return {order_slope, -sums.cosh_moment / sums.total};
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

namespace detail
{

LogBesselWithSlope log_bessel_k_scaled_with_slope(double order, double x)
{
    // At order -1 and 0 < x < kAsymptoticFrom both logs are Boost's (orders 1 and 2 are far from 5/2).
    if (order == -1.0 && x > 0.0 && x < kAsymptoticFrom)
    {
        IgnoreErrors policy;
        double k0 = boost::math::cyl_bessel_k(0.0, x, policy);
        double k1 = boost::math::cyl_bessel_k(1.0, x, policy);
        double value = std::log(k1) + x;
        double below = std::log(k2_from(k0, k1, x)) + x;
        return {value, 1.0 - std::exp(below - value) - order / x};
    }
    double value = log_bessel_k_scaled(order, x);
    return {value, 1.0 - std::exp(log_bessel_k_scaled(order - 1.0, x) - value) - order / x};
}

LogBesselSlopes log_bessel_k_slopes(double order, double x, bool scaled)
{
    constexpr double kNan = std::numeric_limits<double>::quiet_NaN();
    if (std::isnan(order) || !(x > 0.0) || !std::isfinite(x))
    {
        return {kNan, kNan};
    }
    if (std::fabs(order) <= kSlopeMaxOrder && x >= kSlopeMinX && x <= kSlopeMaxX)
    {
        LogBesselSlopes slopes = trapezoid_slopes(order, x);
        return {slopes.order, slopes.x + (scaled ? 1.0 : 0.0)};
    }
    // ponytail: outside the validated domain, Richardson on log K (~1e-10 on the slope in the order) and the ratio
    // K_(v-1) / K_v, both of the scaled log where asked: widen the trapezoid's domain if a model ever lives there.
    if (scaled)
    {
        double d_order = order_slope([](double v, double y) { return log_bessel_k_scaled(v, y); }, order, x);
        return {d_order, log_bessel_k_scaled_slope(order, x)};
    }
    double d_order = order_slope([](double v, double y) { return log_bessel_k(v, y); }, order, x);
    return {d_order, log_bessel_k_slope(order, x)};
}

}

}
