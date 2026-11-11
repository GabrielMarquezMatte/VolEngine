#include "dist/moments.hpp"

#include <cmath>
#include <cstddef>
#include <numbers>

namespace ve::dist::detail
{

namespace
{

constexpr int kNewtonIterations = 100;
constexpr double kNewtonTolerance = 1e-15;

struct Legendre
{
    double value;
    double slope;
};

// P_n(x) by the three-term recurrence, and P_n'(x) = n (x P_n - P_{n-1}) / (x^2 - 1).
Legendre legendre(double x)
{
    double previous = 1.0;
    double current = x;
    for (std::size_t k = 2; k <= kGaussLegendreNodes; ++k)
    {
        double order = static_cast<double>(k);
        double next = ((2.0 * order - 1.0) * x * current - (order - 1.0) * previous) / order;
        previous = current;
        current = next;
    }
    double n = static_cast<double>(kGaussLegendreNodes);
    return {current, n * (x * current - previous) / (x * x - 1.0)};
}

double legendre_root(double guess)
{
    double x = guess;
    for (int iteration = 0; iteration < kNewtonIterations; ++iteration)
    {
        Legendre p = legendre(x);
        double correction = p.value / p.slope;
        x -= correction;
        if (std::abs(correction) < kNewtonTolerance)
        {
            break;
        }
    }
    return x;
}

// Roots of P_n on [-1, 1] by Newton from cos(pi (i + 3/4) / (n + 1/2)), weights 2 / ((1 - x^2) P_n'(x)^2), both
// mapped to [0, 1]; the roots come in pairs +-x.
std::array<QuadratureNode, kGaussLegendreNodes> compute_gauss_legendre()
{
    constexpr std::size_t n = kGaussLegendreNodes;
    std::array<QuadratureNode, n> nodes{};
    for (std::size_t i = 0; i < n / 2; ++i)
    {
        double guess = std::cos(std::numbers::pi * (static_cast<double>(i) + 0.75) / (static_cast<double>(n) + 0.5));
        double x = legendre_root(guess);
        Legendre p = legendre(x);
        double weight = 1.0 / ((1.0 - x * x) * p.slope * p.slope);
        nodes[i] = {0.5 * (1.0 - x), weight};
        nodes[n - 1 - i] = {0.5 * (1.0 + x), weight};
    }
    return nodes;
}

}

const std::array<QuadratureNode, kGaussLegendreNodes>& gauss_legendre()
{
    static const std::array<QuadratureNode, kGaussLegendreNodes> nodes = compute_gauss_legendre();
    return nodes;
}

}
