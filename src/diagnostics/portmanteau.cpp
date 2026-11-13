#include "diagnostics/portmanteau.hpp"

#include "diagnostics/tails.hpp"

#include <cmath>
#include <numeric>
#include <vector>

namespace ve::diagnostics
{

namespace
{

std::vector<double> centered(std::span<const double> z, bool squared)
{
    std::vector<double> x(z.size());
    for (std::size_t t = 0; t < z.size(); ++t)
    {
        double value = 0.0;
        if (std::isfinite(z[t]))
        {
            value = z[t];
        }
        x[t] = value;
        if (squared)
        {
            x[t] = value * value;
        }
    }
    double mean = std::accumulate(x.begin(), x.end(), 0.0) / static_cast<double>(x.size());
    for (double& value : x)
    {
        value -= mean;
    }
    return x;
}

double autocorrelation(std::span<const double> x, std::size_t lag, double denominator)
{
    double numerator = 0.0;
    for (std::size_t t = 0; t + lag < x.size(); ++t)
    {
        numerator += x[t] * x[t + lag];
    }
    return numerator / denominator;
}

}

TestResult weighted_ljung_box(std::span<const double> z, std::size_t lag, std::size_t fitdf, bool squared)
{
    std::vector<double> x = centered(z, squared);
    double denominator = std::inner_product(x.begin(), x.end(), x.begin(), 0.0);
    double n = static_cast<double>(z.size());
    double length = static_cast<double>(lag);
    double degrees = static_cast<double>(fitdf);
    if (squared)
    {
        degrees = 0.0;
    }
    double statistic = 0.0;
    for (std::size_t k = 1; k <= lag; ++k)
    {
        double rho = autocorrelation(x, k, denominator);
        double weight = (length - static_cast<double>(k) + 1.0) / length;
        statistic += weight * rho * rho / (n - static_cast<double>(k));
    }
    statistic *= n * (n + 2.0);
    double d = 2.0 * length * length + 3.0 * length + 1.0 - 6.0 * length * degrees;
    double shape = 0.75 * (length + 1.0) * (length + 1.0) * length / d;
    double scale = (2.0 / 3.0) * d / (length * (length + 1.0));
    return {statistic, gamma_upper(statistic, shape, scale)};
}

TestResult weighted_arch_lm(std::span<const double> residuals, std::span<const double> sigma,
                            std::size_t lag, std::size_t fitdf)
{
    std::vector<double> z(residuals.size());
    for (std::size_t t = 0; t < z.size(); ++t)
    {
        z[t] = residuals[t] / sigma[t];
    }
    std::vector<double> x = centered(z, true);
    double denominator = std::inner_product(x.begin(), x.end(), x.begin(), 0.0);
    double n = static_cast<double>(z.size());
    double length = static_cast<double>(lag);
    double degrees = static_cast<double>(fitdf);
    double statistic = 0.0;
    for (std::size_t k = fitdf + 1; k <= lag; ++k)
    {
        double rho = autocorrelation(x, k, denominator);
        double weight = (length - static_cast<double>(k) + degrees + 1.0) / length;
        statistic += weight * rho * rho;
    }
    statistic *= n;
    double d = 2.0 * length * length + 3.0 * length + 2.0 * length * degrees +
               2.0 * degrees * degrees + 3.0 * degrees + 1.0;
    double shape = 0.75 * (length + degrees + 1.0) * (length + degrees + 1.0) * (length - degrees) / d;
    double scale = (2.0 / 3.0) * d / (length * (length + degrees + 1.0));
    return {statistic, gamma_upper(statistic, shape, scale)};
}

std::optional<std::string> ljung_box_problem(std::size_t n, std::size_t lag, std::size_t fitdf, bool squared)
{
    if (lag == 0 || lag >= n)
    {
        return "lag must be positive and smaller than the observation count";
    }
    // Division avoids overflow in 3 * fitdf - 1 for untrusted size_t arguments.
    if (!squared && fitdf > lag / 3 + (lag % 3 == 2 ? 1 : 0))
    {
        return "lag must be at least 3 * fitdf - 1";
    }
    return std::nullopt;
}

std::optional<std::string> arch_lm_problem(std::size_t n, std::size_t lag, std::size_t fitdf)
{
    if (fitdf == 0 || fitdf >= lag)
    {
        return "fitdf must be positive and smaller than lag";
    }
    if (lag >= n)
    {
        return "lag must be smaller than the observation count";
    }
    return std::nullopt;
}

}
