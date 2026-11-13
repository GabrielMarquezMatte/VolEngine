#include "dcc/panel.hpp"

#include "linalg/symmetric_eigen.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace ve::dcc
{

namespace
{

constexpr std::size_t kMinObservations = 100;

std::vector<double> column_means(std::span<const double> x, std::size_t t, std::size_t m)
{
    std::vector<double> mean(m, 0.0);
    for (std::size_t row = 0; row < t; ++row)
    {
        for (std::size_t i = 0; i < m; ++i)
        {
            mean[i] += x[row * m + i];
        }
    }
    for (double& v : mean)
    {
        v /= static_cast<double>(t);
    }
    return mean;
}

bool has_estimate(const FitResult* fit)
{
    if (fit == nullptr || fit->estimate.empty() || fit->sigma.empty() || fit->residuals.size() != fit->sigma.size())
    {
        return false;
    }
    const auto finite = [](double v) { return std::isfinite(v); };
    const auto positive = [](double v) { return std::isfinite(v) && v > 0.0; };
    return std::ranges::all_of(fit->estimate, finite) && std::ranges::all_of(fit->sigma, positive) &&
           std::ranges::all_of(fit->residuals, finite);
}

}

std::vector<double> second_moment(std::span<const double> x, std::size_t t, std::size_t m, bool centred, double divisor)
{
    const std::vector<double> mean = centred ? column_means(x, t, m) : std::vector<double>(m, 0.0);
    std::vector<double> out(m * m, 0.0);
    for (std::size_t row = 0; row < t; ++row)
    {
        for (std::size_t i = 0; i < m; ++i)
        {
            for (std::size_t j = 0; j <= i; ++j)
            {
                out[i * m + j] += (x[row * m + i] - mean[i]) * (x[row * m + j] - mean[j]);
            }
        }
    }
    for (std::size_t i = 0; i < m; ++i)
    {
        for (std::size_t j = 0; j <= i; ++j)
        {
            out[i * m + j] /= divisor;
            out[j * m + i] = out[i * m + j];
        }
    }
    return out;
}

std::optional<std::string> panel_problem(std::span<const FitResult* const> fits)
{
    if (fits.size() < 2)
    {
        return "a DCC needs at least 2 series, got " + std::to_string(fits.size());
    }
    for (std::size_t i = 0; i < fits.size(); ++i)
    {
        if (!has_estimate(fits[i]))
        {
            return "series " + std::to_string(i + 1) + " has no finite estimate";
        }
    }
    const std::size_t t = fits[0]->sigma.size();
    for (std::size_t i = 1; i < fits.size(); ++i)
    {
        if (fits[i]->sigma.size() != t)
        {
            return "series " + std::to_string(i + 1) + " has " + std::to_string(fits[i]->sigma.size()) +
                   " observations but series 1 has " + std::to_string(t);
        }
    }
    if (t < kMinObservations)
    {
        return "a DCC needs at least 100 observations, got " + std::to_string(t);
    }
    return std::nullopt;
}

StandardizedPanel standardize(std::span<const FitResult* const> fits)
{
    StandardizedPanel panel;
    panel.m = fits.size();
    panel.t = fits[0]->sigma.size();
    const std::size_t t = panel.t;
    const std::size_t m = panel.m;
    panel.z.resize(t * m);
    panel.sigma.resize(t * m);
    for (std::size_t i = 0; i < m; ++i)
    {
        for (std::size_t row = 0; row < t; ++row)
        {
            const double sigma = fits[i]->sigma[row];
            panel.sigma[row * m + i] = sigma;
            panel.z[row * m + i] = fits[i]->residuals[row] / sigma;
        }
    }
    refresh_moments(panel);
    return panel;
}

void refresh_moments(StandardizedPanel& panel)
{
    const std::size_t t = panel.t;
    const std::size_t m = panel.m;
    std::vector<double> n(t * m);
    std::ranges::transform(panel.z, n.begin(), [](double z) { return z < 0.0 ? z : 0.0; });
    const auto count = static_cast<double>(t);
    panel.qbar = second_moment(panel.z, t, m, true, count - 1.0);
    panel.qbar_raw = second_moment(panel.z, t, m, false, count);
    panel.nbar = second_moment(n, t, m, false, count);
    panel.nbar_centered = second_moment(n, t, m, true, count - 1.0);
    panel.delta = linalg::largest_relative_eigenvalue(panel.qbar, panel.nbar, m);
    panel.delta_centered = linalg::largest_relative_eigenvalue(panel.qbar, panel.nbar_centered, m);
}

}
