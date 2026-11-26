#include "forecast/cumulative.hpp"

#include <algorithm>
#include <cstddef>
#include <utility>
#include <vector>

namespace ve::forecast
{

namespace
{

// psi_0..psi_{horizon-1} of phi(L) x = theta(L) e.
std::vector<double> arma_weights(std::span<const double> ar, std::span<const double> ma, std::size_t horizon)
{
    std::vector<double> psi(horizon, 0.0);
    if (horizon > 0)
    {
        psi[0] = 1.0;
    }
    for (std::size_t j = 1; j < horizon; ++j)
    {
        double value = 0.0;
        if (j <= ma.size())
        {
            value = ma[j - 1];
        }
        for (std::size_t i = 1; i <= std::min(j, ar.size()); ++i)
        {
            value += ar[i - 1] * psi[j - i];
        }
        psi[j] = value;
    }
    return psi;
}

// psi convolved with g_0 = 1, g_j = g_{j-1} (j - 1 + d) / j, the coefficients of (1 - L)^-d, truncated at psi's size.
void integrate_fractionally(std::vector<double>& psi, double d)
{
    std::vector<double> g(psi.size(), 1.0);
    for (std::size_t j = 1; j < g.size(); ++j)
    {
        g[j] = g[j - 1] * (static_cast<double>(j) - 1.0 + d) / static_cast<double>(j);
    }
    std::vector<double> out(psi.size(), 0.0);
    for (std::size_t n = 0; n < out.size(); ++n)
    {
        for (std::size_t j = 0; j <= n; ++j)
        {
            out[n] += psi[j] * g[n - j];
        }
    }
    psi = std::move(out);
}

}

std::vector<double> cumulative_weights(const model::Layout& layout, std::span<const double> theta,
                                       std::size_t horizon)
{
    const auto ar = theta.subspan(layout.ar_begin(), static_cast<std::size_t>(layout.ar));
    const auto ma = theta.subspan(layout.ma_begin(), static_cast<std::size_t>(layout.ma));
    std::vector<double> psi = arma_weights(ar, ma, horizon);
    if (layout.has_arfima())
    {
        integrate_fractionally(psi, theta[layout.arfima_index()]);
    }
    for (std::size_t n = 1; n < psi.size(); ++n)
    {
        psi[n] += psi[n - 1];
    }
    return psi;
}

double cumulative_variance(std::span<const double> weights, std::span<const double> variances, std::size_t h)
{
    double sum = 0.0;
    for (std::size_t k = 1; k <= h; ++k)
    {
        sum += weights[h - k] * weights[h - k] * variances[k - 1];
    }
    return sum;
}

}
