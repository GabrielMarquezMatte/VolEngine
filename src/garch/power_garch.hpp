#pragma once

#include "garch/presample.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <span>

namespace ve::garch
{

// An empty eta1 or eta2 means eta1 = 0 or eta2 = 0. moments[i] = E(|z - eta2_i| - eta1_i (z - eta2_i))^power, the
// expected shock factor of lag i + 1 before t = 0. With abs_smoothing = c > 0, |u| for u = z - eta2 is taken as
// sqrt(u^2 + c), as rugarch's fGARCH does (c = 1e-6); with c = 0 it is exact.
template <class T>
struct PowerGarchParameters
{
    T omega;
    std::span<const T> alpha;
    std::span<const T> beta;
    std::span<const T> eta1;
    std::span<const T> eta2;
    T power;
    std::span<const T> moments;
    double abs_smoothing = 0.0;
};

// sigma |u| for u = z - eta2 and shift = sigma u: |shift| or, smoothed, sqrt(shift^2 + c sigma^2).
template <class T>
T scaled_abs(const T& shift, const T& sigma_power, const PowerGarchParameters<T>& p)
{
    using std::abs;
    using std::pow;
    using std::sqrt;
    if (p.abs_smoothing == 0.0)
    {
        return abs(shift);
    }
    T variance = pow(sigma_power, 2.0 / p.power);
    return sqrt(shift * shift + p.abs_smoothing * variance);
}

// sigma^power (|z - eta2| - eta1 (z - eta2))^power with z = e / sigma, taken as
// (|e - eta2 sigma| - eta1 (e - eta2 sigma))^power.
template <class T>
T power_shock(const T& residual, const T& sigma_power, const PowerGarchParameters<T>& p, std::size_t i)
{
    using std::pow;
    T shift = residual;
    if (!p.eta2.empty())
    {
        shift = residual - p.eta2[i] * pow(sigma_power, 1.0 / p.power);
    }
    T asymmetry = scaled_abs(shift, sigma_power, p);
    if (!p.eta1.empty())
    {
        asymmetry = asymmetry - p.eta1[i] * shift;
    }
    return pow(asymmetry, p.power);
}

template <class T>
T next_power_variance(std::span<const T> residuals, const PowerGarchParameters<T>& p, const T& presample_power,
                      std::span<const T> sigma_power, std::size_t t)
{
    T variance = p.omega;
    for (std::size_t i = 1; i <= p.alpha.size(); ++i)
    {
        if (i > t)
        {
            variance = variance + p.alpha[i - 1] * p.moments[i - 1] * presample_power;
            continue;
        }
        variance = variance + p.alpha[i - 1] * power_shock(residuals[t - i], sigma_power[t - i], p, i - 1);
    }
    for (std::size_t j = 1; j <= p.beta.size(); ++j)
    {
        variance = variance + p.beta[j - 1] * lagged(sigma_power, t, j, presample_power);
    }
    return variance;
}

// sigma_t^power = omega + sum_i alpha_i sigma_{t-i}^power (|z_{t-i} - eta2_i| - eta1_i (z_{t-i} - eta2_i))^power
// + sum_j beta_j sigma_{t-j}^power. sigma2 has the size of residuals and holds sigma^power while the recursion
// runs.
template <class T>
void power_garch_variance(std::span<const T> residuals, const PowerGarchParameters<T>& p,
                          const Presample<T>& presample, std::span<T> sigma2)
{
    using std::pow;
    T presample_power = pow(presample.value, 0.5 * p.power);
    std::size_t held = std::min(presample.held, sigma2.size());
    std::fill(sigma2.begin(), sigma2.begin() + static_cast<std::ptrdiff_t>(held), presample_power);
    for (std::size_t t = held; t < sigma2.size(); ++t)
    {
        sigma2[t] = next_power_variance<T>(residuals, p, presample_power, sigma2, t);
    }
    T exponent = 2.0 / p.power;
    std::ranges::transform(sigma2, sigma2.begin(), [&exponent](const T& power) { return pow(power, exponent); });
}

}
