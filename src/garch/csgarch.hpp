#pragma once

#include "garch/presample.hpp"

#include <algorithm>
#include <cstddef>
#include <span>

namespace ve::garch
{

template <class T>
struct CsGarchParameters
{
    T omega;
    std::span<const T> alpha;
    std::span<const T> beta;
    T rho;
    T phi;
};

// sum_i weights_i (values_{t-i} - permanent_{t-i}). Before t = 0 both lags read the presample variance, so a lag
// there adds nothing.
template <class T>
T lagged_deviations(std::span<const T> weights, std::span<const T> values, std::span<const T> permanent,
                    std::size_t t)
{
    T sum{0.0};
    for (std::size_t i = 1; i <= std::min(weights.size(), t); ++i)
    {
        sum = sum + weights[i - 1] * (values[t - i] - permanent[t - i]);
    }
    return sum;
}

// q_t = omega + rho q_{t-1} + phi (e^2_{t-1} - sigma2_{t-1});
// sigma2_t = q_t + sum_i alpha_i (e^2_{t-i} - q_{t-i}) + sum_j beta_j (sigma2_{t-j} - q_{t-j}).
// sigma2 and permanent (q) have the size of squared_residuals.
template <class T>
void csgarch_variance(std::span<const T> squared_residuals, const CsGarchParameters<T>& p,
                      const Presample<T>& presample, std::span<T> sigma2, std::span<T> permanent)
{
    std::size_t held = std::min(presample.held, sigma2.size());
    std::fill(sigma2.begin(), sigma2.begin() + static_cast<std::ptrdiff_t>(held), presample.value);
    std::fill(permanent.begin(), permanent.begin() + static_cast<std::ptrdiff_t>(held), presample.value);
    std::span<const T> phi(&p.phi, 1);
    for (std::size_t t = held; t < sigma2.size(); ++t)
    {
        permanent[t] = p.omega + p.rho * lagged<T>(permanent, t, 1, presample.value) +
                       lagged_deviations<T>(phi, squared_residuals, sigma2, t);
        sigma2[t] = permanent[t] + lagged_deviations<T>(p.alpha, squared_residuals, permanent, t) +
                    lagged_deviations<T>(p.beta, sigma2, permanent, t);
    }
}

}
