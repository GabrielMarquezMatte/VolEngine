#pragma once

#include "ad/dual.hpp"
#include "garch/presample.hpp"

#include <algorithm>
#include <cstddef>
#include <span>

namespace ve::garch
{

template <class T>
struct GjrGarchParameters
{
    T omega;
    std::span<const T> alpha;
    std::span<const T> beta;
    std::span<const T> gamma;
    std::span<const T> intercept_shift{};
};

// (alpha_i + gamma_i 1[e < 0]) e^2 of lag i; before t = 0, its expected value given the presample variance b,
// (alpha_i + gamma_i / 2) b.
template <class T>
T gjr_shock(std::span<const T> residuals, const GjrGarchParameters<T>& p, const T& presample, std::size_t t,
            std::size_t i)
{
    if (i > t)
    {
        return (p.alpha[i - 1] + 0.5 * p.gamma[i - 1]) * presample;
    }
    const T& e = residuals[t - i];
    if (ad::value_of(e) < 0.0)
    {
        return (p.alpha[i - 1] + p.gamma[i - 1]) * (e * e);
    }
    return p.alpha[i - 1] * (e * e);
}

template <class T>
T next_gjr_variance(std::span<const T> residuals, const GjrGarchParameters<T>& p, const Presample<T>& presample,
                    std::span<const T> sigma2, std::size_t t)
{
    T variance = intercept_at<T>(p.omega, p.intercept_shift, t);
    for (std::size_t i = 1; i <= p.alpha.size(); ++i)
    {
        variance = variance + gjr_shock(residuals, p, presample.value, t, i);
    }
    for (std::size_t j = 1; j <= p.beta.size(); ++j)
    {
        variance = variance + p.beta[j - 1] * lagged(sigma2, t, j, presample.value);
    }
    return variance;
}

// sigma2_t = omega + sum_i (alpha_i + gamma_i 1[e_{t-i} < 0]) e^2_{t-i} + sum_j beta_j sigma2_{t-j}; sigma2 has
// the size of residuals.
template <class T>
void gjrgarch_variance(std::span<const T> residuals, const GjrGarchParameters<T>& p, const Presample<T>& presample,
                       std::span<T> sigma2)
{
    std::size_t held = std::min(presample.held, sigma2.size());
    std::fill(sigma2.begin(), sigma2.begin() + static_cast<std::ptrdiff_t>(held), presample.value);
    for (std::size_t t = held; t < sigma2.size(); ++t)
    {
        sigma2[t] = next_gjr_variance<T>(residuals, p, presample, sigma2, t);
    }
}

}
