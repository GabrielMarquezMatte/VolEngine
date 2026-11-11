#pragma once

#include <algorithm>
#include <cstddef>
#include <span>

namespace ve::garch
{

template <class T>
struct SGarchParameters
{
    T omega;
    std::span<const T> alpha;
    std::span<const T> beta;
};

// sigma2_t = value for t < held; from t = held on the recursion runs, and a lag before t = 0 reads value for
// both the squared residual and the variance.
template <class T>
struct Presample
{
    T value;
    std::size_t held;
};

template <class T>
const T& lagged(std::span<const T> values, std::size_t t, std::size_t lag, const T& presample)
{
    if (lag > t)
    {
        return presample;
    }
    return values[t - lag];
}

template <class T>
T next_variance(std::span<const T> squared_residuals, const SGarchParameters<T>& p,
                const Presample<T>& presample, std::span<const T> sigma2, std::size_t t)
{
    T variance = p.omega;
    for (std::size_t i = 1; i <= p.alpha.size(); ++i)
    {
        variance = variance + p.alpha[i - 1] * lagged(squared_residuals, t, i, presample.value);
    }
    for (std::size_t j = 1; j <= p.beta.size(); ++j)
    {
        variance = variance + p.beta[j - 1] * lagged(sigma2, t, j, presample.value);
    }
    return variance;
}

// sigma2_t = omega + sum_i alpha_i e^2_{t-i} + sum_j beta_j sigma2_{t-j}; sigma2 has the size of
// squared_residuals.
template <class T>
void sgarch_variance(std::span<const T> squared_residuals, const SGarchParameters<T>& p,
                     const Presample<T>& presample, std::span<T> sigma2)
{
    std::size_t held = std::min(presample.held, sigma2.size());
    std::fill(sigma2.begin(), sigma2.begin() + static_cast<std::ptrdiff_t>(held), presample.value);
    for (std::size_t t = held; t < sigma2.size(); ++t)
    {
        sigma2[t] = next_variance<T>(squared_residuals, p, presample, sigma2, t);
    }
}

}
