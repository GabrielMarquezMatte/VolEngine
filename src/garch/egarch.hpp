#pragma once

#include "garch/presample.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numbers>
#include <span>

namespace ve::garch
{

template <class T>
struct EGarchParameters
{
    T omega;
    std::span<const T> alpha;
    std::span<const T> beta;
    std::span<const T> gamma;
    T abs_moment;
};

// A lag before t = 0 has z = 0 and, as in arch, which drops its term in |z| - sqrt(2/pi), |z| = sqrt(2/pi) for
// every distribution.
template <class T>
T next_log_variance(std::span<const T> residuals, const EGarchParameters<T>& p, const T& log_presample,
                    std::span<const T> log_sigma2, std::size_t t)
{
    using std::abs;
    using std::exp;
    const double presample_abs = std::sqrt(2.0 / std::numbers::pi);
    T log_variance = p.omega;
    for (std::size_t i = 1; i <= p.alpha.size(); ++i)
    {
        if (i > t)
        {
            log_variance = log_variance + p.gamma[i - 1] * (presample_abs - p.abs_moment);
            continue;
        }
        T z = residuals[t - i] * exp(-0.5 * log_sigma2[t - i]);
        log_variance = log_variance + p.alpha[i - 1] * z + p.gamma[i - 1] * (abs(z) - p.abs_moment);
    }
    for (std::size_t j = 1; j <= p.beta.size(); ++j)
    {
        log_variance = log_variance + p.beta[j - 1] * lagged(log_sigma2, t, j, log_presample);
    }
    return log_variance;
}

// log sigma2_t = omega + sum_i [alpha_i z_{t-i} + gamma_i (|z_{t-i}| - E|z|)] + sum_j beta_j log sigma2_{t-j},
// with z_t = e_t / sigma_t. sigma2 has the size of residuals and holds log sigma2 while the recursion runs.
template <class T>
void egarch_variance(std::span<const T> residuals, const EGarchParameters<T>& p, const Presample<T>& presample,
                     std::span<T> sigma2)
{
    using std::exp;
    using std::log;
    T log_presample = log(presample.value);
    std::size_t held = std::min(presample.held, sigma2.size());
    std::fill(sigma2.begin(), sigma2.begin() + static_cast<std::ptrdiff_t>(held), log_presample);
    for (std::size_t t = held; t < sigma2.size(); ++t)
    {
        sigma2[t] = next_log_variance<T>(residuals, p, log_presample, sigma2, t);
    }
    std::ranges::transform(sigma2, sigma2.begin(), [](const T& log_variance) { return exp(log_variance); });
}

}
