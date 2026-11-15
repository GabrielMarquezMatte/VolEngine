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
    std::span<const T> intercept_shift{};
};

// A lag before t = 0 has z = 0 and, as in arch, which drops its term in |z| - sqrt(2/pi), |z| = sqrt(2/pi) for
// every distribution.
template <class T>
VE_FORCE_INLINE T egarch_step(std::span<const T> residuals, const EGarchParameters<T>& p,
                              const T& log_presample, std::span<const T> log_sigma2, std::size_t t)
{
    using std::abs;
    using std::exp;
    const double presample_abs = std::sqrt(2.0 / std::numbers::pi);
    T log_variance = intercept_at<T>(p.omega, p.intercept_shift, t);
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
        sigma2[t] = egarch_step<T>(residuals, p, log_presample, sigma2, t);
    }
    std::ranges::transform(sigma2, sigma2.begin(), [](const T& log_variance) { return exp(log_variance); });
}

// The adjoints of the reverse pass of egarch_variance. log_sigma2 holds d/dh_t of h_t = log sigma2_t, and presample
// d/dlog b of the presample b; abs_moment gets d/dE|z|. The others as SGarchAdjoint.
struct EGarchAdjoint
{
    std::span<double> residuals;
    std::span<double> log_sigma2;
    double omega;
    std::span<double> alpha;
    std::span<double> beta;
    std::span<double> gamma;
    double abs_moment;
    double presample;
};

// The forward states the reverse of egarch_variance reads: h_t = log sigma2_t and 1 / sigma_t = exp(-h_t / 2).
struct EGarchPath
{
    std::span<const double> log_sigma2;
    std::span<const double> inverse_sigma;
};

namespace detail
{

// The reverse of the shock term of lag i at step t, with g = d/dh_t. Before t = 0 it is
// gamma_i (sqrt(2/pi) - E|z|). Otherwise, with z = e_{t-i} exp(-h_{t-i} / 2), it is alpha_i z + gamma_i (|z| - E|z|):
// alpha_i gets g z, gamma_i gets g (|z| - E|z|), E|z| gets -g gamma_i, and d/dz = g (alpha_i + gamma_i sign z) (sign 0
// at 0) passes exp(-h_{t-i} / 2) of itself to e_{t-i} and -z / 2 to h_{t-i}.
inline void egarch_shock_backward(std::span<const double> residuals, const EGarchParameters<double>& p,
                                  const EGarchPath& path, LagStep step, EGarchAdjoint& adjoint)
{
    std::size_t i = step.lag;
    double gamma = p.gamma[i - 1];
    adjoint.abs_moment -= step.g * gamma;
    if (i > step.t)
    {
        adjoint.gamma[i - 1] += step.g * (std::sqrt(2.0 / std::numbers::pi) - p.abs_moment);
        return;
    }
    std::size_t lag = step.t - i;
    double scale = path.inverse_sigma[lag];
    double z = residuals[lag] * scale;
    double sign = (z > 0.0) - (z < 0.0);
    adjoint.alpha[i - 1] += step.g * z;
    adjoint.gamma[i - 1] += step.g * (std::fabs(z) - p.abs_moment);
    double z_adjoint = step.g * (p.alpha[i - 1] + gamma * sign);
    adjoint.residuals[lag] += z_adjoint * scale;
    adjoint.log_sigma2[lag] -= 0.5 * z_adjoint * z;
}

}

// The reverse of egarch_variance over h_t = log sigma2_t (path.log_sigma2), t from the end down to held, with
// g = d/dh_t: omega gets g, the shocks go through detail::egarch_shock_backward, and beta_j gets g h_{t-j} while
// d/dh_{t-j} gets g beta_j (log b, and the presample, before t = 0). The steps held at log b pass their adjoint to it.
// No variance regressors (an empty intercept shift).
inline void egarch_variance_backward(std::span<const double> residuals, const EGarchParameters<double>& p,
                                     const Presample<double>& presample, const EGarchPath& path,
                                     EGarchAdjoint& adjoint)
{
    std::span<const double> log_sigma2 = path.log_sigma2;
    std::size_t held = std::min(presample.held, log_sigma2.size());
    double log_presample = std::log(presample.value);
    for (std::size_t t = log_sigma2.size(); t-- > held;)
    {
        double g = adjoint.log_sigma2[t];
        adjoint.omega += g;
        for (std::size_t i = 1; i <= p.alpha.size(); ++i)
        {
            detail::egarch_shock_backward(residuals, p, path, {t, i, g}, adjoint);
        }
        for (std::size_t j = 1; j <= p.beta.size(); ++j)
        {
            adjoint.beta[j - 1] += g * (j > t ? log_presample : log_sigma2[t - j]);
            if (j > t)
            {
                adjoint.presample += g * p.beta[j - 1];
                continue;
            }
            adjoint.log_sigma2[t - j] += g * p.beta[j - 1];
        }
    }
    for (std::size_t t = 0; t < held; ++t)
    {
        adjoint.presample += adjoint.log_sigma2[t];
    }
}

}
