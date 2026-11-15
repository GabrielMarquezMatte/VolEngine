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
VE_FORCE_INLINE T gjrgarch_step(std::span<const T> residuals, const GjrGarchParameters<T>& p,
                                const Presample<T>& presample, std::span<const T> sigma2, std::size_t t)
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
        sigma2[t] = gjrgarch_step<T>(residuals, p, presample, sigma2, t);
    }
}

// The adjoints of the reverse pass of gjrgarch_variance, as SGarchAdjoint with gamma.
struct GjrGarchAdjoint
{
    std::span<double> residuals;
    std::span<double> sigma2;
    double omega;
    std::span<double> alpha;
    std::span<double> beta;
    std::span<double> gamma;
    double presample;
};

namespace detail
{

// The reverse of gjr_shock with g = d/dsigma2_t. Before t = 0 the shock is (alpha_i + gamma_i / 2) b; for
// e = e_{t-i} < 0 it is (alpha_i + gamma_i) e^2, otherwise alpha_i e^2 (the indicator has slope 0).
inline void gjr_shock_backward(std::span<const double> residuals, const GjrGarchParameters<double>& p, double b,
                               std::size_t t, std::size_t i, double g, GjrGarchAdjoint& adjoint)
{
    if (i > t)
    {
        adjoint.alpha[i - 1] += g * b;
        adjoint.gamma[i - 1] += 0.5 * g * b;
        adjoint.presample += g * (p.alpha[i - 1] + 0.5 * p.gamma[i - 1]);
        return;
    }
    double e = residuals[t - i];
    double weight = p.alpha[i - 1];
    adjoint.alpha[i - 1] += g * e * e;
    if (e < 0.0)
    {
        weight += p.gamma[i - 1];
        adjoint.gamma[i - 1] += g * e * e;
    }
    adjoint.residuals[t - i] += 2.0 * g * weight * e;
}

}

// The reverse of gjrgarch_variance, t from the end down to held: the shocks by detail::gjr_shock_backward, omega and
// beta as in sgarch_variance_backward. No variance regressors (an empty intercept shift).
inline void gjrgarch_variance_backward(std::span<const double> residuals, const GjrGarchParameters<double>& p,
                                       const Presample<double>& presample, std::span<const double> sigma2,
                                       GjrGarchAdjoint& adjoint)
{
    std::size_t held = std::min(presample.held, sigma2.size());
    double b = presample.value;
    for (std::size_t t = sigma2.size(); t-- > held;)
    {
        double g = adjoint.sigma2[t];
        adjoint.omega += g;
        for (std::size_t i = 1; i <= p.alpha.size(); ++i)
        {
            detail::gjr_shock_backward(residuals, p, b, t, i, g, adjoint);
        }
        for (std::size_t j = 1; j <= p.beta.size(); ++j)
        {
            adjoint.beta[j - 1] += g * (j > t ? b : sigma2[t - j]);
            if (j > t)
            {
                adjoint.presample += g * p.beta[j - 1];
                continue;
            }
            adjoint.sigma2[t - j] += g * p.beta[j - 1];
        }
    }
    for (std::size_t t = 0; t < held; ++t)
    {
        adjoint.presample += adjoint.sigma2[t];
    }
}

}
