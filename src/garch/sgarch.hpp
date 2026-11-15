#pragma once

#include "garch/presample.hpp"

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
    std::span<const T> intercept_shift{};
};

template <class T>
VE_FORCE_INLINE T sgarch_step(std::span<const T> residuals, const SGarchParameters<T>& p,
                              const Presample<T>& presample, std::span<const T> sigma2, std::size_t t)
{
    T variance = intercept_at<T>(p.omega, p.intercept_shift, t);
    for (std::size_t i = 1; i <= p.alpha.size(); ++i)
    {
        variance = variance + p.alpha[i - 1] * lagged_square(residuals, t, i, presample.value);
    }
    for (std::size_t j = 1; j <= p.beta.size(); ++j)
    {
        variance = variance + p.beta[j - 1] * lagged(sigma2, t, j, presample.value);
    }
    return variance;
}

// sigma2_t = omega + sum_i alpha_i e^2_{t-i} + sum_j beta_j sigma2_{t-j}; sigma2 has the size of residuals. Each
// e^2 is squared where it is read: a stored copy costs an allocation and a pass more than the squares.
template <class T>
void sgarch_variance(std::span<const T> residuals, const SGarchParameters<T>& p, const Presample<T>& presample,
                     std::span<T> sigma2)
{
    std::size_t held = std::min(presample.held, sigma2.size());
    std::fill(sigma2.begin(), sigma2.begin() + static_cast<std::ptrdiff_t>(held), presample.value);
    for (std::size_t t = held; t < sigma2.size(); ++t)
    {
        sigma2[t] = sgarch_step<T>(residuals, p, presample, sigma2, t);
    }
}

// The adjoints of the reverse pass of sgarch_variance. residuals and sigma2 hold d/de_t and d/dsigma2_t; the
// parameter partials and the presample's are accumulated (+=).
struct SGarchAdjoint
{
    std::span<double> residuals;
    std::span<double> sigma2;
    double omega;
    std::span<double> alpha;
    std::span<double> beta;
    double presample;
};

// The reverse of sgarch_variance, t from the end down to held, with g = d/dsigma2_t: omega gets g; alpha_i gets
// g e^2_{t-i} and d/de_{t-i} gets 2 g alpha_i e_{t-i}; beta_j gets g sigma2_{t-j} and d/dsigma2_{t-j} gets
// g beta_j. A lag before t = 0 reads the presample b instead, which gets g alpha_i or g beta_j; the steps held at b
// pass their whole adjoint to it. No variance regressors (an empty intercept shift).
inline void sgarch_variance_backward(std::span<const double> residuals, const SGarchParameters<double>& p,
                                     const Presample<double>& presample, std::span<const double> sigma2,
                                     SGarchAdjoint& adjoint)
{
    std::size_t held = std::min(presample.held, sigma2.size());
    double b = presample.value;
    for (std::size_t t = sigma2.size(); t-- > held;)
    {
        double g = adjoint.sigma2[t];
        adjoint.omega += g;
        for (std::size_t i = 1; i <= p.alpha.size(); ++i)
        {
            double square = i > t ? b : residuals[t - i] * residuals[t - i];
            adjoint.alpha[i - 1] += g * square;
            if (i > t)
            {
                adjoint.presample += g * p.alpha[i - 1];
                continue;
            }
            adjoint.residuals[t - i] += 2.0 * g * p.alpha[i - 1] * residuals[t - i];
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
