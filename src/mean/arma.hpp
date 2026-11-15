#pragma once

#include <algorithm>
#include <cstddef>
#include <span>

namespace ve::mean
{

template <class T>
struct ArmaParameters
{
    T mu;
    std::span<const T> ar;
    std::span<const T> ma;
};

// e_t of the recursion: deviation(t) is d_t = r_t - m_t. rugarch conditions on the first m = ar.size()
// observations: e_t = d_t for t < m. From t = m on, e_t = d_t - sum_i ar_i d_{t-i} - sum_j ma_j e_{t-j}, and an MA
// lag before t = 0 adds nothing. Reads residuals before t only.
template <class T, class Deviation>
T arma_step_of(Deviation deviation, const ArmaParameters<T>& p, std::span<const T> residuals, std::size_t t)
{
    T residual = deviation(t);
    if (t < p.ar.size())
    {
        return residual;
    }
    for (std::size_t i = 1; i <= p.ar.size(); ++i)
    {
        residual = residual - p.ar[i - 1] * deviation(t - i);
    }
    std::size_t ma_lags = std::min(p.ma.size(), t);
    for (std::size_t j = 1; j <= ma_lags; ++j)
    {
        residual = residual - p.ma[j - 1] * residuals[t - j];
    }
    return residual;
}

// The step over d, which already holds r_t - m_t up to t (the mean, regressors and ARCH-in-mean included); p.mu is
// not read.
template <class T>
T arma_step(std::span<const T> d, std::span<const T> residuals, const ArmaParameters<T>& p, std::size_t t)
{
    return arma_step_of<T>([&](std::size_t lag) -> T { return d[lag]; }, p, residuals, t);
}

// The step at every t. residuals has the size of the series.
template <class T, class Deviation>
void arma_recursion(Deviation deviation, const ArmaParameters<T>& p, std::span<T> residuals)
{
    for (std::size_t t = 0; t < residuals.size(); ++t)
    {
        residuals[t] = arma_step_of<T>(deviation, p, residuals, t);
    }
}

// The recursion with d_t = r_t - mu. residuals has the size of r.
template <class T>
void arma_residuals(std::span<const double> r, const ArmaParameters<T>& p, std::span<T> residuals)
{
    arma_recursion<T>([&](std::size_t t) -> T { return r[t] - p.mu; }, p, residuals);
}

// d/dmu, d/dar_i and d/dma_j, accumulated by arma_residuals_backward.
struct ArmaGradient
{
    double mu;
    std::span<double> ar;
    std::span<double> ma;
};

// The reverse of arma_residuals over d_t = r_t - mu. residual_adjoint holds d/de_t on entry and is consumed by the
// sweep, which runs t from the end: with g = d/de_t, a step t >= m (m the AR order) adds g (sum_i ar_i - 1) to mu,
// -g d_{t-i} to ar_i, -g e_{t-j} to ma_j and -g ma_j to d/de_{t-j}; a step t < m, where e_t = d_t, adds -g to mu.
inline void arma_residuals_backward(std::span<const double> r, const ArmaParameters<double>& p,
                                    std::span<const double> residuals, std::span<double> residual_adjoint,
                                    ArmaGradient& gradient)
{
    double ar_sum = 0.0;
    for (double ar : p.ar)
    {
        ar_sum += ar;
    }
    std::size_t m = p.ar.size();
    for (std::size_t t = residuals.size(); t-- > m;)
    {
        double g = residual_adjoint[t];
        gradient.mu += g * (ar_sum - 1.0);
        for (std::size_t i = 1; i <= m; ++i)
        {
            gradient.ar[i - 1] -= g * (r[t - i] - p.mu);
        }
        std::size_t ma_lags = std::min(p.ma.size(), t);
        for (std::size_t j = 1; j <= ma_lags; ++j)
        {
            gradient.ma[j - 1] -= g * residuals[t - j];
            residual_adjoint[t - j] -= g * p.ma[j - 1];
        }
    }
    for (std::size_t t = 0; t < std::min(m, residuals.size()); ++t)
    {
        gradient.mu -= residual_adjoint[t];
    }
}

// The recursion over d, which already holds r_t - m_t (the mean, regressors included); p.mu is not read. d has
// the size of residuals.
template <class T>
void arma_residuals_of_deviations(std::span<const T> d, const ArmaParameters<T>& p, std::span<T> residuals)
{
    arma_recursion<T>([&](std::size_t t) -> T { return d[t]; }, p, residuals);
}

}
