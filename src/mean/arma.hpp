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

// e_t = (r_t - mu) - sum_i ar_i (r_{t-i} - mu) - sum_j ma_j e_{t-j}; a lag before the first observation adds nothing.
// residuals has the size of r.
template <class T>
void arma_residuals(std::span<const double> r, const ArmaParameters<T>& p, std::span<T> residuals)
{
    for (std::size_t t = 0; t < r.size(); ++t)
    {
        T residual = r[t] - p.mu;
        std::size_t ar_lags = std::min(p.ar.size(), t);
        for (std::size_t i = 1; i <= ar_lags; ++i)
        {
            residual = residual - p.ar[i - 1] * (r[t - i] - p.mu);
        }
        std::size_t ma_lags = std::min(p.ma.size(), t);
        for (std::size_t j = 1; j <= ma_lags; ++j)
        {
            residual = residual - p.ma[j - 1] * residuals[t - j];
        }
        residuals[t] = residual;
    }
}

}
