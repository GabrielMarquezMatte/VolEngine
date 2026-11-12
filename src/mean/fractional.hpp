#pragma once

#include <cstddef>
#include <span>
#include <vector>

namespace ve::mean
{

// rugarch's ARFIMA: the truncated (1 - L)^d of the mean residuals, with no presample. In place, residuals[t]
// becomes sum_{k=0}^{t} pi_k residuals[t - k], with pi_0 = 1 and pi_k = pi_{k-1} (k - 1 - d) / k. Walks t down, as
// the sum at t reads residuals before t only.
// ponytail: O(T^2) direct sum; FFT convolution if long series ever need it.
template <class T>
void fractional_difference(std::span<T> residuals, const T& d)
{
    std::size_t count = residuals.size();
    std::vector<T> pi(count, T{1.0});
    for (std::size_t k = 1; k < count; ++k)
    {
        pi[k] = pi[k - 1] * ((static_cast<double>(k) - 1.0) - d) / static_cast<double>(k);
    }
    for (std::size_t t = count; t-- > 0;)
    {
        T sum = residuals[t];
        for (std::size_t k = 1; k <= t; ++k)
        {
            sum = sum + pi[k] * residuals[t - k];
        }
        residuals[t] = sum;
    }
}

}
