#pragma once

#include <algorithm>
#include <cstddef>
#include <span>
#include <vector>

namespace ve::mean
{

namespace detail
{

// out[t] = x[t] * w[0] + sum_{k=1}^{t} w[k] x[t - k], each t summed with k ascending, one product and one sum per
// term (the order of the direct sum, so the result is the same to the bit). Eight t at a time: eight independent
// sums the compiler vectorises across t, where one t at a time is a single chain bound by the latency of the add.
// ponytail: O(T^2); an FFT convolution if long series ever need it, at the price of the bit-for-bit order.
inline void causal_convolution(std::span<const double> w, std::span<const double> x, std::span<double> out)
{
    constexpr std::size_t kBlock = 8;
    const std::size_t count = x.size();
    std::size_t t = 0;
    for (; t + kBlock <= count; t += kBlock)
    {
        double sum[kBlock];
        for (std::size_t i = 0; i < kBlock; ++i)
        {
            sum[i] = w[0] * x[t + i];
        }
        for (std::size_t k = 1; k <= t; ++k)
        {
            const double weight = w[k];
            const double* lagged = x.data() + (t - k);
            for (std::size_t i = 0; i < kBlock; ++i)
            {
                sum[i] = sum[i] + weight * lagged[i];
            }
        }
        // The terms k = t+1..t+i of the later t of the block, still in ascending k.
        for (std::size_t i = 1; i < kBlock; ++i)
        {
            for (std::size_t k = t + 1; k <= t + i; ++k)
            {
                sum[i] = sum[i] + w[k] * x[t + i - k];
            }
        }
        std::copy(sum, sum + kBlock, out.begin() + static_cast<std::ptrdiff_t>(t));
    }
    for (; t < count; ++t)
    {
        double sum = w[0] * x[t];
        for (std::size_t k = 1; k <= t; ++k)
        {
            sum = sum + w[k] * x[t - k];
        }
        out[t] = sum;
    }
}

// pi_0 = 1, pi_k = pi_{k-1} (k - 1 - d) / k.
inline std::vector<double> weights(std::size_t count, double d)
{
    std::vector<double> pi(count, 1.0);
    for (std::size_t k = 1; k < count; ++k)
    {
        pi[k] = pi[k - 1] * ((static_cast<double>(k) - 1.0) - d) / static_cast<double>(k);
    }
    return pi;
}

}

// rugarch's ARFIMA: the truncated (1 - L)^d of the mean residuals, with no presample. In place, residuals[t]
// becomes sum_{k=0}^{t} pi_k residuals[t - k], with pi_0 = 1 and pi_k = pi_{k-1} (k - 1 - d) / k. T is double (the
// specialisation below) or ad::Dual<N> (this template). With derivatives, pi_k depends on d alone, so the d part is
// d' times one convolution with dpi/dd, and the residual part one convolution with pi per lane, skipped for a lane
// no residual moves.
template <class T>
void fractional_difference(std::span<T> residuals, const T& d)
{
    const std::size_t count = residuals.size();
    const std::vector<double> pi = detail::weights(count, d.value);
    // dpi_k/dd = (dpi_{k-1} (k - 1 - d) - pi_{k-1}) / k.
    std::vector<double> dpi(count, 0.0);
    for (std::size_t k = 1; k < count; ++k)
    {
        dpi[k] = (dpi[k - 1] * ((static_cast<double>(k) - 1.0) - d.value) - pi[k - 1]) / static_cast<double>(k);
    }
    std::vector<double> lane(count);
    for (std::size_t t = 0; t < count; ++t)
    {
        lane[t] = residuals[t].value;
    }
    std::vector<double> convolved(count);
    std::vector<double> d_term(count);
    detail::causal_convolution(dpi, lane, d_term);
    detail::causal_convolution(pi, lane, convolved);
    for (std::size_t t = 0; t < count; ++t)
    {
        residuals[t].value = convolved[t];
    }
    for (std::size_t j = 0; j < d.grad.size(); ++j)
    {
        bool moves = false;
        for (std::size_t t = 0; t < count; ++t)
        {
            lane[t] = residuals[t].grad[j];
            moves = moves || lane[t] != 0.0;
        }
        std::fill(convolved.begin(), convolved.end(), 0.0);
        if (moves)
        {
            detail::causal_convolution(pi, lane, convolved);
        }
        const double slope = d.grad[j];
        for (std::size_t t = 0; t < count; ++t)
        {
            residuals[t].grad[j] = convolved[t] + slope * d_term[t];
        }
    }
}

template <>
inline void fractional_difference<double>(std::span<double> residuals, const double& d)
{
    const std::vector<double> pi = detail::weights(residuals.size(), d);
    const std::vector<double> e(residuals.begin(), residuals.end());
    detail::causal_convolution(pi, e, residuals);
}

}
