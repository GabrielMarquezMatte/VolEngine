#pragma once

#include "ad/dual.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
#include <span>

namespace ve::model
{

constexpr double kInvalidLoglik = -std::numeric_limits<double>::infinity();

template <class T>
bool is_valid_variance(const T& sigma2)
{
    double value = ad::value_of(sigma2);
    return std::isfinite(value) && value > 0.0;
}

// The sum over t of log_density_of(e_t, sigma2_t) - log(sigma2_t) / 2, where log_density_of(e, sigma2) is
// log f(e / sqrt(sigma2)); each term goes to terms unless it is empty. -inf when some sigma2_t is not
// positive and finite, or when the sum is not finite: a NaN or +inf term, or finite terms whose sum
// overflows.
template <class T, class LogDensityOf>
T sum_of_terms(std::span<const T> residuals, std::span<const T> sigma2, LogDensityOf log_density_of,
               std::span<T> terms)
{
    using std::log;
    T total{0.0};
    for (std::size_t t = 0; t < residuals.size(); ++t)
    {
        if (!is_valid_variance(sigma2[t]))
        {
            total = T{kInvalidLoglik};
            return total;
        }
        T term = log_density_of(residuals[t], sigma2[t]) - 0.5 * log(sigma2[t]);
        if (!terms.empty())
        {
            terms[t] = term;
        }
        total = total + term;
    }
    if (!std::isfinite(ad::value_of(total)))
    {
        total = T{kInvalidLoglik};
    }
    return total;
}

}
