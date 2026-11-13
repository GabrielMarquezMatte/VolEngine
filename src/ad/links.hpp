#pragma once

#include "ad/dual.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <span>

namespace ve::ad
{

// The maps between bounded natural parameters and the unconstrained space the optimisers search, for T = double
// or ad::Dual.

// 1 / (1 + exp(-x)), without overflow on either side.
template <class T>
T logistic(const T& x)
{
    using std::exp;
    if (value_of(x) >= 0.0)
    {
        return 1.0 / (1.0 + exp(-x));
    }
    T exponential = exp(x);
    return exponential / (1.0 + exponential);
}

// log(numerator / denominator), each floored at DBL_MIN so that a zero weight gives a finite value.
inline double log_ratio(double numerator, double denominator)
{
    constexpr double kFloor = std::numeric_limits<double>::min();
    return std::log(std::max(numerator, kFloor)) - std::log(std::max(denominator, kFloor));
}

// weights[i] = scale softmax(logits, 0)[i]; logits has weights.size() - 1 entries, the last weight the reference.
template <class T>
void scaled_softmax(std::span<const T> logits, const T& scale, std::span<T> weights)
{
    using std::exp;
    std::size_t count = weights.size();
    double peak = 0.0;
    for (const T& logit : logits)
    {
        peak = std::max(peak, value_of(logit));
    }
    T total{};
    for (std::size_t i = 0; i < count; ++i)
    {
        T logit = i + 1 < count ? logits[i] : T{};
        weights[i] = exp(logit - peak);
        total = total + weights[i];
    }
    for (T& weight : weights)
    {
        weight = scale * (weight / total);
    }
}

}
