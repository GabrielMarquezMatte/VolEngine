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

// log(numerator / denominator). It maps start values, and a start on the edge of its domain (a weight of 0, a
// persistence at its ceiling) has no logit: it takes +-kEdgeLogit, inside, where the optimiser can still move it.
// Flooring at DBL_MIN instead gave +-708, where the logistic is flat: from alpha1 = 0 an sGARCH fit stayed there and
// reported convergence 130 below the optimum in loglik. On the rmgarch dcc3 panel a DCC b started at a logit of 5
// below a stayed at b = 0.0004, and from 4 it reached the optimum; 3 leaves a margin. Any other start maps exactly,
// so a fit started at given values evaluates them.
inline double log_ratio(double numerator, double denominator)
{
    constexpr double kEdgeLogit = 3.0;
    if (!(numerator > 0.0))
    {
        return -kEdgeLogit;
    }
    if (!(denominator > 0.0))
    {
        return kEdgeLogit;
    }
    return std::log(numerator) - std::log(denominator);
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
