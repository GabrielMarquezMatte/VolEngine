#pragma once

#include "dcc/panel.hpp"
#include "dcc/recursion.hpp"
#include "dcc/spec.hpp"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <span>

namespace ve::dcc
{

// theta in the order of DccParameters: P, Q and (aDCC) P weights, then the mvt shape; the FDCC has G = max(groups)
// weights a and G weights b. Without the mvt the shape is 0.
template <class T>
DccParameters<T> split_parameters(const DccSpec& spec, std::span<const T> theta)
{
    auto p = static_cast<std::size_t>(spec.p);
    auto q = static_cast<std::size_t>(spec.q);
    std::size_t asymmetric = 0;
    if (spec.model == DccModel::ADcc)
    {
        asymmetric = p;
    }
    if (spec.model == DccModel::FDcc)
    {
        p = static_cast<std::size_t>(std::ranges::max(spec.groups));
        q = p;
    }
    [[maybe_unused]] const auto shape_count = static_cast<std::size_t>(spec.dist == DccDist::MvT);
    assert(theta.size() == p + q + asymmetric + shape_count);
    DccParameters<T> parameters{theta.subspan(0, p), theta.subspan(p, q), theta.subspan(p + q, asymmetric), T{0.0}};
    if (spec.dist == DccDist::MvT)
    {
        parameters.shape = theta[p + q + asymmetric];
    }
    return parameters;
}

// terms: log f(z_t; R_t) of each observation, without - sum_i log sigma_it (size T); r: R_t, T x m x m, written
// unless empty.
template <class T>
struct CorrelationOutput
{
    std::span<T> terms;
    std::span<double> r = {};
};

// Fills out. spec.convention picks Nbar and the presample. Outside the domain of correlation_path every term and
// every R_t is NaN. Instantiated for T = double, ad::Dual<8>, 16, 32 and kMaxParameterCount.
template <class T>
void correlation_terms(const StandardizedPanel& panel, const DccSpec& spec, const DccParameters<T>& theta,
                       Stage stage, const CorrelationOutput<T>& out);

// sum_t (log f(z_t; R_t) - sum_i log sigma_it) at Stage::Reported, writing R_t as correlation_terms does.
double reported_loglik(const StandardizedPanel& panel, const DccSpec& spec, std::span<const double> theta,
                       std::span<double> r = {});

// What the optimiser minimises. Standard: - sum_t log f(z_t; R_t) with every constant. Rmgarch (Stage::Objective):
// the same, but 1/2 sum_t (log |R_t| + z_t' R_t^-1 z_t) for the mvnorm. +infinity outside the domain (see
// correlation_path) or wherever the sum is not finite. Instantiated as correlation_terms.
template <class T>
T objective(const StandardizedPanel& panel, const DccSpec& spec, std::span<const T> theta);

}
