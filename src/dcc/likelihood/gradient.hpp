#pragma once

#include "dcc/panel.hpp"
#include "dcc/spec.hpp"
#include "simd/cholesky_kernels.hpp"

#include <cstddef>
#include <optional>
#include <span>

namespace ve::dcc
{

// True for the specs ObjectiveGradient computes: DCC, aDCC and FDCC with the mvnorm, mvt and mvlaplace, listed
// explicitly; anything else keeps the Dual objective. The copula flag does not matter: the score panel of a copula is
// an ordinary DCC panel, and its fit is the DCC objective on it.
bool has_fast_gradient(const DccSpec& spec);

// The objective of dcc::objective<double> and its gradient over the natural parameters theta, without ad::Dual.
//
// The recursion of Q_t is linear in the earlier Q and in the earlier products of the scores, so its derivative
// with respect to each parameter, dQ_t, follows the same recursion with the derivative of the weights added: one
// packed matrix per parameter, all from the elementwise kernels of the value. The density of an observation is
// pulled back once to Q_t (through log |R_t|, z_t' R_t^-1 z_t and the normalization R = D Q D), and the derivative
// of the objective in each parameter is the sum over t of the dot product of that gradient and dQ_t.
class ObjectiveGradient
{
public:
    // theta has the layout of DccParameters (a, b, g, then the mvt shape); grad has its size. The value is the one of
    // objective<double>, to the bit; outside its domain, or wherever it is not finite, it is +infinity and grad is 0.
    double evaluate(const StandardizedPanel& panel, const DccSpec& spec, std::span<const double> theta,
                    std::span<double> grad);

private:
    std::optional<simd::CorrelationBlock> block_;
    std::size_t block_m_ = 0;
};

}
