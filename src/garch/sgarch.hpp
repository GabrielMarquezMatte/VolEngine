#pragma once

#include "core/var_init.hpp"

#include <cstddef>
#include <span>

namespace ve::garch
{

struct SGarchConfig
{
    bool include_mean = true;
    VarInit var_init = VarInit::Sample;
};

// Each span is either empty (not computed) or exactly sized: sigma2 T, grad k, scores T x k row-major.
struct SGarchOutputs
{
    std::span<double> sigma2{};
    std::span<double> grad{};
    std::span<double> scores{};
};

std::size_t sgarch_param_count(const SGarchConfig& config);

// theta is in natural space, ordered as rugarch names it: mu (only with include_mean), omega, alpha1, beta1.
// Returns a non-finite log-likelihood when some sigma2_t is not positive and finite.
double sgarch_norm_loglik(std::span<const double> r, std::span<const double> theta, const SGarchConfig& config,
                          const SGarchOutputs& out);

}
