#pragma once

#include "ad/dual.hpp"
#include "core/var_init.hpp"
#include "model/layout.hpp"

#include <span>

namespace ve::model
{

// Each span is either empty (not computed) or has the size of r.
template <class T>
struct LoglikOutputs
{
    std::span<T> terms{};
    std::span<T> residuals{};
    std::span<T> sigma2{};
};

// theta is in natural space, in Layout order. Returns the sum of the per-observation terms
// log f(e_t / sigma_t) - log(sigma2_t) / 2. The result is not finite when some sigma2_t is not positive and
// finite, when the shape is outside its domain, or when some term is not finite; the outputs are then
// unspecified. Instantiated only for T = double, ad::Dual<8>, ad::Dual<16>, ad::Dual<32> and ad::Dual<kMaxParameterCount>.
template <class T>
T loglik(std::span<const double> r, std::span<const T> theta, const Layout& layout, VarInit var_init,
         const LoglikOutputs<T>& out);

// omega under variance targeting, from the residuals of the mean at theta: m (1 - P) with m the mean of e^2, or
// of |e|^power in apARCH and fGARCH, and P = persistence(theta); eGARCH: log(mean e^2) (1 - P). theta[omega] is
// not read. Instantiated as loglik.
template <class T>
T targeted_omega(std::span<const T> residuals, std::span<const T> theta, const Layout& layout);

}
