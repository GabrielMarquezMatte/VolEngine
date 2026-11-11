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
// unspecified. Instantiated only for T = double, ad::Dual<8>, ad::Dual<16>, ad::Dual<32> and ad::Dual<40>.
template <class T>
T loglik(std::span<const double> r, std::span<const T> theta, const Layout& layout, VarInit var_init,
         const LoglikOutputs<T>& out);

}
