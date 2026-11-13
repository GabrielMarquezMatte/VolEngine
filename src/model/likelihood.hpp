#pragma once

#include "ad/dual.hpp"
#include "core/var_init.hpp"
#include "model/data.hpp"
#include "model/layout.hpp"

#include <span>

namespace ve::model
{

// Each span is either empty (not computed) or has the size of the series. deviations: d_t, the series the ARMA
// acts on (r_t - mu - sum_k delta_k x_t,k, less the ARCH-in-mean term); permanent: the csGARCH trend q_t, left
// untouched by the other models.
template <class T>
struct LoglikOutputs
{
    std::span<T> terms{};
    std::span<T> residuals{};
    std::span<T> sigma2{};
    std::span<T> deviations{};
    std::span<T> permanent{};
};

// theta is in natural space, in Layout order; data.mean has layout.mxreg columns and data.variance layout.vxreg.
// Returns the sum of the per-observation terms log f(e_t / sigma_t) - log(sigma2_t) / 2. The result is not finite
// when some sigma2_t is not positive and finite, when the shape is outside its domain, or when some term is not
// finite; the outputs are then unspecified. Instantiated only for T = double, ad::Dual<8>, ad::Dual<16>,
// ad::Dual<32> and ad::Dual<kMaxParameterCount>.
template <class T>
T loglik(const Data& data, std::span<const T> theta, const Layout& layout, VarInit var_init,
         const LoglikOutputs<T>& out);

// A series without regressors.
template <class T>
T loglik(std::span<const double> r, std::span<const T> theta, const Layout& layout, VarInit var_init,
         const LoglikOutputs<T>& out)
{
    return loglik<T>(Data{r}, theta, layout, var_init, out);
}

// omega under variance targeting, from the residuals of the mean at theta: m (1 - P) with m the mean of e^2, or
// of |e|^power in apARCH and fGARCH, and P = persistence(theta); eGARCH: log(mean e^2) (1 - P). theta[omega] is
// not read. variance has layout.vxreg columns (none, the default, without variance regressors); with variance
// regressors, minus sum_k zeta_k mean_t v_t,k. Instantiated as loglik.
template <class T>
T targeted_omega(std::span<const T> residuals, std::span<const T> theta, const Layout& layout,
                 const RegressorMatrix& variance = {});

// targeted_omega<double> from the residuals the likelihood itself targets: those of the mean without the
// ARCH-in-mean term (the e^0 of the presample), with the variance regressors of data. theta[omega] is not read.
double targeted_omega_at(const Data& data, std::span<const double> theta, const Layout& layout);

}
