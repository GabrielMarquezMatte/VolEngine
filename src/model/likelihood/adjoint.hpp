#pragma once

#include "core/var_init.hpp"
#include "garch/presample.hpp"
#include "model/data.hpp"
#include "model/layout.hpp"
#include "simd/density_kernels.hpp"

#include <span>
#include <vector>

namespace ve::model
{

// True for the models, densities and means whose log-likelihood gradient the adjoint computes, listed explicitly so
// that a new combination falls back to the forward Dual: sGARCH, iGARCH, gjrGARCH and eGARCH; norm, std and ged;
// ARMA with or without a mean; either VarInit; no regressors, ARCH-in-mean, ARFIMA or variance targeting.
bool has_adjoint(const Layout& layout, VarInit var_init);

// The log-likelihood and its gradient over the natural-space theta by reverse mode: loglik<double> runs forward and
// keeps the residuals and the variance, then each component runs its reverse sweep in the opposite order (density,
// variance recursion, presample, ARMA). The value is that of loglik<double> bit for bit. The buffers live as long as
// the object, which the optimizer calls on every evaluation.
class LoglikAdjoint
{
public:
    // Requires has_adjoint(layout, var_init) and grad.size() == theta.size(). grad is unspecified when the result is
    // not finite.
    double evaluate(const Data& data, std::span<const double> theta, const Layout& layout, VarInit var_init,
                    std::span<double> grad);

private:
    void density_backward(std::span<const double> theta, const Layout& layout,
                          const simd::SymmetricPartials& partials, std::span<double> grad);
    void variance_backward(std::span<const double> theta, const Layout& layout, const Data& data,
                           VarInit var_init, std::span<double> grad);
    double sgarch_backward(std::span<const double> theta, const Layout& layout,
                           const garch::Presample<double>& presample, std::span<double> grad);
    double gjrgarch_backward(std::span<const double> theta, const Layout& layout,
                             const garch::Presample<double>& presample, std::span<double> grad);
    double egarch_backward(std::span<const double> theta, const Layout& layout,
                           const garch::Presample<double>& presample, std::span<double> grad);
    void mean_backward(const Data& data, std::span<const double> theta, const Layout& layout,
                       std::span<double> grad);

    std::vector<double> residuals_;
    std::vector<double> sigma2_;
    std::vector<double> residual_adjoint_;
    std::vector<double> sigma2_adjoint_;
    std::vector<double> log_sigma2_;
    std::vector<double> inverse_sigma_;
};

}
