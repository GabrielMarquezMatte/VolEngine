#include "model/likelihood/adjoint.hpp"

#include "ad/dual.hpp"
#include "dist/moments.hpp"
#include "dist/symmetric.hpp"
#include "garch/egarch.hpp"
#include "garch/gjrgarch.hpp"
#include "garch/sgarch.hpp"
#include "mean/arma.hpp"
#include "model/likelihood.hpp"
#include "model/likelihood/parameters.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>

namespace ve::model
{

namespace
{

bool is_sgarch_family(Model model)
{
    return model == Model::SGarch || model == Model::IGarch;
}

bool is_covered_model(Model model)
{
    return is_sgarch_family(model) || model == Model::GjrGarch || model == Model::EGarch;
}

bool is_covered_dist(Dist dist)
{
    return dist == Dist::Norm || dist == Dist::Std || dist == Dist::Ged;
}

// Exhaustive with no default: a new VarInit is a -Wswitch error here until the adjoint's presample handles it.
bool is_covered_init(VarInit var_init)
{
    switch (var_init)
    {
        case VarInit::Sample:
        case VarInit::Backcast:
            return true;
    }
    return false;
}

// iGARCH reads beta_q = 1 - sum alpha - sum_{j<q} beta_j: the partial of that beta_q moves to the alphas and the
// earlier betas with a minus sign, and the stored beta_q, never read, has none.
void chain_igarch_beta(std::span<double> alpha, std::span<double> beta)
{
    double last = beta.back();
    for (double& partial : alpha)
    {
        partial -= last;
    }
    for (double& partial : beta.first(beta.size() - 1))
    {
        partial -= last;
    }
    beta.back() = 0.0;
}

}

bool has_adjoint(const Layout& layout, VarInit var_init)
{
    if (!is_covered_init(var_init))
    {
        return false;
    }
    bool plain_mean = layout.mxreg == 0 && !layout.has_archm() && !layout.has_arfima();
    return is_covered_model(layout.model) && is_covered_dist(layout.dist) && plain_mean && layout.vxreg == 0 &&
           !layout.variance_targeting;
}

double LoglikAdjoint::evaluate(const Data& data, std::span<const double> theta, const Layout& layout,
                               VarInit var_init, std::span<double> grad)
{
    assert(has_adjoint(layout, var_init) && grad.size() == theta.size());
    std::size_t count = data.r.size();
    residuals_.resize(count);
    sigma2_.resize(count);
    residual_adjoint_.resize(count);
    sigma2_adjoint_.resize(count);
    simd::SymmetricPartials partials{residual_adjoint_, sigma2_adjoint_};
    double value = loglik<double>(data, theta, layout, var_init,
                                  {.residuals = residuals_, .sigma2 = sigma2_, .density_partials = &partials});
    if (!std::isfinite(value))
    {
        return value;
    }
    std::ranges::fill(grad, 0.0);
    density_backward(theta, layout, partials, grad);
    variance_backward(theta, layout, data, var_init, grad);
    mean_backward(data, theta, layout, grad);
    return value;
}

// The adjoints of e_t and sigma2_t are the partials of the terms, which the likelihood's own vector pass wrote; what
// is left is the shape's: the constants that depend on the shape alone (log constant, scale, power) carry their
// derivative from a Dual<1> of the same function the likelihood calls, weighted by the sums of the partials.
void LoglikAdjoint::density_backward(std::span<const double> theta, const Layout& layout,
                                     const simd::SymmetricPartials& partials, std::span<double> grad)
{
    if (!layout.has_shape())
    {
        return;
    }
    double shape = shape_parameters_of(theta, layout).shape;
    dist::LogDensity<ad::Dual<1>> slopes = dist::make_log_density(layout.dist, ad::variable<1>(shape, 0));
    grad[layout.shape()] += static_cast<double>(residuals_.size()) * slopes.log_constant.grad[0] +
                            partials.d_scale * slopes.scale.grad[0] + partials.d_power * slopes.power.grad[0];
}

// The variance recursion, then its presample: under VarInit::Sample the presample is the mean square of e, which
// passes 2 e_t / T of its adjoint to each e_t; a backcast is fixed by the data.
void LoglikAdjoint::variance_backward(std::span<const double> theta, const Layout& layout, const Data& data,
                                      VarInit var_init, std::span<double> grad)
{
    garch::Presample<double> presample = presample_at(data, residuals_, theta, layout, var_init);
    double presample_adjoint = 0.0;
    if (layout.model == Model::GjrGarch)
    {
        presample_adjoint = gjrgarch_backward(theta, layout, presample, grad);
    }
    if (is_sgarch_family(layout.model))
    {
        presample_adjoint = sgarch_backward(theta, layout, presample, grad);
    }
    if (layout.model == Model::EGarch)
    {
        presample_adjoint = egarch_backward(theta, layout, presample, grad);
    }
    if (var_init == VarInit::Sample)
    {
        double slope = 2.0 * presample_adjoint / static_cast<double>(residuals_.size());
        for (std::size_t t = 0; t < residuals_.size(); ++t)
        {
            residual_adjoint_[t] += slope * residuals_[t];
        }
    }
}

// sGARCH, and iGARCH through its beta_q; returns the presample's adjoint.
double LoglikAdjoint::sgarch_backward(std::span<const double> theta, const Layout& layout,
                                      const garch::Presample<double>& presample, std::span<double> grad)
{
    garch::SGarchParameters<double> p = sgarch_parameters(theta, layout);
    std::array<double, kMaxOrder> beta_storage{};
    if (layout.model == Model::IGarch)
    {
        p.beta = igarch_beta(p, beta_storage);
    }
    std::span<double> alpha = grad.subspan(layout.alpha_begin(), p.alpha.size());
    std::span<double> beta = grad.subspan(layout.beta_begin(), p.beta.size());
    garch::SGarchAdjoint adjoint{residual_adjoint_, sigma2_adjoint_, 0.0, alpha, beta, 0.0};
    garch::sgarch_variance_backward(residuals_, p, presample, sigma2_, adjoint);
    grad[layout.omega()] += adjoint.omega;
    if (layout.model == Model::IGarch)
    {
        chain_igarch_beta(alpha, beta);
    }
    return adjoint.presample;
}

// Returns the presample's adjoint.
double LoglikAdjoint::gjrgarch_backward(std::span<const double> theta, const Layout& layout,
                                        const garch::Presample<double>& presample, std::span<double> grad)
{
    garch::GjrGarchParameters<double> p = gjrgarch_parameters(theta, layout);
    garch::GjrGarchAdjoint adjoint{residual_adjoint_,
                                   sigma2_adjoint_,
                                   0.0,
                                   grad.subspan(layout.alpha_begin(), p.alpha.size()),
                                   grad.subspan(layout.beta_begin(), p.beta.size()),
                                   grad.subspan(layout.gamma_begin(), p.gamma.size()),
                                   0.0};
    garch::gjrgarch_variance_backward(residuals_, p, presample, sigma2_, adjoint);
    grad[layout.omega()] += adjoint.omega;
    return adjoint.presample;
}

// The recursion runs on h_t = log sigma2_t, whose adjoint is sigma2_t d/dsigma2_t; returns the presample's adjoint,
// d/db = (d/dlog b) / b. E|z| depends on the shape alone and passes its adjoint on through a Dual<1>.
double LoglikAdjoint::egarch_backward(std::span<const double> theta, const Layout& layout,
                                      const garch::Presample<double>& presample, std::span<double> grad)
{
    garch::EGarchParameters<double> p = egarch_parameters(theta, layout);
    log_sigma2_.resize(sigma2_.size());
    inverse_sigma_.resize(sigma2_.size());
    for (std::size_t t = 0; t < sigma2_.size(); ++t)
    {
        log_sigma2_[t] = std::log(sigma2_[t]);
        inverse_sigma_[t] = 1.0 / std::sqrt(sigma2_[t]);
        sigma2_adjoint_[t] *= sigma2_[t];
    }
    garch::EGarchAdjoint adjoint{residual_adjoint_,
                                 sigma2_adjoint_,
                                 0.0,
                                 grad.subspan(layout.alpha_begin(), p.alpha.size()),
                                 grad.subspan(layout.beta_begin(), p.beta.size()),
                                 grad.subspan(layout.gamma_begin(), p.gamma.size()),
                                 0.0,
                                 0.0};
    garch::egarch_variance_backward(residuals_, p, presample, {log_sigma2_, inverse_sigma_}, adjoint);
    grad[layout.omega()] += adjoint.omega;
    if (layout.has_shape())
    {
        dist::ShapeParameters<ad::Dual<1>> shape{};
        shape.shape = ad::variable<1>(theta[layout.shape()], 0);
        ad::Dual<1> moment = dist::abs_moment(layout.dist, shape, ad::constant<1>(1.0));
        grad[layout.shape()] += adjoint.abs_moment * moment.grad[0];
    }
    return adjoint.presample / presample.value;
}

void LoglikAdjoint::mean_backward(const Data& data, std::span<const double> theta, const Layout& layout,
                                  std::span<double> grad)
{
    mean::ArmaParameters<double> p = arma_parameters(theta, layout);
    mean::ArmaGradient gradient{0.0, grad.subspan(layout.ar_begin(), p.ar.size()),
                                grad.subspan(layout.ma_begin(), p.ma.size())};
    mean::arma_residuals_backward(data.r, p, residuals_, residual_adjoint_, gradient);
    if (layout.include_mean)
    {
        grad[layout.mu()] += gradient.mu;
    }
}

}
