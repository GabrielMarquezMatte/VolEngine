#include "model/likelihood.hpp"

#include "dist/densities.hpp"
#include "dist/moments.hpp"
#include "garch/csgarch.hpp"
#include "garch/egarch.hpp"
#include "garch/gjrgarch.hpp"
#include "garch/power_garch.hpp"
#include "garch/sgarch.hpp"
#include "mean/arma.hpp"
#include "model/parameter_map.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace ve::model
{

namespace
{

// arch backcast: exponentially weighted mean of the first min(75, T) squared residuals, decay 0.94.
constexpr std::size_t kBackcastWindow = 75;
constexpr double kBackcastDecay = 0.94;

constexpr double kInvalid = -std::numeric_limits<double>::infinity();

double sample_mean(std::span<const double> r)
{
    double sum = 0.0;
    for (double value : r)
    {
        sum += value;
    }
    return sum / static_cast<double>(r.size());
}

// arch computes the backcast once, from the residuals at its starting mean (the sample mean, or zero without
// a mean), and keeps it fixed during the optimization: it depends on no parameter.
double backcast(std::span<const double> r, bool include_mean)
{
    double center = 0.0;
    if (include_mean)
    {
        center = sample_mean(r);
    }
    double weight = 1.0;
    double weight_sum = 0.0;
    double weighted_square_sum = 0.0;
    for (double value : r.first(std::min(kBackcastWindow, r.size())))
    {
        double residual = value - center;
        weight_sum += weight;
        weighted_square_sum += weight * residual * residual;
        weight *= kBackcastDecay;
    }
    return weighted_square_sum / weight_sum;
}

template <class T>
std::span<const T> block(std::span<const T> theta, std::size_t begin, int count)
{
    return theta.subspan(begin, static_cast<std::size_t>(count));
}

template <class T>
mean::ArmaParameters<T> arma_parameters(std::span<const T> theta, const Layout& layout)
{
    T mu{0.0};
    if (layout.include_mean)
    {
        mu = theta[layout.mu()];
    }
    return {mu, block(theta, layout.ar_begin(), layout.ar), block(theta, layout.ma_begin(), layout.ma)};
}

template <class T>
garch::SGarchParameters<T> sgarch_parameters(std::span<const T> theta, const Layout& layout)
{
    return {theta[layout.omega()], block(theta, layout.alpha_begin(), layout.arch),
            block(theta, layout.beta_begin(), layout.garch)};
}

template <class T>
garch::EGarchParameters<T> egarch_parameters(std::span<const T> theta, const Layout& layout)
{
    garch::SGarchParameters<T> s = sgarch_parameters(theta, layout);
    return {s.omega, s.alpha, s.beta, block(theta, layout.gamma_begin(), layout.gamma),
            dist::abs_moment(layout.dist, shape_of(theta, layout), T{1.0})};
}

template <class T>
garch::GjrGarchParameters<T> gjrgarch_parameters(std::span<const T> theta, const Layout& layout)
{
    garch::SGarchParameters<T> s = sgarch_parameters(theta, layout);
    return {s.omega, s.alpha, s.beta, block(theta, layout.gamma_begin(), layout.gamma)};
}

template <class T>
garch::CsGarchParameters<T> csgarch_parameters(std::span<const T> theta, const Layout& layout)
{
    garch::SGarchParameters<T> s = sgarch_parameters(theta, layout);
    return {s.omega, s.alpha, s.beta, theta[layout.eta1_begin()], theta[layout.eta2_begin()]};
}

// rugarch's fGARCH takes |z - eta2| as sqrt((z - eta2)^2 + 1e-6) in every submodel; its apARCH is exact.
constexpr double kFGarchAbsSmoothing = 1e-6;

double abs_smoothing_of(const Layout& layout)
{
    if (layout.model == Model::ApArch)
    {
        return 0.0;
    }
    return kFGarchAbsSmoothing;
}

// moments must outlive the result, whose moments span views it; shock_moments fills it.
template <class T>
garch::PowerGarchParameters<T> power_garch_parameters(std::span<const T> theta, const Layout& layout,
                                                      std::span<const T> moments)
{
    garch::SGarchParameters<T> s = sgarch_parameters(theta, layout);
    return {s.omega,
            s.alpha,
            s.beta,
            theta.subspan(layout.asymmetry_begin(), layout.asymmetry_count()),
            block(theta, layout.eta2_begin(), layout.eta2),
            power_of(theta, layout),
            moments.first(s.alpha.size()),
            abs_smoothing_of(layout),
            layout.free_power};
}

// False when a kappa is not finite (std with shape <= power), which can come back as NaN as well as +inf.
template <class T>
bool shock_moments(std::span<const T> theta, const Layout& layout, std::span<T> moments)
{
    for (std::size_t i = 0; i < static_cast<std::size_t>(layout.arch); ++i)
    {
        moments[i] = shock_moment(theta, layout, i);
        if (!std::isfinite(ad::value_of(moments[i])))
        {
            return false;
        }
    }
    return true;
}

template <class T>
T mean_square(std::span<const T> residuals)
{
    T sum{0.0};
    for (const T& e : residuals)
    {
        sum = sum + e * e;
    }
    return sum / static_cast<double>(residuals.size());
}

// (mean |e|^power)^(2 / power): the variance whose sigma^power is the sample mean of |e|^power.
template <class T>
T mean_power_variance(std::span<const T> residuals, const T& power, bool free_power)
{
    using std::abs;
    T sum{0.0};
    for (const T& e : residuals)
    {
        sum = sum + garch::raise(abs(e), power, free_power);
    }
    return garch::raise(sum / static_cast<double>(residuals.size()), 2.0 / power, free_power);
}

// rugarch holds sigma2_t for t < max(m, n, p, q), the ARMA orders included, at the sample mean of the squared
// residuals taken at the current mean parameters; in apARCH and fGARCH it holds sigma^power at the sample mean of
// |e|^power instead.
template <class T>
garch::Presample<T> presample(std::span<const double> r, std::span<const T> residuals, std::span<const T> theta,
                              const Layout& layout, VarInit var_init)
{
    if (var_init == VarInit::Backcast)
    {
        return {T{backcast(r, layout.include_mean)}, 0};
    }
    std::size_t held = static_cast<std::size_t>(std::max({layout.ar, layout.ma, layout.arch, layout.garch}));
    if (layout.is_power_family())
    {
        return {mean_power_variance(residuals, power_of(theta, layout), layout.free_power), held};
    }
    return {mean_square(residuals), held};
}

// The sGARCH recursion with beta_q = 1 - sum alpha - sum_{j<q} beta_j; the stored beta_q is never read.
template <class T>
void igarch_variance(std::span<const T> residuals, std::span<const T> theta, const Layout& layout,
                     const garch::Presample<T>& initial, std::span<T> sigma2)
{
    garch::SGarchParameters<T> p = sgarch_parameters(theta, layout);
    std::array<T, kMaxOrder> beta{};
    std::ranges::copy(p.beta, beta.begin());
    T last{1.0};
    for (const T& alpha : p.alpha)
    {
        last = last - alpha;
    }
    for (const T& earlier : p.beta.first(p.beta.size() - 1))
    {
        last = last - earlier;
    }
    beta[p.beta.size() - 1] = last;
    p.beta = std::span<const T>(beta).first(p.beta.size());
    garch::sgarch_variance<T>(residuals, p, initial, sigma2);
}

template <class T>
void csgarch_from_residuals(std::span<const T> residuals, const garch::CsGarchParameters<T>& p,
                            const garch::Presample<T>& initial, std::span<T> sigma2)
{
    std::vector<T> squared_residuals(residuals.size());
    std::vector<T> permanent(residuals.size());
    std::ranges::transform(residuals, squared_residuals.begin(), [](const T& e) { return e * e; });
    garch::csgarch_variance<T>(squared_residuals, p, initial, sigma2, permanent);
}

// False when a shock moment is not finite.
template <class T>
bool power_variance(std::span<const T> residuals, std::span<const T> theta, const Layout& layout,
                    const garch::Presample<T>& initial, std::span<T> sigma2)
{
    std::array<T, kMaxOrder> moments{};
    garch::PowerGarchParameters<T> p = power_garch_parameters<T>(theta, layout, moments);
    if (!shock_moments(theta, layout, std::span<T>(moments)))
    {
        return false;
    }
    garch::power_garch_variance<T>(residuals, p, initial, sigma2);
    return true;
}

// False when the point is outside the domain of the recursion.
template <class T>
bool fill_variance(std::span<const T> residuals, std::span<const T> theta, const Layout& layout,
                   const garch::Presample<T>& initial, std::span<T> sigma2)
{
    switch (layout.model)
    {
        case Model::SGarch:
            garch::sgarch_variance<T>(residuals, sgarch_parameters(theta, layout), initial, sigma2);
            return true;
        case Model::IGarch:
            igarch_variance<T>(residuals, theta, layout, initial, sigma2);
            return true;
        case Model::EGarch:
            garch::egarch_variance<T>(residuals, egarch_parameters(theta, layout), initial, sigma2);
            return true;
        case Model::GjrGarch:
            garch::gjrgarch_variance<T>(residuals, gjrgarch_parameters(theta, layout), initial, sigma2);
            return true;
        case Model::CsGarch:
            csgarch_from_residuals<T>(residuals, csgarch_parameters(theta, layout), initial, sigma2);
            return true;
        case Model::ApArch:
        case Model::FGarchGarch:
        case Model::FGarchTGarch:
        case Model::FGarchAvGarch:
        case Model::FGarchNGarch:
        case Model::FGarchNaGarch:
        case Model::FGarchApArch:
        case Model::FGarchGjrGarch:
        case Model::FGarchAllGarch:
            return power_variance<T>(residuals, theta, layout, initial, sigma2);
    }
    return false;
}

template <class T>
std::span<T> provided_or(std::span<T> provided, std::vector<T>& storage, std::size_t size)
{
    if (!provided.empty())
    {
        return provided;
    }
    storage.resize(size);
    return storage;
}

template <class T>
bool is_valid_variance(const T& sigma2)
{
    double value = ad::value_of(sigma2);
    return std::isfinite(value) && value > 0.0;
}

// log_density_of(e, sigma2) is log f(e / sqrt(sigma2)).
template <class T, class LogDensityOf>
T sum_of_terms(std::span<const T> residuals, std::span<const T> sigma2, LogDensityOf log_density_of,
               std::span<T> terms)
{
    using std::log;
    T total{0.0};
    for (std::size_t t = 0; t < residuals.size(); ++t)
    {
        if (!is_valid_variance(sigma2[t]))
        {
            total = T{kInvalid};
            return total;
        }
        T term = log_density_of(residuals[t], sigma2[t]) - 0.5 * log(sigma2[t]);
        if (!terms.empty())
        {
            terms[t] = term;
        }
        total = total + term;
    }
    return total;
}

// norm and std read z^2 = e^2 / sigma2, which needs neither sqrt nor a second division.
template <class T>
T sum_of_terms(std::span<const T> residuals, std::span<const T> sigma2, const dist::LogDensity<T>& density,
               std::span<T> terms)
{
    using std::abs;
    using std::sqrt;
    switch (density.dist)
    {
        case Dist::Norm:
            return sum_of_terms<T>(residuals, sigma2,
                                   [&density](const T& e, const T& variance)
                                   { return dist::normal_log_density_at(density, e * e / variance); },
                                   terms);
        case Dist::Std:
            return sum_of_terms<T>(residuals, sigma2,
                                   [&density](const T& e, const T& variance)
                                   { return dist::student_log_density_at(density, e * e / variance); },
                                   terms);
        case Dist::Ged:
            return sum_of_terms<T>(residuals, sigma2,
                                   [&density](const T& e, const T& variance)
                                   { return dist::ged_log_density_at(density, abs(e) / sqrt(variance)); },
                                   terms);
    }
    return T{kInvalid};
}

}

template <class T>
T loglik(std::span<const double> r, std::span<const T> theta, const Layout& layout, VarInit var_init,
         const LoglikOutputs<T>& out)
{
    assert(theta.size() == layout.count());
    assert(out.terms.empty() || out.terms.size() == r.size());
    assert(out.residuals.empty() || out.residuals.size() == r.size());
    assert(out.sigma2.empty() || out.sigma2.size() == r.size());
    T shape = shape_of(theta, layout);
    if (!dist::in_domain(layout.dist, ad::value_of(shape)))
    {
        return T{kInvalid};
    }
    std::vector<T> residual_storage;
    std::vector<T> variance_storage;
    std::span<T> residuals = provided_or(out.residuals, residual_storage, r.size());
    std::span<T> sigma2 = provided_or(out.sigma2, variance_storage, r.size());
    mean::arma_residuals(r, arma_parameters(theta, layout), residuals);
    garch::Presample<T> initial = presample<T>(r, residuals, theta, layout, var_init);
    if (!fill_variance<T>(residuals, theta, layout, initial, sigma2))
    {
        return T{kInvalid};
    }
    return sum_of_terms<T>(residuals, sigma2, dist::make_log_density(layout.dist, shape), out.terms);
}

template double loglik<double>(std::span<const double>, std::span<const double>, const Layout&, VarInit,
                               const LoglikOutputs<double>&);
template ad::Dual<8> loglik<ad::Dual<8>>(std::span<const double>, std::span<const ad::Dual<8>>, const Layout&,
                                         VarInit, const LoglikOutputs<ad::Dual<8>>&);
template ad::Dual<16> loglik<ad::Dual<16>>(std::span<const double>, std::span<const ad::Dual<16>>,
                                           const Layout&, VarInit, const LoglikOutputs<ad::Dual<16>>&);
template ad::Dual<32> loglik<ad::Dual<32>>(std::span<const double>, std::span<const ad::Dual<32>>,
                                           const Layout&, VarInit, const LoglikOutputs<ad::Dual<32>>&);
template ad::Dual<40> loglik<ad::Dual<40>>(std::span<const double>, std::span<const ad::Dual<40>>,
                                           const Layout&, VarInit, const LoglikOutputs<ad::Dual<40>>&);
}
