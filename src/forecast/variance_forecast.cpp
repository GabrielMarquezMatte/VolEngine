#include "forecast/variance_forecast.hpp"

#include "dist/moments.hpp"
#include "garch/power_garch.hpp"
#include "model/parameter_map.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <span>
#include <vector>

namespace ve::forecast::detail
{

namespace
{

// omega + c_h, with c_h = sum_k zeta_k v_{h,k} over the future variance regressors.
double intercept(const Inputs& in, std::size_t h)
{
    std::span<const double> zeta = block(in.theta, in.layout.vxreg_begin(), in.layout.vxreg);
    double shift = 0.0;
    for (std::size_t k = 0; k < zeta.size(); ++k)
    {
        shift = shift + zeta[k] * in.future.variance.at(h - 1, k);
    }
    return in.theta[in.layout.omega()] + shift;
}

// The beta of the point; in iGARCH beta_q = 1 - sum alpha - sum_{j<q} beta_j, as the likelihood derives it.
std::vector<double> betas(const Inputs& in)
{
    std::span<const double> beta = block(in.theta, in.layout.beta_begin(), in.layout.garch);
    std::vector<double> result(beta.begin(), beta.end());
    if (in.layout.model != Model::IGarch)
    {
        return result;
    }
    double last = 1.0;
    for (double alpha : block(in.theta, in.layout.alpha_begin(), in.layout.arch))
    {
        last = last - alpha;
    }
    for (std::size_t j = 0; j + 1 < result.size(); ++j)
    {
        last = last - result[j];
    }
    result.back() = last;
    return result;
}

// Shock i of step h: (alpha_i + gamma_i 1[e < 0]) e^2 in the sample, (alpha_i + gamma_i pi) sigma2 after it; gamma
// is 0 but in gjrGARCH.
double quadratic_shock(const Inputs& in, std::span<const double> forecast, std::size_t h, std::size_t i, double pi)
{
    double alpha = in.theta[in.layout.alpha_begin() + i - 1];
    double gamma = 0.0;
    if (in.layout.model == Model::GjrGarch)
    {
        gamma = in.theta[in.layout.gamma_begin() + i - 1];
    }
    if (i < h)
    {
        return (alpha + gamma * pi) * forecast[h - 1 - i];
    }
    double e = lagged(in.state.residuals, forecast, h, i);
    double weight = e < 0.0 ? alpha + gamma : alpha;
    return weight * (e * e);
}

// sGARCH, iGARCH and gjrGARCH: sigma2_h = omega_h + sum_i shock_i + sum_j beta_j sigma2_{h-j}.
std::vector<double> quadratic_variance(const Inputs& in)
{
    double pi = in.moments.below_zero;
    std::vector<double> beta = betas(in);
    std::vector<double> sigma2(in.horizon);
    for (std::size_t h = 1; h <= in.horizon; ++h)
    {
        double value = intercept(in, h);
        for (std::size_t i = 1; i <= static_cast<std::size_t>(in.layout.arch); ++i)
        {
            value = value + quadratic_shock(in, sigma2, h, i, pi);
        }
        for (std::size_t j = 1; j <= beta.size(); ++j)
        {
            value = value + beta[j - 1] * lagged(in.state.sigma2, sigma2, h, j);
        }
        sigma2[h - 1] = value;
    }
    return sigma2;
}

// csGARCH, as rugarch forecasts it: h = 1 is the recursion of the fit at n; from h = 2 on, the closed form from the
// last observation, q_h = (1 - rho^h) / (1 - rho) omega_h + rho^h q and sigma2_h = q_h + (sum alpha + sum
// beta)^h (sigma2 - q), which leaves out the shock of that observation.
std::vector<double> component_variance(const Inputs& in)
{
    const ForecastState& s = in.state;
    std::size_t last = s.sigma2.size() - 1;
    double rho = in.theta[in.layout.eta1_begin()];
    std::span<const double> alpha = block(in.theta, in.layout.alpha_begin(), in.layout.arch);
    std::span<const double> beta = block(in.theta, in.layout.beta_begin(), in.layout.garch);
    double first = next_permanent(in);
    for (std::size_t i = 1; i <= alpha.size(); ++i)
    {
        double e = s.residuals[last + 1 - i];
        first = first + alpha[i - 1] * (e * e - s.permanent[last + 1 - i]);
    }
    for (std::size_t j = 1; j <= beta.size(); ++j)
    {
        first = first + beta[j - 1] * (s.sigma2[last + 1 - j] - s.permanent[last + 1 - j]);
    }
    double weight = 0.0;
    std::ranges::for_each(alpha, [&weight](double a) { weight += a; });
    std::ranges::for_each(beta, [&weight](double b) { weight += b; });
    std::vector<double> sigma2(in.horizon, first);
    for (std::size_t h = 2; h <= in.horizon; ++h)
    {
        double power = std::pow(rho, static_cast<double>(h));
        double trend = (1.0 - power) / (1.0 - rho) * intercept(in, h) + power * s.permanent[last];
        sigma2[h - 1] = trend + std::pow(weight, static_cast<double>(h)) * (s.sigma2[last] - s.permanent[last]);
    }
    return sigma2;
}

// eGARCH in log sigma2: alpha_i z + gamma_i (|z| - E|z|) in the sample, 0 after it, as rugarch's forecast takes
// the shock of a future step at its expectation.
std::vector<double> log_variance(const Inputs& in)
{
    std::span<const double> alpha = block(in.theta, in.layout.alpha_begin(), in.layout.arch);
    std::span<const double> gamma = block(in.theta, in.layout.gamma_begin(), in.layout.gamma);
    std::vector<double> beta = betas(in);
    double abs_moment = in.moments.abs_moment;
    std::vector<double> sample(in.state.sigma2.size());
    std::ranges::transform(in.state.sigma2, sample.begin(), [](double variance) { return std::log(variance); });
    std::vector<double> logs(in.horizon);
    for (std::size_t h = 1; h <= in.horizon; ++h)
    {
        double value = intercept(in, h);
        for (std::size_t i = h; i <= alpha.size(); ++i)
        {
            double z = lagged(in.state.residuals, {}, h, i) / std::sqrt(lagged(in.state.sigma2, {}, h, i));
            value = value + alpha[i - 1] * z + gamma[i - 1] * (std::abs(z) - abs_moment);
        }
        for (std::size_t j = 1; j <= beta.size(); ++j)
        {
            value = value + beta[j - 1] * lagged(sample, logs, h, j);
        }
        logs[h - 1] = value;
    }
    std::ranges::transform(logs, logs.begin(), [](double value) { return std::exp(value); });
    return logs;
}

garch::PowerGarchParameters<double> exact_power_parameters(const Inputs& in, std::span<const double> moments)
{
    return {in.theta[in.layout.omega()],
            block(in.theta, in.layout.alpha_begin(), in.layout.arch),
            block(in.theta, in.layout.beta_begin(), in.layout.garch),
            in.theta.subspan(in.layout.asymmetry_begin(), in.layout.asymmetry_count()),
            block(in.theta, in.layout.eta2_begin(), in.layout.eta2),
            model::power_of<double>(in.theta, in.layout),
            moments,
            0.0,
            in.layout.free_power};
}

// apARCH and fGARCH in sigma^delta: the shock of power_garch_step in the sample, with the exact |.| of rugarch's
// forecast (the filter of the fit smooths it in fGARCH), and kappa_i sigma^delta after it.
std::vector<double> power_variance(const Inputs& in)
{
    garch::PowerGarchParameters<double> p = exact_power_parameters(in, in.moments.shock);
    std::vector<double> sample(in.state.sigma2.size());
    std::ranges::transform(in.state.sigma2, sample.begin(),
                           [&p](double variance) { return garch::raise(variance, 0.5 * p.power, p.free_power); });
    std::vector<double> x(in.horizon);
    for (std::size_t h = 1; h <= in.horizon; ++h)
    {
        double value = intercept(in, h);
        for (std::size_t i = 1; i <= p.alpha.size(); ++i)
        {
            double shock = i < h ? p.moments[i - 1] * x[h - 1 - i]
                                 : garch::power_shock(lagged(in.state.residuals, {}, h, i), lagged(sample, {}, h, i),
                                                      p, i - 1);
            value = value + p.alpha[i - 1] * shock;
        }
        for (std::size_t j = 1; j <= p.beta.size(); ++j)
        {
            value = value + p.beta[j - 1] * lagged(sample, x, h, j);
        }
        x[h - 1] = value;
    }
    std::ranges::transform(x, x.begin(),
                           [&p](double value) { return garch::raise(value, 2.0 / p.power, p.free_power); });
    return x;
}

}

Moments moments_of(const model::Layout& layout, std::span<const double> theta)
{
    Moments moments;
    dist::ShapeParameters<double> shape = model::shape_parameters_of(theta, layout);
    if (layout.model == Model::GjrGarch)
    {
        moments.below_zero = dist::probability_below_zero(layout.dist, shape);
    }
    if (layout.model == Model::EGarch)
    {
        moments.abs_moment = dist::abs_moment(layout.dist, shape, 1.0);
    }
    if (layout.is_power_family())
    {
        moments.shock.resize(static_cast<std::size_t>(layout.arch));
        for (std::size_t i = 0; i < moments.shock.size(); ++i)
        {
            moments.shock[i] = model::shock_moment<double>(theta, layout, i);
        }
    }
    return moments;
}

double next_permanent(const Inputs& in)
{
    const ForecastState& s = in.state;
    std::size_t last = s.sigma2.size() - 1;
    double square = s.residuals[last] * s.residuals[last];
    return intercept(in, 1) + in.theta[in.layout.eta1_begin()] * s.permanent[last] +
           in.theta[in.layout.eta2_begin()] * (square - s.sigma2[last]);
}

std::vector<double> variance_forecast(const Inputs& in)
{
    switch (in.layout.model)
    {
        case Model::SGarch:
        case Model::IGarch:
        case Model::GjrGarch:
            return quadratic_variance(in);
        case Model::CsGarch:
            return component_variance(in);
        case Model::EGarch:
            return log_variance(in);
        case Model::ApArch:
        case Model::FGarchGarch:
        case Model::FGarchTGarch:
        case Model::FGarchAvGarch:
        case Model::FGarchNGarch:
        case Model::FGarchNaGarch:
        case Model::FGarchApArch:
        case Model::FGarchGjrGarch:
        case Model::FGarchAllGarch:
            return power_variance(in);
    }
    assert(false && "unknown variance model");
    return std::vector<double>(in.horizon, 0.0);
}

}
