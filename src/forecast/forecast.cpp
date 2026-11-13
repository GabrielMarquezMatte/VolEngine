#include "forecast/forecast.hpp"

#include "dist/moments.hpp"
#include "model/layout.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <span>
#include <vector>

namespace ve::forecast
{

namespace
{

// What every rule reads: the point and its layout, the final state of the fit and the future regressors.
struct Inputs
{
    const ForecastState& state;
    model::Layout layout;
    std::span<const double> theta;
    const FutureRegressors& future;
    std::size_t horizon;
};

std::span<const double> block(std::span<const double> theta, std::size_t begin, int count)
{
    return theta.subspan(begin, static_cast<std::size_t>(count));
}

// x_{n-1+h-lag}: the forecast when it lies after the sample (lag < h), else the last values of the sample.
double lagged(std::span<const double> sample, std::span<const double> forecast, std::size_t h, std::size_t lag)
{
    if (lag < h)
    {
        return forecast[h - 1 - lag];
    }
    return sample[sample.size() - 1 - (lag - h)];
}

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
    double pi = 0.0;
    if (in.layout.model == Model::GjrGarch)
    {
        pi = dist::probability_below_zero(in.layout.dist, model::shape_parameters_of(in.theta, in.layout));
    }
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
    double phi = in.theta[in.layout.eta2_begin()];
    std::span<const double> alpha = block(in.theta, in.layout.alpha_begin(), in.layout.arch);
    std::span<const double> beta = block(in.theta, in.layout.beta_begin(), in.layout.garch);
    double square = s.residuals[last] * s.residuals[last];
    double q = intercept(in, 1) + rho * s.permanent[last] + phi * (square - s.sigma2[last]);
    double first = q;
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
        case Model::ApArch:
        case Model::FGarchGarch:
        case Model::FGarchTGarch:
        case Model::FGarchAvGarch:
        case Model::FGarchNGarch:
        case Model::FGarchNaGarch:
        case Model::FGarchApArch:
        case Model::FGarchGjrGarch:
        case Model::FGarchAllGarch:
            break;
    }
    assert(false && "variance forecast not implemented for this model");
    return std::vector<double>(in.horizon, 0.0);
}

// mu + sum_k delta_k x_{h,k} + xi sigma_h^p: the mean of step h before the ARMA part.
double level(const Inputs& in, double sigma, std::size_t h)
{
    double value = in.layout.include_mean ? in.theta[in.layout.mu()] : 0.0;
    std::span<const double> delta = block(in.theta, in.layout.mxreg_begin(), in.layout.mxreg);
    for (std::size_t k = 0; k < delta.size(); ++k)
    {
        value = value + delta[k] * in.future.mean.at(h - 1, k);
    }
    if (in.layout.archm == 1)
    {
        value = value + in.theta[in.layout.archm_index()] * sigma;
    }
    if (in.layout.archm == 2)
    {
        value = value + in.theta[in.layout.archm_index()] * (sigma * sigma);
    }
    return value;
}

// d_h = sum_i phi_i d_{h-i} + sum_{j >= h} theta_j e_{n-1+h-j}: the residuals after the sample are 0.
std::vector<double> mean_forecast(const Inputs& in, std::span<const double> sigma)
{
    std::span<const double> ar = block(in.theta, in.layout.ar_begin(), in.layout.ar);
    std::span<const double> ma = block(in.theta, in.layout.ma_begin(), in.layout.ma);
    std::vector<double> deviations(in.horizon);
    std::vector<double> mean(in.horizon);
    for (std::size_t h = 1; h <= in.horizon; ++h)
    {
        double d = 0.0;
        for (std::size_t i = 1; i <= ar.size(); ++i)
        {
            d = d + ar[i - 1] * lagged(in.state.deviations, deviations, h, i);
        }
        for (std::size_t j = h; j <= ma.size(); ++j)
        {
            d = d + ma[j - 1] * lagged(in.state.residuals, {}, h, j);
        }
        deviations[h - 1] = d;
        mean[h - 1] = level(in, sigma[h - 1], h) + d;
    }
    return mean;
}

}

Forecast forecast(const FitResult& fit, std::size_t horizon, const FutureRegressors& future)
{
    model::Layout layout = model::layout_of(fit.spec);
    assert(horizon > 0 && !fit.state.residuals.empty() && !layout.has_arfima());
    assert(future.mean.columns == static_cast<std::size_t>(layout.mxreg));
    assert(future.mean.values.size() == horizon * future.mean.columns);
    assert(future.variance.columns == static_cast<std::size_t>(layout.vxreg));
    assert(future.variance.values.size() == horizon * future.variance.columns);
    Inputs in{fit.state, layout, fit.estimate, future, horizon};
    Forecast result;
    result.sigma = variance_forecast(in);
    std::ranges::transform(result.sigma, result.sigma.begin(), [](double variance) { return std::sqrt(variance); });
    result.mean = mean_forecast(in, result.sigma);
    return result;
}

}
