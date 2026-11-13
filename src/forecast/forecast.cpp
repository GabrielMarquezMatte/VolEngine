#include "forecast/forecast.hpp"

#include "forecast/inputs.hpp"
#include "forecast/step.hpp"
#include "forecast/variance_forecast.hpp"
#include "mean/fractional.hpp"
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

using detail::Inputs;
using detail::block;
using detail::lagged;

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

// The differenced deviations w_s = sum_{k>=0} pi_k d_{s-k} at the last m observations of the sample.
std::vector<double> differenced_tail(std::span<const double> series, std::span<const double> pi, std::size_t m)
{
    std::size_t n = series.size();
    std::vector<double> differenced(m);
    for (std::size_t i = 0; i < m; ++i)
    {
        std::size_t s = n - m + i;
        double w = 0.0;
        for (std::size_t k = 0; k <= s; ++k)
        {
            w = w + pi[k] * series[s - k];
        }
        differenced[i] = w;
    }
    return differenced;
}

// rugarch's ARFIMA, phi(L) (1 - L)^d d = theta(L) e over the deviations (not the ARMA residuals): F_t = -sum_{k>=1}
// pi_k d_{t-k}, w = the differenced deviations (d - F after the sample), and d_t = F_t + sum_i phi_i w_{t-i} +
// sum_{j >= h} theta_j e_{t-j}.
// ponytail: O(n H) direct sum; a recursion over the pi if long horizons ever need it.
std::vector<double> fractional_deviations(const Inputs& in)
{
    std::span<const double> ar = block(in.theta, in.layout.ar_begin(), in.layout.ar);
    std::span<const double> ma = block(in.theta, in.layout.ma_begin(), in.layout.ma);
    std::size_t n = in.state.deviations.size();
    std::vector<double> pi = mean::detail::weights(n + in.horizon, in.theta[in.layout.arfima_index()]);
    std::vector<double> series(in.state.deviations.begin(), in.state.deviations.end());
    std::vector<double> differenced = differenced_tail(series, pi, ar.size());
    std::vector<double> future(in.horizon);
    for (std::size_t h = 1; h <= in.horizon; ++h)
    {
        std::size_t t = n - 1 + h;
        double f = 0.0;
        for (std::size_t k = 1; k <= t; ++k)
        {
            f = f - pi[k] * series[t - k];
        }
        double d = f;
        for (std::size_t i = 1; i <= ar.size(); ++i)
        {
            d = d + ar[i - 1] * differenced[differenced.size() - i];
        }
        for (std::size_t j = h; j <= ma.size(); ++j)
        {
            d = d + ma[j - 1] * lagged(in.state.residuals, {}, h, j);
        }
        series.push_back(d);
        differenced.push_back(d - f);
        future[h - 1] = d;
    }
    return future;
}

// d_h = sum_i phi_i d_{h-i} + sum_{j >= h} theta_j e_{n-1+h-j}: the residuals after the sample are 0.
std::vector<double> arma_deviations(const Inputs& in)
{
    std::span<const double> ar = block(in.theta, in.layout.ar_begin(), in.layout.ar);
    std::span<const double> ma = block(in.theta, in.layout.ma_begin(), in.layout.ma);
    std::vector<double> deviations(in.horizon);
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
    }
    return deviations;
}

std::vector<double> mean_forecast(const Inputs& in, std::span<const double> sigma)
{
    std::vector<double> mean = in.layout.has_arfima() ? fractional_deviations(in) : arma_deviations(in);
    for (std::size_t h = 1; h <= in.horizon; ++h)
    {
        mean[h - 1] = level(in, sigma[h - 1], h) + mean[h - 1];
    }
    return mean;
}

}

Forecast forecast(const FitResult& fit, std::size_t horizon, const FutureRegressors& future)
{
    model::Layout layout = model::layout_of(fit.spec);
    assert(horizon > 0 && !fit.state.residuals.empty());
    assert(future.mean.columns == static_cast<std::size_t>(layout.mxreg));
    assert(future.mean.values.size() == horizon * future.mean.columns);
    assert(future.variance.columns == static_cast<std::size_t>(layout.vxreg));
    assert(future.variance.values.size() == horizon * future.variance.columns);
    detail::Moments moments = detail::moments_of(layout, fit.estimate);
    Inputs in{fit.state, layout, fit.estimate, future, horizon, moments};
    Forecast result;
    result.sigma = detail::variance_forecast(in);
    std::ranges::transform(result.sigma, result.sigma.begin(), [](double variance) { return std::sqrt(variance); });
    result.mean = mean_forecast(in, result.sigma);
    return result;
}

namespace detail
{

Step first_step(const Inputs& in)
{
    assert(in.horizon == 1);
    double sigma2 = variance_forecast(in)[0];
    double level_of_step = level(in, std::sqrt(sigma2), 1);
    std::vector<double> deviation = in.layout.has_arfima() ? fractional_deviations(in) : arma_deviations(in);
    double permanent = in.layout.model == Model::CsGarch ? next_permanent(in) : 0.0;
    return {sigma2, level_of_step + deviation[0], level_of_step, permanent};
}

}

}
