#include "dcc/future/forecast.hpp"

#include "dcc/future/state.hpp"
#include "dcc/likelihood.hpp"
#include "forecast/cumulative.hpp"
#include "model/layout.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <span>

namespace ve::dcc::future
{

namespace
{

std::optional<std::string> regressor_problem(const model::RegressorMatrix& rows, int columns, std::size_t horizon,
                                             const std::string& name)
{
    if (columns == 0 && rows.columns != 0)
    {
        return name + " was given but the spec has no " + name;
    }
    if (columns != 0 && rows.columns == 0)
    {
        return name + " must be given: the spec has " + std::to_string(columns) + " columns";
    }
    if (rows.columns != static_cast<std::size_t>(columns) || rows.values.size() != horizon * rows.columns)
    {
        return name + " must have " + std::to_string(horizon) + " rows x " + std::to_string(columns) + " columns";
    }
    return std::nullopt;
}

std::optional<std::string> series_problem(const SeriesEnd& end, const forecast::FutureRegressors& rows,
                                          std::size_t horizon)
{
    const model::Layout layout = model::layout_of(end.spec);
    if (std::optional<std::string> problem = regressor_problem(rows.mean, layout.mxreg, horizon, "mxreg"))
    {
        return problem;
    }
    return regressor_problem(rows.variance, layout.vxreg, horizon, "vxreg");
}

// diag(Qbar)^-1/2 Qbar diag(Qbar)^-1/2 of a packed Qbar, m x m with an exact unit diagonal, evaluated as the
// recursion does.
std::vector<double> bar_correlation(const std::vector<double>& qbar, std::size_t m)
{
    std::vector<double> scale(m);
    for (std::size_t i = 0; i < m; ++i)
    {
        scale[i] = 1.0 / std::sqrt(qbar[detail::packed_index(i, i)]);
    }
    std::vector<double> r(m * m);
    for (std::size_t i = 0; i < m; ++i)
    {
        r[i * m + i] = 1.0;
        for (std::size_t j = 0; j < i; ++j)
        {
            r[i * m + j] = qbar[detail::packed_index(i, j)] * scale[i] * scale[j];
            r[j * m + i] = r[i * m + j];
        }
    }
    return r;
}

// K_ij of the decay of R_h towards Rbar: the total persistence sum a + sum b, or for the FDCC the products of its
// group weights alpha_i alpha_j + beta_i beta_j, the same weights as the recursion.
std::vector<double> persistence(const DccFitResult& fit)
{
    const std::size_t m = fit.series;
    const DccParameters<double> theta = split_parameters<double>(fit.spec, fit.end.theta);
    std::vector<double> k(m * m);
    if (fit.spec.model != DccModel::FDcc)
    {
        double total = 0.0;
        for (double a : theta.a)
        {
            total = total + a;
        }
        for (double b : theta.b)
        {
            total = total + b;
        }
        std::ranges::fill(k, total);
        return k;
    }
    for (std::size_t i = 0; i < m; ++i)
    {
        for (std::size_t j = 0; j < m; ++j)
        {
            const auto gi = static_cast<std::size_t>(fit.spec.groups[i] - 1);
            const auto gj = static_cast<std::size_t>(fit.spec.groups[j] - 1);
            k[i * m + j] = theta.a[gi] * theta.a[gj] + theta.b[gi] * theta.b[gj];
        }
    }
    return k;
}

// R_1..R_horizon, row-major.
std::vector<double> correlation_forecast(const DccFitResult& fit, std::size_t horizon)
{
    const std::size_t m = fit.series;
    const std::size_t size = m * m;
    std::vector<double> r(horizon * size);
    if (!fit.end.static_r.empty())
    {
        for (std::size_t h = 0; h < horizon; ++h)
        {
            std::ranges::copy(fit.end.static_r, r.begin() + static_cast<std::ptrdiff_t>(h * size));
        }
        return r;
    }
    CorrelationStepper stepper(fit.end, fit.spec, m);
    const std::span<const double> first = stepper.next();
    std::ranges::copy(first, r.begin());
    const std::vector<double> bar = bar_correlation(fit.end.qbar, m);
    const std::vector<double> k = persistence(fit);
    for (std::size_t h = 1; h < horizon; ++h)
    {
        // k is one value outside FDCC and a few per group pair in it: pow only where it changes.
        double base = std::numeric_limits<double>::quiet_NaN();
        double weight = 0.0;
        for (std::size_t e = 0; e < size; ++e)
        {
            if (k[e] != base)
            {
                base = k[e];
                weight = std::pow(base, static_cast<double>(h));
            }
            r[h * size + e] = (1.0 - weight) * bar[e] + weight * first[e];
        }
        for (std::size_t i = 0; i < m; ++i)
        {
            r[h * size + i * m + i] = 1.0;
        }
    }
    return r;
}

}

std::optional<std::string> forecast_problem(const DccFitResult& fit, std::size_t horizon,
                                            const SeriesRegressors& future)
{
    if (fit.r == nullptr)
    {
        return "the DCC fit has no finite point to forecast from (see its status)";
    }
    if (fit.series_end == nullptr)
    {
        return "the DCC fit keeps no final state: fit it with the batch";
    }
    if (horizon == 0)
    {
        return "horizon must be at least 1";
    }
    if (future.size() != fit.series)
    {
        return "future regressors must have one entry per series (" + std::to_string(fit.series) + "), got " +
               std::to_string(future.size()) + " series";
    }
    for (std::size_t i = 0; i < fit.series; ++i)
    {
        if (std::optional<std::string> problem = series_problem((*fit.series_end)[i], future[i], horizon))
        {
            return "series " + std::to_string(i + 1) + ": " + *problem;
        }
    }
    return std::nullopt;
}

DccForecast forecast_dcc(const DccFitResult& fit, std::size_t horizon, const SeriesRegressors& future)
{
    const std::size_t m = fit.series;
    DccForecast out;
    out.horizon = horizon;
    out.series = m;
    out.sigma.resize(horizon * m);
    out.mean.resize(horizon * m);
    out.cumulative_mean.resize(horizon * m);
    out.weights.resize(horizon * m);
    for (std::size_t i = 0; i < m; ++i)
    {
        const SeriesEnd& end = (*fit.series_end)[i];
        const forecast::Forecast own = forecast::forecast(end.spec, end.estimate, end.state, horizon, future[i]);
        const std::vector<double> a = forecast::cumulative_weights(model::layout_of(end.spec), end.estimate, horizon);
        for (std::size_t h = 0; h < horizon; ++h)
        {
            out.sigma[h * m + i] = own.sigma[h];
            out.mean[h * m + i] = own.mean[h];
            out.cumulative_mean[h * m + i] = own.cumulative_mean[h];
            out.weights[h * m + i] = a[h];
        }
    }
    out.r = correlation_forecast(fit, horizon);
    out.h.resize(out.r.size());
    for (std::size_t h = 0; h < horizon; ++h)
    {
        const double* sigma = out.sigma.data() + h * m;
        for (std::size_t i = 0; i < m; ++i)
        {
            for (std::size_t j = 0; j < m; ++j)
            {
                const std::size_t k = (h * m + i) * m + j;
                out.h[k] = sigma[i] * out.r[k] * sigma[j];
            }
        }
    }
    return out;
}

namespace
{

// sum_{k=1..h} a_i(h-k) a_j(h-k) H_k,ij.
double cumulative_entry(const DccForecast& forecast, std::size_t h, std::size_t i, std::size_t j)
{
    const std::size_t m = forecast.series;
    double sum = 0.0;
    for (std::size_t k = 1; k <= h; ++k)
    {
        const std::size_t lag = (h - k) * m;
        sum += forecast.weights[lag + i] * forecast.weights[lag + j] * forecast.h[(k - 1) * m * m + i * m + j];
    }
    return sum;
}

}

std::vector<double> cumulative_covariance(const DccForecast& forecast)
{
    const std::size_t m = forecast.series;
    const std::size_t size = m * m;
    std::vector<double> c(forecast.horizon * size);
    // ponytail: O(h^2 m^2) with an ARMA/ARFIMA mean; without one every weight is 1 and each entry is a running
    // sum of H, the same additions in the same order as cumulative_entry.
    const bool plain = std::ranges::all_of(forecast.weights, [](double a) { return a == 1.0; });
    for (std::size_t h = 1; h <= forecast.horizon; ++h)
    {
        for (std::size_t i = 0; i < m; ++i)
        {
            for (std::size_t j = i; j < m; ++j)
            {
                const std::size_t entry = i * m + j;
                const double previous = h == 1 ? 0.0 : c[(h - 2) * size + entry];
                const double value = plain ? previous + forecast.h[(h - 1) * size + entry]
                                           : cumulative_entry(forecast, h, i, j);
                c[(h - 1) * size + i * m + j] = value;
                c[(h - 1) * size + j * m + i] = value;
            }
        }
    }
    return c;
}

}
