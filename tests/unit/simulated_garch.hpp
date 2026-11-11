#pragma once

#include "core/spec.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numbers>
#include <random>
#include <vector>

namespace ve::testing
{

inline std::vector<double> simulate_garch_series(std::uint64_t seed, std::size_t count)
{
    std::mt19937_64 engine(seed);
    std::normal_distribution<double> shock(0.0, 1.0);
    const double mu = 0.05;
    const double omega = 0.01;
    const double alpha = 0.08;
    const double beta = 0.9;
    double sigma2 = omega / (1.0 - alpha - beta);
    std::vector<double> r(count);
    for (double& value : r)
    {
        const double residual = std::sqrt(sigma2) * shock(engine);
        value = mu + residual;
        sigma2 = omega + alpha * residual * residual + beta * sigma2;
    }
    return r;
}

// std::normal_distribution differs across standard libraries; Box-Muller on the engine's raw output gives the
// same series everywhere, up to the last bits of std::log and std::cos.
inline std::vector<double> simulate_portable_garch_series(std::uint64_t seed, std::size_t count)
{
    std::mt19937_64 engine(seed);
    auto uniform = [&engine] { return (static_cast<double>(engine() >> 11) + 0.5) * 0x1.0p-53; };
    const double mu = 0.05;
    const double omega = 0.01;
    const double alpha = 0.08;
    const double beta = 0.9;
    double sigma2 = omega / (1.0 - alpha - beta);
    std::vector<double> r(count);
    for (double& value : r)
    {
        const double radius = std::sqrt(-2.0 * std::log(uniform()));
        const double shock = radius * std::cos(2.0 * std::numbers::pi * uniform());
        const double residual = std::sqrt(sigma2) * shock;
        value = mu + residual;
        sigma2 = omega + alpha * residual * residual + beta * sigma2;
    }
    return r;
}

// ARMA(m, n)-sGARCH(p, q) with standardized (mean 0, variance 1) innovations. Lags before the first
// observation take the unconditional value: zero for the mean terms, omega / (1 - persistence) for the
// variance terms.
struct SimulatedModel
{
    double mu = 0.05;
    std::vector<double> ar;
    std::vector<double> ma;
    double omega = 0.01;
    std::vector<double> alpha{0.08};
    std::vector<double> beta{0.9};
    ve::Dist dist = ve::Dist::Norm;
    double shape = 0.0;
};

// Draws from the engine's raw output only, so the series is the same with every standard library. Each draw
// is its own statement: the evaluation order of operands is unspecified, and compilers consume the engine in
// different orders.
class PortableShocks
{
public:
    explicit PortableShocks(std::uint64_t seed) : engine_(seed)
    {
    }

    double uniform()
    {
        return (static_cast<double>(engine_() >> 11) + 0.5) * 0x1.0p-53;
    }

    double normal()
    {
        const double radius = std::sqrt(-2.0 * std::log(uniform()));
        return radius * std::cos(2.0 * std::numbers::pi * uniform());
    }

    // Marsaglia and Tsang (2000), unit scale; a shape below one is boosted by U^(1 / shape).
    double gamma(double shape)
    {
        if (shape < 1.0)
        {
            const double boosted = gamma(shape + 1.0);
            return boosted * std::pow(uniform(), 1.0 / shape);
        }
        const double d = shape - 1.0 / 3.0;
        const double c = 1.0 / std::sqrt(9.0 * d);
        while (true)
        {
            const double x = normal();
            const double v = std::pow(1.0 + c * x, 3.0);
            if (v > 0.0 && std::log(uniform()) < 0.5 * x * x + d - d * v + d * std::log(v))
            {
                return d * v;
            }
        }
    }

    double standardized(ve::Dist dist, double shape)
    {
        if (dist == ve::Dist::Std)
        {
            const double numerator = normal();
            return numerator * std::sqrt((shape - 2.0) / (2.0 * gamma(0.5 * shape)));
        }
        if (dist == ve::Dist::Ged)
        {
            const double lambda =
                std::sqrt(std::pow(2.0, -2.0 / shape) * std::tgamma(1.0 / shape) / std::tgamma(3.0 / shape));
            const double sign = uniform() < 0.5 ? -1.0 : 1.0;
            return sign * lambda * std::pow(2.0 * gamma(1.0 / shape), 1.0 / shape);
        }
        return normal();
    }

private:
    std::mt19937_64 engine_;
};

inline double lagged(const std::vector<double>& values, std::size_t t, std::size_t lag, double presample)
{
    if (t < lag)
    {
        return presample;
    }
    return values[t - lag];
}

inline double simulated_variance(const SimulatedModel& model, const std::vector<double>& squared_residuals,
                                 const std::vector<double>& sigma2, std::size_t t)
{
    double persistence = 0.0;
    for (double weight : model.alpha)
    {
        persistence += weight;
    }
    for (double weight : model.beta)
    {
        persistence += weight;
    }
    const double unconditional = model.omega / (1.0 - persistence);
    double variance = model.omega;
    for (std::size_t i = 0; i < model.alpha.size(); ++i)
    {
        variance += model.alpha[i] * lagged(squared_residuals, t, i + 1, unconditional);
    }
    for (std::size_t j = 0; j < model.beta.size(); ++j)
    {
        variance += model.beta[j] * lagged(sigma2, t, j + 1, unconditional);
    }
    return variance;
}

inline double simulated_mean(const SimulatedModel& model, const std::vector<double>& centered,
                             const std::vector<double>& residuals, std::size_t t)
{
    double mean = model.mu;
    for (std::size_t i = 0; i < model.ar.size(); ++i)
    {
        mean += model.ar[i] * lagged(centered, t, i + 1, 0.0);
    }
    for (std::size_t j = 0; j < model.ma.size(); ++j)
    {
        mean += model.ma[j] * lagged(residuals, t, j + 1, 0.0);
    }
    return mean;
}

inline std::vector<double> simulate_arma_garch(const SimulatedModel& model, std::uint64_t seed,
                                               std::size_t count)
{
    PortableShocks shocks(seed);
    std::vector<double> r(count);
    std::vector<double> centered(count);
    std::vector<double> residuals(count);
    std::vector<double> squared_residuals(count);
    std::vector<double> sigma2(count);
    for (std::size_t t = 0; t < count; ++t)
    {
        sigma2[t] = simulated_variance(model, squared_residuals, sigma2, t);
        residuals[t] = std::sqrt(sigma2[t]) * shocks.standardized(model.dist, model.shape);
        squared_residuals[t] = residuals[t] * residuals[t];
        r[t] = simulated_mean(model, centered, residuals, t) + residuals[t];
        centered[t] = r[t] - model.mu;
    }
    return r;
}

// C-order count x n matrix whose column j is the series simulated with seed j + 1.
inline std::vector<double> simulate_garch_matrix(std::size_t count, std::size_t n)
{
    std::vector<double> matrix(count * n);
    for (std::size_t j = 0; j < n; ++j)
    {
        const std::vector<double> series = simulate_garch_series(j + 1, count);
        for (std::size_t t = 0; t < count; ++t)
        {
            matrix[t * n + j] = series[t];
        }
    }
    return matrix;
}

}
