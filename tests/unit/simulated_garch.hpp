#pragma once

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

// std::normal_distribution differs across standard libraries; Box-Muller on the engine's raw output gives the same
// series everywhere, up to the last bits of std::log and std::cos.
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
