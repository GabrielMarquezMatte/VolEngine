#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
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
