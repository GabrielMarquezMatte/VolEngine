#pragma once

#include "core/spec.hpp"
#include "simulated_garch.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace ve::bench
{

inline double innovation_shape(ve::Dist dist)
{
    if (dist == ve::Dist::Std)
    {
        return 8.0;
    }
    if (dist == ve::Dist::Ged)
    {
        return 1.5;
    }
    return 0.0;
}

// A data-generating process with the structure of spec: persistence 0.98 split evenly over the ARCH and GARCH lags, and
// ARMA terms of moderate size. Built on the portable generators of the unit tests, so every standard library produces
// the same series.
inline ve::testing::SimulatedModel simulated_model_for(const ve::Spec& spec)
{
    ve::testing::SimulatedModel model;
    const std::vector<double> ar{0.3, -0.1, 0.05, 0.02, 0.01, 0.01};
    const std::vector<double> ma{0.2, 0.1, 0.05, 0.02, 0.01, 0.01};
    model.ar.assign(ar.begin(), ar.begin() + spec.ar);
    model.ma.assign(ma.begin(), ma.begin() + spec.ma);
    model.alpha.assign(static_cast<std::size_t>(spec.p), 0.08 / spec.p);
    model.beta.assign(static_cast<std::size_t>(spec.q), 0.9 / spec.q);
    model.dist = spec.dist;
    model.shape = innovation_shape(spec.dist);
    return model;
}

inline std::vector<double> simulate_series(const ve::Spec& spec, std::uint64_t seed, std::size_t count)
{
    return ve::testing::simulate_arma_garch(simulated_model_for(spec), seed, count);
}

// C-order count x n matrix whose column j is the series simulated from model with seed j + 1.
inline std::vector<double> simulate_panel(const ve::testing::SimulatedModel& model, std::size_t count, std::size_t n)
{
    std::vector<double> matrix(count * n);
    for (std::size_t j = 0; j < n; ++j)
    {
        const std::vector<double> series = ve::testing::simulate_arma_garch(model, j + 1, count);
        for (std::size_t t = 0; t < count; ++t)
        {
            matrix[t * n + j] = series[t];
        }
    }
    return matrix;
}

}
