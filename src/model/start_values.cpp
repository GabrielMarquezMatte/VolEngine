#include "model/start_values.hpp"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <numeric>

namespace ve::model
{

namespace
{

constexpr double kInitialArch = 0.05;
constexpr double kInitialGarch = 0.90;
constexpr double kInitialStdShape = 8.0;
constexpr double kInitialGedShape = 2.0;

double sample_variance(std::span<const double> r, double mean)
{
    double sum = 0.0;
    for (double value : r)
    {
        sum += (value - mean) * (value - mean);
    }
    return sum / static_cast<double>(r.size());
}

double sample_mean(std::span<const double> r)
{
    return std::accumulate(r.begin(), r.end(), 0.0) / static_cast<double>(r.size());
}

}

Standardization standardization_of(std::span<const double> r)
{
    double mean = sample_mean(r);
    return {mean, std::sqrt(sample_variance(r, mean))};
}

std::vector<double> start_values(std::span<const double> r, const Layout& layout)
{
    assert(!r.empty());
    double mean = sample_mean(r);
    std::vector<double> theta(layout.count(), 0.0);
    if (layout.include_mean)
    {
        theta[layout.mu()] = mean;
    }
    theta[layout.omega()] = sample_variance(r, mean) * (1.0 - (kInitialArch + kInitialGarch));
    for (int i = 0; i < layout.arch; ++i)
    {
        theta[layout.alpha_begin() + static_cast<std::size_t>(i)] = kInitialArch / layout.arch;
    }
    for (int j = 0; j < layout.garch; ++j)
    {
        theta[layout.beta_begin() + static_cast<std::size_t>(j)] = kInitialGarch / layout.garch;
    }
    if (layout.has_shape())
    {
        theta[layout.shape()] = layout.dist == Dist::Std ? kInitialStdShape : kInitialGedShape;
    }
    return theta;
}

}
