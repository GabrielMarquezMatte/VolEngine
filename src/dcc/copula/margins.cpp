#include "dcc/copula/margins.hpp"

#include "dist/cdf.hpp"
#include "model/layout.hpp"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <numeric>
#include <span>
#include <vector>

namespace ve::dcc::copula
{

namespace
{

constexpr double kRmgarchCeiling = 0.99999;
constexpr double kRmgarchFloor = 0x1p-52;

std::vector<double> column(const StandardizedPanel& panel, std::size_t i)
{
    std::vector<double> out(panel.t);
    for (std::size_t t = 0; t < panel.t; ++t)
    {
        out[t] = panel.z[t * panel.m + i];
    }
    return out;
}

void parametric(const Margin& margin, const StandardizedPanel& panel, std::size_t i, std::span<double> u)
{
    const model::Layout layout = model::layout_of(*margin.spec);
    const dist::ShapeParameters<double> shape = model::shape_parameters_of<double>(margin.estimate, layout);
    for (std::size_t t = 0; t < panel.t; ++t)
    {
        u[t * panel.m + i] = dist::cdf(layout.dist, shape, panel.z[t * panel.m + i]);
    }
}

// Standard: the position in a stable sort, + 1, over T + 1. Rmgarch: the ecdf, the last position of the value.
void empirical(const StandardizedPanel& panel, std::size_t i, std::span<double> u, Convention convention)
{
    const std::vector<double> z = column(panel, i);
    std::vector<std::size_t> order(panel.t);
    std::iota(order.begin(), order.end(), std::size_t{0});
    std::ranges::stable_sort(order, [&z](std::size_t left, std::size_t right) { return z[left] < z[right]; });
    const auto count = static_cast<double>(panel.t);
    for (std::size_t rank = 0; rank < panel.t; ++rank)
    {
        std::size_t last = rank;
        if (convention == Convention::Rmgarch)
        {
            while (last + 1 < panel.t && z[order[last + 1]] == z[order[rank]])
            {
                ++last;
            }
        }
        const double position = static_cast<double>(last + 1);
        u[order[rank] * panel.m + i] = convention == Convention::Rmgarch ? position / count : position / (count + 1.0);
    }
}

void cut(std::span<double> u, Convention convention)
{
    for (double& value : u)
    {
        if (convention == Convention::Rmgarch)
        {
            value = value > kRmgarchCeiling ? kRmgarchCeiling : value;
            value = value < 1.5 * kRmgarchFloor ? kRmgarchFloor : value;
            continue;
        }
        value = std::clamp(value, kUniformFloor, 1.0 - kUniformFloor);
    }
}

}

std::vector<double> uniforms(std::span<const FitResult* const> fits, const StandardizedPanel& panel, DccCopula how,
                             Convention convention)
{
    std::vector<Margin> margins;
    for (const FitResult* fit : fits)
    {
        margins.push_back({&fit->spec, fit->estimate});
    }
    return uniforms(std::span<const Margin>(margins), panel, how, convention);
}

std::vector<double> uniforms(std::span<const Margin> margins, const StandardizedPanel& panel, DccCopula how,
                             Convention convention)
{
    assert(how != DccCopula::None);
    std::vector<double> u(panel.t * panel.m);
    for (std::size_t i = 0; i < panel.m; ++i)
    {
        if (how == DccCopula::Parametric)
        {
            parametric(margins[i], panel, i, u);
        }
        if (how == DccCopula::Empirical)
        {
            empirical(panel, i, u, convention);
        }
    }
    cut(u, convention);
    return u;
}

}
