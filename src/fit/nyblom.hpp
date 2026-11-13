#pragma once

#include <array>
#include <cstddef>
#include <limits>
#include <optional>
#include <span>
#include <vector>

namespace ve::fit
{

struct Nyblom
{
    std::vector<double> individual;
    double joint = std::numeric_limits<double>::quiet_NaN();
    std::size_t estimated = 0;
};

// Scores are row-major; fixed and derived parameter columns are excluded by estimated.
Nyblom nyblom_statistics(std::span<const double> scores, std::size_t k, std::span<const bool> estimated);
std::optional<std::array<double, 3>> nyblom_critical(std::size_t parameters);

}
