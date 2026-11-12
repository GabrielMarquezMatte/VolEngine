#pragma once

#include <cstddef>
#include <span>

namespace ve::model
{

// T x columns, row-major: row t holds the regressors of observation t.
struct RegressorMatrix
{
    std::span<const double> values;
    std::size_t columns = 0;

    double at(std::size_t t, std::size_t k) const { return values[t * columns + k]; }
};

struct Data
{
    std::span<const double> r;
    RegressorMatrix mean{};
    RegressorMatrix variance{};
};

}
