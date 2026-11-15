#pragma once

#include "dcc/fit.hpp"
#include "dcc/parameter_map.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
#include <string>
#include <vector>

namespace ve::capi::dcc
{

// The table a ve_dcc_fit reports: every parameter of the model, as ve_fit does, in the order of parameter_names. A
// fixed parameter keeps its value and has NaN standard errors; DccFitResult itself holds the estimated ones only.
struct DccTable
{
    std::vector<std::string> names;
    std::vector<double> estimate;
    std::vector<double> se_hessian;
    std::vector<double> se_robust;
};

// The estimated entries of `values` (one per estimated parameter) spread over every parameter; NaN where `given` has
// no estimated one.
inline std::vector<double> spread(const ve::dcc::DccLayout& layout, const std::vector<double>& values,
                                  const std::vector<double>& fixed)
{
    std::vector<double> all = fixed;
    std::size_t next = 0;
    for (std::size_t index = 0; index < all.size(); ++index)
    {
        if (layout.is_estimated(index))
        {
            all[index] = next < values.size() ? values[next++] : std::numeric_limits<double>::quiet_NaN();
        }
    }
    return all;
}

inline DccTable table_of(const ve::dcc::DccFitResult& result)
{
    const ve::dcc::DccLayout layout = ve::dcc::layout_of(result.spec, result.series);
    const std::vector<double> nan(layout.count(), std::numeric_limits<double>::quiet_NaN());
    DccTable table;
    table.names = ve::dcc::parameter_names(layout);
    table.estimate = spread(layout, result.estimate, layout.given);
    table.se_hessian = spread(layout, result.se_hessian, nan);
    table.se_robust = spread(layout, result.se_robust, nan);
    return table;
}

}
