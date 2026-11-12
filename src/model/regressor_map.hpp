#pragma once

#include "model/layout.hpp"
#include "model/parameter_map.hpp"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <span>

namespace ve::model
{

// archm, mxreg_k and vxreg_k: the Location entries that scale with the data.
inline bool is_regressor(const Layout& layout, std::size_t index)
{
    bool archm = layout.has_archm() && index == layout.archm_index();
    bool mean = index >= layout.mxreg_begin() && index < layout.omega();
    bool variance = index >= layout.vxreg_begin() && index < layout.skew();
    return archm || mean || variance;
}

// The scale S of the variance intercept: s^2 (sGARCH, iGARCH, gjrGARCH, csGARCH), 1 (eGARCH), s^delta with the
// natural delta (apARCH, fGARCH).
template <class T>
T intercept_scale(std::span<const T> theta, const Layout& layout, double scale)
{
    using std::exp;
    if (layout.model == Model::EGarch)
    {
        return T{1.0};
    }
    if (layout.is_power_family())
    {
        return exp(power_of(theta, layout) * std::log(scale));
    }
    return T{scale * scale};
}

// natural = u * regressor_scale; theta supplies the natural delta of the power family.
// archm: 1 (p = 1), 1/s (p = 2); mxreg_k: s/s_x,k; vxreg_k: S/s_v,k.
template <class T>
T regressor_scale(std::span<const T> theta, const Layout& layout, const Standardization& standardization,
                  std::size_t index)
{
    assert(is_regressor(layout, index));
    double scale = standardization.scale;
    if (layout.has_archm() && index == layout.archm_index())
    {
        return T{layout.archm == 1 ? 1.0 : 1.0 / scale};
    }
    if (index < layout.omega())
    {
        return T{scale / standardization.mean_regressor_scale[index - layout.mxreg_begin()]};
    }
    return intercept_scale<T>(theta, layout, scale) /
           standardization.variance_regressor_scale[index - layout.vxreg_begin()];
}

}
