#pragma once

#include "ad/dual.hpp"
#include "core/spec.hpp"
#include "dist/hyperbolic.hpp"
#include "dist/johnson.hpp"
#include "dist/skewed.hpp"
#include "dist/symmetric.hpp"

#include <cmath>
#include <variant>

namespace ve::dist
{

// Distribution parameters of one fit; the ones its distribution lacks are 0.
template <class T>
struct ShapeParameters
{
    T skew{0.0};
    T shape{0.0};
    T ghlambda{0.0};
};

template <class T>
using Density =
    std::variant<LogDensity<T>, SkewedLogDensity<T>, HyperbolicLogDensity<T>, JohnsonLogDensity<T>>;

// norm, std and ged get the LogDensity of make_log_density, so their numbers do not change.
template <class T>
Density<T> make_density(Dist dist, const ShapeParameters<T>& parameters)
{
    switch (dist)
    {
        case Dist::Sstd:
        case Dist::Sged:
            return skewed_log_density(dist, parameters.skew, parameters.shape);
        case Dist::Nig:
            return hyperbolic_log_density(parameters.skew, parameters.shape, T{-0.5}, false);
        case Dist::Ghyp:
            return hyperbolic_log_density(parameters.skew, parameters.shape, parameters.ghlambda, true);
        case Dist::Jsu:
            return johnson_log_density(parameters.skew, parameters.shape);
        case Dist::Norm:
        case Dist::Std:
        case Dist::Ged:
            break;
    }
    return make_log_density(dist, parameters.shape);
}

// A loop over observations visits once, outside the loop, and calls the log_density of the alternative.
template <class T>
T log_density(const Density<T>& density, const T& z)
{
    return std::visit([&z](const auto& alternative) { return log_density(alternative, z); }, density);
}

// The point where the density is not smooth: z* = -shift / stretch for the Fernandez-Steel densities, 0 for
// the others.
template <class T>
T density_kink(const Density<T>& density)
{
    const SkewedLogDensity<T>* skewed = std::get_if<SkewedLogDensity<T>>(&density);
    if (skewed == nullptr)
    {
        return T{0.0};
    }
    return -(skewed->shift / skewed->stretch);
}

// The parameter domains of the "Distribuições" table; the parameters a distribution lacks are not read.
inline bool in_domain(Dist dist, const ShapeParameters<double>& parameters)
{
    bool positive_skew = parameters.skew > 0.0;
    bool correlation_skew = std::fabs(parameters.skew) < 1.0;
    bool positive_shape = parameters.shape > 0.0;
    switch (dist)
    {
        case Dist::Sstd:
            return positive_skew && parameters.shape > 2.0;
        case Dist::Sged:
            return positive_skew && positive_shape;
        case Dist::Nig:
            return correlation_skew && positive_shape;
        case Dist::Ghyp:
            return correlation_skew && positive_shape && std::isfinite(parameters.ghlambda);
        case Dist::Jsu:
            return std::isfinite(parameters.skew) && positive_shape;
        case Dist::Norm:
        case Dist::Std:
        case Dist::Ged:
            break;
    }
    return in_domain(dist, parameters.shape);
}

}
