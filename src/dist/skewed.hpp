#pragma once

#include "ad/dual.hpp"
#include "core/spec.hpp"
#include "dist/symmetric.hpp"

#include <cmath>
#include <numbers>

namespace ve::dist
{

// The Fernandez-Steel skewing of the std (sstd) or ged (sged) kernel g, standardized as rugarch does:
//   log f(z) = log_constant + log g(y / skew) for y >= 0 and log g(y skew) for y < 0, y = shift + stretch z,
// with shift = mu_xi, stretch = sigma_xi and log_constant = log(2 / (xi + 1 / xi)) + log sigma_xi.
// arch's Hansen skew-t is the same family: xi = sqrt((1 + lambda) / (1 - lambda)), dxi/dlambda =
// 1 / (xi (1 - lambda)^2).
template <class T>
struct SkewedLogDensity
{
    LogDensity<T> kernel;
    T log_constant;
    T shift;
    T stretch;
    T skew;
};

namespace detail
{

template <class T>
LogDensity<T> skewed_kernel(Dist dist, const T& shape)
{
    if (dist == Dist::Sstd)
    {
        return student_log_density(shape);
    }
    return ged_log_density(shape);
}

// E|x| under the kernel.
template <class T>
T kernel_abs_mean(Dist dist, const T& shape)
{
    if (dist == Dist::Sstd)
    {
        return student_abs_moment(shape, T{1.0});
    }
    return ged_abs_moment(shape, T{1.0});
}

}

// dist is Sstd or Sged.
template <class T>
SkewedLogDensity<T> skewed_log_density(Dist dist, const T& skew, const T& shape)
{
    using std::log;
    using std::sqrt;
    T inverse = 1.0 / skew;
    T m1 = detail::kernel_abs_mean(dist, shape);
    T m1_squared = m1 * m1;
    T shift = m1 * (skew - inverse);
    T variance = (1.0 - m1_squared) * (skew * skew + inverse * inverse) + 2.0 * m1_squared - 1.0;
    T log_constant = std::numbers::ln2 - log(skew + inverse) + 0.5 * log(variance);
    return {detail::skewed_kernel(dist, shape), log_constant, shift, sqrt(variance), skew};
}

template <class T>
T log_density(const SkewedLogDensity<T>& density, const T& z)
{
    T y = density.shift + density.stretch * z;
    if (ad::value_of(y) >= 0.0)
    {
        return density.log_constant + log_density(density.kernel, y / density.skew);
    }
    return density.log_constant + log_density(density.kernel, y * density.skew);
}

}
