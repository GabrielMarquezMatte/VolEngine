#pragma once

#include "ad/special.hpp"

#include <array>
#include <cmath>
#include <cstddef>

namespace ve::ad
{

template <std::size_t N>
struct Dual
{
    double value = 0.0;
    std::array<double, N> grad{};
};

template <std::size_t N>
Dual<N> variable(double value, std::size_t index)
{
    Dual<N> result{value, {}};
    result.grad[index] = 1.0;
    return result;
}

template <std::size_t N>
Dual<N> constant(double value)
{
    return Dual<N>{value, {}};
}

inline double value_of(double x)
{
    return x;
}

template <std::size_t N>
double value_of(const Dual<N>& x)
{
    return x.value;
}

// Value v with derivative slope * dx.
template <std::size_t N>
Dual<N> chain(const Dual<N>& x, double v, double slope)
{
    Dual<N> result{v, {}};
    for (std::size_t i = 0; i < N; ++i)
    {
        result.grad[i] = slope * x.grad[i];
    }
    return result;
}

// Value v with derivative da * a' + db * b'.
template <std::size_t N>
Dual<N> chain(const Dual<N>& a, const Dual<N>& b, double v, double da, double db)
{
    Dual<N> result{v, {}};
    for (std::size_t i = 0; i < N; ++i)
    {
        result.grad[i] = da * a.grad[i] + db * b.grad[i];
    }
    return result;
}

template <std::size_t N>
Dual<N> operator-(const Dual<N>& x)
{
    return chain(x, -x.value, -1.0);
}

template <std::size_t N>
Dual<N> operator+(const Dual<N>& a, const Dual<N>& b)
{
    return chain(a, b, a.value + b.value, 1.0, 1.0);
}

template <std::size_t N>
Dual<N> operator-(const Dual<N>& a, const Dual<N>& b)
{
    return chain(a, b, a.value - b.value, 1.0, -1.0);
}

template <std::size_t N>
Dual<N> operator*(const Dual<N>& a, const Dual<N>& b)
{
    return chain(a, b, a.value * b.value, b.value, a.value);
}

// The derivative is taken as (a' - q b') / b with q = a / b: the factor q / b of a' / b - (q / b) b'
// underflows or overflows on its own once |b| passes ~1e154.
template <std::size_t N>
Dual<N> operator/(const Dual<N>& a, const Dual<N>& b)
{
    double quotient = a.value / b.value;
    Dual<N> result{quotient, {}};
    for (std::size_t i = 0; i < N; ++i)
    {
        result.grad[i] = (a.grad[i] - quotient * b.grad[i]) / b.value;
    }
    return result;
}

template <std::size_t N>
Dual<N> operator+(const Dual<N>& a, double b)
{
    return chain(a, a.value + b, 1.0);
}

template <std::size_t N>
Dual<N> operator+(double a, const Dual<N>& b)
{
    return b + a;
}

template <std::size_t N>
Dual<N> operator-(const Dual<N>& a, double b)
{
    return chain(a, a.value - b, 1.0);
}

template <std::size_t N>
Dual<N> operator-(double a, const Dual<N>& b)
{
    return chain(b, a - b.value, -1.0);
}

template <std::size_t N>
Dual<N> operator*(const Dual<N>& a, double b)
{
    return chain(a, a.value * b, b);
}

template <std::size_t N>
Dual<N> operator*(double a, const Dual<N>& b)
{
    return b * a;
}

template <std::size_t N>
Dual<N> operator/(const Dual<N>& a, double b)
{
    return chain(a, a.value / b, 1.0 / b);
}

template <std::size_t N>
Dual<N> operator/(double a, const Dual<N>& b)
{
    double quotient = a / b.value;
    Dual<N> result{quotient, {}};
    for (std::size_t i = 0; i < N; ++i)
    {
        result.grad[i] = -(quotient * b.grad[i]) / b.value;
    }
    return result;
}

template <std::size_t N>
bool operator<(const Dual<N>& a, const Dual<N>& b)
{
    return a.value < b.value;
}

template <std::size_t N>
bool operator>(const Dual<N>& a, const Dual<N>& b)
{
    return a.value > b.value;
}

template <std::size_t N>
bool operator<=(const Dual<N>& a, const Dual<N>& b)
{
    return a.value <= b.value;
}

template <std::size_t N>
bool operator>=(const Dual<N>& a, const Dual<N>& b)
{
    return a.value >= b.value;
}

template <std::size_t N>
Dual<N> log(const Dual<N>& x)
{
    return chain(x, std::log(x.value), 1.0 / x.value);
}

template <std::size_t N>
Dual<N> log1p(const Dual<N>& x)
{
    return chain(x, std::log1p(x.value), 1.0 / (1.0 + x.value));
}

template <std::size_t N>
Dual<N> exp(const Dual<N>& x)
{
    double e = std::exp(x.value);
    return chain(x, e, e);
}

template <std::size_t N>
Dual<N> sqrt(const Dual<N>& x)
{
    double root = std::sqrt(x.value);
    return chain(x, root, 0.5 / root);
}

template <std::size_t N>
Dual<N> abs(const Dual<N>& x)
{
    double sign = (x.value > 0.0) - (x.value < 0.0);
    return chain(x, std::fabs(x.value), sign);
}

namespace detail
{

// e x^(e - 1) for v = x^e: e v / x saves a second pow wherever v is a normal number; at x = 0, or where v under-
// or overflows, x^(e - 1) is computed on its own.
inline double pow_slope(double x, double exponent, double v)
{
    if (std::isnormal(v))
    {
        return exponent * (v / x);
    }
    return exponent * std::pow(x, exponent - 1.0);
}

}

template <std::size_t N>
Dual<N> pow(const Dual<N>& x, double exponent)
{
    double v = std::pow(x.value, exponent);
    return chain(x, v, detail::pow_slope(x.value, exponent, v));
}

// A slot skips the term of an operand whose gradient is zero there, and x^e * log(x) takes its limit 0 at x =
// 0, so a zero or negative base keeps finite derivatives wherever they exist. With both slopes finite no term
// needs skipping, and the loop runs without branches.
template <std::size_t N>
Dual<N> pow(const Dual<N>& x, const Dual<N>& exponent)
{
    double v = std::pow(x.value, exponent.value);
    double dx = detail::pow_slope(x.value, exponent.value, v);
    double de = 0.0;
    if (x.value != 0.0)
    {
        de = v * std::log(x.value);
    }
    Dual<N> result{v, {}};
    if (std::isfinite(dx) && std::isfinite(de))
    {
        for (std::size_t i = 0; i < N; ++i)
        {
            result.grad[i] = dx * x.grad[i] + de * exponent.grad[i];
        }
        return result;
    }
    for (std::size_t i = 0; i < N; ++i)
    {
        if (x.grad[i] != 0.0)
        {
            result.grad[i] += dx * x.grad[i];
        }
        if (exponent.grad[i] != 0.0)
        {
            result.grad[i] += de * exponent.grad[i];
        }
    }
    return result;
}

template <std::size_t N>
Dual<N> lgamma(const Dual<N>& x)
{
    return chain(x, std::lgamma(x.value), digamma(x.value));
}

}
