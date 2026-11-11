#pragma once

#include <cstddef>
#include <span>

namespace ve::garch
{

// For t < held the recursion holds its state at value; from t = held on it runs, and a lag before t = 0 reads the
// expected value of its term given value. value is always a variance: each recursion derives its own state from it
// (csGARCH holds its transitory part, sigma2 - q, at value).
template <class T>
struct Presample
{
    T value;
    std::size_t held;
};

template <class T>
const T& lagged(std::span<const T> values, std::size_t t, std::size_t lag, const T& presample)
{
    if (lag > t)
    {
        return presample;
    }
    return values[t - lag];
}

// e^2 of lag `lag`; before t = 0, its expected value, the presample variance.
template <class T>
T lagged_square(std::span<const T> residuals, std::size_t t, std::size_t lag, const T& presample)
{
    if (lag > t)
    {
        return presample;
    }
    const T& e = residuals[t - lag];
    return e * e;
}

}
