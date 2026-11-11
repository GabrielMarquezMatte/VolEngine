#pragma once

#include <cstddef>
#include <span>

namespace ve::garch
{

// sigma2_t = value for t < held; from t = held on the recursion runs, and a lag before t = 0 reads the expected
// value of its term given value. value is always a variance: each recursion derives its own state from it.
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

}
