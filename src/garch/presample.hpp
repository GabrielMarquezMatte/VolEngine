#pragma once

#include <cstddef>
#include <span>

// The steps run in a loop here and, for ARCH-in-mean, in a second one in the likelihood. A step with two callers
// is no longer inlined under GCC -O3, and an un-inlined step costs 30% on the Dual recursion of sGARCH.
#if defined(_MSC_VER) && !defined(__clang__)
#define VE_FORCE_INLINE __forceinline
#else
#define VE_FORCE_INLINE [[gnu::always_inline]] inline
#endif

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

// omega + c_t, the intercept of step t. An empty shift is the plain omega, with no term added.
template <class T>
T intercept_at(const T& omega, std::span<const T> shift, std::size_t t)
{
    if (shift.empty())
    {
        return omega;
    }
    return omega + shift[t];
}

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
