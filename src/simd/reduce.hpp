#pragma once

#include <span>

namespace ve::simd
{

// Sum of x, accumulated in vector lanes; the order of the additions depends on the target the machine dispatches to.
double sum(std::span<const double> x);

// Sum of x[k] y[k], the two spans of the same size, accumulated like sum.
double dot(std::span<const double> x, std::span<const double> y);

}
