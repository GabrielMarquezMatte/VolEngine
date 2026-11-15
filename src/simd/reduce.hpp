#pragma once

#include <span>

namespace ve::simd
{

// Sum of x, accumulated in vector lanes; the order of the additions depends on the target the machine dispatches to.
double sum(std::span<const double> x);

}
