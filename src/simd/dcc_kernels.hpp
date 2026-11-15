#pragma once

#include <cstddef>
#include <span>

namespace ve::simd
{

// The kernels of this file multiply, divide and add in the order of the scalar loops they replace and never fuse, so
// they return the same bits as those loops on every target.

// acc[k] += weights[k] * x[k] for every k; the three spans have the same size. One term of the Q_t recursion.
void multiply_add(std::span<double> acc, std::span<const double> weights, std::span<const double> x);

// acc[i (i + 1) / 2 + j] += weights[i (i + 1) / 2 + j] * (v[i] * v[j]) for j <= i: a weighted term v v' of the Q_t
// recursion over the packed lower triangle, without forming v v'.
void add_outer(std::span<double> acc, std::span<const double> weights, std::span<const double> v);

// R = D Q D with D = diag(Q)^-1/2, from the packed lower triangle of Q (m (m + 1) / 2 entries): R row-major m x m
// with an exact unit diagonal and an exactly symmetric result. scale is m entries of scratch.
void correlation_from_q(std::span<const double> q, std::size_t m, std::span<double> scale, std::span<double> r);

}
