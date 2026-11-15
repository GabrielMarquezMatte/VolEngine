#pragma once

#include <cstddef>
#include <span>

namespace ve::simd
{

// For count matrices Q_t (the packed lower triangle, m (m + 1) / 2 entries, one after the other in q) and count
// vectors z_t (m each, one after the other in z): R_t = D Q_t D with D = diag(Q_t)^-1/2 and an exact unit diagonal,
// then log_det[t] = log |R_t| and quadratic[t] = z_t' R_t^-1 z_t from the Cholesky factor of R_t. Both are NaN, and
// only those, when R_t is not positive definite or the diagonal of Q_t is not positive.
//
// The vector lanes hold different t, not different elements of one matrix: a factor is a chain of short dependent
// inner products that no vector width shortens, while the matrices of different t are independent. Every lane does
// the same operations in the same order, so a result does not depend on the lane, the batch or the other matrices.
// It differs from linalg::cholesky_* by rounding only: the pivots are reciprocals, the sums are fused on AVX2 and the
// log is the vector one.
void log_det_and_quadratic(std::span<const double> q, std::span<const double> z, std::size_t m,
                           std::span<double> log_det, std::span<double> quadratic);

}
