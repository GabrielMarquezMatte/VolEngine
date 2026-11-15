#pragma once

#include <cstddef>
#include <span>
#include <vector>

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

// log_det_and_quadratic for at most kCapacity matrices that keeps the factors, so that the caller can take the
// derivative of its density in log |R_t| and z_t' R_t^-1 z_t, and pull it back to Q_t without factoring again.
// factor() gives the same values as log_det_and_quadratic, to the bit.
class CorrelationBlock
{
public:
    // A whole number of vectors at every target (2, 4 or 8 lanes).
    static constexpr std::size_t kCapacity = 8;

    // Blocks of m x m matrices.
    explicit CorrelationBlock(std::size_t m);

    void factor(std::span<const double> q, std::span<const double> z, std::span<double> log_det,
                std::span<double> quadratic);

    // For the matrices of the last factor() and a function l_t of (log |R_t|, z_t' R_t^-1 z_t) with partials
    // d_log_det[t] and d_quadratic[t]: g[t] is the packed lower triangle (m (m + 1) / 2 entries, one matrix after the
    // other) of the gradient of l_t with respect to the symmetric Q_t, counting an off-diagonal pair once: the
    // derivative along a symmetric dQ_t is the packed dot product of g[t] and dQ_t. Meaningless for the matrices
    // that were not positive definite.
    void pull_back(std::span<const double> d_log_det, std::span<const double> d_quadratic, std::span<double> g);

private:
    std::size_t m_;
    std::size_t count_ = 0;
    std::vector<double> state_;
    std::vector<double> scratch_;
};

}
