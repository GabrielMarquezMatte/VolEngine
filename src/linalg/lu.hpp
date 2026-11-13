#pragma once

#include <cstddef>
#include <span>

namespace ve::linalg
{

// Row-major n x n matrix, any sign pattern: overwritten with P A = L U (L unit lower, below the diagonal; U on and
// above it), pivots[k] the row swapped into row k. Returns false, leaving the contents unspecified, when a pivot is
// zero or not finite.
bool lu_factor(std::span<double> a, std::span<std::size_t> pivots, std::size_t n);

// Solves A x = b in place given the factor and pivots from lu_factor.
void lu_solve(std::span<const double> factor, std::span<const std::size_t> pivots, std::size_t n,
              std::span<double> b);

}
