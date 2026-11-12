#pragma once

#include <cstddef>
#include <span>

namespace ve::linalg
{

// Row-major n x n symmetric matrix; reads and overwrites the lower triangle with L, where A = L L^T.
// Returns false, leaving the contents unspecified, unless A is positive definite.
bool cholesky_factor(std::span<double> a, std::size_t n);

// Solves A x = b in place given the factor from cholesky_factor.
void cholesky_solve(std::span<const double> factor, std::size_t n, std::span<double> b);

}
