#pragma once

#include <cstddef>
#include <span>

namespace ve::linalg
{

// The largest eigenvalue of L^-1 N L^-T, L the Cholesky factor of the symmetric positive definite Q (the spectrum
// of Q^-1/2 N Q^-1/2), N symmetric positive semidefinite; power iteration to 1e-14 relative. n x n row-major.
// Returns NaN unless Q is positive definite.
double largest_relative_eigenvalue(std::span<const double> q, std::span<const double> n, std::size_t size);

}
