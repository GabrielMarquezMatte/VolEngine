#pragma once

#include <cstddef>
#include <span>
#include <vector>

namespace ve::linalg
{

// Minimum-norm solution of min ||y - A b||, with A rows x columns, row-major, and columns <= 9. Solves the normal
// equations by Cholesky; when A'A is not numerically positive definite, by the pseudo-inverse of A'A from its
// Jacobi eigendecomposition, with eigenvalues below 1e-12 times the largest cut.
std::vector<double> least_squares(std::span<const double> a, std::size_t columns, std::span<const double> y);

}
