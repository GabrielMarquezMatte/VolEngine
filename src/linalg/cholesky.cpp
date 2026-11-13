#include "linalg/cholesky.hpp"

namespace ve::linalg
{

bool cholesky_factor(std::span<double> a, std::size_t n)
{
    return cholesky_factor<double>(a, n);
}

void cholesky_solve(std::span<const double> factor, std::size_t n, std::span<double> b)
{
    cholesky_solve<double>(factor, n, b);
}

}
