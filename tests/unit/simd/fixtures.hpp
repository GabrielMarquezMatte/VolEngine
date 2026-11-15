#pragma once

#include <cmath>
#include <cstddef>
#include <vector>

namespace ve::test
{

// A deterministic stream in (-1, 1); the kernels are checked on fixed data, not on a random seed.
class Stream
{
public:
    explicit Stream(unsigned seed) : state_(seed) {}

    double next()
    {
        state_ = state_ * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<double>(state_ >> 11) / 4503599627370496.0 - 1.0;
    }

    std::vector<double> vector(std::size_t n)
    {
        std::vector<double> x(n);
        for (double& v : x)
        {
            v = next();
        }
        return x;
    }

private:
    unsigned long long state_;
};

// Row-major n x n symmetric positive definite matrix, B B' / n + I with B random: condition number below 5.
inline std::vector<double> spd_matrix(Stream& stream, std::size_t n)
{
    const std::vector<double> b = stream.vector(n * n);
    std::vector<double> a(n * n);
    for (std::size_t i = 0; i < n; ++i)
    {
        for (std::size_t j = 0; j < n; ++j)
        {
            double sum = i == j ? 1.0 : 0.0;
            for (std::size_t k = 0; k < n; ++k)
            {
                sum += b[i * n + k] * b[j * n + k] / static_cast<double>(n);
            }
            a[i * n + j] = sum;
        }
    }
    return a;
}

}
