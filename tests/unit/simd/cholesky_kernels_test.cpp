#include "simd/cholesky_kernels.hpp"

#include "linalg/cholesky.hpp"
#include "simd/fixtures.hpp"
#include "simd/targets.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <gtest/gtest.h>
#include <limits>
#include <span>
#include <vector>

namespace
{

// The kernel factors with reciprocal pivots, fused multiply-adds on AVX2 and the vector log, so its log |R| and
// z' R^-1 z differ from linalg's by rounding: relative error (against max(|x|, 1)) under this on matrices of
// condition number below 5.
constexpr double kTolerance = 1e-13;

const std::size_t kSizes[] = {1, 2, 3, 5, 21};
// Below, at, and above the lane counts of every target (2, 4 and 8), and a few blocks.
const std::size_t kCounts[] = {1, 2, 3, 4, 5, 7, 8, 9, 33};

std::size_t packed_count(std::size_t m)
{
    return m * (m + 1) / 2;
}

std::size_t packed_index(std::size_t i, std::size_t j)
{
    return i * (i + 1) / 2 + j;
}

// count matrices Q_t packed lower, one after the other, and their vectors z_t.
struct Batch
{
    std::size_t m;
    std::vector<double> q;
    std::vector<double> z;
};

Batch make_batch(ve::test::Stream& stream, std::size_t m, std::size_t count)
{
    Batch batch{m, {}, {}};
    for (std::size_t t = 0; t < count; ++t)
    {
        const std::vector<double> full = ve::test::spd_matrix(stream, m);
        for (std::size_t i = 0; i < m; ++i)
        {
            for (std::size_t j = 0; j <= i; ++j)
            {
                batch.q.push_back(full[i * m + j]);
            }
        }
    }
    batch.z = stream.vector(count * m);
    return batch;
}

// R = D Q D with an exact unit diagonal, as the scalar recursion forms it, in full.
std::vector<double> correlation_of(std::span<const double> q, std::size_t m)
{
    std::vector<double> r(m * m);
    for (std::size_t i = 0; i < m; ++i)
    {
        r[i * m + i] = 1.0;
        for (std::size_t j = 0; j < i; ++j)
        {
            const double scale_i = 1.0 / std::sqrt(q[packed_index(i, i)]);
            const double scale_j = 1.0 / std::sqrt(q[packed_index(j, j)]);
            r[i * m + j] = q[packed_index(i, j)] * scale_i * scale_j;
            r[j * m + i] = r[i * m + j];
        }
    }
    return r;
}

struct Reference
{
    double log_det;
    double quadratic;
};

// linalg's own factor, solve and log determinant of R; NaN for both when R is not positive definite.
Reference reference_of(std::span<const double> q, std::span<const double> z, std::size_t m)
{
    std::vector<double> factor = correlation_of(q, m);
    if (!ve::linalg::cholesky_factor(factor, m))
    {
        return {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN()};
    }
    std::vector<double> solution(z.begin(), z.end());
    ve::linalg::cholesky_solve(factor, m, solution);
    double quadratic = 0.0;
    for (std::size_t i = 0; i < m; ++i)
    {
        quadratic += z[i] * solution[i];
    }
    return {ve::linalg::cholesky_log_determinant<double>(factor, m), quadratic};
}

struct Output
{
    std::vector<double> log_det;
    std::vector<double> quadratic;
};

Output run(const Batch& batch)
{
    const std::size_t count = batch.z.size() / batch.m;
    Output out{std::vector<double>(count), std::vector<double>(count)};
    ve::simd::log_det_and_quadratic(batch.q, batch.z, batch.m, out.log_det, out.quadratic);
    return out;
}

TEST(CholeskyKernels, LogDetAndQuadraticMatchTheScalarFactorOnEveryTarget)
{
    for (const auto target : ve::test::runnable_targets())
    {
        const ve::test::ForcedTarget forced(target);
        ve::test::Stream stream(7);
        for (const std::size_t m : kSizes)
        {
            for (const std::size_t count : kCounts)
            {
                const Batch batch = make_batch(stream, m, count);
                const Output out = run(batch);
                for (std::size_t t = 0; t < count; ++t)
                {
                    const std::span<const double> q(batch.q.data() + t * packed_count(m), packed_count(m));
                    const std::span<const double> z(batch.z.data() + t * m, m);
                    const Reference expected = reference_of(q, z, m);
                    EXPECT_NEAR(out.log_det[t], expected.log_det, kTolerance * std::max(std::fabs(expected.log_det), 1.0))
                        << "m = " << m << " count = " << count << " t = " << t;
                    EXPECT_NEAR(out.quadratic[t], expected.quadratic,
                                kTolerance * std::max(std::fabs(expected.quadratic), 1.0))
                        << "m = " << m << " count = " << count << " t = " << t;
                }
            }
        }
    }
}

TEST(CholeskyKernels, ANotPositiveDefiniteMatrixGivesNanAloneInItsBatch)
{
    const double nan = std::numeric_limits<double>::quiet_NaN();
    for (const auto target : ve::test::runnable_targets())
    {
        const ve::test::ForcedTarget forced(target);
        ve::test::Stream stream(13);
        for (const std::size_t m : {2U, 3U, 5U, 21U})
        {
            Batch batch = make_batch(stream, m, 9);
            // t = 2: correlations of 1.5 make the second leading minor negative; t = 5: a NaN on the diagonal;
            // t = 8: a zero on the diagonal, the last matrix of the batch (a partial block on every target).
            const std::size_t cells = packed_count(m);
            for (std::size_t i = 0; i < m; ++i)
            {
                for (std::size_t j = 0; j <= i; ++j)
                {
                    batch.q[2 * cells + packed_index(i, j)] = i == j ? 1.0 : 1.5;
                }
            }
            batch.q[5 * cells + packed_index(m - 1, m - 1)] = nan;
            batch.q[8 * cells] = 0.0;
            const Output out = run(batch);
            for (std::size_t t = 0; t < 9; ++t)
            {
                const bool bad = t == 2 || t == 5 || t == 8;
                EXPECT_EQ(std::isnan(out.log_det[t]), bad) << "m = " << m << " t = " << t;
                EXPECT_EQ(std::isnan(out.quadratic[t]), bad) << "m = " << m << " t = " << t;
            }
        }
    }
}

// The same matrix gives the same bits whatever lane, batch or neighbours it comes with: a term never depends on
// where it falls in the sample, so the thread count and the block boundaries cannot change a result.
TEST(CholeskyKernels, AResultDoesNotDependOnItsPositionInTheBatch)
{
    for (const auto target : ve::test::runnable_targets())
    {
        const ve::test::ForcedTarget forced(target);
        ve::test::Stream stream(17);
        for (const std::size_t m : {3U, 21U})
        {
            const Batch batch = make_batch(stream, m, 9);
            const Output together = run(batch);
            const std::size_t cells = packed_count(m);
            for (std::size_t t = 0; t < 9; ++t)
            {
                const Batch alone{m,
                                  {batch.q.begin() + static_cast<std::ptrdiff_t>(t * cells),
                                   batch.q.begin() + static_cast<std::ptrdiff_t>((t + 1) * cells)},
                                  {batch.z.begin() + static_cast<std::ptrdiff_t>(t * m),
                                   batch.z.begin() + static_cast<std::ptrdiff_t>((t + 1) * m)}};
                const Output single = run(alone);
                EXPECT_EQ(single.log_det[0], together.log_det[t]) << "m = " << m << " t = " << t;
                EXPECT_EQ(single.quadratic[0], together.quadratic[t]) << "m = " << m << " t = " << t;
            }
        }
    }
}

}
