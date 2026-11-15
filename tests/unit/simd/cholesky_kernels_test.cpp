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

// The pullback of l = f(log |R|, z' R^-1 z) to the packed Q, by the formulas of the scalar code written out in full:
// G = a R^-1 - b w w' with w = R^-1 z, then through R = D Q D (an exact unit diagonal): 2 G_ij s_i s_j below the
// diagonal and - (sum_(j != i) G_ij R_ij) / Q_ii on it.
std::vector<double> reference_pullback(std::span<const double> q, std::span<const double> z, std::size_t m, double a,
                                       double b)
{
    const std::vector<double> r = correlation_of(q, m);
    std::vector<double> factor = r;
    EXPECT_TRUE(ve::linalg::cholesky_factor(factor, m));
    std::vector<double> inverse(m * m);
    for (std::size_t j = 0; j < m; ++j)
    {
        std::vector<double> column(m, 0.0);
        column[j] = 1.0;
        ve::linalg::cholesky_solve(factor, m, column);
        for (std::size_t i = 0; i < m; ++i)
        {
            inverse[i * m + j] = column[i];
        }
    }
    std::vector<double> w(z.begin(), z.end());
    ve::linalg::cholesky_solve(factor, m, w);
    std::vector<double> g(packed_count(m));
    for (std::size_t i = 0; i < m; ++i)
    {
        double diagonal = 0.0;
        for (std::size_t j = 0; j < m; ++j)
        {
            const double gradient = a * inverse[i * m + j] - b * w[i] * w[j];
            if (j == i)
            {
                continue;
            }
            diagonal += gradient * r[i * m + j];
            if (j < i)
            {
                g[packed_index(i, j)] =
                    2.0 * gradient / std::sqrt(q[packed_index(i, i)]) / std::sqrt(q[packed_index(j, j)]);
            }
        }
        g[packed_index(i, i)] = -diagonal / q[packed_index(i, i)];
    }
    return g;
}

// Three orders of magnitude of slack over the rounding of the kernel on matrices of condition number below 5.
constexpr double kPullbackTolerance = 1e-12;

TEST(CholeskyKernels, PullBackMatchesTheScalarFormulasOnEveryTarget)
{
    for (const auto target : ve::test::runnable_targets())
    {
        const ve::test::ForcedTarget forced(target);
        ve::test::Stream stream(19);
        for (const std::size_t m : kSizes)
        {
            for (const std::size_t count : kCounts)
            {
                if (count > ve::simd::CorrelationBlock::kCapacity)
                {
                    continue;
                }
                const Batch batch = make_batch(stream, m, count);
                const std::vector<double> a = stream.vector(count);
                const std::vector<double> b = stream.vector(count);
                ve::simd::CorrelationBlock block(m);
                std::vector<double> log_det(count);
                std::vector<double> quadratic(count);
                block.factor(batch.q, batch.z, log_det, quadratic);
                std::vector<double> g(count * packed_count(m));
                block.pull_back(a, b, g);
                for (std::size_t t = 0; t < count; ++t)
                {
                    const std::span<const double> q(batch.q.data() + t * packed_count(m), packed_count(m));
                    const std::span<const double> z(batch.z.data() + t * m, m);
                    const std::vector<double> expected = reference_pullback(q, z, m, a[t], b[t]);
                    for (std::size_t k = 0; k < packed_count(m); ++k)
                    {
                        const double actual = g[t * packed_count(m) + k];
                        EXPECT_NEAR(actual, expected[k], kPullbackTolerance * std::max(std::fabs(expected[k]), 1.0))
                            << "m = " << m << " count = " << count << " t = " << t << " k = " << k;
                    }
                }
            }
        }
    }
}

// The block keeps the value of log_det_and_quadratic, to the bit.
TEST(CholeskyKernels, TheBlockGivesTheValuesOfTheValueKernel)
{
    for (const auto target : ve::test::runnable_targets())
    {
        const ve::test::ForcedTarget forced(target);
        ve::test::Stream stream(23);
        for (const std::size_t m : {3U, 21U})
        {
            const Batch batch = make_batch(stream, m, 8);
            const Output expected = run(batch);
            ve::simd::CorrelationBlock block(m);
            Output actual{std::vector<double>(8), std::vector<double>(8)};
            block.factor(batch.q, batch.z, actual.log_det, actual.quadratic);
            EXPECT_EQ(actual.log_det, expected.log_det) << "m = " << m;
            EXPECT_EQ(actual.quadratic, expected.quadratic) << "m = " << m;
        }
    }
}

}
