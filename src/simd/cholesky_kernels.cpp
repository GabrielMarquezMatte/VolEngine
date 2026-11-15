#include "simd/cholesky_kernels.hpp"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "simd/cholesky_kernels.cpp"
#include <hwy/foreach_target.h>
#include <hwy/highway.h>

#include <hwy/contrib/math/math-inl.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <vector>

HWY_BEFORE_NAMESPACE();
namespace ve::simd
{
namespace HWY_NAMESPACE
{
namespace hn = hwy::HWY_NAMESPACE;

constexpr std::size_t PackedIndex(std::size_t i, std::size_t j)
{
    return i * (i + 1) / 2 + j;
}

// The packed lower triangles of up to Lanes matrices, entry k of lane l at lanes * k + l. The lanes of a partial block
// repeat the first matrix, whose results are dropped.
void Transpose(const double* q, std::size_t cells, std::size_t used, std::size_t lanes, double* soa)
{
    for (std::size_t lane = 0; lane < lanes; ++lane)
    {
        const double* source = q + (lane < used ? lane : 0) * cells;
        for (std::size_t k = 0; k < cells; ++k)
        {
            soa[lanes * k + lane] = source[k];
        }
    }
}

// Q becomes R = D Q D in place, with D = diag(Q)^-1/2 and an exact unit diagonal, in the order of the scalar
// recursion: (Q_ij s_i) s_j. A diagonal that is not positive and finite makes its scale, and so R, NaN or infinite,
// which Factor then rejects.
template <class D>
void Normalize(D d, double* a, std::size_t m, double* scales)
{
    const std::size_t lanes = hn::Lanes(d);
    const auto one = hn::Set(d, 1.0);
    for (std::size_t i = 0; i < m; ++i)
    {
        const auto diagonal = hn::LoadU(d, a + lanes * PackedIndex(i, i));
        hn::StoreU(hn::Div(one, hn::Sqrt(diagonal)), d, scales + lanes * i);
    }
    for (std::size_t i = 0; i < m; ++i)
    {
        const auto scale_i = hn::LoadU(d, scales + lanes * i);
        for (std::size_t j = 0; j < i; ++j)
        {
            double* entry = a + lanes * PackedIndex(i, j);
            const auto scaled = hn::Mul(hn::Mul(hn::LoadU(d, entry), scale_i), hn::LoadU(d, scales + lanes * j));
            hn::StoreU(scaled, d, entry);
        }
        hn::StoreU(one, d, a + lanes * PackedIndex(i, i));
    }
}

// Right-looking Cholesky of every lane at once, a = L over the packed lower triangle; inverse_pivots receives 1 / L_kk.
// The returned mask is true in the lanes whose pivots were all positive and finite.
template <class D>
hn::Mask<D> Factor(D d, double* a, std::size_t m, double* inverse_pivots)
{
    const std::size_t lanes = hn::Lanes(d);
    const auto zero = hn::Zero(d);
    const auto infinity = hn::Inf(d);
    auto valid = hn::FirstN(d, lanes);
    for (std::size_t k = 0; k < m; ++k)
    {
        const auto pivot = hn::LoadU(d, a + lanes * PackedIndex(k, k));
        valid = hn::And(valid, hn::And(hn::Gt(pivot, zero), hn::Lt(pivot, infinity)));
        const auto root = hn::Sqrt(pivot);
        const auto inverse = hn::Div(hn::Set(d, 1.0), root);
        hn::StoreU(root, d, a + lanes * PackedIndex(k, k));
        hn::StoreU(inverse, d, inverse_pivots + lanes * k);
        for (std::size_t i = k + 1; i < m; ++i)
        {
            double* column = a + lanes * PackedIndex(i, k);
            hn::StoreU(hn::Mul(hn::LoadU(d, column), inverse), d, column);
        }
        for (std::size_t j = k + 1; j < m; ++j)
        {
            const auto l_jk = hn::LoadU(d, a + lanes * PackedIndex(j, k));
            for (std::size_t i = j; i < m; ++i)
            {
                double* entry = a + lanes * PackedIndex(i, j);
                const auto l_ik = hn::LoadU(d, a + lanes * PackedIndex(i, k));
                hn::StoreU(hn::NegMulAdd(l_ik, l_jk, hn::LoadU(d, entry)), d, entry);
            }
        }
    }
    return valid;
}

// z' R^-1 z = |L^-1 z|^2, with y = L^-1 z by forward substitution, one lane per matrix; log |R| = 2 sum log L_kk.
template <class D>
void Summarise(D d, const double* factor, const double* inverse_pivots, const double* z, std::size_t m,
               hn::Mask<D> valid, double* log_det, double* quadratic, std::size_t used)
{
    const std::size_t lanes = hn::Lanes(d);
    std::vector<double> y(m * lanes);
    auto sum_squares = hn::Zero(d);
    auto sum_logs = hn::Zero(d);
    for (std::size_t i = 0; i < m; ++i)
    {
        std::array<double, HWY_MAX_LANES_D(D)> lane_z;
        for (std::size_t lane = 0; lane < lanes; ++lane)
        {
            lane_z[lane] = z[(lane < used ? lane : 0) * m + i];
        }
        auto sum = hn::LoadU(d, lane_z.data());
        for (std::size_t k = 0; k < i; ++k)
        {
            sum = hn::NegMulAdd(hn::LoadU(d, factor + lanes * PackedIndex(i, k)), hn::LoadU(d, y.data() + lanes * k),
                                sum);
        }
        const auto y_i = hn::Mul(sum, hn::LoadU(d, inverse_pivots + lanes * i));
        hn::StoreU(y_i, d, y.data() + lanes * i);
        sum_squares = hn::MulAdd(y_i, y_i, sum_squares);
        sum_logs = hn::Add(sum_logs, hn::Log(d, hn::LoadU(d, factor + lanes * PackedIndex(i, i))));
    }
    const auto nan = hn::NaN(d);
    std::array<double, HWY_MAX_LANES_D(D)> out;
    hn::StoreU(hn::IfThenElse(valid, hn::Add(sum_logs, sum_logs), nan), d, out.data());
    std::copy_n(out.begin(), used, log_det);
    hn::StoreU(hn::IfThenElse(valid, sum_squares, nan), d, out.data());
    std::copy_n(out.begin(), used, quadratic);
}

void LogDetQuadraticImpl(const double* q, const double* z, std::size_t m, std::size_t count, double* log_det,
                         double* quadratic)
{
    const hn::ScalableTag<double> d;
    const std::size_t lanes = hn::Lanes(d);
    const std::size_t cells = m * (m + 1) / 2;
    std::vector<double> a(cells * lanes);
    // The scales of the normalization, then the reciprocal pivots of the factor.
    std::vector<double> inverse_pivots(m * lanes);
    for (std::size_t first = 0; first < count; first += lanes)
    {
        const std::size_t used = std::min(lanes, count - first);
        Transpose(q + first * cells, cells, used, lanes, a.data());
        Normalize(d, a.data(), m, inverse_pivots.data());
        const auto valid = Factor(d, a.data(), m, inverse_pivots.data());
        Summarise(d, a.data(), inverse_pivots.data(), z + first * m, m, valid, log_det + first, quadratic + first,
                  used);
    }
}

}
}
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace ve::simd
{

HWY_EXPORT(LogDetQuadraticImpl);

void log_det_and_quadratic(std::span<const double> q, std::span<const double> z, std::size_t m,
                           std::span<double> log_det, std::span<double> quadratic)
{
    const std::size_t count = log_det.size();
    assert(q.size() == count * (m * (m + 1) / 2) && z.size() == count * m && quadratic.size() == count);
    HWY_DYNAMIC_DISPATCH(LogDetQuadraticImpl)(q.data(), z.data(), m, count, log_det.data(), quadratic.data());
}

}
#endif
