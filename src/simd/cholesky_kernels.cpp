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

// What one chunk (a vector of matrices, one per lane) keeps between the factor and the pullback. A packed matrix
// holds entry k of lane l at lanes * k + l.
struct Chunk
{
    double* factor;          // L, the packed lower triangle
    double* scales;          // s_i = Q_ii^-1/2
    double* inverse_pivots;  // 1 / L_ii
    double* y;               // L^-1 z
    double* z;
};

// The doubles of the state of one chunk of m x m matrices.
constexpr std::size_t ChunkSize(std::size_t m, std::size_t lanes)
{
    return (m * (m + 1) / 2 + 4 * m) * lanes;
}

Chunk ChunkAt(double* state, std::size_t m, std::size_t lanes)
{
    const std::size_t cells = m * (m + 1) / 2;
    return {state, state + cells * lanes, state + (cells + m) * lanes, state + (cells + 2 * m) * lanes,
            state + (cells + 3 * m) * lanes};
}

// The packed lower triangles of up to Lanes matrices. The lanes of a partial block repeat the first matrix, whose
// results are dropped.
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
// Keeps z and y in the chunk and writes the first `used` lanes of the two results.
template <class D>
void Summarise(D d, const Chunk& chunk, const double* z, std::size_t m, hn::Mask<D> valid, double* log_det,
               double* quadratic, std::size_t used)
{
    const std::size_t lanes = hn::Lanes(d);
    auto sum_squares = hn::Zero(d);
    auto sum_logs = hn::Zero(d);
    for (std::size_t i = 0; i < m; ++i)
    {
        for (std::size_t lane = 0; lane < lanes; ++lane)
        {
            chunk.z[lanes * i + lane] = z[(lane < used ? lane : 0) * m + i];
        }
        auto sum = hn::LoadU(d, chunk.z + lanes * i);
        for (std::size_t k = 0; k < i; ++k)
        {
            sum = hn::NegMulAdd(hn::LoadU(d, chunk.factor + lanes * PackedIndex(i, k)),
                                hn::LoadU(d, chunk.y + lanes * k), sum);
        }
        const auto y_i = hn::Mul(sum, hn::LoadU(d, chunk.inverse_pivots + lanes * i));
        hn::StoreU(y_i, d, chunk.y + lanes * i);
        sum_squares = hn::MulAdd(y_i, y_i, sum_squares);
        sum_logs = hn::Add(sum_logs, hn::Log(d, hn::LoadU(d, chunk.factor + lanes * PackedIndex(i, i))));
    }
    const auto nan = hn::NaN(d);
    std::array<double, HWY_MAX_LANES_D(D)> out;
    hn::StoreU(hn::IfThenElse(valid, hn::Add(sum_logs, sum_logs), nan), d, out.data());
    std::copy_n(out.begin(), used, log_det);
    hn::StoreU(hn::IfThenElse(valid, sum_squares, nan), d, out.data());
    std::copy_n(out.begin(), used, quadratic);
}

void FactorChunk(const double* q, const double* z, std::size_t m, std::size_t used, const Chunk& chunk,
                 double* log_det, double* quadratic)
{
    const hn::ScalableTag<double> d;
    const std::size_t lanes = hn::Lanes(d);
    Transpose(q, m * (m + 1) / 2, used, lanes, chunk.factor);
    Normalize(d, chunk.factor, m, chunk.scales);
    const auto valid = Factor(d, chunk.factor, m, chunk.inverse_pivots);
    Summarise(d, chunk, z, m, valid, log_det, quadratic, used);
}

std::size_t LanesImpl()
{
    return hn::Lanes(hn::ScalableTag<double>());
}

void LogDetQuadraticImpl(const double* q, const double* z, std::size_t m, std::size_t count, double* log_det,
                         double* quadratic)
{
    const std::size_t lanes = LanesImpl();
    std::vector<double> state(ChunkSize(m, lanes));
    const Chunk chunk = ChunkAt(state.data(), m, lanes);
    for (std::size_t first = 0; first < count; first += lanes)
    {
        const std::size_t used = std::min(lanes, count - first);
        FactorChunk(q + first * (m * (m + 1) / 2), z + first * m, m, used, chunk, log_det + first, quadratic + first);
    }
}

void FactorBlockImpl(const double* q, const double* z, std::size_t m, std::size_t count, double* state,
                     double* log_det, double* quadratic)
{
    const std::size_t lanes = LanesImpl();
    for (std::size_t first = 0; first < count; first += lanes)
    {
        const std::size_t used = std::min(lanes, count - first);
        const Chunk chunk = ChunkAt(state + (first / lanes) * ChunkSize(m, lanes), m, lanes);
        FactorChunk(q + first * (m * (m + 1) / 2), z + first * m, m, used, chunk, log_det + first, quadratic + first);
    }
}

// L^-1, packed lower, row by row: (L^-1)_ij = - (1 / L_ii) sum_(k = j..i-1) L_ik (L^-1)_kj.
template <class D>
void Invert(D d, const Chunk& chunk, std::size_t m, double* inverse)
{
    const std::size_t lanes = hn::Lanes(d);
    const auto zero = hn::Zero(d);
    for (std::size_t i = 0; i < m; ++i)
    {
        for (std::size_t j = 0; j < i; ++j)
        {
            hn::StoreU(zero, d, inverse + lanes * PackedIndex(i, j));
        }
        for (std::size_t k = 0; k < i; ++k)
        {
            const auto l_ik = hn::LoadU(d, chunk.factor + lanes * PackedIndex(i, k));
            for (std::size_t j = 0; j <= k; ++j)
            {
                double* entry = inverse + lanes * PackedIndex(i, j);
                hn::StoreU(hn::MulAdd(l_ik, hn::LoadU(d, inverse + lanes * PackedIndex(k, j)), hn::LoadU(d, entry)), d,
                           entry);
            }
        }
        const auto inverse_pivot = hn::LoadU(d, chunk.inverse_pivots + lanes * i);
        const auto negative = hn::Neg(inverse_pivot);
        for (std::size_t j = 0; j < i; ++j)
        {
            double* entry = inverse + lanes * PackedIndex(i, j);
            hn::StoreU(hn::Mul(hn::LoadU(d, entry), negative), d, entry);
        }
        hn::StoreU(inverse_pivot, d, inverse + lanes * PackedIndex(i, i));
    }
}

// R^-1 = (L^-1)' L^-1, packed lower: (R^-1)_ij = sum_(k >= i) (L^-1)_ki (L^-1)_kj for j <= i; and
// w = R^-1 z = (L^-1)' y with y = L^-1 z.
template <class D>
void InvertCorrelation(D d, const Chunk& chunk, std::size_t m, const double* inverse, double* correlation_inverse,
                       double* w)
{
    const std::size_t lanes = hn::Lanes(d);
    const auto zero = hn::Zero(d);
    for (std::size_t k = 0; k < m * (m + 1) / 2; ++k)
    {
        hn::StoreU(zero, d, correlation_inverse + lanes * k);
    }
    for (std::size_t i = 0; i < m; ++i)
    {
        hn::StoreU(zero, d, w + lanes * i);
    }
    for (std::size_t k = 0; k < m; ++k)
    {
        const auto y_k = hn::LoadU(d, chunk.y + lanes * k);
        for (std::size_t i = 0; i <= k; ++i)
        {
            const auto l_ki = hn::LoadU(d, inverse + lanes * PackedIndex(k, i));
            hn::StoreU(hn::MulAdd(l_ki, y_k, hn::LoadU(d, w + lanes * i)), d, w + lanes * i);
            for (std::size_t j = 0; j <= i; ++j)
            {
                double* entry = correlation_inverse + lanes * PackedIndex(i, j);
                hn::StoreU(hn::MulAdd(l_ki, hn::LoadU(d, inverse + lanes * PackedIndex(k, j)), hn::LoadU(d, entry)), d,
                           entry);
            }
        }
    }
}

// dl/dQ from G = a R^-1 - b w w' (a = dl/dlog|R|, b = dl/dq): 2 G_ij s_i s_j below the diagonal and
// - s_i^2 (a (1 - (R^-1)_ii) - b w_i (z_i - w_i)) on it, which is - (sum_(j != i) G_ij R_ij) / Q_ii written with
// R R^-1 = I and R w = z over the unit diagonal of R.
template <class D>
void PullBackChunkTo(D d, const Chunk& chunk, std::size_t m, const double* correlation_inverse, const double* w,
                     const double* a_lanes, const double* b_lanes, double* g)
{
    const std::size_t lanes = hn::Lanes(d);
    const auto one = hn::Set(d, 1.0);
    const auto a = hn::LoadU(d, a_lanes);
    const auto b = hn::LoadU(d, b_lanes);
    for (std::size_t i = 0; i < m; ++i)
    {
        const auto w_i = hn::LoadU(d, w + lanes * i);
        const auto s_i = hn::LoadU(d, chunk.scales + lanes * i);
        const auto b_w_i = hn::Mul(b, w_i);
        for (std::size_t j = 0; j < i; ++j)
        {
            const auto gradient = hn::Sub(hn::Mul(a, hn::LoadU(d, correlation_inverse + lanes * PackedIndex(i, j))),
                                          hn::Mul(b_w_i, hn::LoadU(d, w + lanes * j)));
            const auto s_j = hn::LoadU(d, chunk.scales + lanes * j);
            hn::StoreU(hn::Mul(hn::Mul(hn::Add(gradient, gradient), s_i), s_j), d, g + lanes * PackedIndex(i, j));
        }
        const auto rinv_ii = hn::LoadU(d, correlation_inverse + lanes * PackedIndex(i, i));
        const auto from_inverse = hn::Mul(a, hn::Sub(one, rinv_ii));
        const auto from_w = hn::Mul(b_w_i, hn::Sub(hn::LoadU(d, chunk.z + lanes * i), w_i));
        const auto diagonal = hn::Mul(hn::Mul(s_i, s_i), hn::Sub(from_inverse, from_w));
        hn::StoreU(hn::Neg(diagonal), d, g + lanes * PackedIndex(i, i));
    }
}

// scratch: L^-1, R^-1 and the gradient (cells lanes each), then w (m lanes) and the two partials (a lane each).
void PullBackImpl(double* state, std::size_t m, std::size_t count, const double* d_log_det, const double* d_quadratic,
                  double* g, double* scratch)
{
    const hn::ScalableTag<double> d;
    const std::size_t lanes = hn::Lanes(d);
    const std::size_t cells = m * (m + 1) / 2;
    double* inverse = scratch;
    double* correlation_inverse = inverse + cells * lanes;
    double* gradient = correlation_inverse + cells * lanes;
    double* w = gradient + cells * lanes;
    double* a_lanes = w + m * lanes;
    double* b_lanes = a_lanes + lanes;
    for (std::size_t first = 0; first < count; first += lanes)
    {
        const std::size_t used = std::min(lanes, count - first);
        const Chunk chunk = ChunkAt(state + (first / lanes) * ChunkSize(m, lanes), m, lanes);
        for (std::size_t lane = 0; lane < lanes; ++lane)
        {
            a_lanes[lane] = d_log_det[first + (lane < used ? lane : 0)];
            b_lanes[lane] = d_quadratic[first + (lane < used ? lane : 0)];
        }
        Invert(d, chunk, m, inverse);
        InvertCorrelation(d, chunk, m, inverse, correlation_inverse, w);
        PullBackChunkTo(d, chunk, m, correlation_inverse, w, a_lanes, b_lanes, gradient);
        for (std::size_t lane = 0; lane < used; ++lane)
        {
            for (std::size_t k = 0; k < cells; ++k)
            {
                g[(first + lane) * cells + k] = gradient[lanes * k + lane];
            }
        }
    }
}

}
}
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace ve::simd
{

HWY_EXPORT(LanesImpl);
HWY_EXPORT(LogDetQuadraticImpl);
HWY_EXPORT(FactorBlockImpl);
HWY_EXPORT(PullBackImpl);

void log_det_and_quadratic(std::span<const double> q, std::span<const double> z, std::size_t m,
                           std::span<double> log_det, std::span<double> quadratic)
{
    const std::size_t count = log_det.size();
    assert(q.size() == count * (m * (m + 1) / 2) && z.size() == count * m && quadratic.size() == count);
    HWY_DYNAMIC_DISPATCH(LogDetQuadraticImpl)(q.data(), z.data(), m, count, log_det.data(), quadratic.data());
}

CorrelationBlock::CorrelationBlock(std::size_t m) : m_(m)
{
    const std::size_t lanes = HWY_DYNAMIC_DISPATCH(LanesImpl)();
    const std::size_t cells = m * (m + 1) / 2;
    const std::size_t chunks = (kCapacity + lanes - 1) / lanes;
    state_.resize(chunks * (cells + 4 * m) * lanes);
    scratch_.resize((3 * cells + m + 2) * lanes);
}

void CorrelationBlock::factor(std::span<const double> q, std::span<const double> z, std::span<double> log_det,
                              std::span<double> quadratic)
{
    count_ = log_det.size();
    assert(count_ <= kCapacity && q.size() == count_ * (m_ * (m_ + 1) / 2) && z.size() == count_ * m_ &&
           quadratic.size() == count_);
    HWY_DYNAMIC_DISPATCH(FactorBlockImpl)(q.data(), z.data(), m_, count_, state_.data(), log_det.data(),
                                          quadratic.data());
}

void CorrelationBlock::pull_back(std::span<const double> d_log_det, std::span<const double> d_quadratic,
                                 std::span<double> g)
{
    assert(d_log_det.size() == count_ && d_quadratic.size() == count_ && g.size() == count_ * (m_ * (m_ + 1) / 2));
    HWY_DYNAMIC_DISPATCH(PullBackImpl)(state_.data(), m_, count_, d_log_det.data(), d_quadratic.data(), g.data(),
                                       scratch_.data());
}

}
#endif
