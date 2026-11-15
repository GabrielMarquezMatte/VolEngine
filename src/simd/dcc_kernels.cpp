#include "simd/dcc_kernels.hpp"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "simd/dcc_kernels.cpp"
#include <hwy/foreach_target.h>
#include <hwy/highway.h>

#include <cassert>
#include <cmath>

// The rows of a packed triangle are short and their length is rarely a multiple of the lanes: the tail of each is
// scalar, since a masked store (the vector way) is slow on AMD cores.
HWY_BEFORE_NAMESPACE();
namespace ve::simd
{
namespace HWY_NAMESPACE
{
namespace hn = hwy::HWY_NAMESPACE;

void MultiplyAddImpl(double* acc, const double* weights, const double* x, std::size_t count)
{
    const hn::ScalableTag<double> d;
    const std::size_t lanes = hn::Lanes(d);
    std::size_t k = 0;
    for (; k + lanes <= count; k += lanes)
    {
        const auto product = hn::Mul(hn::LoadU(d, weights + k), hn::LoadU(d, x + k));
        hn::StoreU(hn::Add(hn::LoadU(d, acc + k), product), d, acc + k);
    }
    for (; k < count; ++k)
    {
        acc[k] = acc[k] + weights[k] * x[k];
    }
}

// acc[k] += w[k] * (v[i] * v[k]) for k <= i, one row of the triangle.
HWY_INLINE void AddOuterRow(double* acc, const double* w, const double* v, double v_i, std::size_t i)
{
    const hn::ScalableTag<double> d;
    const std::size_t lanes = hn::Lanes(d);
    const auto scale = hn::Set(d, v_i);
    std::size_t k = 0;
    for (; k + lanes <= i + 1; k += lanes)
    {
        const auto product = hn::Mul(hn::LoadU(d, w + k), hn::Mul(scale, hn::LoadU(d, v + k)));
        hn::StoreU(hn::Add(hn::LoadU(d, acc + k), product), d, acc + k);
    }
    for (; k <= i; ++k)
    {
        acc[k] = acc[k] + w[k] * (v_i * v[k]);
    }
}

void AddOuterImpl(double* acc, const double* weights, const double* v, std::size_t m)
{
    for (std::size_t i = 0; i < m; ++i)
    {
        const std::size_t row = i * (i + 1) / 2;
        AddOuterRow(acc + row, weights + row, v, v[i], i);
    }
}

// out[k] = (q[k] * s_i) * s[k] for k < i, one row below the diagonal.
HWY_INLINE void NormalizeRow(const double* q, double s_i, const double* s, double* out, std::size_t i)
{
    const hn::ScalableTag<double> d;
    const std::size_t lanes = hn::Lanes(d);
    const auto scale = hn::Set(d, s_i);
    std::size_t k = 0;
    for (; k + lanes <= i; k += lanes)
    {
        hn::StoreU(hn::Mul(hn::Mul(hn::LoadU(d, q + k), scale), hn::LoadU(d, s + k)), d, out + k);
    }
    for (; k < i; ++k)
    {
        out[k] = q[k] * s_i * s[k];
    }
}

void CorrelationImpl(const double* q, std::size_t m, double* scale, double* r)
{
    for (std::size_t i = 0; i < m; ++i)
    {
        scale[i] = 1.0 / std::sqrt(q[i * (i + 3) / 2]);
    }
    for (std::size_t i = 0; i < m; ++i)
    {
        r[i * m + i] = 1.0;
        NormalizeRow(q + i * (i + 1) / 2, scale[i], scale, r + i * m, i);
        for (std::size_t j = 0; j < i; ++j)
        {
            r[j * m + i] = r[i * m + j];
        }
    }
}

}
}
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace ve::simd
{

HWY_EXPORT(MultiplyAddImpl);
HWY_EXPORT(AddOuterImpl);
HWY_EXPORT(CorrelationImpl);

void multiply_add(std::span<double> acc, std::span<const double> weights, std::span<const double> x)
{
    assert(weights.size() == acc.size() && x.size() == acc.size());
    HWY_DYNAMIC_DISPATCH(MultiplyAddImpl)(acc.data(), weights.data(), x.data(), acc.size());
}

void add_outer(std::span<double> acc, std::span<const double> weights, std::span<const double> v)
{
    assert(acc.size() == v.size() * (v.size() + 1) / 2 && weights.size() == acc.size());
    HWY_DYNAMIC_DISPATCH(AddOuterImpl)(acc.data(), weights.data(), v.data(), v.size());
}

void correlation_from_q(std::span<const double> q, std::size_t m, std::span<double> scale, std::span<double> r)
{
    assert(q.size() == m * (m + 1) / 2 && scale.size() == m && r.size() == m * m);
    HWY_DYNAMIC_DISPATCH(CorrelationImpl)(q.data(), m, scale.data(), r.data());
}

}
#endif
