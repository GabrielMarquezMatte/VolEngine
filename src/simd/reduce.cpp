#include "simd/reduce.hpp"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "simd/reduce.cpp"
#include <hwy/foreach_target.h>
#include <hwy/highway.h>

#include <cassert>

HWY_BEFORE_NAMESPACE();
namespace ve::simd
{
namespace HWY_NAMESPACE
{
namespace hn = hwy::HWY_NAMESPACE;

double SumImpl(const double* x, std::size_t n)
{
    const hn::ScalableTag<double> d;
    const std::size_t lanes = hn::Lanes(d);
    auto acc = hn::Zero(d);
    std::size_t i = 0;
    for (; i + lanes <= n; i += lanes)
    {
        acc = hn::Add(acc, hn::LoadU(d, x + i));
    }
    acc = hn::Add(acc, hn::LoadN(d, x + i, n - i));
    return hn::ReduceSum(d, acc);
}

double DotImpl(const double* x, const double* y, std::size_t n)
{
    const hn::ScalableTag<double> d;
    const std::size_t lanes = hn::Lanes(d);
    auto acc = hn::Zero(d);
    std::size_t i = 0;
    for (; i + lanes <= n; i += lanes)
    {
        acc = hn::MulAdd(hn::LoadU(d, x + i), hn::LoadU(d, y + i), acc);
    }
    acc = hn::MulAdd(hn::LoadN(d, x + i, n - i), hn::LoadN(d, y + i, n - i), acc);
    return hn::ReduceSum(d, acc);
}

}
}
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace ve::simd
{

HWY_EXPORT(SumImpl);
HWY_EXPORT(DotImpl);

double sum(std::span<const double> x)
{
    return HWY_DYNAMIC_DISPATCH(SumImpl)(x.data(), x.size());
}

double dot(std::span<const double> x, std::span<const double> y)
{
    assert(x.size() == y.size());
    return HWY_DYNAMIC_DISPATCH(DotImpl)(x.data(), y.data(), x.size());
}

}
#endif
