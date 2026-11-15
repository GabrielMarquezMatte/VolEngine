#include "simd/reduce.hpp"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "simd/reduce.cpp"
#include <hwy/foreach_target.h>
#include <hwy/highway.h>

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

}
}
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace ve::simd
{

HWY_EXPORT(SumImpl);

double sum(std::span<const double> x)
{
    return HWY_DYNAMIC_DISPATCH(SumImpl)(x.data(), x.size());
}

}
#endif
