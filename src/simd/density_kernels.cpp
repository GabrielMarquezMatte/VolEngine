#include "simd/density_kernels.hpp"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "simd/density_kernels.cpp"
#include <hwy/foreach_target.h>
#include <hwy/highway.h>

#include <hwy/contrib/math/math-inl.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>

HWY_BEFORE_NAMESPACE();
namespace ve::simd
{
namespace HWY_NAMESPACE
{
namespace hn = hwy::HWY_NAMESPACE;

// The partial sums of the scale and power, and the lanes of a vector of terms in which something went wrong.
template <class D>
struct Accumulators
{
    hn::Vec<D> total;
    hn::Vec<D> scale;
    hn::Vec<D> power;
    hn::Mask<D> valid;
};

// What one vector of residuals and variances gives: the term, and for the fused kernel the partials in e and sigma2
// and the (per lane) partials in the scale and power.
template <class D>
struct Lanes
{
    hn::Vec<D> term;
    hn::Vec<D> d_e;
    hn::Vec<D> d_sigma2;
    hn::Vec<D> d_scale;
    hn::Vec<D> d_power;
};

struct Constants
{
    double log_constant;
    double scale;
    double power;
};

// norm: x = e^2 / sigma2.
template <Dist kDist, bool kPartials, class D>
HWY_INLINE Lanes<D> NormalLanes(D d, const Constants& c, hn::Vec<D> e, hn::Vec<D> sigma2, hn::Vec<D> log_sigma2)
{
    const auto half = hn::Set(d, 0.5);
    const auto x = hn::Div(hn::Mul(e, e), sigma2);
    const auto density = hn::Sub(hn::Set(d, c.log_constant), hn::Mul(half, x));
    Lanes<D> lanes{hn::Sub(density, hn::Mul(half, log_sigma2)), hn::Zero(d), hn::Zero(d), hn::Zero(d), hn::Zero(d)};
    if (kPartials)
    {
        lanes.d_e = hn::Neg(hn::Div(e, sigma2));
        lanes.d_sigma2 = hn::Div(hn::Mul(half, hn::Sub(x, hn::Set(d, 1.0))), sigma2);
    }
    return lanes;
}

// std: u = e^2 / (sigma2 scale), w = power / (1 + u).
template <bool kPartials, class D>
HWY_INLINE Lanes<D> StudentLanes(D d, const Constants& c, hn::Vec<D> e, hn::Vec<D> sigma2, hn::Vec<D> log_sigma2)
{
    const auto one = hn::Set(d, 1.0);
    const auto half = hn::Set(d, 0.5);
    const auto power = hn::Set(d, c.power);
    const auto u = hn::Div(hn::Div(hn::Mul(e, e), sigma2), hn::Set(d, c.scale));
    // The vector log1p is not defined at +infinity (an overflowed e^2): log1p(inf) = inf, and a NaN stays a NaN.
    const auto log1p_u = hn::IfThenElse(hn::Lt(u, hn::Inf(d)), hn::Log1p(d, u), u);
    const auto density = hn::Sub(hn::Set(d, c.log_constant), hn::Mul(power, log1p_u));
    Lanes<D> lanes{hn::Sub(density, hn::Mul(half, log_sigma2)), hn::Zero(d), hn::Zero(d), hn::Zero(d), hn::Zero(d)};
    if (kPartials)
    {
        const auto inverse_variance = hn::Div(one, sigma2);
        const auto inverse_scale = hn::Div(one, hn::Set(d, c.scale));
        const auto w = hn::Div(power, hn::Add(one, u));
        const auto w_u = hn::Mul(w, u);
        lanes.d_e = hn::Mul(hn::Mul(hn::Mul(hn::Mul(hn::Set(d, -2.0), w), e), inverse_variance), inverse_scale);
        lanes.d_sigma2 = hn::Mul(hn::Sub(w_u, half), inverse_variance);
        lanes.d_scale = hn::Mul(w_u, inverse_scale);
        lanes.d_power = hn::Neg(log1p_u);
    }
    return lanes;
}

// ged: y = |e| / (sigma lambda), v = y^power = exp(power log y), slope = power v / y. A lane with y = 0 has v = 0 and
// no partial but the one of the variance.
template <bool kPartials, class D>
HWY_INLINE Lanes<D> GedLanes(D d, const Constants& c, hn::Vec<D> e, hn::Vec<D> sigma2, hn::Vec<D> log_sigma2)
{
    const auto zero = hn::Zero(d);
    const auto one = hn::Set(d, 1.0);
    const auto half = hn::Set(d, 0.5);
    const auto power = hn::Set(d, c.power);
    const auto sigma = hn::Sqrt(sigma2);
    const auto y = hn::Div(hn::Div(hn::Abs(e), sigma), hn::Set(d, c.scale));
    // Only a positive finite y goes through the vector log and exp, which are not defined at 0 and +infinity; for
    // the others y^power is y itself: 0, +infinity (an overflowed term) or NaN.
    const auto positive = hn::And(hn::Gt(y, zero), hn::Lt(y, hn::Inf(d)));
    const auto log_y = hn::Log(d, hn::IfThenElse(positive, y, one));
    // The vector exp wraps above about 709.78: past it y^power is +infinity, as std::pow gives.
    const auto exponent = hn::Mul(power, log_y);
    const auto exp_v = hn::IfThenElse(hn::Gt(exponent, hn::Set(d, 709.78)), hn::Inf(d), hn::Exp(d, exponent));
    const auto v = hn::IfThenElse(positive, exp_v, y);
    const auto density = hn::Sub(hn::Set(d, c.log_constant), hn::Mul(half, v));
    Lanes<D> lanes{hn::Sub(density, hn::Mul(half, log_sigma2)), zero, zero, zero, zero};
    if (kPartials)
    {
        const auto inverse_variance = hn::Div(one, sigma2);
        const auto inverse_sigma = hn::Div(one, sigma);
        const auto inverse_scale = hn::Div(one, hn::Set(d, c.scale));
        const auto negative_half = hn::Set(d, -0.5);
        const auto slope = hn::IfThenElse(positive, hn::Mul(power, hn::Div(v, y)), zero);
        const auto sign = hn::CopySign(one, e);
        lanes.d_e = hn::Mul(hn::Mul(hn::Mul(hn::Mul(negative_half, slope), sign), inverse_sigma), inverse_scale);
        const auto from_slope = hn::Mul(hn::Mul(hn::Mul(hn::Set(d, 0.25), slope), y), inverse_variance);
        lanes.d_sigma2 = hn::Add(hn::Mul(negative_half, inverse_variance), from_slope);
        lanes.d_scale = hn::Mul(hn::Mul(hn::Mul(half, slope), y), inverse_scale);
        lanes.d_power = hn::Mul(hn::Mul(negative_half, v), log_y);
    }
    return lanes;
}

template <Dist kDist, bool kPartials, class D>
HWY_INLINE Lanes<D> LanesOf(D d, const Constants& c, hn::Vec<D> e, hn::Vec<D> sigma2, hn::Vec<D> log_sigma2)
{
    if (kDist == Dist::Std)
    {
        return StudentLanes<kPartials>(d, c, e, sigma2, log_sigma2);
    }
    if (kDist == Dist::Ged)
    {
        return GedLanes<kPartials>(d, c, e, sigma2, log_sigma2);
    }
    return NormalLanes<kDist, kPartials>(d, c, e, sigma2, log_sigma2);
}

template <Dist kDist, bool kPartials>
double Run(const detail::DensityArgs& a)
{
    const hn::ScalableTag<double> d;
    const std::size_t lanes = hn::Lanes(d);
    const Constants constants{a.log_constant, a.scale, a.power};
    const auto zero = hn::Zero(d);
    const auto one = hn::Set(d, 1.0);
    const auto infinity = hn::Inf(d);
    auto total = zero;
    auto sum_scale = zero;
    auto sum_power = zero;
    auto valid = hn::FirstN(d, lanes);
    for (std::size_t i = 0; i < a.count; i += lanes)
    {
        const std::size_t used = std::min(lanes, a.count - i);
        const bool full = used == lanes;
        const auto inside = hn::FirstN(d, used);
        const auto e = full ? hn::LoadU(d, a.e + i) : hn::LoadN(d, a.e + i, used);
        const auto sigma2 = full ? hn::LoadU(d, a.sigma2 + i)
                                 : hn::IfThenElse(inside, hn::LoadN(d, a.sigma2 + i, used), one);
        valid = hn::And(valid, hn::And(hn::Gt(sigma2, zero), hn::Lt(sigma2, infinity)));
        const auto result = LanesOf<kDist, kPartials>(d, constants, e, sigma2, hn::Log(d, sigma2));
        total = hn::Add(total, hn::IfThenElse(inside, result.term, zero));
        if (a.terms != nullptr)
        {
            hn::StoreN(result.term, d, a.terms + i, used);
        }
        if (kPartials)
        {
            hn::StoreN(result.d_e, d, a.d_e + i, used);
            hn::StoreN(result.d_sigma2, d, a.d_sigma2 + i, used);
            sum_scale = hn::Add(sum_scale, hn::IfThenElse(inside, result.d_scale, zero));
            sum_power = hn::Add(sum_power, hn::IfThenElse(inside, result.d_power, zero));
        }
    }
    const double sum = hn::ReduceSum(d, total);
    if (!hn::AllTrue(d, valid) || !std::isfinite(sum))
    {
        return -std::numeric_limits<double>::infinity();
    }
    if (kPartials)
    {
        *a.d_scale = hn::ReduceSum(d, sum_scale);
        *a.d_power = hn::ReduceSum(d, sum_power);
    }
    return sum;
}

template <bool kPartials>
double RunDensity(Dist dist, const detail::DensityArgs& a)
{
    if (dist == Dist::Std)
    {
        return Run<Dist::Std, kPartials>(a);
    }
    if (dist == Dist::Ged)
    {
        return Run<Dist::Ged, kPartials>(a);
    }
    return Run<Dist::Norm, kPartials>(a);
}

double TermsImpl(Dist dist, const detail::DensityArgs& a)
{
    return RunDensity<false>(dist, a);
}

double TermsWithPartialsImpl(Dist dist, const detail::DensityArgs& a)
{
    return RunDensity<true>(dist, a);
}

}
}
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace ve::simd
{

HWY_EXPORT(TermsImpl);
HWY_EXPORT(TermsWithPartialsImpl);

namespace
{

detail::DensityArgs args_of(const dist::LogDensity<double>& density, const SymmetricSeries& series)
{
    assert(series.sigma2.size() == series.e.size() && (series.terms.empty() || series.terms.size() == series.e.size()));
    assert(density.dist == Dist::Norm || density.dist == Dist::Std || density.dist == Dist::Ged);
    return {series.e.data(),
            series.sigma2.data(),
            series.terms.empty() ? nullptr : series.terms.data(),
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            series.e.size(),
            density.log_constant,
            density.scale,
            density.power};
}

}

double symmetric_terms(const dist::LogDensity<double>& density, std::span<const double> e,
                       std::span<const double> sigma2, std::span<double> terms)
{
    return HWY_DYNAMIC_DISPATCH(TermsImpl)(density.dist, args_of(density, {e, sigma2, terms}));
}

double symmetric_terms_with_partials(const dist::LogDensity<double>& density, const SymmetricSeries& series,
                                     SymmetricPartials& partials)
{
    assert(partials.d_e.size() == series.e.size() && partials.d_sigma2.size() == series.e.size());
    detail::DensityArgs args = args_of(density, series);
    args.d_e = partials.d_e.data();
    args.d_sigma2 = partials.d_sigma2.data();
    args.d_scale = &partials.d_scale;
    args.d_power = &partials.d_power;
    return HWY_DYNAMIC_DISPATCH(TermsWithPartialsImpl)(density.dist, args);
}

}
#endif
