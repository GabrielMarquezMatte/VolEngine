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

// The fused kernel's partials of one vector of terms, in e, sigma2, and (per lane) the scale and power, as a Highway
// tuple: SVE and RVV vectors are sizeless and cannot be members of a struct.
enum Partial : std::size_t
{
    kDe,
    kDSigma2,
    kDScale,
    kDPower,
};

struct Constants
{
    double log_constant;
    double scale;
    double power;
};

// norm: x = e^2 / sigma2.
template <bool kPartials, class D>
HWY_INLINE hn::Vec<D> NormalTerm(D d, const Constants& c, hn::Vec<D> e, hn::Vec<D> sigma2, hn::Vec4<D>& partials)
{
    const auto half = hn::Set(d, 0.5);
    const auto x = hn::Div(hn::Mul(e, e), sigma2);
    const auto density = hn::Sub(hn::Set(d, c.log_constant), hn::Mul(half, x));
    if (kPartials)
    {
        const auto d_e = hn::Neg(hn::Div(e, sigma2));
        const auto d_sigma2 = hn::Div(hn::Mul(half, hn::Sub(x, hn::Set(d, 1.0))), sigma2);
        partials = hn::Create4(d, d_e, d_sigma2, hn::Zero(d), hn::Zero(d));
    }
    return hn::Sub(density, hn::Mul(half, hn::Log(d, sigma2)));
}

// std: u = e^2 / (sigma2 scale), w = power / (1 + u).
template <bool kPartials, class D>
HWY_INLINE hn::Vec<D> StudentTerm(D d, const Constants& c, hn::Vec<D> e, hn::Vec<D> sigma2, hn::Vec4<D>& partials)
{
    const auto one = hn::Set(d, 1.0);
    const auto half = hn::Set(d, 0.5);
    const auto power = hn::Set(d, c.power);
    const auto u = hn::Div(hn::Div(hn::Mul(e, e), sigma2), hn::Set(d, c.scale));
    // The vector log1p is not defined at +infinity (an overflowed e^2): log1p(inf) = inf, and a NaN stays a NaN.
    const auto log1p_u = hn::IfThenElse(hn::Lt(u, hn::Inf(d)), hn::Log1p(d, u), u);
    const auto density = hn::Sub(hn::Set(d, c.log_constant), hn::Mul(power, log1p_u));
    if (kPartials)
    {
        const auto inverse_variance = hn::Div(one, sigma2);
        const auto inverse_scale = hn::Div(one, hn::Set(d, c.scale));
        const auto w = hn::Div(power, hn::Add(one, u));
        const auto w_u = hn::Mul(w, u);
        const auto d_e = hn::Mul(hn::Mul(hn::Mul(hn::Mul(hn::Set(d, -2.0), w), e), inverse_variance), inverse_scale);
        const auto d_sigma2 = hn::Mul(hn::Sub(w_u, half), inverse_variance);
        partials = hn::Create4(d, d_e, d_sigma2, hn::Mul(w_u, inverse_scale), hn::Neg(log1p_u));
    }
    return hn::Sub(density, hn::Mul(half, hn::Log(d, sigma2)));
}

// ged: y = |e| / (sigma lambda), v = y^power = exp(power log y), slope = power v / y. A lane with y = 0 has v = 0 and
// no partial but the one of the variance.
template <bool kPartials, class D>
HWY_INLINE hn::Vec<D> GedTerm(D d, const Constants& c, hn::Vec<D> e, hn::Vec<D> sigma2, hn::Vec4<D>& partials)
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
    if (kPartials)
    {
        const auto inverse_variance = hn::Div(one, sigma2);
        const auto inverse_sigma = hn::Div(one, sigma);
        const auto inverse_scale = hn::Div(one, hn::Set(d, c.scale));
        const auto negative_half = hn::Set(d, -0.5);
        const auto slope = hn::IfThenElse(positive, hn::Mul(power, hn::Div(v, y)), zero);
        const auto sign = hn::CopySign(one, e);
        const auto d_e = hn::Mul(hn::Mul(hn::Mul(hn::Mul(negative_half, slope), sign), inverse_sigma), inverse_scale);
        const auto from_slope = hn::Mul(hn::Mul(hn::Mul(hn::Set(d, 0.25), slope), y), inverse_variance);
        const auto d_sigma2 = hn::Add(hn::Mul(negative_half, inverse_variance), from_slope);
        const auto d_scale = hn::Mul(hn::Mul(hn::Mul(half, slope), y), inverse_scale);
        partials = hn::Create4(d, d_e, d_sigma2, d_scale, hn::Mul(hn::Mul(negative_half, v), log_y));
    }
    return hn::Sub(density, hn::Mul(half, hn::Log(d, sigma2)));
}

template <Dist kDist, bool kPartials, class D>
HWY_INLINE hn::Vec<D> TermOf(D d, const Constants& c, hn::Vec<D> e, hn::Vec<D> sigma2, hn::Vec4<D>& partials)
{
    if (kDist == Dist::Std)
    {
        return StudentTerm<kPartials>(d, c, e, sigma2, partials);
    }
    if (kDist == Dist::Ged)
    {
        return GedTerm<kPartials>(d, c, e, sigma2, partials);
    }
    return NormalTerm<kPartials>(d, c, e, sigma2, partials);
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
        auto partials = hn::Create4(d, zero, zero, zero, zero);
        const auto term = TermOf<kDist, kPartials>(d, constants, e, sigma2, partials);
        total = hn::Add(total, hn::IfThenElse(inside, term, zero));
        if (a.terms != nullptr)
        {
            hn::StoreN(term, d, a.terms + i, used);
        }
        if (kPartials)
        {
            hn::StoreN(hn::Get4<kDe>(partials), d, a.d_e + i, used);
            hn::StoreN(hn::Get4<kDSigma2>(partials), d, a.d_sigma2 + i, used);
            sum_scale = hn::Add(sum_scale, hn::IfThenElse(inside, hn::Get4<kDScale>(partials), zero));
            sum_power = hn::Add(sum_power, hn::IfThenElse(inside, hn::Get4<kDPower>(partials), zero));
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
