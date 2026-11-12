#pragma once

#include "ad/dual.hpp"
#include "dist/moments.hpp"
#include "model/layout.hpp"
#include "model/parameter_map.hpp"

#include <array>
#include <cstddef>
#include <limits>
#include <span>

namespace ve::model
{

constexpr double kPersistenceCeiling = 1.0 - 1e-8;
constexpr double kBoundary = 1e-8;
// gjrGARCH: 2 p + q.
constexpr std::size_t kMaxWeightCount = 3 * kMaxOrder;

template <class T>
using WeightArray = std::array<T, kMaxWeightCount>;

// The weight block seen as terms t_j >= 0, one per position, summing to the persistence P, with a ceiling c:
//   sGARCH, csGARCH, iGARCH (alpha, beta), eGARCH (beta): the value itself;
//   apARCH, fGARCH: alpha_i kappa_i, beta_j;
//   gjrGARCH: (1 - pi) alpha_i, pi (alpha_i + gamma_i) in the alpha and gamma positions, but alpha_i - max(0, -gamma_i),
//   max(0, -gamma_i) + pi gamma_i when gamma_i is fixed and alpha_i free; pi = P(z < 0).
// c is 1 in iGARCH, the fixed rho in csGARCH and 1 - 1e-8 otherwise. NaN where pi or kappa is not computable.
struct WeightTerms
{
    std::array<double, 3 * kMaxOrder> terms{};
    std::size_t count = 0;
    double ceiling = 0.0;
};

WeightTerms weight_terms(std::span<const double> theta, const Layout& layout);

// Writes the weight block entries that are not fixed from their terms (the inverse of the table above); the
// iGARCH derived beta_q is 1 - the other weights.
void natural_from_terms(const WeightTerms& terms, std::span<double> theta, const Layout& layout);

struct WeightBlock
{
    std::size_t begin;
    std::size_t count;
};

inline std::size_t count_of(int order)
{
    return static_cast<std::size_t>(order);
}

inline WeightBlock weight_block(const Layout& layout)
{
    std::size_t p = count_of(layout.arch);
    std::size_t q = count_of(layout.garch);
    if (layout.model == Model::EGarch)
    {
        return {layout.beta_begin(), q};
    }
    if (layout.model == Model::GjrGarch)
    {
        return {layout.alpha_begin(), 2 * p + q};
    }
    return {layout.alpha_begin(), p + q};
}

inline double weight_ceiling(const Layout& layout)
{
    if (layout.model == Model::IGarch)
    {
        return 1.0;
    }
    if (layout.model == Model::CsGarch && layout.is_fixed(layout.eta1_begin()))
    {
        return layout.given[layout.eta1_begin()];
    }
    return kPersistenceCeiling;
}

// pi = P(z < 0) splits the gjrGARCH alpha_i + gamma_i between the negative and the positive shocks. alpha_i is
// read back as w / (1 - pi) and gamma_i as w / pi - alpha_i, which lose every digit as pi nears 0 or 1, so a pi
// within kBoundary of either end is not computable: it comes back NaN.
template <class T>
T probability_below_zero_of(std::span<const T> theta, const Layout& layout)
{
    T below = dist::probability_below_zero(layout.dist, shape_parameters_of(theta, layout));
    double value = ad::value_of(below);
    if (!(value >= kBoundary && value <= 1.0 - kBoundary))
    {
        below = T{std::numeric_limits<double>::quiet_NaN()};
    }
    return below;
}

// pi in gjrGARCH; the other models do not read it.
template <class T>
T below_zero_of(std::span<const T> theta, const Layout& layout)
{
    if (layout.model != Model::GjrGarch)
    {
        return T{0.5};
    }
    return probability_below_zero_of<T>(theta, layout);
}

// kappa_i when it is a positive finite double, NaN otherwise: alpha_i is read back as t / kappa_i.
template <class T>
T computable_moment(std::span<const T> theta, const Layout& layout, std::size_t i)
{
    T moment = shock_moment<T>(theta, layout, i);
    double value = ad::value_of(moment);
    if (!(value > 0.0 && value < std::numeric_limits<double>::infinity()))
    {
        moment = T{std::numeric_limits<double>::quiet_NaN()};
    }
    return moment;
}

// A fixed gjrGARCH gamma_i < 0 asks alpha_i >= -gamma_i, so a free alpha_i carries max(0, -gamma_i) outside its
// term, and the term can take any share of the budget.
inline bool alpha_carries_gamma(const Layout& layout, std::size_t i)
{
    return layout.model == Model::GjrGarch && layout.is_fixed(layout.gamma_begin() + i) &&
           layout.is_estimated(layout.alpha_begin() + i);
}

template <class T>
T negative_part(const T& x)
{
    if (ad::value_of(x) < 0.0)
    {
        return -x;
    }
    return T{0.0};
}

template <class T>
T gjr_alpha_term(std::span<const T> theta, const Layout& layout, std::size_t i, const T& below)
{
    const T& alpha = theta[layout.alpha_begin() + i];
    if (alpha_carries_gamma(layout, i))
    {
        return alpha - negative_part(theta[layout.gamma_begin() + i]);
    }
    return (1.0 - below) * alpha;
}

template <class T>
T gjr_gamma_term(std::span<const T> theta, const Layout& layout, std::size_t i, const T& below)
{
    const T& gamma = theta[layout.gamma_begin() + i];
    if (alpha_carries_gamma(layout, i))
    {
        return negative_part(gamma) + below * gamma;
    }
    return below * (theta[layout.alpha_begin() + i] + gamma);
}

// The term at theta[index], a position of the weight block; below = pi.
template <class T>
T term_at(std::span<const T> theta, const Layout& layout, std::size_t index, const T& below)
{
    bool alpha = index < layout.beta_begin();
    bool gjr = layout.model == Model::GjrGarch;
    if (gjr && alpha)
    {
        return gjr_alpha_term<T>(theta, layout, index - layout.alpha_begin(), below);
    }
    if (gjr && index >= layout.gamma_begin())
    {
        return gjr_gamma_term<T>(theta, layout, index - layout.gamma_begin(), below);
    }
    if (layout.is_power_family() && alpha)
    {
        return theta[index] * computable_moment<T>(theta, layout, index - layout.alpha_begin());
    }
    return theta[index];
}

// The natural value at position index from its term t (the inverse of term_at). A gjrGARCH gamma_i reads alpha_i,
// which must hold its natural value already.
template <class T>
T natural_at(std::span<const T> theta, const Layout& layout, std::size_t index, const T& term, const T& below)
{
    bool alpha = index < layout.beta_begin();
    bool gjr = layout.model == Model::GjrGarch;
    std::size_t i = index - layout.alpha_begin();
    if (gjr && alpha && alpha_carries_gamma(layout, i))
    {
        return term + negative_part(theta[layout.gamma_begin() + i]);
    }
    if (gjr && alpha)
    {
        return term / (1.0 - below);
    }
    if (gjr && index >= layout.gamma_begin())
    {
        return term / below - theta[layout.alpha_begin() + (index - layout.gamma_begin())];
    }
    if (layout.is_power_family() && alpha)
    {
        return term / computable_moment<T>(theta, layout, i);
    }
    return term;
}

template <class T>
WeightArray<T> terms_of(std::span<const T> theta, const Layout& layout)
{
    WeightBlock block = weight_block(layout);
    T below = below_zero_of<T>(theta, layout);
    WeightArray<T> terms{};
    for (std::size_t j = 0; j < block.count; ++j)
    {
        terms[j] = term_at<T>(theta, layout, block.begin + j, below);
    }
    return terms;
}

// The ceiling less the terms of the fixed positions: what the free terms share.
template <class T>
T budget_of(std::span<const T> theta, const Layout& layout, const T& below)
{
    WeightBlock block = weight_block(layout);
    T budget{weight_ceiling(layout)};
    for (std::size_t index = block.begin; index < block.begin + block.count; ++index)
    {
        if (layout.is_fixed(index))
        {
            budget = budget - term_at<T>(theta, layout, index, below);
        }
    }
    return budget;
}

// The positions of the weight block whose parameter is estimated, in order.
struct FreePositions
{
    std::array<std::size_t, kMaxWeightCount> index{};
    std::size_t count = 0;
};

inline FreePositions free_positions(const Layout& layout)
{
    WeightBlock block = weight_block(layout);
    FreePositions free;
    for (std::size_t index = block.begin; index < block.begin + block.count; ++index)
    {
        if (layout.is_estimated(index))
        {
            free.index[free.count++] = index;
        }
    }
    return free;
}

template <class T>
void write_derived_weight(std::span<T> theta, const Layout& layout)
{
    T derived{1.0};
    for (const T& weight : theta.subspan(layout.alpha_begin(), layout.derived() - layout.alpha_begin()))
    {
        derived = derived - weight;
    }
    theta[layout.derived()] = derived;
}

}
