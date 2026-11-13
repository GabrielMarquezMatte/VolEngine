#include "dcc/parameter_map.hpp"

#include "ad/dual.hpp"
#include "ad/links.hpp"
#include "core/spec.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numbers>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace ve::dcc
{

namespace
{

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kQuarterTurn = std::numbers::pi / 2.0;

using ad::log_ratio;
using ad::logistic;

// --- layout and names ---

DccLayout counts_of(const DccSpec& spec, std::size_t m)
{
    DccLayout layout;
    layout.model = spec.model;
    layout.has_shape = spec.dist == DccDist::MvT;
    if (spec.model == DccModel::Static)
    {
        layout.c_count = correlation_coefficients(spec, m);
        return layout;
    }
    if (spec.model == DccModel::FDcc)
    {
        layout.a_count = static_cast<std::size_t>(std::ranges::max(spec.groups));
        layout.b_count = layout.a_count;
        return layout;
    }
    layout.a_count = static_cast<std::size_t>(spec.p);
    layout.b_count = static_cast<std::size_t>(spec.q);
    layout.g_count = spec.model == DccModel::ADcc ? layout.a_count : 0;
    return layout;
}

void append_block(std::vector<std::string>& names, const char* prefix, std::size_t count)
{
    for (std::size_t i = 1; i <= count; ++i)
    {
        names.push_back(prefix + std::to_string(i));
    }
}

std::optional<std::size_t> index_of(const std::vector<std::string>& names, const std::string& name)
{
    auto found = std::ranges::find(names, name);
    if (found == names.end())
    {
        return std::nullopt;
    }
    return static_cast<std::size_t>(found - names.begin());
}

// The counts with the fixed and start values of spec, whose names the caller checked.
DccLayout layout_with_values(const DccSpec& spec, std::size_t m)
{
    DccLayout layout = counts_of(spec, m);
    const std::vector<std::string> names = parameter_names(layout);
    layout.fixed.assign(names.size(), false);
    layout.started.assign(names.size(), false);
    layout.given.assign(names.size(), 0.0);
    for (const ParameterValue& entry : spec.fixed)
    {
        const std::size_t index = *index_of(names, entry.name);
        layout.fixed[index] = true;
        layout.given[index] = entry.value;
    }
    for (const ParameterValue& entry : spec.start)
    {
        const std::size_t index = *index_of(names, entry.name);
        if (layout.is_estimated(index))
        {
            layout.started[index] = true;
            layout.given[index] = entry.value;
        }
    }
    return layout;
}

// --- problems ---

std::string joined(const std::vector<std::string>& names)
{
    std::string text;
    for (const std::string& name : names)
    {
        if (!text.empty())
        {
            text += ' ';
        }
        text += name;
    }
    return text;
}

std::optional<std::string> kind_problem(const DccSpec& spec)
{
    const auto model = static_cast<int>(spec.model);
    const auto dist = static_cast<int>(spec.dist);
    const auto copula = static_cast<int>(spec.copula);
    const auto method = static_cast<int>(spec.static_method);
    if (model < 0 || model > static_cast<int>(DccModel::Static))
    {
        return "unknown dcc model " + std::to_string(model);
    }
    if (dist < 0 || dist > static_cast<int>(DccDist::MvLaplace))
    {
        return "unknown dcc dist " + std::to_string(dist);
    }
    if (copula < 0 || copula > static_cast<int>(DccCopula::Empirical))
    {
        return "unknown dcc copula " + std::to_string(copula);
    }
    if (method < 0 || method > static_cast<int>(StaticMethod::Ml))
    {
        return "unknown static_method " + std::to_string(method);
    }
    return std::nullopt;
}

bool starts_shape(const DccSpec& spec)
{
    return std::ranges::any_of(spec.start, [](const ParameterValue& entry) { return entry.name == "mshape"; });
}

// The combinations of plan 4b's "Entrada e erros" table.
std::optional<std::string> copula_problem(const DccSpec& spec)
{
    const bool copula = spec.copula != DccCopula::None;
    if (spec.model == DccModel::Static && !copula)
    {
        return "the static model needs a copula (parametric or empirical)";
    }
    if (spec.static_method != StaticMethod::Default && spec.model != DccModel::Static)
    {
        return "static_method is only for the static model";
    }
    if (!copula)
    {
        return std::nullopt;
    }
    if (spec.model == DccModel::FDcc)
    {
        return "the fdcc model has no copula";
    }
    if (spec.dist == DccDist::MvLaplace)
    {
        return "the mvlaplace copula is not supported; use mvnorm or mvt";
    }
    if (spec.dist == DccDist::MvT && starts_shape(spec))
    {
        return "a copula mshape takes no start: a bracketing search finds it; fix it or leave it free";
    }
    return std::nullopt;
}

std::optional<std::string> order_problem(const DccSpec& spec)
{
    const bool in_range = spec.p >= 1 && spec.p <= kMaxOrder && spec.q >= 1 && spec.q <= kMaxOrder;
    if (!in_range)
    {
        return "dcc orders must be between 1 and " + std::to_string(kMaxOrder);
    }
    if (spec.model == DccModel::FDcc && (spec.p != 1 || spec.q != 1))
    {
        return "the fdcc model has order (1, 1)";
    }
    if (spec.model == DccModel::Static && (spec.p != 1 || spec.q != 1))
    {
        return "the static model takes no order; leave it at (1, 1)";
    }
    return std::nullopt;
}

// FDCC: m labels in 1..G, every group with a series.
std::optional<std::string> groups_problem(const DccSpec& spec, std::size_t m)
{
    if (spec.model != DccModel::FDcc)
    {
        if (spec.groups.empty())
        {
            return std::nullopt;
        }
        return "groups are only for the fdcc model";
    }
    if (spec.groups.empty() || spec.groups.size() != m)
    {
        return "groups must have one group per series (" + std::to_string(m) + "), not " +
               std::to_string(spec.groups.size());
    }
    if (std::ranges::min(spec.groups) < 1)
    {
        return "groups must be labelled from 1";
    }
    const int count = std::ranges::max(spec.groups);
    for (int group = 1; group <= count; ++group)
    {
        if (std::ranges::find(spec.groups, group) == spec.groups.end())
        {
            return "group " + std::to_string(group) + " has no series";
        }
    }
    return std::nullopt;
}

std::optional<std::string> unknown_name_problem(const DccSpec& spec, const std::vector<std::string>& names)
{
    for (const auto* list : {&spec.fixed, &spec.start})
    {
        for (const ParameterValue& entry : *list)
        {
            if (!index_of(names, entry.name))
            {
                return "unknown parameter " + entry.name + "; the spec has: " + joined(names);
            }
        }
    }
    return std::nullopt;
}

// The optimiser differentiates with ad::Dual, built up to kMaxParameterCount entries; the static fit does not.
std::optional<std::string> width_problem(const DccSpec& spec, std::size_t m)
{
    const std::size_t count = layout_with_values(spec, m).estimated_count();
    if (count <= kMaxParameterCount || spec.model == DccModel::Static)
    {
        return std::nullopt;
    }
    return "the spec estimates " + std::to_string(count) + " dcc parameters; at most " +
           std::to_string(kMaxParameterCount) + " are supported";
}

// --- weights (DCC, aDCC) ---

// The factor that takes a weight to its term: delta for g, 1 for a and b.
double term_scale(const DccLayout& layout, std::size_t index, double delta)
{
    if (index >= layout.g_begin())
    {
        return delta;
    }
    return 1.0;
}

std::vector<std::size_t> free_weights(const DccLayout& layout)
{
    std::vector<std::size_t> free;
    for (std::size_t index = 0; index < layout.shape(); ++index)
    {
        if (layout.is_estimated(index))
        {
            free.push_back(index);
        }
    }
    return free;
}

// The ceiling less the fixed terms: what the free terms share.
double weight_budget(const DccLayout& layout, double delta)
{
    double budget = kPersistenceCeiling;
    for (std::size_t index = 0; index < layout.shape(); ++index)
    {
        if (layout.is_fixed(index))
        {
            budget -= term_scale(layout, index, delta) * layout.given[index];
        }
    }
    return budget;
}

// theta[free[i]] = term_i, terms = budget logistic(u[free[0]]) softmax(u[free[1..]], 0).
template <class T>
void free_terms_to_natural(std::span<const T> u, std::span<T> theta, const std::vector<std::size_t>& free,
                           double budget)
{
    std::vector<T> logits(free.size() - 1);
    for (std::size_t i = 1; i < free.size(); ++i)
    {
        logits[i - 1] = u[free[i]];
    }
    std::vector<T> terms(free.size());
    ad::scaled_softmax<T>(logits, budget * logistic(u[free[0]]), terms);
    for (std::size_t i = 0; i < free.size(); ++i)
    {
        theta[free[i]] = terms[i];
    }
}

template <class T>
void weights_to_natural(std::span<const T> u, std::span<T> theta, const DccLayout& layout, double delta)
{
    const std::vector<std::size_t> free = free_weights(layout);
    if (free.empty())
    {
        return;
    }
    const double budget = weight_budget(layout, delta);
    if (!(budget > 0.0))
    {
        for (std::size_t index : free)
        {
            theta[index] = T{kNaN};
        }
        return;
    }
    free_terms_to_natural<T>(u, theta, free, budget);
    for (std::size_t index : free)
    {
        theta[index] = theta[index] / term_scale(layout, index, delta);
    }
}

void weights_to_unconstrained(std::span<const double> theta, std::span<double> u, const DccLayout& layout,
                              double delta)
{
    const std::vector<std::size_t> free = free_weights(layout);
    if (free.empty())
    {
        return;
    }
    std::vector<double> terms(free.size());
    double shared = 0.0;
    for (std::size_t i = 0; i < free.size(); ++i)
    {
        terms[i] = term_scale(layout, free[i], delta) * theta[free[i]];
        shared += terms[i];
    }
    u[free[0]] = log_ratio(shared, weight_budget(layout, delta) - shared);
    for (std::size_t i = 1; i < free.size(); ++i)
    {
        u[free[i]] = log_ratio(terms[i - 1], terms.back());
    }
}

bool weights_valid(std::span<const double> theta, const DccLayout& layout, double delta)
{
    if (layout.g_count > 0 && !std::isfinite(delta))
    {
        return false;
    }
    double persistence = 0.0;
    for (std::size_t index = 0; index < layout.shape(); ++index)
    {
        if (!(theta[index] >= 0.0))
        {
            return false;
        }
        persistence += term_scale(layout, index, delta) * theta[index];
    }
    return persistence < kPersistenceCeiling;
}

// --- groups (FDCC) ---

// How far the free weight of a group can go when the other one is fixed at `fixed`: sqrt(c^2 - fixed^2), NaN
// unless positive.
double room_beside(double fixed)
{
    const double room = kPersistenceCeiling * kPersistenceCeiling - fixed * fixed;
    if (!(room > 0.0))
    {
        return kNaN;
    }
    return std::sqrt(room);
}

template <class T>
void group_to_natural(std::span<const T> u, std::span<T> theta, std::size_t ia, std::size_t ib,
                      const DccLayout& layout)
{
    using std::cos;
    using std::sin;
    if (layout.is_estimated(ia) && layout.is_estimated(ib))
    {
        const T radius = kPersistenceCeiling * logistic(u[ia]);
        const T angle = kQuarterTurn * logistic(u[ib]);
        theta[ia] = radius * cos(angle);
        theta[ib] = radius * sin(angle);
        return;
    }
    if (layout.is_estimated(ib))
    {
        theta[ib] = room_beside(layout.given[ia]) * logistic(u[ib]);
    }
    if (layout.is_estimated(ia))
    {
        theta[ia] = room_beside(layout.given[ib]) * logistic(u[ia]);
    }
}

void group_to_unconstrained(std::span<const double> theta, std::span<double> u, std::size_t ia, std::size_t ib,
                            const DccLayout& layout)
{
    const double a = theta[ia];
    const double b = theta[ib];
    if (layout.is_estimated(ia) && layout.is_estimated(ib))
    {
        const double radius = std::hypot(a, b);
        const double angle = std::atan2(b, a);
        u[ia] = log_ratio(radius, kPersistenceCeiling - radius);
        u[ib] = log_ratio(angle, kQuarterTurn - angle);
        return;
    }
    if (layout.is_estimated(ib))
    {
        u[ib] = log_ratio(b, room_beside(a) - b);
    }
    if (layout.is_estimated(ia))
    {
        u[ia] = log_ratio(a, room_beside(b) - a);
    }
}

bool groups_valid(std::span<const double> theta, const DccLayout& layout)
{
    for (std::size_t group = 0; group < layout.a_count; ++group)
    {
        const double a = theta[group];
        const double b = theta[layout.b_begin() + group];
        if (!(a > 0.0 && b > 0.0 && a * a + b * b < kPersistenceCeiling * kPersistenceCeiling))
        {
            return false;
        }
    }
    return true;
}

template <class T>
void write_fixed(std::span<T> theta, const DccLayout& layout)
{
    for (std::size_t index = 0; index < theta.size(); ++index)
    {
        if (layout.is_fixed(index))
        {
            theta[index] = T{layout.given[index]};
        }
    }
}

}

std::size_t DccLayout::estimated_count() const
{
    return static_cast<std::size_t>(std::ranges::count(fixed, false));
}

DccLayout layout_of(const DccSpec& spec, std::size_t m)
{
    assert(!spec_problem(spec, m));
    return layout_with_values(spec, m);
}

std::vector<std::string> parameter_names(const DccLayout& layout)
{
    std::vector<std::string> names;
    names.reserve(layout.count());
    const bool fdcc = layout.model == DccModel::FDcc;
    append_block(names, fdcc ? "fdcca" : "dcca", layout.a_count);
    append_block(names, fdcc ? "fdccb" : "dccb", layout.b_count);
    append_block(names, "dccg", layout.g_count);
    append_block(names, "C", layout.c_count);
    if (layout.has_shape)
    {
        names.emplace_back("mshape");
    }
    return names;
}

std::vector<std::string> estimated_names(const DccLayout& layout)
{
    std::vector<std::string> names = parameter_names(layout);
    std::vector<std::string> estimated;
    for (std::size_t index = 0; index < names.size(); ++index)
    {
        if (layout.is_estimated(index))
        {
            estimated.push_back(std::move(names[index]));
        }
    }
    return estimated;
}

std::optional<std::string> spec_problem(const DccSpec& spec, std::size_t m)
{
    std::optional<std::string> problem = kind_problem(spec);
    if (!problem)
    {
        problem = order_problem(spec);
    }
    if (!problem)
    {
        problem = groups_problem(spec, m);
    }
    if (!problem)
    {
        problem = copula_problem(spec);
    }
    if (!problem)
    {
        problem = unknown_name_problem(spec, parameter_names(counts_of(spec, m)));
    }
    if (!problem)
    {
        problem = width_problem(spec, m);
    }
    return problem;
}

std::size_t correlation_coefficients(const DccSpec& spec, std::size_t m)
{
    if (spec.model != DccModel::Static || static_method_of(spec) != StaticMethod::Ml)
    {
        return 0;
    }
    return m * (m - 1) / 2;
}

StaticMethod static_method_of(const DccSpec& spec)
{
    if (spec.static_method == StaticMethod::Default)
    {
        return StaticMethod::Kendall;
    }
    return spec.static_method;
}

double constraint_delta(const StandardizedPanel& panel, const DccSpec& spec)
{
    if (spec.convention == Convention::Rmgarch)
    {
        return panel.delta_centered;
    }
    return panel.delta;
}

std::vector<double> start_values(const DccLayout& layout)
{
    std::vector<double> theta(layout.count());
    double a = 0.01;
    double b = 0.9;
    if (layout.model != DccModel::FDcc)
    {
        a = 0.05 / static_cast<double>(layout.a_count);
        b = 0.9 / static_cast<double>(layout.b_count);
    }
    std::fill_n(theta.begin(), layout.a_count, a);
    std::fill_n(theta.begin() + static_cast<std::ptrdiff_t>(layout.b_begin()), layout.b_count, b);
    std::fill_n(theta.begin() + static_cast<std::ptrdiff_t>(layout.g_begin()), layout.g_count, a);
    if (layout.has_shape)
    {
        theta[layout.shape()] = 5.0;
    }
    for (std::size_t index = 0; index < theta.size(); ++index)
    {
        if (layout.is_fixed(index) || layout.started[index])
        {
            theta[index] = layout.given[index];
        }
    }
    return theta;
}

template <class T>
void to_natural(std::span<const T> u, std::span<T> theta, const DccLayout& layout, double delta)
{
    assert(u.size() == layout.count() && theta.size() == layout.count());
    write_fixed<T>(theta, layout);
    for (std::size_t index = layout.c_begin(); index < layout.shape(); ++index)
    {
        if (layout.is_estimated(index))
        {
            theta[index] = u[index];
        }
    }
    if (layout.model == DccModel::FDcc)
    {
        for (std::size_t group = 0; group < layout.a_count; ++group)
        {
            group_to_natural<T>(u, theta, group, layout.b_begin() + group, layout);
        }
    }
    if (layout.model != DccModel::FDcc && layout.model != DccModel::Static)
    {
        weights_to_natural<T>(u, theta, layout, delta);
    }
    if (layout.has_shape && layout.is_estimated(layout.shape()))
    {
        theta[layout.shape()] = kShapeFloor + (kShapeCeiling - kShapeFloor) * logistic(u[layout.shape()]);
    }
}

template void to_natural<double>(std::span<const double>, std::span<double>, const DccLayout&, double);
template void to_natural<ad::Dual<8>>(std::span<const ad::Dual<8>>, std::span<ad::Dual<8>>, const DccLayout&,
                                      double);
template void to_natural<ad::Dual<16>>(std::span<const ad::Dual<16>>, std::span<ad::Dual<16>>, const DccLayout&,
                                       double);
template void to_natural<ad::Dual<32>>(std::span<const ad::Dual<32>>, std::span<ad::Dual<32>>, const DccLayout&,
                                       double);
using DualMax = ad::Dual<kMaxParameterCount>;
template void to_natural<DualMax>(std::span<const DualMax>, std::span<DualMax>, const DccLayout&, double);

void to_unconstrained(std::span<const double> theta, std::span<double> u, const DccLayout& layout, double delta)
{
    assert(theta.size() == layout.count() && u.size() == layout.count());
    std::ranges::fill(u, 0.0);
    for (std::size_t index = layout.c_begin(); index < layout.shape(); ++index)
    {
        if (layout.is_estimated(index))
        {
            u[index] = theta[index];
        }
    }
    if (layout.model == DccModel::FDcc)
    {
        for (std::size_t group = 0; group < layout.a_count; ++group)
        {
            group_to_unconstrained(theta, u, group, layout.b_begin() + group, layout);
        }
    }
    if (layout.model != DccModel::FDcc && layout.model != DccModel::Static)
    {
        weights_to_unconstrained(theta, u, layout, delta);
    }
    if (layout.has_shape && layout.is_estimated(layout.shape()))
    {
        const double shape = theta[layout.shape()];
        u[layout.shape()] = log_ratio(shape - kShapeFloor, kShapeCeiling - shape);
    }
}

bool is_valid(std::span<const double> theta, const DccLayout& layout, double delta)
{
    assert(theta.size() == layout.count());
    if (!std::ranges::all_of(theta, [](double value) { return std::isfinite(value); }))
    {
        return false;
    }
    if (layout.has_shape && !(theta[layout.shape()] > kShapeFloor && theta[layout.shape()] < kShapeCeiling))
    {
        return false;
    }
    if (layout.model == DccModel::FDcc)
    {
        return groups_valid(theta, layout);
    }
    if (layout.model == DccModel::Static)
    {
        return true;
    }
    return weights_valid(theta, layout, delta);
}

std::vector<std::size_t> estimated_positions(const DccLayout& layout)
{
    std::vector<std::size_t> free;
    for (std::size_t index = 0; index < layout.count(); ++index)
    {
        if (layout.is_estimated(index))
        {
            free.push_back(index);
        }
    }
    return free;
}

std::vector<double> estimated_part(const DccLayout& layout, std::span<const double> theta)
{
    assert(theta.size() == layout.count());
    std::vector<double> estimated;
    estimated.reserve(layout.estimated_count());
    for (std::size_t index = 0; index < theta.size(); ++index)
    {
        if (layout.is_estimated(index))
        {
            estimated.push_back(theta[index]);
        }
    }
    return estimated;
}

std::vector<double> full_parameters(const DccLayout& layout, std::span<const double> estimated)
{
    assert(estimated.size() == layout.estimated_count());
    std::vector<double> theta = layout.given;
    std::size_t next = 0;
    for (std::size_t index = 0; index < theta.size(); ++index)
    {
        if (layout.is_estimated(index))
        {
            theta[index] = estimated[next++];
        }
    }
    return theta;
}

}
