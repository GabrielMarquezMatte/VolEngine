#pragma once

#include "dcc/panel.hpp"
#include "dcc/spec.hpp"

#include <cstddef>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace ve::dcc
{

// The DCC/aDCC persistence and the radius of each FDCC group stay below it.
constexpr double kPersistenceCeiling = 1.0 - 1e-8;
constexpr double kShapeFloor = 4.0;
constexpr double kShapeCeiling = 50.0;

// The positions of theta (and of u): a_1..a_A, b_1..b_B, g_1..g_G, then the mvt shape. DCC and aDCC: A = P,
// B = Q, G = P (aDCC) or 0; FDCC: A = B = the number of groups, G = 0.
struct DccLayout
{
    DccModel model = DccModel::Dcc;
    std::size_t a_count = 0;
    std::size_t b_count = 0;
    std::size_t g_count = 0;
    bool has_shape = false;
    std::vector<bool> fixed;
    std::vector<bool> started;
    // The fixed value or the user's start value; 0 elsewhere.
    std::vector<double> given;

    std::size_t b_begin() const { return a_count; }
    std::size_t g_begin() const { return a_count + b_count; }
    std::size_t shape() const { return g_begin() + g_count; }
    std::size_t count() const { return shape() + (has_shape ? 1 : 0); }
    bool is_fixed(std::size_t index) const { return fixed[index]; }
    bool is_estimated(std::size_t index) const { return !fixed[index]; }
    std::size_t estimated_count() const;
};

// The caller checked spec_problem(spec, m).
DccLayout layout_of(const DccSpec& spec, std::size_t m);

// Every parameter, fixed or not: dcca1.., dccb1.., dccg1.., mshape; fdcca1..G, fdccb1..G, mshape.
std::vector<std::string> parameter_names(const DccLayout& layout);

// The estimated ones: the names of a fit's table.
std::vector<std::string> estimated_names(const DccLayout& layout);

// Why spec cannot be fitted to m series, or nullopt: a model or dist out of range, orders outside 1..6, an FDCC
// that is not (1, 1) or whose groups do not label the m series 1..G with no group empty, groups on another
// model, an unknown fixed or start name, or more estimated parameters than kMaxParameterCount.
std::optional<std::string> spec_problem(const DccSpec& spec, std::size_t m);

// The delta that scales the aDCC g weights in the persistence: of the uncentered Nbar (Standard) or of the
// centered one (Rmgarch).
double constraint_delta(const StandardizedPanel& panel, const DccSpec& spec);

// The natural start: a_j 0.05/P, b_k 0.9/Q, g_j 0.05/P, shape 5, FDCC a 0.01 and b 0.9, then the user's start
// values and the fixed values.
std::vector<double> start_values(const DccLayout& layout);

// u and theta follow the layout, each u entry at the position of the natural one it maps to; fixed entries hold
// layout.given and their u is unused (0).
//   DCC/aDCC: the terms t = (a, b, delta g) are >= 0 and sum to P = logistic(v) B, with B = 1 - 1e-8 less the
//   fixed terms: v at the first free position and the free terms B logistic(v) softmax(w, 0), w at the others in
//   order, the last free term the reference. B <= 0 gives NaN free weights.
//   FDCC, per group: (a, b) = r (cos phi, sin phi), r = (1 - 1e-8) logistic(u_a), phi = (pi / 2) logistic(u_b);
//   with one of them fixed at f, the other is sqrt(c^2 - f^2) logistic(u), c = 1 - 1e-8 (NaN when f >= c). The
//   positive definite intercept is left to the objective.
//   shape = 4 + 46 logistic(u).
// delta > 0 is read by the aDCC only. Instantiated for T = double, ad::Dual<8>, 16, 32 and kMaxParameterCount.
template <class T>
void to_natural(std::span<const T> u, std::span<T> theta, const DccLayout& layout, double delta);

// The inverse of to_natural on a theta that satisfies is_valid; u is 0 at the fixed entries.
void to_unconstrained(std::span<const double> theta, std::span<double> u, const DccLayout& layout, double delta);

// Every entry finite; DCC/aDCC: weights >= 0 with sum a + sum b + delta sum g < 1 - 1e-8 (a finite delta in the
// aDCC); FDCC: a_g, b_g > 0 with a_g^2 + b_g^2 < (1 - 1e-8)^2; shape in (4, 50).
bool is_valid(std::span<const double> theta, const DccLayout& layout, double delta);

// The estimated entries of a full theta, in order, and back (the fixed ones from layout.given).
std::vector<double> estimated_part(const DccLayout& layout, std::span<const double> theta);
std::vector<double> full_parameters(const DccLayout& layout, std::span<const double> estimated);

}
