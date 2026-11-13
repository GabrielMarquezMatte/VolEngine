#pragma once

#include "core/spec.hpp"
#include "dcc/copula/margins.hpp"
#include "dcc/copula/scores.hpp"
#include "dcc/fit.hpp"
#include "dcc/panel.hpp"
#include "dcc/spec.hpp"
#include "fit/fit_result.hpp"

#include <array>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace ve::dcc::copula
{

// What the copula specs of a batch share: the univariate panel (z, and sigma for H_t), the u of each transformation
// and convention a spec asks for (computed once; uniforms in margins.hpp), and sum_i l_i of the fits.
struct CopulaInputs
{
    std::shared_ptr<const StandardizedPanel> univariate;
    // Indexed by uniform_index; empty unless some spec asks for it.
    std::array<std::vector<double>, 4> u;
    // sin(pi tau_b / 2) of each u a static Kendall spec reads; nullopt when it is not positive definite (the spec
    // then fails) or no spec reads it.
    std::array<std::optional<std::vector<double>>, 4> kendall;
    double univariate_loglik = 0.0;
    // The scores of the one spec these inputs were built for (inputs_at), at the shape scores_shape; copula_terms
    // reuses them when theta's shape has the same bits, which every step but one on the shape does.
    std::shared_ptr<const CopulaPanel> scores;
    double scores_shape = 0.0;

    // The u of spec (spec.copula is not None, and copula_inputs saw a spec like it).
    std::span<const double> uniforms_of(const DccSpec& spec) const;
};

// The fits form a panel (panel_problem) and specs are the batch's; specs without a copula ask for nothing.
CopulaInputs copula_inputs(std::span<const FitResult* const> fits, std::span<const DccSpec> specs);

// sum_i l_i + sum_t c_t of plan 4b's "ℓ da cópula" rule at theta (every parameter, natural, layout order), writing
// R_t (T x m x m) unless r is empty. The dynamic models run the plan 4a recursion on the scores (Rmgarch: at
// Stage::Objective, the zero presample, for everything); the static model reads R from theta (C1..) or from the
// scores (Kendall). NaN outside the domain.
double copula_loglik(const CopulaInputs& inputs, const DccSpec& spec, std::span<const double> theta,
                     std::span<double> r = {});

// The summands c_t of copula_loglik without the univariate sum, NaN where it is NaN.
std::vector<double> copula_terms(const CopulaInputs& inputs, const DccSpec& spec, std::span<const double> theta,
                                 std::span<double> r = {});

// The inputs of spec at a perturbed univariate point (z and sigma of univariate, each series' estimate in margins),
// for the two-stage standard errors. Standard keeps the empirical ranks and the Kendall matrix of base (ranks are
// piecewise constant in theta); Rmgarch recomputes both, as rmgarch's finite differences do (plan 4b notes,
// section 6). univariate_loglik is 0.
// The scores are computed once there, at shape (unused without the mvt).
CopulaInputs inputs_at(std::shared_ptr<const StandardizedPanel> univariate, std::span<const Margin> margins,
                       const DccSpec& spec, const CopulaInputs& base, double shape);

// A DccFitResult like fit_dcc's: names, estimate and status of spec, loglik = copula_loglik at the estimate, r,
// and the univariate panel (H_t from its sigma). A free mvt shape is the Brent maximum of the profile over
// [4, 50] (30 bits, 100 iterations; MaxIter when they run out), each profile point a fit_dcc with the shape fixed
// from the default start. The caller checked spec_problem(spec, m); spec.copula is not None.
DccFitResult fit_copula(const CopulaInputs& inputs, const DccSpec& spec, const FitConfig& config);

}
