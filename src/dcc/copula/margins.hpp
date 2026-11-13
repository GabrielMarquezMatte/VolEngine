#pragma once

#include "dcc/panel.hpp"
#include "dcc/spec.hpp"
#include "fit/fit_result.hpp"

#include <span>
#include <vector>

namespace ve::dcc::copula
{

// Standard keeps u in [2^-53, 1 - 2^-53], so that no quantile is infinite.
constexpr double kUniformFloor = 0x1p-53;

// u, T x m row-major like panel.z, of each series (plan 4b, "Marginais"):
//   Parametric: F_i(z_it), the cdf of fit i's distribution at its estimated shape (fits[i] gives it);
//   Empirical: rank / (T + 1), ties broken by the order of t (fits unused).
// Rmgarch: empirical ecdf = #{z_is <= z_it} / T, then u > 0.99999 becomes 0.99999 and u < 1.5 * 2^-52 becomes
// 2^-52; Standard: the [kUniformFloor, 1 - kUniformFloor] cut. how is not None; the caller checked panel_problem.
std::vector<double> uniforms(std::span<const FitResult* const> fits, const StandardizedPanel& panel, DccCopula how,
                             Convention convention);

}
