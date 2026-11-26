#pragma once

#include "dcc/panel.hpp"
#include "dcc/spec.hpp"

#include <memory>
#include <span>
#include <vector>

namespace ve::dcc::copula
{

// The copula scores epsilon as a panel: z = epsilon (T x m), sigma of the univariate panel (for H_t), and the
// moments of epsilon (refresh_moments). marginal_terms[t] = sum_i log f1(epsilon_it), f1 the standard normal or the
// standardized t, and marginal_log_density their sum.
struct CopulaPanel
{
    std::shared_ptr<const StandardizedPanel> panel;
    double marginal_log_density = 0.0;
    std::vector<double> marginal_terms;
    // log f1(epsilon_it) of each entry, T x m: what copula_panel_with_column sums again.
    std::vector<double> marginal_entries;
};

// epsilon = Phi^-1(u) (MvNorm), or the standardized t quantile t_nu^-1(u) sqrt((nu - 2) / nu) (MvT, shape = nu > 2).
// u is T x m like univariate.z, inside (0, 1).
CopulaPanel copula_panel(std::span<const double> u, const StandardizedPanel& univariate, DccDist dist, double shape);

// copula_panel(u, univariate, dist, shape) when only column `column` of u and of univariate differ from those base
// was built from at the same dist and shape (one series refiltered by the standard errors): the other scores and
// their moments are base's, and the result has the bits of copula_panel at O(T m) instead of O(T m^2).
CopulaPanel copula_panel_with_column(const CopulaPanel& base, std::span<const double> u,
                                     const StandardizedPanel& univariate, std::size_t column, DccDist dist,
                                     double shape);

}
