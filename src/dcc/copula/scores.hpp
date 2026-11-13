#pragma once

#include "dcc/panel.hpp"
#include "dcc/spec.hpp"

#include <memory>
#include <span>

namespace ve::dcc::copula
{

// The copula scores epsilon as a panel: z = epsilon (T x m), sigma of the univariate panel (for H_t), and the
// moments of epsilon (refresh_moments). marginal_log_density = sum_t sum_i log f1(epsilon_it), f1 the standard
// normal or the standardized t.
struct CopulaPanel
{
    std::shared_ptr<const StandardizedPanel> panel;
    double marginal_log_density = 0.0;
};

// epsilon = Phi^-1(u) (MvNorm), or the standardized t quantile t_nu^-1(u) sqrt((nu - 2) / nu) (MvT, shape = nu > 2).
// u is T x m like univariate.z, inside (0, 1).
CopulaPanel copula_panel(std::span<const double> u, const StandardizedPanel& univariate, DccDist dist, double shape);

}
