#pragma once

#include "core/panel.hpp"

#include <vector>

namespace ve::cov
{

// Covariance estimators over a t x n panel of returns, each series' mean removed. The caller guarantees a rectangular
// panel with t >= 2, n >= 1 and finite values. The matrices are n x n, row-major and exactly symmetric.

// Z'Z / (t - 1), as numpy.cov(rowvar=False).
std::vector<double> sample(const PanelView& panel);

struct Shrunk
{
    std::vector<double> covariance;
    double shrinkage = 0.0;
};

// The Ledoit-Wolf (2004) shrinkage of E = Z'Z / t towards mu * I, mu = tr(E) / n, as
// sklearn.covariance.ledoit_wolf(assume_centered=False); the intensity is in [0, 1], and 0 when E is mu * I.
Shrunk ledoit_wolf(const PanelView& panel);

}
