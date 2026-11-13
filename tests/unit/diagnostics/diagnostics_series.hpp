#pragma once

#include <cmath>
#include <vector>

namespace ve::testing
{

struct DiagnosticsSeries
{
    std::vector<double> z;
    std::vector<double> sigma;
    std::vector<double> residuals;
};

inline DiagnosticsSeries diagnostics_series()
{
    DiagnosticsSeries series;
    for (int t = 1; t <= 40; ++t)
    {
        double z = std::sin(0.7 * t) + 0.3 * std::cos(1.9 * t);
        double sigma = 1.0 + 0.2 * std::sin(0.3 * t);
        series.z.push_back(z);
        series.sigma.push_back(sigma);
        series.residuals.push_back(z * sigma);
    }
    return series;
}

}
