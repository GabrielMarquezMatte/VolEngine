#pragma once

#include "core/spec.hpp"
#include "fit/fit.hpp"
#include "fit/fit_result.hpp"

#include <array>
#include <cstddef>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace ve::testing
{

constexpr std::size_t kCopula3Rows = 2000;
constexpr std::size_t kCopula3Series = 3;

// rugarch's sGARCH(1,1)-std estimates (mu, omega, alpha1, beta1, shape) of the columns A, B and C of copula3.csv,
// the data of the rmgarch Copula-DCC oracle (plan 4b notes, section 1).
constexpr std::array<std::array<double, 5>, kCopula3Series> kCopula3Coefficients{{
    {-0.0181865826694867, 0.0188827760630150, 0.0542214285957902, 0.9267446862078732, 6.9885886575033203},
    {0.0145422387458537, 0.0560886489944989, 0.0957365591243709, 0.8661253621409556, 5.3111792779986891},
    {0.00843170064471263, 0.04917700882790111, 0.06558797670577575, 0.89906386979536290, 6.68792316409210841},
}};

constexpr double kCopula3UnivariateLoglik = -8664.90170655746;

// The columns of tests/reference/data/copula3.csv. Throws unless the file has kCopula3Rows rows.
inline std::vector<std::vector<double>> read_copula3()
{
    const std::string path = std::string(VE_TEST_DATA_DIR) + "/copula3.csv";
    std::ifstream file(path);
    std::string line;
    if (!file || !std::getline(file, line))
    {
        throw std::runtime_error("cannot read " + path);
    }
    std::vector<std::vector<double>> columns(kCopula3Series);
    while (std::getline(file, line))
    {
        std::istringstream row(line);
        std::string cell;
        for (std::vector<double>& column : columns)
        {
            std::getline(row, cell, ',');
            column.push_back(std::stod(cell));
        }
    }
    if (columns[0].size() != kCopula3Rows)
    {
        throw std::runtime_error(path + " has " + std::to_string(columns[0].size()) + " rows");
    }
    return columns;
}

inline Spec copula3_spec(std::size_t series)
{
    const std::array<double, 5>& c = kCopula3Coefficients[series];
    Spec spec;
    spec.dist = Dist::Std;
    spec.fixed = {{"mu", c[0]}, {"omega", c[1]}, {"alpha1", c[2]}, {"beta1", c[3]}, {"shape", c[4]}};
    return spec;
}

// The univariate fits fixed at rugarch's estimates.
inline std::vector<FitResult> copula3_fits()
{
    const std::vector<std::vector<double>> columns = read_copula3();
    std::vector<FitResult> fits;
    for (std::size_t i = 0; i < kCopula3Series; ++i)
    {
        fits.push_back(fit_one(columns[i], copula3_spec(i), FitConfig{}));
    }
    return fits;
}

// The same fits estimated (started at rugarch's estimates with tol 1, which stops the optimiser there), so that
// their Hessians and scores exist for the two-stage standard errors.
inline std::vector<FitResult> copula3_estimated_fits()
{
    const std::vector<std::vector<double>> columns = read_copula3();
    std::vector<FitResult> fits;
    FitConfig config;
    config.tol = 1.0;
    for (std::size_t i = 0; i < kCopula3Series; ++i)
    {
        Spec spec = copula3_spec(i);
        spec.start = spec.fixed;
        spec.fixed.clear();
        fits.push_back(fit_one(columns[i], spec, config));
    }
    return fits;
}

}
