#pragma once

#include "core/spec.hpp"
#include "dcc/panel.hpp"
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

constexpr std::size_t kDcc3Rows = 1500;
constexpr std::size_t kDcc3Series = 3;

// rugarch's sGARCH(1,1)-norm estimates (mu, omega, alpha1, beta1) of the columns A, B and C of dcc3.csv, the data
// of the rmgarch DCC oracle.
constexpr std::array<std::array<double, 4>, kDcc3Series> kDcc3Coefficients{{
    {-0.00891999125885, 0.01947914006937, 0.07282178352751, 0.90657081832066},
    {-0.02778587421953, 0.06144179210890, 0.08407703380533, 0.84625809773984},
    {-0.00997194689148, 0.02565498174998, 0.05037006752284, 0.93387233803435},
}};

// The columns of tests/reference/data/dcc3.csv. Throws unless the file has kDcc3Rows rows of kDcc3Series values.
inline std::vector<std::vector<double>> read_dcc3()
{
    const std::string path = std::string(VE_TEST_DATA_DIR) + "/dcc3.csv";
    std::ifstream file(path);
    std::string line;
    if (!file || !std::getline(file, line))
    {
        throw std::runtime_error("cannot read " + path);
    }
    std::vector<std::vector<double>> columns(kDcc3Series);
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
    if (columns[0].size() != kDcc3Rows)
    {
        throw std::runtime_error(path + " has " + std::to_string(columns[0].size()) + " rows");
    }
    return columns;
}

inline std::vector<FitResult> dcc3_fits()
{
    const std::vector<std::vector<double>> columns = read_dcc3();
    std::vector<FitResult> fits;
    for (std::size_t i = 0; i < kDcc3Series; ++i)
    {
        const std::array<double, 4>& c = kDcc3Coefficients[i];
        Spec spec;
        spec.fixed = {{"mu", c[0]}, {"omega", c[1]}, {"alpha1", c[2]}, {"beta1", c[3]}};
        fits.push_back(fit_one(columns[i], spec, FitConfig{}));
    }
    return fits;
}

// The same fits estimated rather than fixed, so that their Hessians and scores exist: started at rugarch's
// estimates with tol 1, which stops the optimiser there.
inline std::vector<FitResult> dcc3_estimated_fits()
{
    const std::vector<std::vector<double>> columns = read_dcc3();
    std::vector<FitResult> fits;
    FitConfig config;
    config.tol = 1.0;
    for (std::size_t i = 0; i < kDcc3Series; ++i)
    {
        const std::array<double, 4>& c = kDcc3Coefficients[i];
        Spec spec;
        spec.start = {{"mu", c[0]}, {"omega", c[1]}, {"alpha1", c[2]}, {"beta1", c[3]}};
        fits.push_back(fit_one(columns[i], spec, config));
    }
    return fits;
}

inline dcc::StandardizedPanel dcc3_panel()
{
    const std::vector<FitResult> fits = dcc3_fits();
    const std::vector<const FitResult*> pointers{&fits[0], &fits[1], &fits[2]};
    if (dcc::panel_problem(pointers).has_value())
    {
        throw std::runtime_error("the dcc3 fits do not form a panel");
    }
    return dcc::standardize(pointers);
}

}
