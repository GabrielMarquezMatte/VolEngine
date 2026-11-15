#ifndef VOLENGINE_CPP_TESTS_FIXTURE_DATA_HPP
#define VOLENGINE_CPP_TESTS_FIXTURE_DATA_HPP

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <volengine/volengine.hpp>

namespace ve::testing
{

// The columns of a CSV file of the repository's reference data (tests/reference/data): the header line is skipped and
// an empty or non-numeric cell is NaN.
inline std::vector<std::vector<double>> read_csv_columns(const std::string& name)
{
    std::ifstream file(std::string(VOLENGINE_FIXTURE_DIR) + "/" + name);
    if (!file)
    {
        throw std::runtime_error("cannot open fixture " + name);
    }
    std::vector<std::vector<double>> columns;
    std::string line;
    std::getline(file, line);
    while (std::getline(file, line))
    {
        std::stringstream cells(line);
        std::string cell;
        for (std::size_t column = 0; std::getline(cells, cell, ','); ++column)
        {
            columns.resize(std::max(columns.size(), column + 1));
            char* end = nullptr;
            const double value = std::strtod(cell.c_str(), &end);
            columns[column].push_back(end == cell.c_str() ? std::numeric_limits<double>::quiet_NaN() : value);
        }
    }
    return columns;
}

// The three univariate sGARCH(1,1) fits of the rmgarch panel dcc3 (tests/reference/data/dcc3.csv, columns A, B, C):
// each starts at the oracle's estimates and stops there (tol = 1), as in python/tests/oracles/test_dcc_oracles.py.
inline std::vector<ve::Fit> dcc3_univariate_fits(const ve::Context& context)
{
    constexpr double estimates[3][4] = {
        {-0.008920131111225086, 0.019478881309471997, 0.07282124034137887, 0.9065714955528129},
        {-0.02778242404762519, 0.061441088509364916, 0.084080533476537, 0.8462651839828331},
        {-0.009972193379566455, 0.025655072442689313, 0.050370071713741006, 0.9338721906860878},
    };
    constexpr const char* names[4] = {"mu", "omega", "alpha1", "beta1"};
    const auto columns = read_csv_columns("dcc3.csv");
    std::vector<ve::Fit> fits;
    for (std::size_t i = 0; i < 3; ++i)
    {
        ve::NamedValues start;
        for (std::size_t k = 0; k < 4; ++k)
        {
            start.emplace_back(names[k], estimates[i][k]);
        }
        fits.push_back(context.fit(ve::Spec{.start = start}, columns[i], ve::FitOptions{.tol = 1.0}).value());
    }
    return fits;
}

} // namespace ve::testing

#endif
