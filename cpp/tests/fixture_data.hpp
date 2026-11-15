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

} // namespace ve::testing

#endif
