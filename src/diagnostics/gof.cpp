#include "diagnostics/gof.hpp"

#include "diagnostics/tails.hpp"
#include "dist/cdf.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace ve::diagnostics
{

namespace
{

std::vector<double> sorted_finite(std::span<const double> z)
{
    std::vector<double> sorted(z.begin(), z.end());
    for (double& value : sorted)
    {
        if (!std::isfinite(value))
        {
            value = 0.0;
        }
    }
    std::ranges::sort(sorted);
    return sorted;
}

std::size_t group_of(double probability, std::size_t groups)
{
    if (probability >= 1.0)
    {
        return groups - 1;
    }
    double index = probability * static_cast<double>(groups);
    if (index >= static_cast<double>(groups))
    {
        return groups - 1;
    }
    return std::min(static_cast<std::size_t>(index), groups - 1);
}

double pearson_statistic(std::span<const double> probabilities, std::size_t groups)
{
    double expected = static_cast<double>(probabilities.size()) / static_cast<double>(groups);
    std::size_t previous = 0;
    std::size_t count = 0;
    double statistic = 0.0;
    for (double probability : probabilities)
    {
        if (!std::isfinite(probability))
        {
            return std::numeric_limits<double>::quiet_NaN();
        }
        std::size_t group = group_of(probability, groups);
        if (group != previous)
        {
            double difference = static_cast<double>(count) - expected;
            statistic += difference * difference / expected + static_cast<double>(group - previous - 1) * expected;
            count = 0;
            previous = group;
        }
        ++count;
    }
    double difference = static_cast<double>(count) - expected;
    return statistic + difference * difference / expected;
}

}

TestResult pearson_gof(std::span<const double> z, Dist dist, const dist::ShapeParameters<double>& parameters,
                       std::size_t groups)
{
    std::vector<double> sorted = sorted_finite(z);
    std::vector<double> probabilities = dist::sorted_cdf(dist, parameters, sorted);
    double statistic = pearson_statistic(probabilities, groups);
    return {statistic, chi_squared_upper(statistic, static_cast<double>(groups - 1))};
}

}
