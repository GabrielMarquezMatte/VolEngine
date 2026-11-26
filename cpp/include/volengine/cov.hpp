#ifndef VOLENGINE_CPP_COV_HPP
#define VOLENGINE_CPP_COV_HPP

#include <cstddef>
#include <vector>
#include <volengine/error.hpp>
#include <volengine/input.hpp>

// Covariance estimators over a T x N panel of returns (series in columns), each series' mean removed. The matrices
// are N x N, row-major and exactly symmetric. A panel with ranges, fewer than 2 rows or a value that is not finite is
// an error with Status::invalid_argument.
namespace ve::cov
{

struct LedoitWolf
{
    std::vector<double> covariance;
    double shrinkage = 0.0;
};

// Z'Z / (T - 1), as numpy.cov(rowvar=False).
inline Result<std::vector<double>> sample(const Panel& panel)
{
    std::vector<double> covariance(panel.cols() * panel.cols());
    if (auto computed = detail::check(ve_cov_sample(panel.native(), covariance.data())); !computed)
    {
        return detail::failure(computed);
    }
    return covariance;
}

// The Ledoit-Wolf (2004) shrinkage of Z'Z / T towards mu * I, mu the mean variance, with its intensity in [0, 1], as
// sklearn.covariance.ledoit_wolf(assume_centered=False).
inline Result<LedoitWolf> ledoit_wolf(const Panel& panel)
{
    LedoitWolf result{.covariance = std::vector<double>(panel.cols() * panel.cols())};
    if (auto computed = detail::check(ve_cov_ledoit_wolf(panel.native(), result.covariance.data(), &result.shrinkage));
        !computed)
    {
        return detail::failure(computed);
    }
    return result;
}

}

#endif
