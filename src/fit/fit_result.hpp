#pragma once

#include "core/spec.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace ve
{

enum class FitStatus : int32_t
{
    Converged = 0,
    MaxIter = 1,
    Failed = 2,
    InsufficientData = 3
};

// Series (sigma, residuals, std_residuals, fitted) have length nobs, or are empty without a finite estimate.
// sigma is the conditional standard deviation, not the variance.
struct FitResult
{
    FitStatus status = FitStatus::Failed;
    Spec spec{};
    FitConfig config{};
    std::vector<std::string> names;
    std::vector<double> estimate;
    std::vector<double> se_hessian;
    std::vector<double> se_robust;
    double loglik = std::numeric_limits<double>::quiet_NaN();
    std::size_t nobs = 0;
    std::vector<double> sigma;
    std::vector<double> residuals;
    std::vector<double> std_residuals;
    std::vector<double> fitted;
};

}
