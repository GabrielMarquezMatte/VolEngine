#pragma once

#include "fit/fit_result.hpp"

#include <cstddef>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace ve::dcc
{

// z = e / sigma of each fit (T x m row-major), sigma (T x m), and what every spec shares (both conventions).
struct StandardizedPanel
{
    std::size_t t = 0;
    std::size_t m = 0;
    std::vector<double> z;
    std::vector<double> sigma;
    // cov(z) with divisor T-1; z'z/T (Rmgarch FDCC objective); m x m.
    std::vector<double> qbar;
    std::vector<double> qbar_raw;
    // T^-1 sum n n' (Standard); cov(n) with divisor T-1 (Rmgarch); m x m. n = 1(z < 0) z.
    std::vector<double> nbar;
    std::vector<double> nbar_centered;
    // Largest eigenvalue of Qbar^-1/2 Nbar Qbar^-1/2 for each Nbar; NaN unless Qbar is positive definite.
    double delta = 0.0;
    double delta_centered = 0.0;
};

// The caller checked the fits with panel_problem.
StandardizedPanel standardize(std::span<const FitResult* const> fits);

// Why the fits cannot form a panel (at least 2 series, each with a finite estimate, one common T >= 100), or nullopt.
// Series are numbered from 1.
std::optional<std::string> panel_problem(std::span<const FitResult* const> fits);

}
