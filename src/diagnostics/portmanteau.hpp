#pragma once

#include <cstddef>
#include <optional>
#include <span>
#include <string>

namespace ve::diagnostics
{

struct TestResult
{
    double statistic;
    double p_value;
};

TestResult weighted_ljung_box(std::span<const double> z, std::size_t lag, std::size_t fitdf, bool squared);
TestResult weighted_arch_lm(std::span<const double> residuals, std::span<const double> sigma,
                            std::size_t lag, std::size_t fitdf);
std::optional<std::string> ljung_box_problem(std::size_t n, std::size_t lag, std::size_t fitdf, bool squared);
std::optional<std::string> arch_lm_problem(std::size_t n, std::size_t lag, std::size_t fitdf);

}
