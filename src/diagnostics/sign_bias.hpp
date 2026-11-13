#pragma once

#include "diagnostics/portmanteau.hpp"

#include <span>

namespace ve::diagnostics
{

struct SignBias
{
    TestResult sign;
    TestResult negative;
    TestResult positive;
    TestResult joint;
};

SignBias sign_bias(std::span<const double> z, std::span<const double> residuals);

}
