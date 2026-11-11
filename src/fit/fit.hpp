#pragma once

#include "fit/fit_result.hpp"

#include <span>

namespace ve
{

FitResult fit_one(std::span<const double> r, const Spec& spec, const FitConfig& config);

}
