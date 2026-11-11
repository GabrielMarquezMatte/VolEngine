#pragma once

#include "fit/fit_result.hpp"

namespace ve
{

enum class IcScale
{
    PerObs,
    Total
};

// k is the number of estimated parameters and n the number of observations. NaN without a finite loglik.
double aic(const FitResult& result, IcScale scale);
double bic(const FitResult& result, IcScale scale);
double hqic(const FitResult& result, IcScale scale);
double sic(const FitResult& result, IcScale scale);

}
