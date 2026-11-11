#include "fit/criteria.hpp"

#include <cmath>
#include <cstddef>
#include <limits>

namespace ve
{

namespace
{

struct Counts
{
    double loglik;
    double parameters;
    double observations;
};

bool is_usable(const FitResult& result)
{
    return std::isfinite(result.loglik) && result.nobs > 0;
}

Counts counts_of(const FitResult& result)
{
    return {result.loglik, static_cast<double>(result.estimate.size()), static_cast<double>(result.nobs)};
}

double rescale(double per_observation, const Counts& counts, IcScale scale)
{
    if (scale == IcScale::Total)
    {
        return per_observation * counts.observations;
    }
    return per_observation;
}

double with_penalty(const FitResult& result, IcScale scale, double penalty_per_parameter)
{
    if (!is_usable(result))
    {
        return std::numeric_limits<double>::quiet_NaN();
    }
    Counts counts = counts_of(result);
    double total = -2.0 * counts.loglik + penalty_per_parameter * counts.parameters;
    return rescale(total / counts.observations, counts, scale);
}

}

double aic(const FitResult& result, IcScale scale)
{
    return with_penalty(result, scale, 2.0);
}

double bic(const FitResult& result, IcScale scale)
{
    return with_penalty(result, scale, std::log(static_cast<double>(result.nobs)));
}

double hqic(const FitResult& result, IcScale scale)
{
    return with_penalty(result, scale, 2.0 * std::log(std::log(static_cast<double>(result.nobs))));
}

double sic(const FitResult& result, IcScale scale)
{
    if (!is_usable(result))
    {
        return std::numeric_limits<double>::quiet_NaN();
    }
    Counts counts = counts_of(result);
    double fit_term = -2.0 * counts.loglik / counts.observations;
    double penalty_term = std::log((counts.observations + 2.0 * counts.parameters) / counts.observations);
    return rescale(fit_term + penalty_term, counts, scale);
}

}
