#include "optim/transform.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <limits>

namespace ve::optim
{

namespace
{

constexpr double kPersistenceCeiling = 1.0 - 1e-8;
constexpr std::size_t kVarianceParamCount = 3;

struct PersistenceSplit
{
    double persistence;
    double persistence_slope;
    double share;
    double share_slope;
};

std::size_t mean_param_count(bool include_mean)
{
    return include_mean ? 1 : 0;
}

double logistic(double value)
{
    if (value >= 0.0)
    {
        return 1.0 / (1.0 + std::exp(-value));
    }
    double exponential = std::exp(value);
    return exponential / (1.0 + exponential);
}

double log_ratio(double numerator, double denominator)
{
    constexpr double kFloor = std::numeric_limits<double>::min();
    return std::log(std::max(numerator, kFloor)) - std::log(std::max(denominator, kFloor));
}

PersistenceSplit split_persistence(double v, double w)
{
    double persistence_unit = logistic(v);
    double share = logistic(w);
    double persistence = persistence_unit * kPersistenceCeiling;
    return {persistence, persistence * (1.0 - persistence_unit), share, share * (1.0 - share)};
}

}

void sgarch_to_natural(std::span<const double> u, std::span<double> theta, bool include_mean,
                       const Standardization& standardization)
{
    std::size_t offset = mean_param_count(include_mean);
    assert(u.size() == offset + kVarianceParamCount);
    assert(theta.size() == u.size());
    assert(standardization.scale > 0.0);
    double scale = standardization.scale;
    if (include_mean)
    {
        theta[0] = standardization.location + scale * u[0];
    }
    PersistenceSplit split = split_persistence(u[offset + 1], u[offset + 2]);
    theta[offset] = scale * scale * std::exp(u[offset]);
    theta[offset + 1] = split.persistence * split.share;
    theta[offset + 2] = split.persistence * (1.0 - split.share);
}

void sgarch_to_unconstrained(std::span<const double> theta, std::span<double> u, bool include_mean,
                             const Standardization& standardization)
{
    std::size_t offset = mean_param_count(include_mean);
    assert(theta.size() == offset + kVarianceParamCount);
    assert(u.size() == theta.size());
    double omega = theta[offset];
    double alpha = theta[offset + 1];
    double beta = theta[offset + 2];
    double persistence = alpha + beta;
    assert(omega > 0.0 && alpha >= 0.0 && beta >= 0.0 && persistence < kPersistenceCeiling);
    assert(standardization.scale > 0.0);
    double scale = standardization.scale;
    if (include_mean)
    {
        u[0] = (theta[0] - standardization.location) / scale;
    }
    u[offset] = std::log(omega / (scale * scale));
    u[offset + 1] = log_ratio(persistence, kPersistenceCeiling - persistence);
    u[offset + 2] = log_ratio(alpha, beta);
}

void sgarch_chain_gradient(std::span<const double> u, std::span<const double> grad_theta,
                           std::span<double> grad_u, bool include_mean, const Standardization& standardization)
{
    std::size_t offset = mean_param_count(include_mean);
    assert(u.size() == offset + kVarianceParamCount);
    assert(grad_theta.size() == u.size() && grad_u.size() == u.size());
    double scale = standardization.scale;
    if (include_mean)
    {
        grad_u[0] = scale * grad_theta[0];
    }
    PersistenceSplit split = split_persistence(u[offset + 1], u[offset + 2]);
    double grad_alpha = grad_theta[offset + 1];
    double grad_beta = grad_theta[offset + 2];
    double persistence_difference = split.share * grad_alpha + (1.0 - split.share) * grad_beta;
    double share_difference = grad_alpha - grad_beta;
    grad_u[offset] = scale * scale * std::exp(u[offset]) * grad_theta[offset];
    grad_u[offset + 1] = split.persistence_slope * persistence_difference;
    grad_u[offset + 2] = split.persistence * split.share_slope * share_difference;
}

}
