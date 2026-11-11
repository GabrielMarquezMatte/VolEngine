#include "garch/sgarch.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <limits>
#include <numbers>

namespace ve::garch
{

namespace
{

constexpr std::size_t kFullParamCount = 4;
constexpr std::size_t kMu = 0;
constexpr std::size_t kOmega = 1;
constexpr std::size_t kAlpha = 2;
constexpr std::size_t kBeta = 3;

// arch backcast: exponentially weighted mean of the first min(75, T) squared residuals, decay 0.94.
constexpr std::size_t kBackcastWindow = 75;
constexpr double kBackcastDecay = 0.94;

using FullGradient = std::array<double, kFullParamCount>;

struct Parameters
{
    double mu;
    double omega;
    double alpha;
    double beta;
};

// dsigma2 is always over (mu, omega, alpha, beta); without a mean the mu entry is unused.
struct VarianceState
{
    double sigma2;
    FullGradient dsigma2;
};

struct MeanSquare
{
    double value;
    double dmu;
};

Parameters unpack(std::span<const double> theta, const SGarchConfig& config)
{
    if (!config.include_mean)
    {
        return {0.0, theta[0], theta[1], theta[2]};
    }
    return {theta[0], theta[1], theta[2], theta[3]};
}

MeanSquare decayed_mean_square(std::span<const double> r, double mu, double decay)
{
    double weight = 1.0;
    double weight_sum = 0.0;
    double weighted_square_sum = 0.0;
    double weighted_residual_sum = 0.0;
    for (double value : r)
    {
        double residual = value - mu;
        weight_sum += weight;
        weighted_square_sum += weight * residual * residual;
        weighted_residual_sum += weight * residual;
        weight *= decay;
    }
    return {weighted_square_sum / weight_sum, -2.0 * weighted_residual_sum / weight_sum};
}

// rugarch initializes the variance recursion with the sample mean of the squared residuals.
VarianceState sample_initial_state(std::span<const double> r, const Parameters& p)
{
    MeanSquare mean_square = decayed_mean_square(r, p.mu, 1.0);
    VarianceState state{mean_square.value, {}};
    state.dsigma2[kMu] = mean_square.dmu;
    return state;
}

double sample_mean(std::span<const double> r)
{
    double sum = 0.0;
    for (double value : r)
    {
        sum += value;
    }
    return sum / static_cast<double>(r.size());
}

// arch computes the backcast once, from the residuals at its starting mean (the sample mean, or zero without
// a mean), and keeps it fixed during the optimization: it does not depend on mu.
VarianceState backcast_initial_state(std::span<const double> r, const Parameters& p, bool include_mean)
{
    double center = 0.0;
    if (include_mean)
    {
        center = sample_mean(r);
    }
    std::size_t window = std::min(kBackcastWindow, r.size());
    double b = decayed_mean_square(r.first(window), center, kBackcastDecay).value;
    VarianceState state{p.omega + p.alpha * b + p.beta * b, {}};
    state.dsigma2[kOmega] = 1.0;
    state.dsigma2[kAlpha] = b;
    state.dsigma2[kBeta] = b;
    return state;
}

VarianceState initial_state(std::span<const double> r, const Parameters& p, const SGarchConfig& config)
{
    if (config.var_init == VarInit::Backcast)
    {
        return backcast_initial_state(r, p, config.include_mean);
    }
    return sample_initial_state(r, p);
}

VarianceState next_state(const VarianceState& previous, double previous_residual, const Parameters& p)
{
    double previous_square = previous_residual * previous_residual;
    double arch_term = p.alpha * previous_square;
    double garch_term = p.beta * previous.sigma2;
    VarianceState next{p.omega + arch_term + garch_term, {}};
    next.dsigma2[kMu] = -2.0 * p.alpha * previous_residual;
    next.dsigma2[kOmega] = 1.0;
    next.dsigma2[kAlpha] = previous_square;
    next.dsigma2[kBeta] = previous.sigma2;
    for (std::size_t i = 0; i < kFullParamCount; ++i)
    {
        next.dsigma2[i] += p.beta * previous.dsigma2[i];
    }
    return next;
}

bool is_valid_variance(double sigma2)
{
    return std::isfinite(sigma2) && sigma2 > 0.0;
}

double normal_log_density(double residual, double sigma2)
{
    double log_two_pi = std::log(2.0 * std::numbers::pi);
    double log_variance = std::log(sigma2);
    double quadratic_term = residual * residual / sigma2;
    return -0.5 * (log_two_pi + log_variance + quadratic_term);
}

FullGradient normal_score(double residual, const VarianceState& state)
{
    double quadratic_term = residual * residual / state.sigma2;
    double variance_sensitivity = 0.5 * (quadratic_term - 1.0) / state.sigma2;
    FullGradient score{};
    for (std::size_t i = 0; i < kFullParamCount; ++i)
    {
        score[i] = variance_sensitivity * state.dsigma2[i];
    }
    score[kMu] += residual / state.sigma2;
    return score;
}

void store_score(const FullGradient& score, std::size_t t, std::size_t k, const SGarchOutputs& out)
{
    std::size_t offset = kFullParamCount - k;
    for (std::size_t i = 0; i < k; ++i)
    {
        double value = score[offset + i];
        if (!out.grad.empty())
        {
            out.grad[i] += value;
        }
        if (!out.scores.empty())
        {
            out.scores[t * k + i] = value;
        }
    }
}

}

std::size_t sgarch_param_count(const SGarchConfig& config)
{
    if (!config.include_mean)
    {
        return kFullParamCount - 1;
    }
    return kFullParamCount;
}

double sgarch_norm_loglik(std::span<const double> r, std::span<const double> theta, const SGarchConfig& config,
                          const SGarchOutputs& out)
{
    std::size_t k = sgarch_param_count(config);
    assert(theta.size() == k);
    assert(out.sigma2.empty() || out.sigma2.size() == r.size());
    assert(out.grad.empty() || out.grad.size() == k);
    assert(out.scores.empty() || out.scores.size() == r.size() * k);
    bool wants_score = !out.grad.empty() || !out.scores.empty();
    Parameters p = unpack(theta, config);
    std::ranges::fill(out.grad, 0.0);
    VarianceState state = initial_state(r, p, config);
    double loglik = 0.0;
    double previous_residual = 0.0;
    for (std::size_t t = 0; t < r.size(); ++t)
    {
        if (t > 0)
        {
            state = next_state(state, previous_residual, p);
        }
        if (!is_valid_variance(state.sigma2))
        {
            return -std::numeric_limits<double>::infinity();
        }
        double residual = r[t] - p.mu;
        loglik += normal_log_density(residual, state.sigma2);
        if (!out.sigma2.empty())
        {
            out.sigma2[t] = state.sigma2;
        }
        if (wants_score)
        {
            store_score(normal_score(residual, state), t, k, out);
        }
        previous_residual = residual;
    }
    return loglik;
}

}
