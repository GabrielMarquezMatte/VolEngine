#pragma once

#include "core/var_init.hpp"
#include "fit/fit_result.hpp"
#include "model/layout.hpp"

#include <span>

namespace ve
{

FitResult fit_one(std::span<const double> r, const Spec& spec, const FitConfig& config);

}

namespace ve::fit
{

struct LikelihoodProblem
{
    std::span<const double> r;
    model::Layout layout;
    VarInit var_init;
};

// Writes the natural-space gradient of the log-likelihood at theta to grad (k entries). Returns false when theta is
// not valid or the log-likelihood or its gradient is not finite.
bool loglik_gradient(const LikelihoodProblem& problem, std::span<const double> theta, std::span<double> grad);

// residuals and sigma2 have the size of r; scores is the T x k row-major matrix of per-observation gradients.
struct LoglikSeries
{
    std::span<double> residuals;
    std::span<double> sigma2;
    std::span<double> scores;
};

// The log-likelihood at the natural-space theta, filling every series of out in the same pass. The series are
// unspecified when the result is not finite.
double loglik_with_scores(const LikelihoodProblem& problem, std::span<const double> theta, const LoglikSeries& out);

}
