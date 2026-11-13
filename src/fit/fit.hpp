#pragma once

#include "core/var_init.hpp"
#include "fit/fit_result.hpp"
#include "model/data.hpp"
#include "model/layout.hpp"

#include <cstddef>
#include <optional>
#include <span>
#include <vector>

namespace ve
{

// data.mean and data.variance have as many columns as the spec has mxreg and vxreg.
FitResult fit_one(const model::Data& data, const Spec& spec, const FitConfig& config);

// A series without regressors.
FitResult fit_one(std::span<const double> r, const Spec& spec, const FitConfig& config);

}

namespace ve::fit
{

struct LikelihoodProblem
{
    model::Data data;
    model::Layout layout;
    VarInit var_init;
};

// The univariate problem of a fit without regressors, rebuilt from its result: r = fitted + residuals, written to
// storage, which the problem views. nullopt with mxreg or vxreg, which the result does not keep.
std::optional<LikelihoodProblem> problem_of(const FitResult& fit, std::vector<double>& storage);

// The Hessian of the log-likelihood over `indices` of the natural-space theta, row-major, by the differences of the
// fit's standard errors; nullopt when some gradient on the way is not finite.
std::optional<std::vector<double>> loglik_hessian(const LikelihoodProblem& problem, std::span<const double> theta,
                                                  std::span<const std::size_t> indices);

// Writes the natural-space gradient of the log-likelihood at theta to grad (k entries). Returns false when
// theta is not valid or the log-likelihood or its gradient is not finite.
bool loglik_gradient(const LikelihoodProblem& problem, std::span<const double> theta, std::span<double> grad);

// residuals and sigma2 have the size of data.r; scores is the T x k row-major matrix of per-observation gradients;
// deviations and permanent as in model::LoglikOutputs, empty when not asked for.
struct LoglikSeries
{
    std::span<double> residuals;
    std::span<double> sigma2;
    std::span<double> scores;
    std::span<double> deviations{};
    std::span<double> permanent{};
};

// The log-likelihood at the natural-space theta, filling every series of out in the same pass. The series are
// unspecified when the result is not finite.
double loglik_with_scores(const LikelihoodProblem& problem, std::span<const double> theta,
                          const LoglikSeries& out);
}
