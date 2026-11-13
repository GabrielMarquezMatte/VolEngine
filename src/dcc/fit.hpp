#pragma once

#include "core/spec.hpp"
#include "dcc/panel.hpp"
#include "dcc/spec.hpp"
#include "fit/fit_result.hpp"

#include <cstddef>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace ve::dcc
{

// names, estimate, se_hessian and se_robust cover the estimated DCC parameters only: a fixed one leaves the table.
// Without a finite estimate (status Failed) the estimate and loglik are NaN and r is null.
struct DccFitResult
{
    FitStatus status = FitStatus::Failed;
    DccSpec spec{};
    std::vector<std::string> names;
    std::vector<double> estimate;
    // NaN until standard errors are computed.
    std::vector<double> se_hessian;
    std::vector<double> se_robust;
    // The reported log-likelihood at the estimate (Stage::Reported of the spec's convention).
    double loglik = std::numeric_limits<double>::quiet_NaN();
    std::size_t nobs = 0;
    std::size_t series = 0;
    // Filled from the univariate fits by the batch; fit_dcc, which sees only the panel, leaves 0 and empty.
    std::size_t univariate_parameters = 0;
    // R_t, T x m x m, shared with copies.
    std::shared_ptr<const std::vector<double>> r;
    // sigma for H_t; shared by the specs of a batch.
    std::shared_ptr<const StandardizedPanel> panel;
    // The univariate estimates, concatenated in series order.
    std::vector<double> univariate_estimates;
};

// Maximises the DCC likelihood of spec on the panel: L-BFGS on the unconstrained parameters of parameter_map.hpp,
// from start_values, minimising objective / T. The status follows the univariate fit: Failed when the start is
// not valid or the objective is not finite there, or when the reported loglik at the estimate is not finite;
// otherwise Converged or MaxIter. With nothing to estimate the spec is evaluated at its fixed values. The caller
// checked spec_problem(spec, panel->m).
DccFitResult fit_dcc(std::shared_ptr<const StandardizedPanel> panel, const DccSpec& spec, const FitConfig& config);

}
