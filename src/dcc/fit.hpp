#pragma once

#include "core/context.hpp"
#include "core/spec.hpp"
#include "dcc/future/state.hpp"
#include "dcc/panel.hpp"
#include "dcc/spec.hpp"
#include "fit/fit_result.hpp"

#include <cstddef>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace ve::dcc
{

// names, estimate, se_hessian and se_robust cover the estimated DCC parameters only: a fixed one leaves the table.
// A fit that fails before or at the estimate (see fit_dcc) has a NaN estimate and loglik and a null r; one whose
// line search fails is Failed too, but keeps the finite estimate where it stopped, its loglik and r.
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
    // Filled by fit_dcc_batch (null and empty from fit_dcc alone, and when the estimate is NaN): the final state of
    // each series, one for the whole batch, and the final state of the correlation recursion (plan 4c).
    std::shared_ptr<const std::vector<future::SeriesEnd>> series_end;
    future::DccEnd end;
};

// Maximises the DCC likelihood of spec on the panel: L-BFGS on the unconstrained parameters of parameter_map.hpp,
// from start_values, minimising objective / T. As in the univariate fit, the status is the optimiser's: Converged,
// MaxIter, or Failed when the line search fails (with a finite estimate). It is Failed with a NaN estimate when the
// start is not valid, the objective is not finite there, or the reported loglik at the estimate is not finite.
// With nothing to estimate the spec is evaluated at its fixed values. The caller checked
// spec_problem(spec, panel->m).
DccFitResult fit_dcc(std::shared_ptr<const StandardizedPanel> panel, const DccSpec& spec, const FitConfig& config);

// standard_errors asks the batch for the two-stage tables (se_hessian, se_robust) of each converged spec, those of
// dcc_standard_errors to the bit; the other specs keep NaN. With it, fits with regressors fail check_dcc_batch.
struct DccOptions
{
    FitConfig config{};
    bool standard_errors = true;
};

// The message of the first problem that makes the batch unrunnable, nothing when it can run: the fits do not form
// a panel (panel_problem), there is no spec, a spec cannot be fitted to the panel (spec_problem, named "spec i"
// from 1), or options.standard_errors is on and the fits cannot give them (standard_errors_problem). As the
// univariate check_batch, it is asked before the batch, which assumes it passes.
std::optional<std::string> check_dcc_batch(std::span<const FitResult* const> fits, std::span<const DccSpec> specs,
                                           const DccOptions& options);

// One panel, many specs: the panel is built once and shared by the results, and the specs are split among the
// threads of the context (inside context.with_pool, so a concurrent set_threads waits). Results are indexed like
// specs and do not depend on the thread count. A spec that does not converge is its result's status, not an
// error; an exception raised by a fit is rethrown after every started fit has finished. Unlike fit_dcc, the
// results carry univariate_parameters and univariate_estimates. The batch must pass check_dcc_batch.
std::vector<DccFitResult> fit_dcc_batch(Context& context, std::span<const FitResult* const> fits,
                                        std::span<const DccSpec> specs, const DccOptions& options);

enum class Criterion
{
    Aic,
    Bic,
    Hqic,
    Sic
};

// The information criteria of the univariate ones with k = the estimated univariate parameters of all the series +
// the estimated DCC parameters + m (m - 1) / 2 (Qbar counts, Nbar does not), per observation or, with per_obs
// false, total. NaN without a finite loglik.
double dcc_criterion(const DccFitResult& fit, Criterion which, bool per_obs);

// H_t = D_t R_t D_t with D_t = diag(sigma_t) of the panel, T x m x m like r; empty without r. A pure function: the
// caller that wants it kept computes it once.
std::vector<double> covariance_path(const DccFitResult& fit);

}
