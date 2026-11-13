#pragma once

#include "dcc/fit.hpp"
#include "dcc/spec.hpp"
#include "fit/fit_result.hpp"

#include <cstddef>
#include <expected>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace ve::dcc
{

// The two-stage problem (Engle-Sheppard) with theta = (theta_1, ..., theta_m, theta_D): the parameters of each
// univariate fit that its own standard errors cover (estimated, off the boundary), in series order, then the
// estimated DCC parameters. parameters = n_p. Signs as rmgarch builds them:
//   a, n_p x n_p row-major: the univariate diagonal blocks are the Hessians of -l_i, the other univariate blocks
//   and the univariate rows of the DCC columns 0; the DCC rows the forward-difference Hessian of the reported
//   loglik (+l) over all of theta, each univariate perturbation refiltering its series and recomputing z, Qbar and
//   Nbar, with steps eps^(1/3) max(|theta|, 1);
//   scores, T x n_p row-major: the univariate columns the scores of -l_it, the DCC columns central differences
//   of +l_t with the univariate parameters at their estimates and steps eps^(1/3) max(|theta|/2, 0.01).
struct TwoStagePieces
{
    std::size_t parameters = 0;
    std::vector<double> a;
    std::vector<double> scores;
};

// The pieces of fit at its estimate, or why there are none: fits_problem, a fit without a finite estimate, or
// standard_errors_problem. A numerical failure leaves NaN entries.
std::expected<TwoStagePieces, std::string> two_stage_pieces(const DccFitResult& fit,
                                                            std::span<const FitResult* const> fits);

// V of the sandwich, n_p x n_p, with B = cov(scores) (centred, divisor T - 1) and A/T in place of A:
//   Standard: A^-1 B A^-T / T, symmetrized, the Newey-McFadden form; it is the same whether or not the univariate
//   rows of A and columns of the scores change sign, so the pieces need no flip;
//   Rmgarch: A^-1 B A^-1 / T, rmgarch's literal form with its mixed signs, not symmetric.
// NaN when A is singular.
std::vector<double> sandwich(const TwoStagePieces& pieces, Convention convention);

// The DCC rows of the fit's table: hessian = sqrt(diag((-H_DD)^-1)), H_DD the DCC block of A (NaN unless -H_DD is
// positive definite); robust = sqrt(|diag V|) of the sandwich of the fit's convention.
struct DccStandardErrors
{
    std::vector<double> hessian;
    std::vector<double> robust;
};

std::expected<DccStandardErrors, std::string> dcc_standard_errors(const DccFitResult& fit,
                                                                  std::span<const FitResult* const> fits);

// Why the two-stage standard errors cannot be computed from fits, or nullopt: a series with mxreg or vxreg, whose
// regressors its fit does not keep for the refiltering (the message names "regressors").
std::optional<std::string> standard_errors_problem(std::span<const FitResult* const> fits);

// Why fits are not those fit was estimated on, or nullopt: another number of series or of observations, or
// univariate estimates that differ, bit for bit, from fit.univariate_estimates (as filled by fit_dcc_batch).
std::optional<std::string> fits_problem(const DccFitResult& fit, std::span<const FitResult* const> fits);

}
