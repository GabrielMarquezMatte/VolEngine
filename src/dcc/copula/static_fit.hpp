#pragma once

#include "core/spec.hpp"
#include "dcc/panel.hpp"
#include "dcc/parameter_map.hpp"
#include "dcc/spec.hpp"
#include "fit/fit_result.hpp"

#include <cstddef>
#include <optional>
#include <span>
#include <vector>

namespace ve::dcc::copula
{

// sin(pi tau_b / 2) of each pair of columns of u (T x m), m x m with a unit diagonal; nullopt unless positive
// definite. ponytail: O(T^2) pairs per entry; Knight's O(T log T) algorithm if large panels need it.
std::optional<std::vector<double>> kendall_correlation(std::span<const double> u, std::size_t t, std::size_t m);

// R = cov2cor(A A'), A unit lower-triangular with x in column-major lower-triangle order (A21, A31, .., Am1, A32,
// ..): rmgarch's .Pconstruct. Row i of A over its norm is row i of the Cholesky factor of R.
std::vector<double> correlation_from_free(std::span<const double> x, std::size_t m);

// dl/dx from dl/dR (m x m, symmetric, the derivative as if every entry of R were free), by R = L L'.
void free_gradient(std::span<const double> x, std::span<const double> dl_dr, std::span<double> dl_dx, std::size_t m);

// sum_t log f(epsilon_t; R[, nu]) of the scores (mvnorm, or the standardized mvt with shape) and, unless gradient is
// empty, dl/dR = -(T/2) R^-1 + 1/2 sum_t w_t y_t y_t' with y_t = R^-1 epsilon_t and w_t = 1 (mvnorm) or
// (nu + m) / (nu - 2 + q_t) (mvt). NaN unless R is positive definite.
double static_terms(const StandardizedPanel& scores, std::span<const double> r, DccDist dist, double shape,
                    std::span<double> gradient);

// log f(epsilon_t; R[, nu]) of each observation (static_terms' summands); NaN unless R is positive definite.
std::vector<double> static_observations(const StandardizedPanel& scores, std::span<const double> r, DccDist dist,
                                        double shape);

// The ML R of the scores: L-BFGS over the free x of the layout's C block (fixed ones kept), from the user's start or
// 0, minimising -static_terms / T. x holds every C entry.
struct StaticCorrelation
{
    FitStatus status = FitStatus::Failed;
    std::vector<double> r;
    std::vector<double> x;
};

StaticCorrelation static_ml(const StandardizedPanel& scores, const DccLayout& layout, double shape,
                            const FitConfig& config);

}
