#pragma once

#include "ad/dual.hpp"
#include "model/layout.hpp"

#include <cstddef>
#include <span>

namespace ve::model
{

// Location and scale of the series, so that the unconstrained problem does not depend on its units.
// scale must be > 0.
struct Standardization
{
    double location = 0.0;
    double scale = 1.0;
};

// Both spaces follow the Layout order, and each unconstrained entry sits at the position of the natural one it
// maps to; s = scale. mu = location + s m; ARMA unchanged; skew: xi = exp(x) (sstd, sged),
// rho = 2 logistic(x) - 1 (nig, ghyp), nu = x (jsu); shape: 2 + exp(x) (std, sstd), exp(x) (the others);
// ghlambda = x.
// The weights are entries >= 0 summing to the persistence P: the first weight position holds v, with
// P = logistic(v) (1 - 1e-8), and the others the logits of softmax(w, 0):
//   sGARCH, csGARCH: (alpha, beta); eGARCH: beta (alpha and gamma unchanged);
//   gjrGARCH: ((1 - pi) alpha, beta, pi (alpha + gamma)), in the alpha, beta, gamma positions, with
//   pi = P(z < 0) (0.5 for norm, std and ged), so that P = sum alpha + sum beta + pi sum gamma;
//   apARCH, fGARCH: (alpha_i kappa_i, beta), with kappa_i the shock_moment;
//   iGARCH: (alpha, beta) = softmax(w, 0) without v, its p + q - 1 logits first; the derived beta_q position of u
//   is unused, and theta[derived] = 1 - the other weights.
// eta1 and the apARCH gamma: 2 logistic(x) - 1; eta2 unchanged; free delta or lambda: exp(x).
// omega = s^2 exp(u), or s^power exp(u) in apARCH and fGARCH, or u + (1 - sum beta) log(s^2) in eGARCH.
// csGARCH: rho = P + (1 - 1e-8 - P) logistic(x_rho), phi = sum(beta) logistic(x_phi).
// Fixed entries hold layout.given, and every other map writes only estimated entries (theta[omega] is not
// written under variance targeting). With a fixed weight, or a fixed csGARCH rho, the weights are the terms of
// weight_terms.hpp: the fixed terms sum to F, B = c - F, and the free terms are logistic(v) B softmax(w, 0) over
// the free positions in order, v at the first, the last the reference (iGARCH: B softmax(w, 0) without v, the
// derived beta_q the reference). B <= 0 gives NaN free weights.
// Instantiated only for T = double, ad::Dual<8>, ad::Dual<16>, ad::Dual<32> and ad::Dual<kMaxParameterCount>.
template <class T>
void to_natural(std::span<const T> u, std::span<T> theta, const Layout& layout,
                const Standardization& standardization);

// theta must satisfy is_valid. Weights enter log(weight / reference weight) floored at DBL_MIN, so zero
// weights give finite values. u is 0 at the fixed and derived entries.
void to_unconstrained(std::span<const double> theta, std::span<double> u, const Layout& layout,
                      const Standardization& standardization);

// All entries finite; skew, shape and ghlambda in their domain; |eta1|, |apARCH gamma| < 1; delta,
// lambda > 0; omega > 0 (but in eGARCH); weights >= 0 with P < 1 - 1e-8 (iGARCH: alpha, beta_{<q} >= 0
// summing to at most 1), so kappa is finite; gjrGARCH: 0 < pi < 1; csGARCH: P < rho < 1 - 1e-8 and
// 0 <= phi <= sum(beta). The derived entries (the iGARCH beta_q, omega under variance targeting) are not read.
bool is_valid(std::span<const double> theta, const Layout& layout);

// Whether theta[index] sits on the boundary of its domain, where it leaves the Hessian: a weight below 1e-8
// (gjrGARCH: alpha_i below 1e-8 marks alpha_i, alpha_i + gamma_i below 1e-8 marks gamma_i, and with gamma_i
// fixed and alpha_i free, alpha_i - max(0, -gamma_i) below 1e-8 marks alpha_i); |eta1_i| or
// |apARCH gamma_i| above 1 - 1e-8; csGARCH phi below 1e-8, or rho - P below 1e-8; the iGARCH derived beta_q;
// a std, sstd, nig, ghyp or jsu shape whose 1/shape is below 1e-8, which marks the ghyp ghlambda as well;
// a nig or ghyp |rho| above 1 - 1e-8.
bool on_boundary(const Layout& layout, std::span<const double> theta, std::size_t index);

// kappa_i = E(|z - eta2_i| - eta1_i (z - eta2_i))^power for apARCH (eta1 = gamma) and fGARCH; may be NaN as well
// as +inf when not finite. Instantiated as to_natural.
template <class T>
T shock_moment(std::span<const T> theta, const Layout& layout, std::size_t i);

// P = the sum of the weight terms (weight_terms.hpp); NaN where pi or kappa is not computable. Instantiated as
// to_natural.
template <class T>
T persistence(std::span<const T> theta, const Layout& layout);

}
