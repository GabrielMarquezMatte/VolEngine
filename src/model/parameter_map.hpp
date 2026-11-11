#pragma once

#include "ad/dual.hpp"
#include "model/layout.hpp"

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

// Both spaces follow the Layout order. Unconstrained: m (mean), phi and theta unchanged, u (omega), then v
// followed by p + q - 1 softmax logits w (the weights of alpha and beta), then x (shape). mu = location +
// scale m; omega = scale^2 exp(u); persistence P = logistic(v) (1 - 1e-8); (alpha, beta) = P softmax(w, 0);
// std: nu = 2 + exp(x); ged: nu = exp(x). Instantiated only for T = double, ad::Dual<8>, ad::Dual<16> and
// ad::Dual<32>.
template <class T>
void to_natural(std::span<const T> u, std::span<T> theta, const Layout& layout,
                const Standardization& standardization);

// theta must satisfy is_valid. Weights enter log(weight / reference weight) floored at DBL_MIN, so zero
// weights give finite values.
void to_unconstrained(std::span<const double> theta, std::span<double> u, const Layout& layout,
                      const Standardization& standardization);

// All entries finite, omega > 0, alpha, beta >= 0, sum(alpha) + sum(beta) < 1 - 1e-8, shape in its domain.
bool is_valid(std::span<const double> theta, const Layout& layout);

}
