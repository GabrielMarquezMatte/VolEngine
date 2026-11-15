#pragma once

#include "dist/symmetric.hpp"

#include <span>

namespace ve::simd
{

// The residuals e_t and variances sigma2_t of a series, and where the terms go (empty: not stored).
struct SymmetricSeries
{
    std::span<const double> e;
    std::span<const double> sigma2;
    std::span<double> terms = {};
};

// The derivatives of the terms log f(e_t / sigma_t) - log(sigma2_t) / 2 in e_t and sigma2_t, written to d_e and d_sigma2
// (the size of the series), and the sums over t of the derivatives in the density's scale and power, the semantics of
// dist::TermAdjoint (the derivative in the log constant is 1). Where the term has a kink they follow ad::Dual: |e|
// has slope 0 at e = 0, and so does the ged power there.
struct SymmetricPartials
{
    std::span<double> d_e;
    std::span<double> d_sigma2;
    double d_scale = 0.0;
    double d_power = 0.0;
};

namespace detail
{

struct DensityArgs
{
    const double* e;
    const double* sigma2;
    double* terms;
    double* d_e;
    double* d_sigma2;
    double* d_scale;
    double* d_power;
    std::size_t count;
    double log_constant;
    double scale;
    double power;
};

}

// sum_t log f(e_t / sigma_t) - log(sigma2_t) / 2 under the norm, std or ged density (the others are not covered),
// writing the terms when series.terms is not empty. The residuals and the variances are read a vector of lanes at
// a time: the exponent of the ged is exp(power log y), the std reads log1p. -infinity (model::kInvalidLoglik) when
// some sigma2_t is not positive and finite or the sum is not finite, as model::sum_of_terms. The sum is accumulated
// by lanes, so it depends on the target the machine dispatches to, not on the number of threads.
double symmetric_terms(const dist::LogDensity<double>& density, std::span<const double> e,
                       std::span<const double> sigma2, std::span<double> terms);

// The same terms and sum, bit for bit, plus the partials, from the transcendentals the terms already computed: the
// reverse sweep of the likelihood does not evaluate log1p or log again. The partials are unspecified when the
// result is not finite.
double symmetric_terms_with_partials(const dist::LogDensity<double>& density, const SymmetricSeries& series,
                                     SymmetricPartials& partials);

}
