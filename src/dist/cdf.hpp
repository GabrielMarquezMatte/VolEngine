#pragma once

#include "dist/densities.hpp"

#include <cstddef>
#include <span>
#include <vector>

namespace ve::dist
{

double cdf(Dist dist, const ShapeParameters<double>& parameters, double z);
std::vector<double> sorted_cdf(Dist dist, const ShapeParameters<double>& parameters,
                               std::span<const double> sorted);

// The z with cdf(dist, parameters, z) == u, u in (0, 1), of the standardized density of make_density: -inf at 0,
// +inf at 1, NaN outside [0, 1] or outside the domain. Internal (plan 4c): the copula simulation's margins; no C or
// Python surface.
double quantile(Dist dist, const ShapeParameters<double>& parameters, double u);

namespace detail
{

// What the quadrature cdf of a nig or ghyp computes anew at every call and that does not depend on z: the mass below
// 0, and the scales that join the tail quadratures to the central one at -1 and 1.
struct CdfConstants
{
    double below_zero = 0.0;
    double lower_scale = 0.0;
    double upper_scale = 0.0;
};

}

// The quantile of one distribution for many probabilities (plan 5c-2). A nig or ghyp quantile is otherwise a root
// solve over the quadrature cdf (~20 cdfs); this builds once a table of that cdf on a grid in z, takes from it a
// first guess by Hermite interpolation and finishes with Newton steps on the same cdf (its z-independent parts kept),
// so it answers as quantile does (to ~1e-13) in about one cdf. A probability beyond the table, outside [0, 1] or NaN, and any other distribution, go to
// quantile. Immutable once built, so one table serves every thread.
class QuantileTable
{
public:
    QuantileTable(Dist dist, const ShapeParameters<double>& parameters);

    double operator()(double u) const;

private:
    double hermite_guess(std::size_t segment, double u) const;

    Dist dist_;
    ShapeParameters<double> parameters_;
    Density<double> density_;
    detail::CdfConstants constants_;
    // The grid: z increasing, with the cdf and the density there; empty when quantile does the work.
    std::vector<double> z_;
    std::vector<double> cdf_;
    std::vector<double> pdf_;
};

}
