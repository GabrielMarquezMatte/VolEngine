#pragma once

#include "core/spec.hpp"
#include "dist/densities.hpp"
#include "sim/philox.hpp"

namespace ve::sim
{

// Standard normal by Box-Muller, the cosine branch only: two uniforms per draw.
// ponytail: a ziggurat if sampling ever dominates a simulation (plan 5).
double normal(Stream& stream);

// Gamma(shape, 1) by Marsaglia and Tsang (2000), "A simple method for generating gamma variables"; below shape 1,
// G(shape + 1) u^(1 / shape).
double gamma(Stream& stream, double shape);

// The generalized inverse Gaussian with density proportional to w^(lambda - 1) exp(-(chi / w + psi w) / 2), chi and
// psi positive, by Hormann and Leydold (2014), "Generating generalized inverse Gaussian random variates".
double gig(Stream& stream, double lambda, double chi, double psi);

// z of the standardized density of a fit, in the parametrization of make_density.
class Sampler
{
public:
    Sampler(Dist dist, const dist::ShapeParameters<double>& parameters);
    double draw(Stream& stream) const;

private:
    dist::Density<double> density_;
};

}
