#include "dcc/copula/scores.hpp"

#include <boost/math/distributions/normal.hpp>
#include <boost/math/distributions/students_t.hpp>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <memory>
#include <numbers>
#include <span>
#include <utility>

namespace ve::dcc::copula
{

namespace
{

double normal_scores(std::span<const double> u, std::span<double> epsilon)
{
    const boost::math::normal normal;
    const double half_log_two_pi = 0.5 * std::log(2.0 * std::numbers::pi);
    double marginal = 0.0;
    for (std::size_t k = 0; k < u.size(); ++k)
    {
        epsilon[k] = boost::math::quantile(normal, u[k]);
        marginal -= half_log_two_pi + 0.5 * epsilon[k] * epsilon[k];
    }
    return marginal;
}

// log f1(e) = log t_nu(e / s) - log s, s = sqrt((nu - 2) / nu).
double student_scores(std::span<const double> u, double nu, std::span<double> epsilon)
{
    const boost::math::students_t student(nu);
    const double scale = std::sqrt((nu - 2.0) / nu);
    const double constant = std::lgamma((nu + 1.0) / 2.0) - std::lgamma(nu / 2.0) -
                            0.5 * std::log(std::numbers::pi * (nu - 2.0));
    double marginal = 0.0;
    for (std::size_t k = 0; k < u.size(); ++k)
    {
        epsilon[k] = boost::math::quantile(student, u[k]) * scale;
        marginal += constant - 0.5 * (nu + 1.0) * std::log1p(epsilon[k] * epsilon[k] / (nu - 2.0));
    }
    return marginal;
}

}

CopulaPanel copula_panel(std::span<const double> u, const StandardizedPanel& univariate, DccDist dist, double shape)
{
    assert(u.size() == univariate.t * univariate.m);
    auto panel = std::make_shared<StandardizedPanel>();
    panel->t = univariate.t;
    panel->m = univariate.m;
    panel->sigma = univariate.sigma;
    panel->z.resize(u.size());
    double marginal = 0.0;
    if (dist == DccDist::MvT)
    {
        marginal = student_scores(u, shape, panel->z);
    }
    if (dist != DccDist::MvT)
    {
        marginal = normal_scores(u, panel->z);
    }
    refresh_moments(*panel);
    return {std::move(panel), marginal};
}

}
