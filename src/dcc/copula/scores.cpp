#include "dcc/copula/scores.hpp"

#include <boost/math/distributions/normal.hpp>
#include <boost/math/policies/policy.hpp>
#include <boost/math/distributions/students_t.hpp>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <memory>
#include <numbers>
#include <span>
#include <utility>
#include <vector>

namespace ve::dcc::copula
{

namespace
{

// Computing in double, not promoted to long double, keeps the quantiles to the bit whatever the thread's x87
// precision (the Python main thread and the pool threads differ on MinGW).
using DoublePolicy = boost::math::policies::policy<boost::math::policies::promote_double<false>>;

// Each log f1(epsilon_k) goes to marginal[k].
void normal_scores(std::span<const double> u, std::span<double> epsilon, std::span<double> marginal)
{
    const boost::math::normal_distribution<double, DoublePolicy> normal;
    const double half_log_two_pi = 0.5 * std::log(2.0 * std::numbers::pi);
    for (std::size_t k = 0; k < u.size(); ++k)
    {
        epsilon[k] = boost::math::quantile(normal, u[k]);
        marginal[k] = -(half_log_two_pi + 0.5 * epsilon[k] * epsilon[k]);
    }
}

// log f1(e) = log t_nu(e / s) - log s, s = sqrt((nu - 2) / nu).
void student_scores(std::span<const double> u, double nu, std::span<double> epsilon, std::span<double> marginal)
{
    const boost::math::students_t_distribution<double, DoublePolicy> student(nu);
    const double scale = std::sqrt((nu - 2.0) / nu);
    const double constant = std::lgamma((nu + 1.0) / 2.0) - std::lgamma(nu / 2.0) -
                            0.5 * std::log(std::numbers::pi * (nu - 2.0));
    for (std::size_t k = 0; k < u.size(); ++k)
    {
        epsilon[k] = boost::math::quantile(student, u[k]) * scale;
        marginal[k] = constant - 0.5 * (nu + 1.0) * std::log1p(epsilon[k] * epsilon[k] / (nu - 2.0));
    }
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
    std::vector<double> each(u.size());
    if (dist == DccDist::MvT)
    {
        student_scores(u, shape, panel->z, each);
    }
    if (dist != DccDist::MvT)
    {
        normal_scores(u, panel->z, each);
    }
    refresh_moments(*panel);
    CopulaPanel out{std::move(panel), 0.0, std::vector<double>(univariate.t, 0.0)};
    for (std::size_t k = 0; k < each.size(); ++k)
    {
        out.marginal_terms[k / univariate.m] += each[k];
    }
    for (const double term : out.marginal_terms)
    {
        out.marginal_log_density += term;
    }
    return out;
}

}
