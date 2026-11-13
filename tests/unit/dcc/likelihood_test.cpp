#include "dcc/likelihood.hpp"

#include "ad/dual.hpp"
#include "dcc/panel.hpp"
#include "dcc/spec.hpp"
#include "dcc3_panel.hpp"
#include "linalg/cholesky.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <gtest/gtest.h>
#include <limits>
#include <numbers>
#include <span>
#include <string>
#include <vector>

namespace
{

using ve::dcc::Convention;
using ve::dcc::DccDist;
using ve::dcc::DccModel;
using ve::dcc::DccSpec;
using ve::dcc::StandardizedPanel;
using ve::testing::kDcc3Rows;
using ve::testing::kDcc3Series;

constexpr std::size_t kCells = kDcc3Series * kDcc3Series;

DccSpec spec_of(DccModel model, DccDist dist, Convention convention)
{
    DccSpec spec;
    spec.model = model;
    spec.dist = dist;
    spec.convention = convention;
    if (model == DccModel::FDcc)
    {
        spec.groups = {1, 1, 2};
    }
    return spec;
}

double log_sigma_sum(const StandardizedPanel& panel, std::size_t t)
{
    double sum = 0.0;
    for (std::size_t i = 0; i < panel.m; ++i)
    {
        sum += std::log(panel.sigma[t * panel.m + i]);
    }
    return sum;
}

double total_log_sigma(const StandardizedPanel& panel)
{
    double sum = 0.0;
    for (std::size_t t = 0; t < panel.t; ++t)
    {
        sum += log_sigma_sum(panel, t);
    }
    return sum;
}

TEST(DccLikelihood, UnivariateFitsAreRugarchs)
{
    const std::vector<ve::FitResult> fits = ve::testing::dcc3_fits();
    const std::array<double, kDcc3Series> rugarch{-2001.64239768346, -1991.03468862373, -2433.41590886058};
    for (std::size_t i = 0; i < kDcc3Series; ++i)
    {
        EXPECT_NEAR(fits[i].loglik, rugarch[i], 1e-10 * std::fabs(rugarch[i])) << "series " << i + 1;
    }
}

TEST(DccLikelihood, MatchesRmgarchAtItsPoint)
{
    const StandardizedPanel panel = ve::testing::dcc3_panel();
    const DccSpec spec = spec_of(DccModel::Dcc, DccDist::MvNorm, Convention::Rmgarch);
    const std::vector<double> theta{0.0431630001835819, 0.9142701826310855};
    std::vector<double> r(kDcc3Rows * kCells);
    const double loglik = ve::dcc::reported_loglik(panel, spec, theta, r);
    EXPECT_NEAR(loglik, -6082.94087772754, 1e-10 * 6082.94087772754);

    std::vector<double> terms(kDcc3Rows);
    const auto parameters = ve::dcc::split_parameters(spec, std::span<const double>(theta));
    ve::dcc::correlation_terms<double>(panel, spec, parameters, ve::dcc::Stage::Reported, {terms});
    const std::array<double, 3> first{-3.87527571695479, -2.9806184777153, -5.51904233840933};
    for (std::size_t t = 0; t < first.size(); ++t)
    {
        EXPECT_NEAR(terms[t] - log_sigma_sum(panel, t), first[t], 1e-10) << "t = " << t + 1;
    }

    EXPECT_NEAR(r[1], 0.497294779040641, 1e-12);
    EXPECT_NEAR(r[2], 0.323589762766861, 1e-12);
    EXPECT_NEAR(r[5], 0.368658387124292, 1e-12);
    const double objective = ve::dcc::objective<double>(panel, spec, theta);
    EXPECT_NEAR(objective, 1911.09985833825, 1e-10 * 1911.09985833825);
}

struct Variant
{
    std::string name;
    DccSpec spec;
    std::vector<double> theta;
    double loglik;
};

TEST(DccLikelihood, OtherDistributionsAndModelsAtRmgarchPoints)
{
    const StandardizedPanel panel = ve::testing::dcc3_panel();
    DccSpec order21 = spec_of(DccModel::Dcc, DccDist::MvNorm, Convention::Rmgarch);
    order21.p = 2;
    const std::vector<Variant> variants{
        {"DCC/mvt", spec_of(DccModel::Dcc, DccDist::MvT, Convention::Rmgarch),
         {0.04306741677, 0.91353820409, 49.99998817}, -6083.64532606},
        {"aDCC/mvnorm", spec_of(DccModel::ADcc, DccDist::MvNorm, Convention::Rmgarch),
         {0.04243321505, 0.91398307950, 0.00240804736}, -6082.91639426},
        {"FDCC/mvnorm (1,1,2)", spec_of(DccModel::FDcc, DccDist::MvNorm, Convention::Rmgarch),
         {0.19004678480, 0.24108698852, 0.96463050567, 0.94296237137}, -6082.56656677},
        {"DCC/mvlaplace", spec_of(DccModel::Dcc, DccDist::MvLaplace, Convention::Rmgarch),
         {0.03872917063, 0.90870637605}, -6364.40615222},
        {"DCC(2,1)/mvnorm", order21, {0.04060218, 0.00292010, 0.91353221}, -6082.93058943},
    };
    for (const Variant& variant : variants)
    {
        const double loglik = ve::dcc::reported_loglik(panel, variant.spec, variant.theta);
        EXPECT_NEAR(loglik, variant.loglik, 1e-9 * std::fabs(variant.loglik)) << variant.name;
    }
}

// The rule written out for m = 3 and Q = 1, without the library's recursion, Cholesky or Bessel function.
using Matrix3 = std::array<double, kCells>;

struct DirectPoint
{
    std::vector<double> a;
    std::vector<double> g;
    double b;
    double shape;
};

double determinant(const Matrix3& x)
{
    return x[0] * (x[4] * x[8] - x[5] * x[7]) - x[1] * (x[3] * x[8] - x[5] * x[6]) +
           x[2] * (x[3] * x[7] - x[4] * x[6]);
}

// z' X^-1 z by the adjugate of the symmetric X.
double quadratic_form(const Matrix3& x, std::span<const double> z)
{
    const Matrix3 adjugate{x[4] * x[8] - x[5] * x[7], x[2] * x[7] - x[1] * x[8], x[1] * x[5] - x[2] * x[4],
                           x[5] * x[6] - x[3] * x[8], x[0] * x[8] - x[2] * x[6], x[2] * x[3] - x[0] * x[5],
                           x[3] * x[7] - x[4] * x[6], x[1] * x[6] - x[0] * x[7], x[0] * x[4] - x[1] * x[3]};
    double sum = 0.0;
    for (std::size_t i = 0; i < kDcc3Series; ++i)
    {
        for (std::size_t j = 0; j < kDcc3Series; ++j)
        {
            sum += z[i] * adjugate[i * kDcc3Series + j] * z[j];
        }
    }
    return sum / determinant(x);
}

double direct_log_density(DccDist dist, const Matrix3& r, std::span<const double> z, double shape)
{
    const double m = 3.0;
    const double log_det = std::log(determinant(r));
    const double q = quadratic_form(r, z);
    const double log_two_pi = std::log(2.0 * std::numbers::pi);
    if (dist == DccDist::MvT)
    {
        return std::lgamma((shape + m) / 2.0) - std::lgamma(shape / 2.0) -
               m / 2.0 * std::log(std::numbers::pi * (shape - 2.0)) - log_det / 2.0 -
               (shape + m) / 2.0 * std::log(1.0 + q / (shape - 2.0));
    }
    if (dist == DccDist::MvLaplace)
    {
        const double v = 1.0 - m / 2.0;
        const double x = std::sqrt(2.0 * q);
        const double log_k_half = 0.5 * std::log(std::numbers::pi / (2.0 * x)) - x;
        return std::log(2.0) - m / 2.0 * log_two_pi - log_det / 2.0 + v / 2.0 * std::log(q / 2.0) + log_k_half;
    }
    return -0.5 * (m * log_two_pi + log_det + q);
}

Matrix3 outer(std::span<const double> x, bool negative_part)
{
    Matrix3 out{};
    for (std::size_t i = 0; i < kDcc3Series; ++i)
    {
        for (std::size_t j = 0; j < kDcc3Series; ++j)
        {
            const bool kept = !negative_part || (x[i] < 0.0 && x[j] < 0.0);
            if (kept)
            {
                out[i * kDcc3Series + j] = x[i] * x[j];
            }
        }
    }
    return out;
}

Matrix3 to_matrix(const std::vector<double>& x)
{
    Matrix3 out{};
    std::copy(x.begin(), x.end(), out.begin());
    return out;
}

double sum_of(const std::vector<double>& x)
{
    double sum = 0.0;
    for (const double v : x)
    {
        sum += v;
    }
    return sum;
}

// Q_t of the Standard convention: z_s z_s' = Qbar, n_s n_s' = Nbar and Q_s = Qbar for s <= 0.
Matrix3 direct_q(const StandardizedPanel& panel, const DirectPoint& point, const std::vector<Matrix3>& q, std::size_t t)
{
    const Matrix3 qbar = to_matrix(panel.qbar);
    const Matrix3 nbar = to_matrix(panel.nbar);
    const double intercept = 1.0 - sum_of(point.a) - point.b;
    Matrix3 previous = qbar;
    if (t > 0)
    {
        previous = q[t - 1];
    }
    Matrix3 out{};
    for (std::size_t k = 0; k < kCells; ++k)
    {
        out[k] = intercept * qbar[k] - sum_of(point.g) * nbar[k] + point.b * previous[k];
    }
    for (std::size_t lag = 1; lag <= point.a.size(); ++lag)
    {
        Matrix3 zz = qbar;
        Matrix3 nn = nbar;
        if (t >= lag)
        {
            const std::span<const double> z(panel.z.data() + (t - lag) * kDcc3Series, kDcc3Series);
            zz = outer(z, false);
            nn = outer(z, true);
        }
        double g = 0.0;
        if (!point.g.empty())
        {
            g = point.g[lag - 1];
        }
        for (std::size_t k = 0; k < kCells; ++k)
        {
            out[k] += point.a[lag - 1] * zz[k] + g * nn[k];
        }
    }
    return out;
}

Matrix3 correlation_of(const Matrix3& q)
{
    Matrix3 r{};
    for (std::size_t i = 0; i < kDcc3Series; ++i)
    {
        for (std::size_t j = 0; j < kDcc3Series; ++j)
        {
            r[i * kDcc3Series + j] = q[i * kDcc3Series + j] / std::sqrt(q[i * 4] * q[j * 4]);
        }
    }
    return r;
}

double direct_loglik(const StandardizedPanel& panel, const DirectPoint& point, DccDist dist)
{
    std::vector<Matrix3> q;
    double loglik = 0.0;
    for (std::size_t t = 0; t < panel.t; ++t)
    {
        q.push_back(direct_q(panel, point, q, t));
        const std::span<const double> z(panel.z.data() + t * kDcc3Series, kDcc3Series);
        loglik += direct_log_density(dist, correlation_of(q.back()), z, point.shape) - log_sigma_sum(panel, t);
    }
    return loglik;
}

std::vector<double> theta_of(const DirectPoint& point, DccDist dist)
{
    std::vector<double> theta = point.a;
    theta.push_back(point.b);
    theta.insert(theta.end(), point.g.begin(), point.g.end());
    if (dist == DccDist::MvT)
    {
        theta.push_back(point.shape);
    }
    return theta;
}

TEST(DccLikelihood, StandardMatchesDirectEvaluation)
{
    const StandardizedPanel panel = ve::testing::dcc3_panel();
    const Matrix3 first_r = correlation_of(to_matrix(panel.qbar));
    const std::array<DirectPoint, 2> points{DirectPoint{{0.04}, {}, 0.91, 7.0},
                                            DirectPoint{{0.03, 0.01}, {0.02, 0.005}, 0.9, 7.0}};
    for (const DirectPoint& point : points)
    {
        for (const DccDist dist : {DccDist::MvNorm, DccDist::MvT, DccDist::MvLaplace})
        {
            DccModel model = DccModel::ADcc;
            if (point.g.empty())
            {
                model = DccModel::Dcc;
            }
            DccSpec spec = spec_of(model, dist, Convention::Standard);
            spec.p = static_cast<int>(point.a.size());
            const std::vector<double> theta = theta_of(point, dist);
            std::vector<double> r(kDcc3Rows * kCells);
            const double loglik = ve::dcc::reported_loglik(panel, spec, theta, r);
            const double expected = direct_loglik(panel, point, dist);
            const std::string label = "p = " + std::to_string(spec.p) + ", dist " + std::to_string(int(dist));
            EXPECT_NEAR(loglik, expected, 1e-12 * std::fabs(expected)) << label;
            for (std::size_t k = 0; k < kCells; ++k)
            {
                EXPECT_NEAR(r[k], first_r[k], 1e-15) << label << ", cell " << k;
            }
            const double objective = ve::dcc::objective<double>(panel, spec, theta);
            const double minus_log_f = -(loglik + total_log_sigma(panel));
            EXPECT_NEAR(objective, minus_log_f, 1e-12 * std::fabs(minus_log_f)) << label;
        }
    }
}

std::vector<double> with_shape(std::vector<double> theta, DccDist dist)
{
    if (dist == DccDist::MvT)
    {
        theta.push_back(7.0);
    }
    return theta;
}

TEST(DccLikelihood, FdccOneGroupIsDcc)
{
    const StandardizedPanel panel = ve::testing::dcc3_panel();
    const double a = 0.2;
    const double b = 0.95;
    for (const Convention convention : {Convention::Standard, Convention::Rmgarch})
    {
        for (const DccDist dist : {DccDist::MvNorm, DccDist::MvT, DccDist::MvLaplace})
        {
            DccSpec fdcc = spec_of(DccModel::FDcc, dist, convention);
            fdcc.groups = {1, 1, 1};
            const DccSpec dcc = spec_of(DccModel::Dcc, dist, convention);
            std::vector<double> r_fdcc(kDcc3Rows * kCells);
            std::vector<double> r_dcc(kDcc3Rows * kCells);
            const double fdcc_loglik = ve::dcc::reported_loglik(panel, fdcc, with_shape({a, b}, dist), r_fdcc);
            const double dcc_loglik = ve::dcc::reported_loglik(panel, dcc, with_shape({a * a, b * b}, dist), r_dcc);
            const std::string label = "convention " + std::to_string(int(convention)) + ", dist " +
                                      std::to_string(int(dist));
            EXPECT_NEAR(fdcc_loglik, dcc_loglik, 1e-13 * std::fabs(dcc_loglik)) << label;
            double worst = 0.0;
            for (std::size_t k = 0; k < r_dcc.size(); ++k)
            {
                worst = std::max(worst, std::fabs(r_fdcc[k] - r_dcc[k]));
            }
            EXPECT_LT(worst, 1e-13) << label;
        }
    }
}

// Qbar o (11' - alpha alpha' - beta beta') for groups (1, 1, 2).
std::vector<double> fdcc_intercept(const StandardizedPanel& panel, std::array<double, 2> a, std::array<double, 2> b)
{
    const std::array<std::size_t, kDcc3Series> group{0, 0, 1};
    std::vector<double> intercept(kCells);
    for (std::size_t i = 0; i < kDcc3Series; ++i)
    {
        for (std::size_t j = 0; j < kDcc3Series; ++j)
        {
            const double weight = 1.0 - a[group[i]] * a[group[j]] - b[group[i]] * b[group[j]];
            intercept[i * kDcc3Series + j] = weight * panel.qbar[i * kDcc3Series + j];
        }
    }
    return intercept;
}

TEST(DccLikelihood, FdccIntercept)
{
    const StandardizedPanel panel = ve::testing::dcc3_panel();
    const DccSpec spec = spec_of(DccModel::FDcc, DccDist::MvNorm, Convention::Standard);

    // Each group inside a^2 + b^2 < 1, the intercept positive definite: a finite objective.
    std::vector<double> inside = fdcc_intercept(panel, {0.6, 0.05}, {0.6, 0.95});
    ASSERT_TRUE(ve::linalg::cholesky_factor(inside, kDcc3Series));
    EXPECT_TRUE(std::isfinite(ve::dcc::objective<double>(panel, spec, std::vector<double>{0.6, 0.05, 0.6, 0.95})));

    // Each group still inside a^2 + b^2 < 1 (0.9125 and 0.74), but the intercept is not positive definite.
    std::vector<double> outside = fdcc_intercept(panel, {0.1, 0.7}, {0.95, 0.5});
    ASSERT_FALSE(ve::linalg::cholesky_factor(outside, kDcc3Series));
    const std::vector<double> theta{0.1, 0.7, 0.95, 0.5};
    const double infinity = std::numeric_limits<double>::infinity();
    EXPECT_EQ(ve::dcc::objective<double>(panel, spec, theta), infinity);
    std::vector<ve::ad::Dual<8>> dual_theta;
    for (std::size_t i = 0; i < theta.size(); ++i)
    {
        dual_theta.push_back(ve::ad::variable<8>(theta[i], i));
    }
    EXPECT_EQ(ve::dcc::objective<ve::ad::Dual<8>>(panel, spec, dual_theta).value, infinity);
    EXPECT_TRUE(std::isnan(ve::dcc::reported_loglik(panel, spec, theta)));
}

struct GradientCase
{
    std::string name;
    DccSpec spec;
    std::vector<double> theta;
};

double central_difference(const StandardizedPanel& panel, const GradientCase& c, std::size_t i, double h)
{
    std::vector<double> up = c.theta;
    std::vector<double> down = c.theta;
    up[i] += h;
    down[i] -= h;
    return (ve::dcc::objective<double>(panel, c.spec, up) - ve::dcc::objective<double>(panel, c.spec, down)) /
           (2.0 * h);
}

// Richardson extrapolation of central differences at h and 2h, accurate to O(h^4): the third derivative of the
// FDCC objective in b is large enough that a plain central difference misses the gradient by more than 1e-6.
double extrapolated_difference(const StandardizedPanel& panel, const GradientCase& c, std::size_t i)
{
    const double h = 1e-5 * std::max(std::fabs(c.theta[i]), 1.0);
    return (4.0 * central_difference(panel, c, i, h) - central_difference(panel, c, i, 2.0 * h)) / 3.0;
}

TEST(DccLikelihood, DualGradientMatchesDifferences)
{
    using Number = ve::ad::Dual<8>;
    const StandardizedPanel panel = ve::testing::dcc3_panel();
    const std::vector<GradientCase> cases{
        {"DCC/mvnorm", spec_of(DccModel::Dcc, DccDist::MvNorm, Convention::Standard), {0.04, 0.9}},
        {"aDCC/mvt", spec_of(DccModel::ADcc, DccDist::MvT, Convention::Standard), {0.03, 0.9, 0.02, 7.0}},
        {"FDCC/mvlaplace", spec_of(DccModel::FDcc, DccDist::MvLaplace, Convention::Standard), {0.2, 0.25, 0.95, 0.94}},
        {"Rmgarch DCC/mvnorm", spec_of(DccModel::Dcc, DccDist::MvNorm, Convention::Rmgarch), {0.04, 0.9}},
    };
    for (const GradientCase& c : cases)
    {
        std::vector<Number> theta;
        for (std::size_t i = 0; i < c.theta.size(); ++i)
        {
            theta.push_back(ve::ad::variable<8>(c.theta[i], i));
        }
        const Number value = ve::dcc::objective<Number>(panel, c.spec, theta);
        EXPECT_DOUBLE_EQ(value.value, ve::dcc::objective<double>(panel, c.spec, c.theta)) << c.name;
        for (std::size_t i = 0; i < c.theta.size(); ++i)
        {
            const double difference = extrapolated_difference(panel, c, i);
            EXPECT_NEAR(value.grad[i], difference, 1e-6 * std::max(std::fabs(difference), 1.0))
                << c.name << ", parameter " << i;
        }
    }
}

}
