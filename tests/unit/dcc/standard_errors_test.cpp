#include "dcc/standard_errors.hpp"

#include "core/context.hpp"
#include "core/spec.hpp"
#include "dcc/fit.hpp"
#include "dcc/spec.hpp"
#include "dcc3_panel.hpp"
#include "fit/fit.hpp"
#include "fit/fit_result.hpp"
#include "linalg/cholesky.hpp"
#include "model/data.hpp"

#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <gtest/gtest.h>
#include <span>
#include <string>
#include <vector>

namespace
{

using ve::FitConfig;
using ve::FitResult;
using ve::FitStatus;
using ve::dcc::Convention;
using ve::dcc::DccDist;
using ve::dcc::DccFitResult;
using ve::dcc::DccModel;
using ve::dcc::DccOptions;
using ve::dcc::DccSpec;
using ve::dcc::DccStandardErrors;
using ve::dcc::TwoStagePieces;

using Pointers = std::vector<const FitResult*>;

Pointers pointers_of(const std::vector<FitResult>& fits)
{
    Pointers pointers;
    for (const FitResult& fit : fits)
    {
        pointers.push_back(&fit);
    }
    return pointers;
}

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

// The dcc3 fits estimated at rugarch's point, computed once.
const std::vector<FitResult>& rugarch_fits()
{
    static const std::vector<FitResult> fits = ve::testing::dcc3_estimated_fits();
    return fits;
}

std::vector<DccFitResult> run_batch(const Pointers& fits, const std::vector<DccSpec>& specs, const DccOptions& options,
                                    int threads)
{
    ve::Context context;
    context.set_threads(threads);
    EXPECT_FALSE(ve::dcc::check_dcc_batch(fits, specs).has_value());
    return ve::dcc::fit_dcc_batch(context, fits, specs, options);
}

// The batch result of spec, moved to estimate (rmgarch's point).
DccFitResult at_point(const std::vector<FitResult>& fits, const DccSpec& spec, std::vector<double> estimate)
{
    DccFitResult fit = run_batch(pointers_of(fits), {spec}, DccOptions{FitConfig{}, false}, 1)[0];
    EXPECT_EQ(fit.estimate.size(), estimate.size());
    fit.estimate = std::move(estimate);
    return fit;
}

void expect_relative(std::span<const double> actual, std::span<const double> expected, double tolerance)
{
    ASSERT_EQ(actual.size(), expected.size());
    for (std::size_t i = 0; i < actual.size(); ++i)
    {
        EXPECT_NEAR(actual[i], expected[i], tolerance * std::fabs(expected[i])) << "parameter " << i;
    }
}

bool same_bits(std::span<const double> a, std::span<const double> b)
{
    if (a.size() != b.size())
    {
        return false;
    }
    for (std::size_t i = 0; i < a.size(); ++i)
    {
        if (std::bit_cast<uint64_t>(a[i]) != std::bit_cast<uint64_t>(b[i]))
        {
            return false;
        }
    }
    return true;
}

const std::vector<double> kRmgarchDcc{0.0431630001835819, 0.9142701826310855};

TEST(DccStandardErrors, MatchRmgarchSandwich)
{
    const std::vector<FitResult>& fits = rugarch_fits();
    for (std::size_t i = 0; i < fits.size(); ++i)
    {
        ASSERT_EQ(fits[i].status, FitStatus::Converged);
        for (std::size_t p = 0; p < 4; ++p)
        {
            const double expected = ve::testing::kDcc3Coefficients[i][p];
            EXPECT_NEAR(fits[i].estimate[p], expected, 1e-12 * std::fabs(expected)) << i << " " << p;
        }
    }
    const DccFitResult fit = at_point(fits, spec_of(DccModel::Dcc, DccDist::MvNorm, Convention::Rmgarch), kRmgarchDcc);
    const std::expected<DccStandardErrors, std::string> errors = ve::dcc::dcc_standard_errors(fit, pointers_of(fits));
    ASSERT_TRUE(errors.has_value()) << errors.error();
    // rmgarch's univariate Hessians and scores are numerical, ours come from automatic derivatives.
    expect_relative(errors->robust, std::vector<double>{0.00627467301294802, 0.0132675580840498}, 2e-3);
    expect_relative(errors->hessian, std::vector<double>{0.00684873, 0.01556525}, 2e-3);
}

// Entry (row, column) of the row-major matrix with `columns` columns.
double at(std::span<const double> matrix, std::size_t columns, std::size_t row, std::size_t column)
{
    return matrix[row * columns + column];
}

// The block rows [first_row, first_row + rows) x columns [first_column, first_column + columns) of an n x n matrix.
std::vector<double> block(std::span<const double> a, std::size_t n, std::array<std::size_t, 2> first,
                          std::array<std::size_t, 2> size)
{
    std::vector<double> out;
    for (std::size_t i = 0; i < size[0]; ++i)
    {
        for (std::size_t j = 0; j < size[1]; ++j)
        {
            out.push_back(at(a, n, first[0] + i, first[1] + j));
        }
    }
    return out;
}

// cov of the T x k rows of x, centred, divisor T - 1.
std::vector<double> covariance(std::span<const double> x, std::size_t k)
{
    const std::size_t t = x.size() / k;
    std::vector<double> mean(k, 0.0);
    for (std::size_t row = 0; row < t; ++row)
    {
        for (std::size_t i = 0; i < k; ++i)
        {
            mean[i] += at(x, k, row, i) / static_cast<double>(t);
        }
    }
    std::vector<double> out(k * k, 0.0);
    for (std::size_t row = 0; row < t; ++row)
    {
        for (std::size_t i = 0; i < k * k; ++i)
        {
            out[i] += (at(x, k, row, i / k) - mean[i / k]) * (at(x, k, row, i % k) - mean[i % k]);
        }
    }
    for (double& v : out)
    {
        v /= static_cast<double>(t - 1);
    }
    return out;
}

// a_t = s2_t - H21 H11^-1 s1_t in the signs of +l: H11 = -A11 and s1 = -S1, so H11^-1 s1 = A11^-1 S1.
std::vector<double> efficient_scores(const TwoStagePieces& pieces, std::size_t univariate)
{
    const std::size_t n = pieces.parameters;
    const std::size_t dcc = n - univariate;
    const std::size_t t = pieces.scores.size() / n;
    std::vector<double> factor = block(pieces.a, n, {0, 0}, {univariate, univariate});
    EXPECT_TRUE(ve::linalg::cholesky_factor(factor, univariate));
    const std::vector<double> h21 = block(pieces.a, n, {univariate, 0}, {dcc, univariate});
    std::vector<double> a(t * dcc);
    for (std::size_t row = 0; row < t; ++row)
    {
        std::vector<double> solved = block(pieces.scores, n, {row, 0}, {1, univariate});
        ve::linalg::cholesky_solve(factor, univariate, solved);
        for (std::size_t j = 0; j < dcc; ++j)
        {
            double projection = 0.0;
            for (std::size_t i = 0; i < univariate; ++i)
            {
                projection += at(h21, univariate, j, i) * solved[i];
            }
            a[row * dcc + j] = at(pieces.scores, n, row, univariate + j) - projection;
        }
    }
    return a;
}

// T H22^-1 cov(a) H22^-1, from the Cholesky factor of -H22 (H22^-1 M H22^-1 = (-H22)^-1 M (-H22)^-1).
std::vector<double> textbook_dcc_block(const TwoStagePieces& pieces, std::size_t univariate)
{
    const std::size_t n = pieces.parameters;
    const std::size_t dcc = n - univariate;
    const std::size_t t = pieces.scores.size() / n;
    std::vector<double> factor = block(pieces.a, n, {univariate, univariate}, {dcc, dcc});
    for (double& v : factor)
    {
        v = -v;
    }
    EXPECT_TRUE(ve::linalg::cholesky_factor(factor, dcc));
    const std::vector<double> middle = covariance(efficient_scores(pieces, univariate), dcc);
    std::vector<double> left(dcc * dcc);
    for (std::size_t j = 0; j < dcc; ++j)
    {
        std::vector<double> column = block(middle, dcc, {j, 0}, {1, dcc});
        ve::linalg::cholesky_solve(factor, dcc, column);
        for (std::size_t i = 0; i < dcc; ++i)
        {
            left[i * dcc + j] = column[i] * static_cast<double>(t);
        }
    }
    std::vector<double> out(dcc * dcc);
    for (std::size_t j = 0; j < dcc; ++j)
    {
        std::vector<double> row = block(left, dcc, {j, 0}, {1, dcc});
        ve::linalg::cholesky_solve(factor, dcc, row);
        for (std::size_t i = 0; i < dcc; ++i)
        {
            out[j * dcc + i] = row[i];
        }
    }
    return out;
}

TEST(DccStandardErrors, TextbookSandwichOnRmgarchPieces)
{
    const std::vector<FitResult>& fits = rugarch_fits();
    const DccFitResult fit = at_point(fits, spec_of(DccModel::Dcc, DccDist::MvNorm, Convention::Rmgarch), kRmgarchDcc);
    const std::expected<TwoStagePieces, std::string> pieces = ve::dcc::two_stage_pieces(fit, pointers_of(fits));
    ASSERT_TRUE(pieces.has_value()) << pieces.error();
    const std::size_t n = pieces->parameters;
    ASSERT_EQ(n, 14U);
    ASSERT_EQ(pieces->a.size(), n * n);
    ASSERT_EQ(pieces->scores.size(), ve::testing::kDcc3Rows * n);
    const std::size_t univariate = 12;
    const std::vector<double> v = ve::dcc::sandwich(*pieces, Convention::Standard);
    ASSERT_EQ(v.size(), n * n);
    const std::vector<double> se{std::sqrt(std::fabs(at(v, n, 12, 12))), std::sqrt(std::fabs(at(v, n, 13, 13)))};
    expect_relative(se, std::vector<double>{0.00639279, 0.01334010}, 2e-3);
    for (std::size_t i = 0; i < n; ++i)
    {
        for (std::size_t j = 0; j < i; ++j)
        {
            const double scale = std::sqrt(std::fabs(at(v, n, i, i) * at(v, n, j, j)));
            EXPECT_NEAR(at(v, n, i, j), at(v, n, j, i), 1e-12 * scale) << i << " " << j;
        }
    }
    const std::vector<double> textbook = textbook_dcc_block(*pieces, univariate);
    const std::vector<double> ours = block(v, n, {univariate, univariate}, {2, 2});
    for (std::size_t k = 0; k < ours.size(); ++k)
    {
        EXPECT_NEAR(ours[k], textbook[k], 1e-10 * std::fabs(textbook[k])) << k;
    }
    const std::vector<double> rmgarch = ve::dcc::sandwich(*pieces, Convention::Rmgarch);
    const std::expected<DccStandardErrors, std::string> errors = ve::dcc::dcc_standard_errors(fit, pointers_of(fits));
    ASSERT_TRUE(errors.has_value());
    EXPECT_EQ(errors->robust[0], std::sqrt(std::fabs(at(rmgarch, n, 12, 12))));
    EXPECT_EQ(errors->robust[1], std::sqrt(std::fabs(at(rmgarch, n, 13, 13))));
}

struct Variant
{
    DccModel model;
    DccDist dist;
    std::vector<double> estimate;
    std::vector<double> se;
};

// rmgarch's se(mshape) of the mvt, 10.0908, is not compared: at the ceiling 50 the forward-difference curvature in
// mshape, with rmgarch's steps, is rounding noise of the summed loglik, in rmgarch as in VolEngine.
const std::array<Variant, 4> kVariants{{
    {DccModel::Dcc, DccDist::MvT, {0.04306741677, 0.91353820409, 49.99998817}, {0.00637055, 0.01371052}},
    {DccModel::ADcc,
     DccDist::MvNorm,
     {0.04243321505, 0.91398307950, 0.00240804736},
     {0.00695453, 0.01347095, 0.01040230}},
    {DccModel::FDcc,
     DccDist::MvNorm,
     {0.19004678480, 0.24108698852, 0.96463050567, 0.94296237137},
     {0.02580607, 0.04327988, 0.01296845, 0.01708654}},
    {DccModel::Dcc, DccDist::MvLaplace, {0.03872917063, 0.90870637605}, {0.00679018, 0.01707081}},
}};

TEST(DccStandardErrors, OtherVariants)
{
    const std::vector<FitResult>& fits = rugarch_fits();
    const Pointers pointers = pointers_of(fits);
    for (const Variant& variant : kVariants)
    {
        SCOPED_TRACE(static_cast<int>(variant.model) * 10 + static_cast<int>(variant.dist));
        const DccFitResult rmgarch = at_point(fits, spec_of(variant.model, variant.dist, Convention::Rmgarch),
                                              variant.estimate);
        const std::expected<DccStandardErrors, std::string> errors = ve::dcc::dcc_standard_errors(rmgarch, pointers);
        ASSERT_TRUE(errors.has_value()) << errors.error();
        ASSERT_EQ(errors->robust.size(), variant.estimate.size());
        expect_relative(std::span(errors->robust).first(variant.se.size()), variant.se, 5e-3);
        EXPECT_TRUE(std::isfinite(errors->robust.back()) && errors->robust.back() > 0.0);
        const DccSpec standard = spec_of(variant.model, variant.dist, Convention::Standard);
        const DccFitResult fit = run_batch(pointers, {standard}, DccOptions{FitConfig{}, true}, 1)[0];
        ASSERT_EQ(fit.status, FitStatus::Converged);
        for (std::size_t i = 0; i < fit.estimate.size(); ++i)
        {
            EXPECT_TRUE(std::isfinite(fit.se_robust[i]) && fit.se_robust[i] > 0.0) << fit.names[i];
            EXPECT_TRUE(std::isfinite(fit.se_hessian[i]) && fit.se_hessian[i] > 0.0) << fit.names[i];
        }
    }
}

std::vector<FitResult> default_fits()
{
    std::vector<FitResult> fits;
    for (const std::vector<double>& column : ve::testing::read_dcc3())
    {
        fits.push_back(ve::fit_one(column, ve::Spec{}, FitConfig{}));
    }
    return fits;
}

void expect_rejected(const DccFitResult& fit, const Pointers& fits, const std::string& needle = "")
{
    const std::expected<DccStandardErrors, std::string> errors = ve::dcc::dcc_standard_errors(fit, fits);
    ASSERT_FALSE(errors.has_value());
    EXPECT_FALSE(errors.error().empty());
    EXPECT_NE(errors.error().find(needle), std::string::npos) << errors.error();
    EXPECT_TRUE(ve::dcc::fits_problem(fit, fits).has_value() || !needle.empty());
    EXPECT_FALSE(ve::dcc::two_stage_pieces(fit, fits).has_value());
}

// Series 1 with a variance regressor, the squared lagged return of series 2.
std::vector<FitResult> fits_with_variance_regressor(const std::vector<std::vector<double>>& columns,
                                                    std::vector<double>& regressor)
{
    regressor.assign(columns[1].size(), 0.0);
    for (std::size_t t = 1; t < regressor.size(); ++t)
    {
        regressor[t] = columns[1][t - 1] * columns[1][t - 1];
    }
    ve::Spec spec;
    spec.vxreg = 1;
    const ve::model::Data data{columns[0], {}, {regressor, 1}};
    std::vector<FitResult> fits{ve::fit_one(data, spec, FitConfig{})};
    fits.push_back(ve::fit_one(columns[1], ve::Spec{}, FitConfig{}));
    fits.push_back(ve::fit_one(columns[2], ve::Spec{}, FitConfig{}));
    return fits;
}

TEST(DccStandardErrors, RejectsOtherFits)
{
    const std::vector<FitResult>& fits = rugarch_fits();
    const DccFitResult fit = at_point(fits, spec_of(DccModel::Dcc, DccDist::MvNorm, Convention::Rmgarch), kRmgarchDcc);
    EXPECT_FALSE(ve::dcc::fits_problem(fit, pointers_of(fits)).has_value());
    const std::vector<FitResult> others = default_fits();
    expect_rejected(fit, pointers_of(others));
    expect_rejected(fit, {&fits[1], &fits[0], &fits[2]});
    expect_rejected(fit, {&fits[0], &fits[1]});
    std::vector<FitResult> refitted = fits;
    ve::Spec gjr;
    gjr.model = ve::Model::GjrGarch;
    refitted[1] = ve::fit_one(ve::testing::read_dcc3()[1], gjr, FitConfig{});
    expect_rejected(fit, pointers_of(refitted));
    expect_rejected(ve::dcc::fit_dcc(fit.panel, fit.spec, FitConfig{}), pointers_of(fits));

    std::vector<double> regressor;
    const std::vector<FitResult> regressed = fits_with_variance_regressor(ve::testing::read_dcc3(), regressor);
    const DccFitResult with_regressor =
        run_batch(pointers_of(regressed), {fit.spec}, DccOptions{FitConfig{}, true}, 1)[0];
    ASSERT_EQ(with_regressor.status, FitStatus::Converged);
    EXPECT_TRUE(std::isnan(with_regressor.se_robust[0]) && std::isnan(with_regressor.se_hessian[1]));
    EXPECT_FALSE(ve::dcc::fits_problem(with_regressor, pointers_of(regressed)).has_value());
    expect_rejected(with_regressor, pointers_of(regressed), "regressors");
}

TEST(DccStandardErrors, OptionMatchesExplicit)
{
    const std::vector<FitResult> fits = default_fits();
    const Pointers pointers = pointers_of(fits);
    const std::vector<DccSpec> specs{spec_of(DccModel::Dcc, DccDist::MvNorm, Convention::Standard),
                                     spec_of(DccModel::FDcc, DccDist::MvNorm, Convention::Standard)};
    const std::vector<DccFitResult> one = run_batch(pointers, specs, DccOptions{FitConfig{}, true}, 1);
    const std::vector<DccFitResult> three = run_batch(pointers, specs, DccOptions{FitConfig{}, true}, 3);
    const std::vector<DccFitResult> off = run_batch(pointers, specs, DccOptions{FitConfig{}, false}, 1);
    for (std::size_t i = 0; i < specs.size(); ++i)
    {
        SCOPED_TRACE(i);
        ASSERT_EQ(one[i].status, FitStatus::Converged);
        const std::expected<DccStandardErrors, std::string> errors = ve::dcc::dcc_standard_errors(one[i], pointers);
        ASSERT_TRUE(errors.has_value()) << errors.error();
        EXPECT_TRUE(same_bits(one[i].se_robust, errors->robust));
        EXPECT_TRUE(same_bits(one[i].se_hessian, errors->hessian));
        EXPECT_TRUE(same_bits(three[i].se_robust, errors->robust));
        EXPECT_TRUE(same_bits(three[i].se_hessian, errors->hessian));
        EXPECT_TRUE(same_bits(off[i].estimate, one[i].estimate));
        for (std::size_t k = 0; k < one[i].estimate.size(); ++k)
        {
            EXPECT_TRUE(std::isfinite(one[i].se_robust[k]));
            EXPECT_TRUE(std::isnan(off[i].se_robust[k]) && std::isnan(off[i].se_hessian[k]));
        }
    }
    FitConfig short_config;
    short_config.max_iter = 2;
    const DccFitResult stopped = run_batch(pointers, {specs[0]}, DccOptions{short_config, true}, 1)[0];
    ASSERT_EQ(stopped.status, FitStatus::MaxIter);
    EXPECT_TRUE(std::isnan(stopped.se_robust[0]) && std::isnan(stopped.se_hessian[0]));
}

}
