#include "dcc/standard_errors.hpp"

#include "copula3_panel.hpp"
#include "core/context.hpp"
#include "dcc/fit.hpp"
#include "dcc/panel.hpp"
#include "dcc/spec.hpp"
#include "fit/fit_result.hpp"
#include "linalg/lu.hpp"

#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <gtest/gtest.h>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace
{

using ve::FitConfig;
using ve::FitResult;
using ve::FitStatus;
using ve::dcc::Convention;
using ve::dcc::DccCopula;
using ve::dcc::DccDist;
using ve::dcc::DccFitResult;
using ve::dcc::DccModel;
using ve::dcc::DccOptions;
using ve::dcc::DccSpec;
using ve::dcc::DccStandardErrors;
using ve::dcc::StaticMethod;
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

DccSpec copula_spec(DccModel model, DccDist dist, DccCopula copula, Convention convention,
                    StaticMethod method = StaticMethod::Default)
{
    DccSpec spec;
    spec.model = model;
    spec.dist = dist;
    spec.copula = copula;
    spec.convention = convention;
    spec.static_method = method;
    return spec;
}

const std::vector<FitResult>& estimated_fits()
{
    static const std::vector<FitResult> fits = ve::testing::copula3_estimated_fits();
    return fits;
}

std::vector<DccFitResult> run_batch(const std::vector<DccSpec>& specs, bool standard_errors, int threads = 1)
{
    const Pointers fits = pointers_of(estimated_fits());
    const DccOptions options{FitConfig{}, standard_errors};
    EXPECT_FALSE(ve::dcc::check_dcc_batch(fits, specs, options).has_value());
    ve::Context context;
    context.set_threads(threads);
    return ve::dcc::fit_dcc_batch(context, fits, specs, options);
}

// The batch result of spec, moved to estimate (rmgarch's point).
DccFitResult at_point(const DccSpec& spec, std::vector<double> estimate)
{
    DccFitResult fit = run_batch({spec}, false)[0];
    EXPECT_EQ(fit.estimate.size(), estimate.size());
    fit.estimate = std::move(estimate);
    return fit;
}

struct Oracle
{
    const char* id;
    DccSpec spec;
    std::vector<double> estimate;
    std::vector<double> robust;
};

TEST(CopulaStandardErrors, MatchRmgarch)
{
    // rmgarch's [Joint] estimates and robust errors on copula3 (plan 4b notes, section 7).
    const std::vector<Oracle> oracles{
        {"dcc_mvnorm_param", copula_spec(DccModel::Dcc, DccDist::MvNorm, DccCopula::Parametric, Convention::Rmgarch),
         {0.0473819588956065, 0.9175354409917849}, {0.00688260476128296, 0.01506025946251671}},
        {"dcc_mvt_param", copula_spec(DccModel::Dcc, DccDist::MvT, DccCopula::Parametric, Convention::Rmgarch),
         {0.0444504443322488, 0.9194035843020166, 5.8841836499319067},
         {0.0058219978418709, 0.0121850853069757, 0.4971728250148147}},
        {"adcc_mvnorm_param", copula_spec(DccModel::ADcc, DccDist::MvNorm, DccCopula::Parametric, Convention::Rmgarch),
         {0.0453138390863231, 0.9178306244174149, 0.0048824360257134},
         {0.00800333265116977, 0.01490734007035691, 0.00975448437638895}},
        {"static_mvt_kendall_param",
         copula_spec(DccModel::Static, DccDist::MvT, DccCopula::Parametric, Convention::Rmgarch),
         {4.7793578808066}, {0.3902939004744}},
        {"static_mvt_ml_param",
         copula_spec(DccModel::Static, DccDist::MvT, DccCopula::Parametric, Convention::Rmgarch, StaticMethod::Ml),
         {0.594217920400736, 0.436936698871691, 0.277810026573495, 4.790827028666584},
         {0.0272004100371396, 0.0269005277361837, 0.0253538144732677, 0.4064715629142370}},
        {"static_mvnorm_ml_param",
         copula_spec(DccModel::Static, DccDist::MvNorm, DccCopula::Parametric, Convention::Rmgarch, StaticMethod::Ml),
         {0.598672396279854, 0.449366756278866, 0.265832497450360},
         {0.0286473454512667, 0.0280579284266360, 0.0269658866816793}},
    };
    for (const Oracle& oracle : oracles)
    {
        SCOPED_TRACE(oracle.id);
        const DccFitResult fit = at_point(oracle.spec, oracle.estimate);
        const std::expected<DccStandardErrors, std::string> errors =
            ve::dcc::dcc_standard_errors(fit, pointers_of(estimated_fits()));
        ASSERT_TRUE(errors.has_value()) << errors.error();
        ASSERT_EQ(errors->robust.size(), oracle.robust.size());
        for (std::size_t i = 0; i < oracle.robust.size(); ++i)
        {
            EXPECT_NEAR(errors->robust[i], oracle.robust[i], 5e-3 * oracle.robust[i]) << fit.names[i];
            EXPECT_GT(errors->hessian[i], 0.0) << fit.names[i];
        }
    }
}

TEST(CopulaStandardErrors, EmpiricalHasNoCrossBlock)
{
    const DccFitResult fit =
        run_batch({copula_spec(DccModel::Dcc, DccDist::MvNorm, DccCopula::Empirical, Convention::Standard)}, false)[0];
    ASSERT_EQ(fit.status, FitStatus::Converged);
    const auto pieces = ve::dcc::two_stage_pieces(fit, pointers_of(estimated_fits()));
    ASSERT_TRUE(pieces.has_value()) << pieces.error();
    const std::size_t n = pieces->parameters;
    const std::size_t dcc = fit.estimate.size();
    const std::size_t u = n - dcc;
    for (std::size_t row = u; row < n; ++row)
    {
        for (std::size_t column = 0; column < u; ++column)
        {
            EXPECT_EQ(pieces->a[row * n + column], 0.0) << row << " " << column;
        }
    }
    // With no cross block, the copula block of V is A22^-1 cov(s2) A22^-T / T, the second stage alone.
    const std::size_t t = pieces->scores.size() / n;
    std::vector<double> s2(t * dcc);
    for (std::size_t row = 0; row < t; ++row)
    {
        for (std::size_t j = 0; j < dcc; ++j)
        {
            s2[row * dcc + j] = pieces->scores[row * n + u + j];
        }
    }
    const std::vector<double> b = ve::dcc::second_moment(s2, t, dcc, true, static_cast<double>(t) - 1.0);
    std::vector<double> a22(dcc * dcc);
    for (std::size_t k = 0; k < dcc * dcc; ++k)
    {
        a22[k] = pieces->a[(u + k / dcc) * n + u + k % dcc] / static_cast<double>(t);
    }
    std::vector<std::size_t> pivots(dcc);
    ASSERT_TRUE(ve::linalg::lu_factor(a22, pivots, dcc));
    // V = A^-1 B A^-T / T: solve the columns of B, transpose, solve again.
    std::vector<double> left = b;
    for (std::size_t j = 0; j < dcc; ++j)
    {
        std::vector<double> column(dcc);
        for (std::size_t i = 0; i < dcc; ++i)
        {
            column[i] = left[i * dcc + j];
        }
        ve::linalg::lu_solve(a22, pivots, dcc, column);
        for (std::size_t i = 0; i < dcc; ++i)
        {
            left[i * dcc + j] = column[i];
        }
    }
    const std::expected<DccStandardErrors, std::string> errors =
        ve::dcc::dcc_standard_errors(fit, pointers_of(estimated_fits()));
    ASSERT_TRUE(errors.has_value());
    for (std::size_t i = 0; i < dcc; ++i)
    {
        std::vector<double> row(dcc);
        for (std::size_t k = 0; k < dcc; ++k)
        {
            row[k] = left[i * dcc + k];
        }
        ve::linalg::lu_solve(a22, pivots, dcc, row);
        const double expected = std::sqrt(std::abs(row[i]) / static_cast<double>(t));
        EXPECT_NEAR(errors->robust[i], expected, 1e-10 * expected) << fit.names[i];
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

TEST(CopulaStandardErrors, OptionMatchesExplicit)
{
    const std::vector<DccSpec> specs{
        copula_spec(DccModel::Dcc, DccDist::MvT, DccCopula::Parametric, Convention::Standard),
        copula_spec(DccModel::ADcc, DccDist::MvNorm, DccCopula::Empirical, Convention::Standard),
        copula_spec(DccModel::Static, DccDist::MvT, DccCopula::Parametric, Convention::Standard),
        copula_spec(DccModel::Static, DccDist::MvNorm, DccCopula::Empirical, Convention::Standard, StaticMethod::Ml),
    };
    const std::vector<DccFitResult> with = run_batch(specs, true, 3);
    for (const DccFitResult& fit : with)
    {
        SCOPED_TRACE(fit.names.size());
        ASSERT_EQ(fit.status, FitStatus::Converged);
        const std::expected<DccStandardErrors, std::string> errors =
            ve::dcc::dcc_standard_errors(fit, pointers_of(estimated_fits()));
        ASSERT_TRUE(errors.has_value());
        EXPECT_TRUE(same_bits(fit.se_robust, errors->robust));
        EXPECT_TRUE(same_bits(fit.se_hessian, errors->hessian));
        for (double value : fit.se_robust)
        {
            EXPECT_TRUE(std::isfinite(value) && value > 0.0);
        }
    }
}

}
