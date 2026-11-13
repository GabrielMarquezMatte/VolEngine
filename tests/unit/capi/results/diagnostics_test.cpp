#include "capi/handles.hpp"
#include "capi_scope.hpp"
#include "diagnostics/gof.hpp"
#include "diagnostics/portmanteau.hpp"
#include "diagnostics/sign_bias.hpp"
#include "fit/fit.hpp"
#include "model/layout.hpp"
#include "simulated_garch.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <string>
#include <vector>
#include <gtest/gtest.h>

namespace
{

void expect_test(const std::array<double, 2>& actual, ve::diagnostics::TestResult expected)
{
    EXPECT_EQ(actual[0], expected.statistic);
    EXPECT_EQ(actual[1], expected.p_value);
}

ve::testing::FitSet fit_series(const std::vector<double>& series)
{
    auto context = ve::testing::make_context(1);
    auto spec = ve::testing::make_spec(1);
    EXPECT_EQ(ve_spec_set_arma(spec.get(), 1, 1), VE_OK);
    auto panel = ve::testing::make_panel(series.data(), series.size(), 1);
    ve::testing::BatchInputs inputs;
    inputs.specs = {spec.get()};
    inputs.panel = panel.get();
    inputs.job_spec = {0};
    inputs.job_series = {0};
    auto batch = ve::testing::build_batch(inputs);
    return ve::testing::fit_all(context.get(), batch.get());
}

}

TEST(CapiDiagnostics, MatchCore)
{
    const auto series = ve::testing::simulate_garch_series(1, 800);
    auto fits = fit_series(series);
    const auto* f = fits[0];
    const auto& core = f->result;
    std::array<double, 2> out{};
    for (const int squared : {0, 1})
    {
        ASSERT_EQ(ve_fit_ljung_box(f, squared, 5, 2, out.data()), VE_OK);
        expect_test(out, ve::diagnostics::weighted_ljung_box(core.std_residuals, 5, 2, squared != 0));
    }
    ASSERT_EQ(ve_fit_arch_lm(f, 5, 2, out.data()), VE_OK);
    expect_test(out, ve::diagnostics::weighted_arch_lm(core.residuals, core.sigma, 5, 2));
    ASSERT_EQ(ve_fit_gof(f, 20, out.data()), VE_OK);
    expect_test(out, ve::diagnostics::pearson_gof(core.std_residuals, ve::Dist::Norm, {}, 20));
    std::array<double, 8> sign{};
    ASSERT_EQ(ve_fit_sign_bias(f, sign.data()), VE_OK);
    const auto bias = ve::diagnostics::sign_bias(core.std_residuals, core.residuals);
    const std::array tests{bias.sign, bias.negative, bias.positive, bias.joint};
    for (std::size_t i = 0; i < tests.size(); ++i)
    {
        expect_test({sign[2 * i], sign[2 * i + 1]}, tests[i]);
    }
    std::vector<double> individual(core.names.size());
    double joint = 0.0;
    std::vector<int32_t> estimated(core.names.size(), -1);
    ASSERT_EQ(ve_fit_nyblom(f, individual.data(), &joint, estimated.data()), VE_OK);
    EXPECT_EQ(individual, core.nyblom.individual);
    EXPECT_EQ(joint, core.nyblom.joint);
    EXPECT_EQ(estimated, std::vector<int32_t>(core.names.size(), 1));
}

TEST(CapiDiagnostics, InvalidArguments)
{
    ve_fit empty;
    double out[8]{};
    int32_t flags[8]{};
    EXPECT_EQ(ve_fit_ljung_box(nullptr, 0, 1, 0, out), VE_INVALID_HANDLE);
    EXPECT_EQ(ve_fit_arch_lm(nullptr, 3, 2, out), VE_INVALID_HANDLE);
    EXPECT_EQ(ve_fit_nyblom(nullptr, out, out, flags), VE_INVALID_HANDLE);
    EXPECT_EQ(ve_fit_sign_bias(nullptr, out), VE_INVALID_HANDLE);
    EXPECT_EQ(ve_fit_gof(nullptr, 20, out), VE_INVALID_HANDLE);
    EXPECT_EQ(ve_fit_gof(&empty, 20, out), VE_INVALID_ARGUMENT);
    EXPECT_NE(std::string(ve_last_error()).find("finite"), std::string::npos);
    const auto series = ve::testing::simulate_garch_series(1, 10);
    auto fits = fit_series(series);
    auto* f = fits[0];
    ASSERT_FALSE(f->result.std_residuals.empty());
    EXPECT_EQ(ve_fit_ljung_box(f, 0, 20, 0, out), VE_INVALID_ARGUMENT);
    EXPECT_NE(std::string(ve_last_error()).find("lag"), std::string::npos);
    EXPECT_EQ(ve_fit_arch_lm(f, 5, 0, out), VE_INVALID_ARGUMENT);
    EXPECT_NE(std::string(ve_last_error()).find("fitdf"), std::string::npos);
    EXPECT_EQ(ve_fit_gof(f, 1, out), VE_INVALID_ARGUMENT);
    EXPECT_EQ(ve_fit_gof(f, 50, out), VE_INVALID_ARGUMENT);
    EXPECT_NE(std::string(ve_last_error()).find("groups"), std::string::npos);
    EXPECT_EQ(ve_fit_ljung_box(f, 2, 1, 0, out), VE_INVALID_ARGUMENT);
    EXPECT_EQ(ve_fit_ljung_box(f, 0, 1, 0, nullptr), VE_INVALID_ARGUMENT);
    EXPECT_EQ(ve_fit_arch_lm(f, 3, 2, nullptr), VE_INVALID_ARGUMENT);
    EXPECT_EQ(ve_fit_gof(f, 2, nullptr), VE_INVALID_ARGUMENT);
    EXPECT_EQ(ve_fit_sign_bias(f, nullptr), VE_INVALID_ARGUMENT);
    EXPECT_EQ(ve_fit_nyblom(f, nullptr, out, flags), VE_INVALID_ARGUMENT);
    EXPECT_EQ(ve_fit_nyblom(f, out, nullptr, flags), VE_INVALID_ARGUMENT);
    EXPECT_EQ(ve_fit_nyblom(f, out, out, nullptr), VE_INVALID_ARGUMENT);
    EXPECT_EQ(ve_nyblom_critical(0, out), VE_INVALID_ARGUMENT);
    EXPECT_EQ(ve_nyblom_critical(21, out), VE_INVALID_ARGUMENT);
    EXPECT_EQ(ve_nyblom_critical(1, nullptr), VE_INVALID_ARGUMENT);
    f->result.nobs = 5;
    f->result.std_residuals.resize(5);
    f->result.residuals.resize(5);
    EXPECT_EQ(ve_fit_sign_bias(f, out), VE_INVALID_ARGUMENT);
}

TEST(CapiDiagnostics, NyblomWithFixedParameters)
{
    ve::Spec spec;
    spec.fixed = {{"mu", 0.05}};
    ve_fit f;
    f.result = ve::fit_one(ve::testing::simulate_garch_series(1, 800), spec, ve::FitConfig{});
    std::vector<double> individual(f.result.names.size());
    double joint = 0.0;
    std::vector<int32_t> estimated(individual.size());
    ASSERT_EQ(ve_fit_nyblom(&f, individual.data(), &joint, estimated.data()), VE_OK);
    EXPECT_TRUE(std::isnan(individual[0]));
    EXPECT_EQ(estimated[0], 0);
    for (std::size_t i = 1; i < individual.size(); ++i)
    {
        EXPECT_TRUE(std::isfinite(individual[i]));
    }
    std::array<double, 3> critical{};
    ASSERT_EQ(ve_nyblom_critical(1, critical.data()), VE_OK);
    EXPECT_EQ(critical, (std::array<double, 3>{0.353, 0.470, 0.748}));
}

TEST(CapiDiagnostics, NoFinitePointRejectsEveryDiagnostic)
{
    ve_fit f;
    f.result = ve::fit_one(std::vector<double>(40, 1.0), ve::Spec{}, ve::FitConfig{});
    ASSERT_TRUE(f.result.std_residuals.empty());
    std::array<double, 8> out{};
    double joint = 123.0;
    EXPECT_EQ(ve_fit_ljung_box(&f, 0, 1, 0, out.data()), VE_INVALID_ARGUMENT);
    EXPECT_EQ(std::string(ve_last_error()), "the fit has no finite point");
    EXPECT_EQ(ve_fit_arch_lm(&f, 3, 2, out.data()), VE_INVALID_ARGUMENT);
    EXPECT_EQ(std::string(ve_last_error()), "the fit has no finite point");
    std::array<int32_t, 8> flags{};
    EXPECT_EQ(ve_fit_nyblom(&f, out.data(), &joint, flags.data()), VE_INVALID_ARGUMENT);
    EXPECT_EQ(std::string(ve_last_error()), "the fit has no finite point");
    EXPECT_EQ(ve_fit_sign_bias(&f, out.data()), VE_INVALID_ARGUMENT);
    EXPECT_EQ(std::string(ve_last_error()), "the fit has no finite point");
    EXPECT_EQ(ve_fit_gof(&f, 20, out.data()), VE_INVALID_ARGUMENT);
    EXPECT_EQ(std::string(ve_last_error()), "the fit has no finite point");
    EXPECT_EQ(joint, 123.0);
    EXPECT_EQ(out, (std::array<double, 8>{}));
}

TEST(CapiDiagnostics, HugeCountsAreRejectedBeforeComputation)
{
    auto fits = fit_series(ve::testing::simulate_garch_series(1, 40));
    auto* f = fits[0];
    const auto huge = std::numeric_limits<std::size_t>::max();
    std::array<double, 2> out{123.0, 456.0};
    EXPECT_EQ(ve_fit_ljung_box(f, 0, huge, 0, out.data()), VE_INVALID_ARGUMENT);
    EXPECT_NE(std::string(ve_last_error()).find("lag"), std::string::npos);
    EXPECT_EQ(ve_fit_ljung_box(f, 0, 5, huge, out.data()), VE_INVALID_ARGUMENT);
    EXPECT_NE(std::string(ve_last_error()).find("fitdf"), std::string::npos);
    EXPECT_EQ(ve_fit_arch_lm(f, huge, 2, out.data()), VE_INVALID_ARGUMENT);
    EXPECT_NE(std::string(ve_last_error()).find("lag"), std::string::npos);
    EXPECT_EQ(ve_fit_arch_lm(f, 5, huge, out.data()), VE_INVALID_ARGUMENT);
    EXPECT_NE(std::string(ve_last_error()).find("fitdf"), std::string::npos);
    EXPECT_EQ(ve_fit_gof(f, huge, out.data()), VE_INVALID_ARGUMENT);
    EXPECT_NE(std::string(ve_last_error()).find("groups"), std::string::npos);
    EXPECT_EQ(out, (std::array<double, 2>{123.0, 456.0}));
    EXPECT_EQ(ve_fit_ljung_box(f, 1, 5, huge, out.data()), VE_OK);
    std::array<double, 2> expected{};
    ASSERT_EQ(ve_fit_ljung_box(f, 1, 5, 0, expected.data()), VE_OK);
    EXPECT_EQ(out, expected);
}

TEST(CapiDiagnostics, NyblomExcludesDerivedAndAllFixedParameters)
{
    const auto series = ve::testing::simulate_garch_series(1, 800);
    ve::Spec targeted;
    targeted.variance_targeting = true;
    ve_fit f;
    f.result = ve::fit_one(series, targeted, ve::FitConfig{});
    std::vector<double> individual(f.result.names.size());
    double joint = 0.0;
    std::vector<int32_t> estimated(individual.size());
    ASSERT_EQ(ve_fit_nyblom(&f, individual.data(), &joint, estimated.data()), VE_OK);
    const auto layout = ve::model::layout_of(targeted);
    EXPECT_TRUE(std::isnan(individual[layout.omega()]));
    EXPECT_EQ(f.result.nyblom.estimated, layout.count() - 1);
    EXPECT_EQ(estimated[layout.omega()], 0);
    EXPECT_EQ(std::count(estimated.begin(), estimated.end(), 1), static_cast<long>(layout.count() - 1));
    ve::Spec integrated{.model = ve::Model::IGarch};
    ve_fit igarch;
    igarch.result = ve::fit_one(series, integrated, ve::FitConfig{});
    std::vector<double> igarch_individual(igarch.result.names.size());
    std::vector<int32_t> igarch_estimated(igarch_individual.size());
    ASSERT_EQ(ve_fit_nyblom(&igarch, igarch_individual.data(), &joint, igarch_estimated.data()), VE_OK);
    EXPECT_EQ(igarch_estimated.back(), 0);
    ve::Spec fixed;
    fixed.fixed = {{"mu", 0.05}, {"omega", 0.01}, {"alpha1", 0.08}, {"beta1", 0.9}};
    f.result = ve::fit_one(series, fixed, ve::FitConfig{});
    ASSERT_EQ(ve_fit_nyblom(&f, individual.data(), &joint, estimated.data()), VE_OK);
    EXPECT_EQ(f.result.nyblom.estimated, 0);
    EXPECT_EQ(estimated, std::vector<int32_t>(estimated.size(), 0));
    EXPECT_TRUE(std::isnan(joint));
    for (double value : individual)
    {
        EXPECT_TRUE(std::isnan(value));
    }
    std::array<double, 3> critical{};
    ASSERT_EQ(ve_nyblom_critical(20, critical.data()), VE_OK);
    EXPECT_EQ(critical, (std::array<double, 3>{4.22, 4.52, 5.13}));
}
