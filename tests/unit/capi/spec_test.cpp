#include "capi/handles.hpp"
#include "capi_scope.hpp"
#include "simulated_garch.hpp"

#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <string>
#include <volengine/volengine.h>

namespace
{

bool error_mentions(const std::string& text)
{
    return std::string(ve_last_error()).find(text) != std::string::npos;
}

constexpr double kFixedAlpha = 0.08;
constexpr std::size_t kFitLength = 600;
constexpr std::size_t kAlphaIndex = 2;

struct AlphaFit
{
    ve::testing::Snapshot snapshot;
    double alpha_error = 0.0;
};

// sGARCH(1,1) with alpha1 fixed at kFixedAlpha and given the start value 0.4, in either setter order.
AlphaFit fit_with_fixed_and_start_alpha(bool fixed_first)
{
    ve::testing::SpecPtr spec = ve::testing::make_spec(1);
    if (fixed_first)
    {
        EXPECT_EQ(ve_spec_set_fixed(spec.get(), "alpha1", kFixedAlpha), VE_OK);
    }
    EXPECT_EQ(ve_spec_set_start(spec.get(), "alpha1", 0.4), VE_OK);
    if (!fixed_first)
    {
        EXPECT_EQ(ve_spec_set_fixed(spec.get(), "alpha1", kFixedAlpha), VE_OK);
    }
    EXPECT_EQ(ve_spec_validate(spec.get()), VE_OK);
    const std::vector<double> data = ve::testing::simulate_garch_matrix(kFitLength, 1);
    ve::testing::PanelPtr panel = ve::testing::make_panel(data.data(), kFitLength, 1);
    ve::testing::BatchInputs inputs;
    inputs.specs = {spec.get()};
    inputs.panel = panel.get();
    inputs.job_spec = {0};
    inputs.job_series = {0};
    ve::testing::BatchPtr batch = ve::testing::build_batch(inputs);
    ve::testing::ContextPtr context = ve::testing::make_context(1);
    ve::testing::FitSet fits = ve::testing::fit_all(context.get(), batch.get());
    AlphaFit result;
    result.snapshot = ve::testing::read_snapshot(fits[0]);
    std::vector<double> errors(result.snapshot.estimate.size());
    EXPECT_EQ(ve_fit_std_errors(fits[0], VE_SE_HESSIAN, errors.data()), VE_OK);
    result.alpha_error = errors.at(kAlphaIndex);
    return result;
}

}

TEST(CapiSpec, CreateAndConfigure)
{
    ve_spec* spec = nullptr;
    ASSERT_EQ(ve_spec_create(VE_MODEL_SGARCH, &spec), VE_OK);
    ASSERT_NE(spec, nullptr);
    EXPECT_EQ(ve_spec_set_order(spec, 1, 1), VE_OK);
    EXPECT_EQ(ve_spec_set_dist(spec, VE_DIST_NORM), VE_OK);
    EXPECT_EQ(ve_spec_set_include_mean(spec, 0), VE_OK);
    EXPECT_FALSE(spec->spec.include_mean);
    EXPECT_EQ(ve_spec_set_include_mean(spec, 1), VE_OK);
    EXPECT_TRUE(spec->spec.include_mean);
    ve_spec_free(spec);
}

TEST(CapiSpec, UnknownModelIsInvalid)
{
    ve_spec* spec = reinterpret_cast<ve_spec*>(0x1);
    EXPECT_EQ(ve_spec_create(99, &spec), VE_INVALID_ARGUMENT);
    EXPECT_EQ(spec, nullptr);
    EXPECT_TRUE(error_mentions("model"));
    EXPECT_TRUE(error_mentions("99"));
}

TEST(CapiSpec, AllModels)
{
    for (int32_t model = 0; model <= VE_MODEL_FGARCH_ALLGARCH; ++model)
    {
        ve_spec* spec = nullptr;
        ASSERT_EQ(ve_spec_create(model, &spec), VE_OK) << "model " << model;
        ASSERT_NE(spec, nullptr);
        EXPECT_EQ(static_cast<int32_t>(spec->spec.model), model);
        ve_spec_free(spec);
    }
    for (const int32_t model : {VE_MODEL_FGARCH_ALLGARCH + 1, -1})
    {
        ve_spec* spec = reinterpret_cast<ve_spec*>(0x1);
        EXPECT_EQ(ve_spec_create(model, &spec), VE_INVALID_ARGUMENT);
        EXPECT_EQ(spec, nullptr);
        EXPECT_TRUE(error_mentions("model"));
    }
}

TEST(CapiSpec, CreateNullOutIsInvalid)
{
    EXPECT_EQ(ve_spec_create(VE_MODEL_SGARCH, nullptr), VE_INVALID_ARGUMENT);
    EXPECT_TRUE(error_mentions("out"));
}

TEST(CapiSpec, OrdersUpToMax)
{
    ve_spec* spec = nullptr;
    ASSERT_EQ(ve_spec_create(VE_MODEL_SGARCH, &spec), VE_OK);
    EXPECT_EQ(ve_spec_set_order(spec, VE_MAX_ORDER, VE_MAX_ORDER), VE_OK);
    EXPECT_EQ(spec->spec.p, VE_MAX_ORDER);
    EXPECT_EQ(spec->spec.q, VE_MAX_ORDER);
    EXPECT_EQ(ve_spec_set_order(spec, 2, 1), VE_OK);
    EXPECT_EQ(ve_spec_set_order(spec, VE_MAX_ORDER + 1, 1), VE_INVALID_ARGUMENT);
    EXPECT_TRUE(error_mentions("VE_MAX_ORDER"));
    EXPECT_EQ(ve_spec_set_order(spec, 1, VE_MAX_ORDER + 1), VE_INVALID_ARGUMENT);
    EXPECT_EQ(ve_spec_set_order(spec, 0, 1), VE_INVALID_ARGUMENT);
    EXPECT_EQ(ve_spec_set_order(spec, 1, -1), VE_INVALID_ARGUMENT);
    EXPECT_EQ(spec->spec.p, 2);
    EXPECT_EQ(spec->spec.q, 1);
    ve_spec_free(spec);
}

TEST(CapiSpec, Arma)
{
    ve_spec* spec = nullptr;
    ASSERT_EQ(ve_spec_create(VE_MODEL_SGARCH, &spec), VE_OK);
    EXPECT_EQ(spec->spec.ar, 0);
    EXPECT_EQ(spec->spec.ma, 0);
    EXPECT_EQ(ve_spec_set_arma(spec, 0, 0), VE_OK);
    EXPECT_EQ(ve_spec_set_arma(spec, VE_MAX_ORDER, VE_MAX_ORDER), VE_OK);
    EXPECT_EQ(ve_spec_set_arma(spec, 2, 1), VE_OK);
    EXPECT_EQ(ve_spec_set_arma(spec, -1, 0), VE_INVALID_ARGUMENT);
    EXPECT_TRUE(error_mentions("VE_MAX_ORDER"));
    EXPECT_EQ(ve_spec_set_arma(spec, 0, -1), VE_INVALID_ARGUMENT);
    EXPECT_EQ(ve_spec_set_arma(spec, VE_MAX_ORDER + 1, 0), VE_INVALID_ARGUMENT);
    EXPECT_EQ(ve_spec_set_arma(spec, 0, VE_MAX_ORDER + 1), VE_INVALID_ARGUMENT);
    EXPECT_EQ(spec->spec.ar, 2);
    EXPECT_EQ(spec->spec.ma, 1);
    EXPECT_EQ(ve_spec_set_arma(nullptr, 0, 0), VE_INVALID_HANDLE);
    ve_spec_free(spec);
}

TEST(CapiSpec, NewDistributions)
{
    ve_spec* spec = nullptr;
    ASSERT_EQ(ve_spec_create(VE_MODEL_SGARCH, &spec), VE_OK);
    EXPECT_EQ(ve_spec_set_dist(spec, VE_DIST_STD), VE_OK);
    EXPECT_EQ(spec->spec.dist, ve::Dist::Std);
    EXPECT_EQ(ve_spec_set_dist(spec, VE_DIST_GED), VE_OK);
    EXPECT_EQ(spec->spec.dist, ve::Dist::Ged);
    EXPECT_EQ(ve_spec_set_dist(spec, 8), VE_INVALID_ARGUMENT);
    EXPECT_EQ(spec->spec.dist, ve::Dist::Ged);
    ve_spec_free(spec);
}

TEST(CapiSpec, AllDistributions)
{
    ve_spec* spec = nullptr;
    ASSERT_EQ(ve_spec_create(VE_MODEL_SGARCH, &spec), VE_OK);
    for (int32_t dist = 0; dist <= 7; ++dist)
    {
        EXPECT_EQ(ve_spec_set_dist(spec, dist), VE_OK) << dist;
        EXPECT_EQ(static_cast<int32_t>(spec->spec.dist), dist);
    }
    for (const int32_t bad : {8, -1})
    {
        EXPECT_EQ(ve_spec_set_dist(spec, bad), VE_INVALID_ARGUMENT) << bad;
        EXPECT_TRUE(error_mentions("dist"));
        EXPECT_EQ(spec->spec.dist, ve::Dist::Jsu);
    }
    ve_spec_free(spec);
}

TEST(CapiSpec, DistAndIncludeMeanValidation)
{
    ve_spec* spec = nullptr;
    ASSERT_EQ(ve_spec_create(VE_MODEL_SGARCH, &spec), VE_OK);
    EXPECT_EQ(ve_spec_set_dist(spec, 9), VE_INVALID_ARGUMENT);
    EXPECT_TRUE(error_mentions("dist"));
    EXPECT_EQ(ve_spec_set_include_mean(spec, 2), VE_INVALID_ARGUMENT);
    EXPECT_EQ(ve_spec_set_include_mean(spec, -1), VE_INVALID_ARGUMENT);
    EXPECT_TRUE(spec->spec.include_mean);
    ve_spec_free(spec);
}

TEST(CapiSpec, FixedAndStartSetters)
{
    ve_spec* spec = nullptr;
    ASSERT_EQ(ve_spec_create(VE_MODEL_SGARCH, &spec), VE_OK);
    for (const char* name : {static_cast<const char*>(nullptr), ""})
    {
        EXPECT_EQ(ve_spec_set_fixed(spec, name, 1.0), VE_INVALID_ARGUMENT);
        EXPECT_EQ(ve_spec_set_start(spec, name, 1.0), VE_INVALID_ARGUMENT);
        EXPECT_TRUE(error_mentions("name"));
    }
    for (const double bad : {std::nan(""), std::numeric_limits<double>::infinity()})
    {
        EXPECT_EQ(ve_spec_set_fixed(spec, "mu", bad), VE_INVALID_ARGUMENT);
        EXPECT_EQ(ve_spec_set_start(spec, "mu", bad), VE_INVALID_ARGUMENT);
        EXPECT_TRUE(error_mentions("value"));
    }
    EXPECT_TRUE(spec->spec.fixed.empty());
    EXPECT_TRUE(spec->spec.start.empty());

    ASSERT_EQ(ve_spec_set_start(spec, "alpha1", 0.1), VE_OK);
    ASSERT_EQ(ve_spec_set_fixed(spec, "alpha1", 0.2), VE_OK);
    EXPECT_TRUE(spec->spec.start.empty());
    ASSERT_EQ(spec->spec.fixed.size(), 1u);
    EXPECT_EQ(spec->spec.fixed[0].name, "alpha1");
    EXPECT_EQ(spec->spec.fixed[0].value, 0.2);

    ASSERT_EQ(ve_spec_set_fixed(spec, "alpha1", 0.3), VE_OK);
    ASSERT_EQ(spec->spec.fixed.size(), 1u);
    EXPECT_EQ(spec->spec.fixed[0].value, 0.3);

    ASSERT_EQ(ve_spec_set_start(spec, "alpha1", 0.4), VE_OK);
    ASSERT_EQ(spec->spec.fixed.size(), 1u);
    EXPECT_EQ(spec->spec.fixed[0].value, 0.3);
    ASSERT_EQ(spec->spec.start.size(), 1u);
    EXPECT_EQ(spec->spec.start[0].value, 0.4);

    EXPECT_EQ(ve_spec_set_fixed(nullptr, "mu", 1.0), VE_INVALID_HANDLE);
    EXPECT_EQ(ve_spec_set_start(nullptr, "mu", 1.0), VE_INVALID_HANDLE);
    EXPECT_EQ(ve_spec_set_variance_targeting(nullptr, 1), VE_INVALID_HANDLE);
    EXPECT_EQ(ve_spec_validate(nullptr), VE_INVALID_HANDLE);

    EXPECT_EQ(ve_spec_set_variance_targeting(spec, 1), VE_OK);
    EXPECT_TRUE(spec->spec.variance_targeting);
    EXPECT_EQ(ve_spec_set_variance_targeting(spec, 2), VE_INVALID_ARGUMENT);
    EXPECT_TRUE(error_mentions("variance_targeting"));
    EXPECT_EQ(ve_spec_set_variance_targeting(spec, -1), VE_INVALID_ARGUMENT);
    EXPECT_TRUE(spec->spec.variance_targeting);
    EXPECT_EQ(ve_spec_set_variance_targeting(spec, 0), VE_OK);
    EXPECT_FALSE(spec->spec.variance_targeting);
    ve_spec_free(spec);
}

TEST(CapiSpec, StartValueNeverUnfixesInEitherOrder)
{
    const AlphaFit fixed_first = fit_with_fixed_and_start_alpha(true);
    const AlphaFit start_first = fit_with_fixed_and_start_alpha(false);
    for (const AlphaFit* fit : {&fixed_first, &start_first})
    {
        EXPECT_EQ(fit->snapshot.status, VE_FIT_CONVERGED);
        EXPECT_EQ(fit->snapshot.estimate.at(kAlphaIndex), kFixedAlpha);
        EXPECT_TRUE(std::isnan(fit->alpha_error));
    }
    EXPECT_EQ(fixed_first.snapshot, start_first.snapshot);
}

TEST(CapiSpec, ValidateReportsSpecProblems)
{
    ve_spec* spec = nullptr;
    ASSERT_EQ(ve_spec_create(VE_MODEL_SGARCH, &spec), VE_OK);
    EXPECT_EQ(ve_spec_validate(spec), VE_OK);
    ASSERT_EQ(ve_spec_set_dist(spec, VE_DIST_STD), VE_OK);
    ASSERT_EQ(ve_spec_set_fixed(spec, "shape", 5.0), VE_OK);
    EXPECT_EQ(ve_spec_validate(spec), VE_OK);
    ASSERT_EQ(ve_spec_set_dist(spec, VE_DIST_NORM), VE_OK);
    EXPECT_EQ(ve_spec_validate(spec), VE_INVALID_ARGUMENT);
    EXPECT_TRUE(error_mentions("shape"));
    ve_spec_free(spec);

    ASSERT_EQ(ve_spec_create(VE_MODEL_IGARCH, &spec), VE_OK);
    ASSERT_EQ(ve_spec_set_variance_targeting(spec, 1), VE_OK);
    EXPECT_EQ(ve_spec_validate(spec), VE_INVALID_ARGUMENT);
    EXPECT_TRUE(error_mentions("variance targeting"));
    ve_spec_free(spec);

    ASSERT_EQ(ve_spec_create(VE_MODEL_SGARCH, &spec), VE_OK);
    ASSERT_EQ(ve_spec_set_variance_targeting(spec, 1), VE_OK);
    ASSERT_EQ(ve_spec_set_fixed(spec, "omega", 0.1), VE_OK);
    EXPECT_EQ(ve_spec_validate(spec), VE_INVALID_ARGUMENT);
    EXPECT_TRUE(error_mentions("derived"));
    EXPECT_TRUE(error_mentions("omega"));
    ve_spec_free(spec);
}

TEST(CapiOptions, DefaultsAndValidation)
{
    ve_fit_options* options = nullptr;
    ASSERT_EQ(ve_fit_options_create(&options), VE_OK);
    EXPECT_EQ(ve_fit_options_set_max_iter(options, 0), VE_INVALID_ARGUMENT);
    EXPECT_EQ(ve_fit_options_set_max_iter(options, -4), VE_INVALID_ARGUMENT);
    EXPECT_EQ(options->config.max_iter, 500);
    EXPECT_EQ(ve_fit_options_set_max_iter(options, 7), VE_OK);
    EXPECT_EQ(options->config.max_iter, 7);
    EXPECT_EQ(ve_fit_options_set_tolerance(options, -1.0), VE_INVALID_ARGUMENT);
    EXPECT_EQ(ve_fit_options_set_tolerance(options, 0.0), VE_INVALID_ARGUMENT);
    EXPECT_EQ(ve_fit_options_set_tolerance(options, std::nan("")), VE_INVALID_ARGUMENT);
    EXPECT_EQ(ve_fit_options_set_tolerance(options, std::numeric_limits<double>::infinity()),
        VE_INVALID_ARGUMENT);
    EXPECT_EQ(options->config.tol, 1e-6);
    EXPECT_EQ(ve_fit_options_set_tolerance(options, 1e-9), VE_OK);
    EXPECT_EQ(options->config.tol, 1e-9);
    EXPECT_EQ(ve_fit_options_set_var_init(options, 7), VE_INVALID_ARGUMENT);
    EXPECT_TRUE(error_mentions("var_init"));
    EXPECT_EQ(ve_fit_options_set_var_init(options, VE_VARINIT_BACKCAST), VE_OK);
    EXPECT_EQ(options->config.var_init, ve::VarInit::Backcast);
    ve_fit_options_free(options);
}

TEST(CapiOptions, CreateNullOutIsInvalid)
{
    EXPECT_EQ(ve_fit_options_create(nullptr), VE_INVALID_ARGUMENT);
}

TEST(CapiHandles, NullHandleIsInvalidHandle)
{
    EXPECT_EQ(ve_spec_set_dist(nullptr, 0), VE_INVALID_HANDLE);
    EXPECT_EQ(ve_spec_set_order(nullptr, 1, 1), VE_INVALID_HANDLE);
    EXPECT_EQ(ve_spec_set_include_mean(nullptr, 1), VE_INVALID_HANDLE);
    EXPECT_EQ(ve_fit_options_set_max_iter(nullptr, 1), VE_INVALID_HANDLE);
    EXPECT_EQ(ve_fit_options_set_tolerance(nullptr, 1.0), VE_INVALID_HANDLE);
    EXPECT_EQ(ve_fit_options_set_var_init(nullptr, 0), VE_INVALID_HANDLE);
    EXPECT_EQ(ve_panel_set_strides(nullptr, 1, 1), VE_INVALID_HANDLE);
    EXPECT_EQ(ve_panel_set_ranges(nullptr, nullptr, nullptr), VE_INVALID_HANDLE);
}

TEST(CapiHandles, FreeNullIsNoop)
{
    ve_spec_free(nullptr);
    ve_fit_options_free(nullptr);
    ve_panel_free(nullptr);
    SUCCEED();
}
