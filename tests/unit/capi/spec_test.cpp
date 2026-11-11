#include "capi/handles.hpp"

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

TEST(CapiSpec, CreateNullOutIsInvalid)
{
    EXPECT_EQ(ve_spec_create(VE_MODEL_SGARCH, nullptr), VE_INVALID_ARGUMENT);
    EXPECT_TRUE(error_mentions("out"));
}

TEST(CapiSpec, OrderOtherThanOneOneIsUnsupported)
{
    ve_spec* spec = nullptr;
    ASSERT_EQ(ve_spec_create(VE_MODEL_SGARCH, &spec), VE_OK);
    EXPECT_EQ(ve_spec_set_order(spec, 2, 1), VE_UNSUPPORTED);
    EXPECT_TRUE(error_mentions("(1, 1)"));
    EXPECT_EQ(ve_spec_set_order(spec, 1, 3), VE_UNSUPPORTED);
    EXPECT_EQ(ve_spec_set_order(spec, 0, 1), VE_INVALID_ARGUMENT);
    EXPECT_EQ(ve_spec_set_order(spec, 1, -1), VE_INVALID_ARGUMENT);
    EXPECT_EQ(spec->spec.p, 1);
    EXPECT_EQ(spec->spec.q, 1);
    ve_spec_free(spec);
}

TEST(CapiSpec, DistAndIncludeMeanValidation)
{
    ve_spec* spec = nullptr;
    ASSERT_EQ(ve_spec_create(VE_MODEL_SGARCH, &spec), VE_OK);
    EXPECT_EQ(ve_spec_set_dist(spec, 5), VE_INVALID_ARGUMENT);
    EXPECT_TRUE(error_mentions("dist"));
    EXPECT_EQ(ve_spec_set_include_mean(spec, 2), VE_INVALID_ARGUMENT);
    EXPECT_EQ(ve_spec_set_include_mean(spec, -1), VE_INVALID_ARGUMENT);
    EXPECT_TRUE(spec->spec.include_mean);
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
