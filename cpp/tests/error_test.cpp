#include <gtest/gtest.h>
#include <string>
#include <volengine/volengine.hpp>

namespace
{

TEST(Error, CheckOkIsSuccess)
{
    EXPECT_TRUE(ve::detail::check(VE_OK).has_value());
}

TEST(Error, CheckErrorCarriesStatusAndMessage)
{
    ve_spec* spec = nullptr;
    const auto result = ve::detail::check(ve_spec_create(999, &spec));
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().status, ve::Status::invalid_argument);
    EXPECT_FALSE(result.error().message.empty());
}

TEST(Error, MessageSurvivesLaterCalls)
{
    ve_spec* spec = nullptr;
    const auto result = ve::detail::check(ve_spec_create(999, &spec));
    ASSERT_FALSE(result.has_value());
    const std::string before = result.error().message;
    EXPECT_EQ(ve_abi_version(), VE_ABI_VERSION);
    EXPECT_EQ(result.error().message, before);
}

TEST(Error, AbiVersionMismatchNamesBothVersions)
{
    const auto result = ve::detail::check_abi_version(7, 0);
    ASSERT_FALSE(result.has_value());
    EXPECT_NE(result.error().message.find('7'), std::string::npos);
    EXPECT_NE(result.error().message.find('0'), std::string::npos);
    EXPECT_TRUE(ve::detail::check_abi_version(0, 0).has_value());
}

TEST(Context, CreateAndThreads)
{
    auto context = ve::Context::create(2);
    ASSERT_TRUE(context.has_value());
    EXPECT_EQ(context->threads().value(), 2);
    EXPECT_TRUE(context->set_threads(0).has_value());
    const auto invalid = context->set_threads(-1);
    ASSERT_FALSE(invalid.has_value());
    EXPECT_EQ(invalid.error().status, ve::Status::invalid_argument);
}

} // namespace
