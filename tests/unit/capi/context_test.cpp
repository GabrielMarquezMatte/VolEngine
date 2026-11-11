#include <algorithm>
#include <cstdint>
#include <gtest/gtest.h>
#include <string>
#include <thread>
#include <volengine/volengine.h>

namespace
{

int32_t hardware_threads()
{
    return static_cast<int32_t>(std::max(1u, std::thread::hardware_concurrency()));
}

int32_t threads_of(const ve_context* context)
{
    int32_t threads = -1;
    EXPECT_EQ(ve_context_threads(context, &threads), VE_OK);
    return threads;
}

}

TEST(Context, CreateDefaultsToHardwareThreads)
{
    ve_context* context = nullptr;
    ASSERT_EQ(ve_context_create(&context), VE_OK);
    ASSERT_NE(context, nullptr);
    EXPECT_EQ(threads_of(context), hardware_threads());
    ve_context_free(context);
}

TEST(Context, SetThreadsStoresCountAndZeroMeansHardware)
{
    ve_context* context = nullptr;
    ASSERT_EQ(ve_context_create(&context), VE_OK);
    ASSERT_EQ(ve_context_set_threads(context, 3), VE_OK);
    EXPECT_EQ(threads_of(context), 3);
    ASSERT_EQ(ve_context_set_threads(context, 0), VE_OK);
    EXPECT_EQ(threads_of(context), hardware_threads());
    ve_context_free(context);
}

TEST(Context, NegativeThreadsIsInvalidArgument)
{
    ve_context* context = nullptr;
    ASSERT_EQ(ve_context_create(&context), VE_OK);
    EXPECT_EQ(ve_context_set_threads(context, -1), VE_INVALID_ARGUMENT);
    EXPECT_NE(std::string(ve_last_error()).find("threads"), std::string::npos);
    ve_context_free(context);
}

TEST(Context, NullOutIsInvalidArgument)
{
    EXPECT_EQ(ve_context_create(nullptr), VE_INVALID_ARGUMENT);
    EXPECT_NE(std::string(ve_last_error()).find("out"), std::string::npos);
}

TEST(Context, NullHandleIsInvalidHandle)
{
    int32_t threads = 0;
    EXPECT_EQ(ve_context_threads(nullptr, &threads), VE_INVALID_HANDLE);
    EXPECT_EQ(ve_context_set_threads(nullptr, 1), VE_INVALID_HANDLE);
}

TEST(Context, NullThreadsOutIsInvalidArgument)
{
    ve_context* context = nullptr;
    ASSERT_EQ(ve_context_create(&context), VE_OK);
    EXPECT_EQ(ve_context_threads(context, nullptr), VE_INVALID_ARGUMENT);
    ve_context_free(context);
}

TEST(Context, SuccessClearsLastError)
{
    ve_context* context = nullptr;
    ASSERT_EQ(ve_context_create(&context), VE_OK);
    ASSERT_EQ(ve_context_set_threads(context, -1), VE_INVALID_ARGUMENT);
    ASSERT_STRNE(ve_last_error(), "");
    ASSERT_EQ(ve_context_set_threads(context, 1), VE_OK);
    EXPECT_STREQ(ve_last_error(), "");
    ve_context_free(context);
}

TEST(Context, LastErrorIsPerThread)
{
    ASSERT_EQ(ve_context_create(nullptr), VE_INVALID_ARGUMENT);
    std::string other_thread_error = "unset";
    std::thread reader([&] { other_thread_error = ve_last_error(); });
    reader.join();
    EXPECT_EQ(other_thread_error, "");
    EXPECT_STRNE(ve_last_error(), "");
}

TEST(Context, FreeNullIsNoop)
{
    ve_context_free(nullptr);
    SUCCEED();
}
