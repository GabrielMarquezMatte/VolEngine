#include <gtest/gtest.h>
#include <volengine/volengine.h>

TEST(Abi, RuntimeVersionMatchesHeader)
{
    EXPECT_EQ(ve_abi_version(), VE_ABI_VERSION);
}
