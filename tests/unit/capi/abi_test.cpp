#include <gtest/gtest.h>
#include <volengine/volengine.h>

// 1.0 froze the interface: the version moves with the semver major (see docs/superpowers/notes/2026-11-15-abi-audit-6c.md).
TEST(Abi, ThisIsTheFirstStableInterface)
{
    EXPECT_EQ(VE_ABI_VERSION, 1);
}

TEST(Abi, RuntimeVersionMatchesHeader)
{
    EXPECT_EQ(ve_abi_version(), VE_ABI_VERSION);
}
