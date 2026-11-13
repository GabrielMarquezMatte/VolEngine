#include "sim/philox.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>

namespace
{

using Block = std::array<std::uint32_t, 4>;

// Random123 kat_vectors, philox4x32 with 10 rounds.
TEST(Philox, KnownAnswers)
{
    EXPECT_EQ(ve::sim::philox4x32({0, 0, 0, 0}, {0, 0}), (Block{0x6627e8d5, 0xe169c58d, 0xbc57ac4c, 0x9b00dbd8}));
    EXPECT_EQ(ve::sim::philox4x32({0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff}, {0xffffffff, 0xffffffff}),
              (Block{0x408f276d, 0x41c83b0e, 0xa20bc7c6, 0x6d5451fd}));
    EXPECT_EQ(ve::sim::philox4x32({0x243f6a88, 0x85a308d3, 0x13198a2e, 0x03707344}, {0xa4093822, 0x299f31d0}),
              (Block{0xd16cfe09, 0x94fdcceb, 0x5001e420, 0x24126ea1}));
}

TEST(Philox, StreamsArePerPath)
{
    ve::sim::Stream a(7, 3);
    ve::sim::Stream b(7, 3);
    for (int i = 0; i < 1000; ++i)
    {
        ASSERT_EQ(a.next(), b.next()) << "i = " << i;
    }
    EXPECT_NE(ve::sim::Stream(7, 3).next(), ve::sim::Stream(7, 4).next());
    EXPECT_NE(ve::sim::Stream(7, 3).next(), ve::sim::Stream(8, 3).next());
    constexpr std::uint64_t kLast = std::numeric_limits<std::uint64_t>::max();
    EXPECT_NE(ve::sim::Stream(0, 0).next(), ve::sim::Stream(kLast, 0).next());
    ve::sim::Stream stream(42, kLast);
    for (int i = 0; i < 1000000; ++i)
    {
        double u = stream.uniform();
        ASSERT_GT(u, 0.0);
        ASSERT_LT(u, 1.0);
    }
}

}
