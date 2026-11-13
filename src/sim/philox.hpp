#pragma once

#include <array>
#include <cstdint>

namespace ve::sim
{

// Philox4x32-10 of Salmon, Moraes, Dror and Shaw (2011), "Parallel random numbers: as easy as 1, 2, 3", as in
// Random123.
inline std::array<std::uint32_t, 4> philox4x32(std::array<std::uint32_t, 4> counter, std::array<std::uint32_t, 2> key)
{
    constexpr std::uint64_t kMultiplier0 = 0xD2511F53;
    constexpr std::uint64_t kMultiplier1 = 0xCD9E8D57;
    constexpr std::uint32_t kWeyl0 = 0x9E3779B9;
    constexpr std::uint32_t kWeyl1 = 0xBB67AE85;
    for (int round = 0; round < 10; ++round)
    {
        std::uint64_t product0 = kMultiplier0 * counter[0];
        std::uint64_t product1 = kMultiplier1 * counter[2];
        counter = {static_cast<std::uint32_t>(product1 >> 32) ^ counter[1] ^ key[0],
                   static_cast<std::uint32_t>(product1),
                   static_cast<std::uint32_t>(product0 >> 32) ^ counter[3] ^ key[1],
                   static_cast<std::uint32_t>(product0)};
        key = {key[0] + kWeyl0, key[1] + kWeyl1};
    }
    return counter;
}

// The stream of one path: Philox under the key seed, at the counter (block, path), two 64-bit words per block.
// Paths never share a counter, so each stream is the same however the paths are split among threads.
class Stream
{
public:
    Stream(std::uint64_t seed, std::uint64_t path)
        : key_{static_cast<std::uint32_t>(seed), static_cast<std::uint32_t>(seed >> 32)},
          path_{static_cast<std::uint32_t>(path), static_cast<std::uint32_t>(path >> 32)}
    {
    }

    std::uint64_t next()
    {
        if (used_ == 2)
        {
            refill();
        }
        std::uint64_t word = words_[used_];
        ++used_;
        return word;
    }

    // ((u >> 11) + 1/2) 2^-53, in the open interval (0, 1).
    double uniform()
    {
        constexpr double kUnit = 0x1.0p-53;
        return (static_cast<double>(next() >> 11) + 0.5) * kUnit;
    }

private:
    void refill()
    {
        std::array<std::uint32_t, 4> block = philox4x32(
            {static_cast<std::uint32_t>(block_), static_cast<std::uint32_t>(block_ >> 32), path_[0], path_[1]}, key_);
        ++block_;
        words_ = {block[0] | (std::uint64_t{block[1]} << 32), block[2] | (std::uint64_t{block[3]} << 32)};
        used_ = 0;
    }

    std::array<std::uint32_t, 2> key_;
    std::array<std::uint32_t, 2> path_;
    std::uint64_t block_ = 0;
    std::array<std::uint64_t, 2> words_{};
    int used_ = 2;
};

}
