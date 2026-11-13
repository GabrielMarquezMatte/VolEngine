#pragma once

#include "core/spec.hpp"

#include <cstdint>
#include <vector>

namespace ve::dcc
{

enum class DccModel : int32_t
{
    Dcc = 0,
    ADcc = 1,
    FDcc = 2
};

enum class DccDist : int32_t
{
    MvNorm = 0,
    MvT = 1,
    MvLaplace = 2
};

// Standard follows the papers; Rmgarch reproduces rmgarch 1.4.3 (internal, tests only).
enum class Convention
{
    Standard,
    Rmgarch
};

struct DccSpec
{
    DccModel model = DccModel::Dcc;
    int p = 1;
    int q = 1;
    DccDist dist = DccDist::MvNorm;
    std::vector<int> groups{};
    std::vector<ParameterValue> fixed{};
    std::vector<ParameterValue> start{};
    // The C API never sets it.
    Convention convention = Convention::Standard;
};

}
