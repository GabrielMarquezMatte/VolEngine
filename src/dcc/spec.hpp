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
    FDcc = 2,
    // A constant correlation; only with a copula.
    Static = 3
};

enum class DccDist : int32_t
{
    MvNorm = 0,
    MvT = 1,
    MvLaplace = 2
};

// How the margins become uniforms for a copula; None is the DCC of the standardized residuals themselves.
enum class DccCopula : int32_t
{
    None = 0,
    Parametric = 1,
    Empirical = 2
};

// The estimator of a static copula's correlation; Default is Kendall on the static model and unused elsewhere.
enum class StaticMethod : int32_t
{
    Default = 0,
    Kendall = 1,
    Ml = 2
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
    DccCopula copula = DccCopula::None;
    StaticMethod static_method = StaticMethod::Default;
    // The C API never sets it.
    Convention convention = Convention::Standard;
};

}
