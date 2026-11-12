#pragma once

#include "core/var_init.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace ve
{

enum class Model : int32_t
{
    SGarch = 0,
    EGarch = 1,
    GjrGarch = 2,
    ApArch = 3,
    IGarch = 4,
    CsGarch = 5,
    FGarchGarch = 6,
    FGarchTGarch = 7,
    FGarchAvGarch = 8,
    FGarchNGarch = 9,
    FGarchNaGarch = 10,
    FGarchApArch = 11,
    FGarchGjrGarch = 12,
    FGarchAllGarch = 13
};

enum class Dist : int32_t
{
    Norm = 0,
    Std = 1,
    Ged = 2,
    Sstd = 3,
    Sged = 4,
    Nig = 5,
    Ghyp = 6,
    Jsu = 7
};

constexpr int kMaxOrder = 6;
constexpr int kMaxRegressors = 8;
// mu, ar, ma, archm (or arfima, never both), mxreg, omega, alpha, beta, gamma, eta1, eta2, power, vxreg, skew, shape and ghlambda of the
// widest layout.
constexpr std::size_t kMaxParameterCount = 6 * kMaxOrder + 6 + 1 + 2 * kMaxRegressors;

struct ParameterValue
{
    std::string name;
    double value;
};

struct Spec
{
    Model model = Model::SGarch;
    int p = 1;
    int q = 1;
    int ar = 0;
    int ma = 0;
    Dist dist = Dist::Norm;
    bool include_mean = true;
    std::vector<ParameterValue> fixed{};
    std::vector<ParameterValue> start{};
    bool variance_targeting = false;
    int mxreg = 0;
    int vxreg = 0;
    // 0 off, 1 sigma, 2 sigma^2.
    int archm = 0;
    // rugarch's arfima = TRUE; cannot be combined with archm.
    bool arfima = false;
};

struct FitConfig
{
    int max_iter = 500;
    double tol = 1e-6;
    VarInit var_init = VarInit::Sample;
};

}
