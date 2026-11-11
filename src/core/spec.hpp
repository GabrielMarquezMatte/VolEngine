#pragma once

#include "garch/sgarch.hpp"

#include <cstdint>

namespace ve
{

enum class Model : int32_t
{
    SGarch = 0
};

enum class Dist : int32_t
{
    Norm = 0
};

struct Spec
{
    Model model = Model::SGarch;
    int p = 1;
    int q = 1;
    Dist dist = Dist::Norm;
    bool include_mean = true;
};

struct FitConfig
{
    int max_iter = 500;
    double tol = 1e-6;
    garch::VarInit var_init = garch::VarInit::Sample;
};

}
