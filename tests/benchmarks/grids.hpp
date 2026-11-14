#pragma once

#include "core/spec.hpp"

#include <array>
#include <cstddef>
#include <span>
#include <string>
#include <vector>

namespace ve::bench
{

// p is the ARCH order and q the GARCH order: s21 is sGARCH(2, 1).
inline ve::Spec make_spec(int p, int q, int ar, int ma, ve::Dist dist)
{
    ve::Spec spec;
    spec.p = p;
    spec.q = q;
    spec.ar = ar;
    spec.ma = ma;
    spec.dist = dist;
    return spec;
}

constexpr std::array<ve::Dist, 3> kSymmetricDists{ve::Dist::Norm, ve::Dist::Std, ve::Dist::Ged};
constexpr std::array<ve::Dist, 8> kAllDists{ve::Dist::Norm, ve::Dist::Std,  ve::Dist::Ged, ve::Dist::Sstd,
                                            ve::Dist::Sged, ve::Dist::Nig, ve::Dist::Ghyp, ve::Dist::Jsu};

constexpr std::size_t kModelCount = 14;
constexpr std::array<const char*, kModelCount> kModelNames{
    "sgarch",       "egarch",        "gjrgarch",      "aparch",         "igarch",
    "csgarch",      "fgarch_garch",  "fgarch_tgarch", "fgarch_avgarch", "fgarch_ngarch",
    "fgarch_nagarch", "fgarch_aparch", "fgarch_gjrgarch", "fgarch_allgarch"};
constexpr std::array<const char*, 8> kDistNames{"norm", "std", "ged", "sstd", "sged", "nig", "ghyp", "jsu"};

inline void append_orders(std::vector<ve::Spec>& grid, ve::Dist dist, int arma)
{
    for (int p = 1; p <= 3; ++p)
    {
        for (int q = 1; q <= 3; ++q)
        {
            grid.push_back(make_spec(p, q, arma, arma, dist));
        }
    }
}

// {p, q in 1..3} x {ARMA(0,0), ARMA(1,1)} x dists, dist-major.
inline std::vector<ve::Spec> tcc_grid(std::span<const ve::Dist> dists = kSymmetricDists)
{
    std::vector<ve::Spec> grid;
    for (const ve::Dist dist : dists)
    {
        for (const int arma : {0, 1})
        {
            append_orders(grid, dist, arma);
        }
    }
    return grid;
}

// "sgarch_p1q1_arma00_norm": model, orders, ARMA orders and distribution.
inline std::string spec_label(const ve::Spec& spec)
{
    return std::string(kModelNames[static_cast<std::size_t>(spec.model)]) + "_p" + std::to_string(spec.p) + "q"
        + std::to_string(spec.q) + "_arma" + std::to_string(spec.ar) + std::to_string(spec.ma) + "_"
        + kDistNames[static_cast<std::size_t>(spec.dist)];
}

}
