#include "model/layout.hpp"

#include <array>
#include <cassert>
#include <string>

namespace ve::model
{

namespace
{

enum class Block
{
    None,
    PerArch,
    One
};

struct Blocks
{
    Block gamma;
    Block eta1;
    Block eta2;
    bool free_power;
    double fixed_power;
};

constexpr Block N = Block::None;
constexpr Block P = Block::PerArch;
constexpr Block O = Block::One;

// Indexed by Model.
constexpr std::array<Blocks, 14> kBlocks{{
    {N, N, N, false, 2.0}, // SGarch
    {P, N, N, false, 2.0}, // EGarch
    {P, N, N, false, 2.0}, // GjrGarch
    {P, N, N, true, 2.0},  // ApArch
    {N, N, N, false, 2.0}, // IGarch
    {N, O, O, false, 2.0}, // CsGarch
    {N, N, N, false, 2.0}, // FGarchGarch
    {N, P, N, false, 1.0}, // FGarchTGarch
    {N, P, P, false, 1.0}, // FGarchAvGarch
    {N, N, N, true, 2.0},  // FGarchNGarch
    {N, N, P, false, 2.0}, // FGarchNaGarch
    {N, P, N, true, 2.0},  // FGarchApArch
    {N, P, N, false, 2.0}, // FGarchGjrGarch
    {N, P, P, true, 2.0},  // FGarchAllGarch
}};

int block_size(Block block, int arch)
{
    if (block == Block::PerArch)
    {
        return arch;
    }
    return block == Block::One ? 1 : 0;
}

void append_block(std::vector<std::string>& names, const char* prefix, int count)
{
    for (int i = 1; i <= count; ++i)
    {
        names.push_back(prefix + std::to_string(i));
    }
}

}

std::size_t Layout::ar_begin() const
{
    return include_mean ? 1 : 0;
}

std::size_t Layout::ma_begin() const
{
    return ar_begin() + static_cast<std::size_t>(ar);
}

std::size_t Layout::omega() const
{
    return ma_begin() + static_cast<std::size_t>(ma);
}

std::size_t Layout::alpha_begin() const
{
    return omega() + 1;
}

std::size_t Layout::beta_begin() const
{
    return alpha_begin() + static_cast<std::size_t>(arch);
}

bool Layout::has_shape() const
{
    return dist != Dist::Norm;
}

std::size_t Layout::gamma_begin() const
{
    return beta_begin() + static_cast<std::size_t>(garch);
}

std::size_t Layout::eta1_begin() const
{
    return gamma_begin() + static_cast<std::size_t>(gamma);
}

std::size_t Layout::eta2_begin() const
{
    return eta1_begin() + static_cast<std::size_t>(eta1);
}

std::size_t Layout::power() const
{
    return eta2_begin() + static_cast<std::size_t>(eta2);
}

std::size_t Layout::shape() const
{
    return power() + (free_power ? 1 : 0);
}

std::size_t Layout::count() const
{
    return shape() + (has_shape() ? 1 : 0);
}

bool Layout::has_derived() const
{
    return model == Model::IGarch;
}

std::size_t Layout::derived() const
{
    return beta_begin() + static_cast<std::size_t>(garch) - 1;
}

std::size_t Layout::estimated_count() const
{
    return count() - (has_derived() ? 1 : 0);
}

bool Layout::is_power_family() const
{
    switch (model)
    {
        case Model::SGarch:
        case Model::EGarch:
        case Model::GjrGarch:
        case Model::IGarch:
        case Model::CsGarch:
            return false;
        case Model::ApArch:
        case Model::FGarchGarch:
        case Model::FGarchTGarch:
        case Model::FGarchAvGarch:
        case Model::FGarchNGarch:
        case Model::FGarchNaGarch:
        case Model::FGarchApArch:
        case Model::FGarchGjrGarch:
        case Model::FGarchAllGarch:
            return true;
    }
    return false;
}

std::size_t Layout::asymmetry_begin() const
{
    if (model == Model::ApArch)
    {
        return gamma_begin();
    }
    return eta1_begin();
}

std::size_t Layout::asymmetry_count() const
{
    if (model == Model::ApArch)
    {
        return static_cast<std::size_t>(gamma);
    }
    if (model == Model::CsGarch)
    {
        return 0;
    }
    return static_cast<std::size_t>(eta1);
}

ParameterKind Layout::kind(std::size_t index) const
{
    bool egarch = model == Model::EGarch;
    bool cs = model == Model::CsGarch;
    if (include_mean && index == mu())
    {
        return ParameterKind::Location;
    }
    if (index < omega())
    {
        return ParameterKind::Real;
    }
    if (index == omega())
    {
        return egarch ? ParameterKind::Real : ParameterKind::Positive;
    }
    if (index < beta_begin())
    {
        return egarch ? ParameterKind::Real : ParameterKind::Weight;
    }
    if (index < gamma_begin())
    {
        return ParameterKind::Weight;
    }
    if (index < eta1_begin())
    {
        return ParameterKind::Real;
    }
    if (index < eta2_begin())
    {
        return cs ? ParameterKind::Positive : ParameterKind::Real;
    }
    if (index < power())
    {
        return cs ? ParameterKind::Weight : ParameterKind::Real;
    }
    return ParameterKind::Positive;
}

Layout layout_of(const Spec& spec)
{
    auto model_index = static_cast<std::size_t>(spec.model);
    assert(model_index < kBlocks.size());
    const Blocks& blocks = kBlocks[model_index];
    return Layout{.model = spec.model,
                  .include_mean = spec.include_mean,
                  .ar = spec.ar,
                  .ma = spec.ma,
                  .arch = spec.p,
                  .garch = spec.q,
                  .gamma = block_size(blocks.gamma, spec.p),
                  .eta1 = block_size(blocks.eta1, spec.p),
                  .eta2 = block_size(blocks.eta2, spec.p),
                  .free_power = blocks.free_power,
                  .fixed_power = blocks.fixed_power,
                  .dist = spec.dist};
}

std::vector<std::string> parameter_names(const Layout& layout)
{
    std::vector<std::string> names;
    names.reserve(layout.count());
    if (layout.include_mean)
    {
        names.emplace_back("mu");
    }
    append_block(names, "ar", layout.ar);
    append_block(names, "ma", layout.ma);
    names.emplace_back("omega");
    append_block(names, "alpha", layout.arch);
    append_block(names, "beta", layout.garch);
    append_block(names, "gamma", layout.gamma);
    append_block(names, "eta1", layout.eta1);
    append_block(names, "eta2", layout.eta2);
    if (layout.free_power)
    {
        names.emplace_back(layout.model == Model::ApArch ? "delta" : "lambda");
    }
    if (layout.has_shape())
    {
        names.emplace_back("shape");
    }
    return names;
}

}
