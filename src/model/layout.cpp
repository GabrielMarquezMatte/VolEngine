#include "model/layout.hpp"

#include <array>
#include <algorithm>
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

Layout base_layout(const Spec& spec)
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

std::optional<std::size_t> index_of(const std::vector<std::string>& names, const std::string& name)
{
    auto found = std::find(names.begin(), names.end(), name);
    if (found == names.end())
    {
        return std::nullopt;
    }
    return static_cast<std::size_t>(found - names.begin());
}

std::string joined(const std::vector<std::string>& names)
{
    std::string text;
    for (const std::string& name : names)
    {
        if (!text.empty())
        {
            text += ' ';
        }
        text += name;
    }
    return text;
}

bool is_fixed_by_name(const Spec& spec, const std::string& name)
{
    return std::any_of(spec.fixed.begin(), spec.fixed.end(),
                       [&](const ParameterValue& entry) { return entry.name == name; });
}

std::optional<std::string> unknown_name_problem(const Spec& spec, const std::vector<std::string>& names)
{
    for (const auto* list : {&spec.fixed, &spec.start})
    {
        for (const ParameterValue& entry : *list)
        {
            if (!index_of(names, entry.name))
            {
                return "unknown parameter " + entry.name + "; the spec has: " + joined(names);
            }
        }
    }
    return std::nullopt;
}

std::optional<std::string> derived_beta_problem(const Spec& spec, const std::vector<std::string>& names)
{
    if (spec.model != Model::IGarch)
    {
        return std::nullopt;
    }
    const std::string& name = names[base_layout(spec).derived()];
    if (!is_fixed_by_name(spec, name))
    {
        return std::nullopt;
    }
    return name + " is derived in the igarch model and cannot be fixed";
}

std::optional<std::string> variance_targeting_problem(const Spec& spec)
{
    if (!spec.variance_targeting)
    {
        return std::nullopt;
    }
    if (spec.model == Model::IGarch)
    {
        return "variance targeting is not supported by the igarch model";
    }
    if (spec.model == Model::CsGarch)
    {
        return "variance targeting is not supported by the csgarch model";
    }
    if (is_fixed_by_name(spec, "omega"))
    {
        return "omega is derived under variance targeting and cannot be fixed";
    }
    return std::nullopt;
}

std::optional<std::string> fixed_phi_problem(const Spec& spec, const std::vector<std::string>& names)
{
    if (spec.model != Model::CsGarch || !is_fixed_by_name(spec, "eta21"))
    {
        return std::nullopt;
    }
    Layout layout = base_layout(spec);
    for (std::size_t index = layout.beta_begin(); index < layout.gamma_begin(); ++index)
    {
        if (!is_fixed_by_name(spec, names[index]))
        {
            return "eta21 is fixed in the csgarch model, so every beta must be fixed too";
        }
    }
    return std::nullopt;
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

bool Layout::has_skew() const
{
    return dist != Dist::Norm && dist != Dist::Std && dist != Dist::Ged;
}

bool Layout::has_shape() const
{
    return dist != Dist::Norm;
}

bool Layout::has_ghlambda() const
{
    return dist == Dist::Ghyp;
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

std::size_t Layout::skew() const
{
    return power() + (free_power ? 1 : 0);
}

std::size_t Layout::shape() const
{
    return skew() + (has_skew() ? 1 : 0);
}

std::size_t Layout::ghlambda() const
{
    return shape() + (has_shape() ? 1 : 0);
}

std::size_t Layout::count() const
{
    return ghlambda() + (has_ghlambda() ? 1 : 0);
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
    std::size_t derived_count = (has_derived() ? 1 : 0) + (variance_targeting ? 1 : 0);
    return count() - fixed.count() - derived_count;
}

bool Layout::is_fixed(std::size_t index) const
{
    return fixed.test(index);
}

bool Layout::is_started(std::size_t index) const
{
    return started.test(index);
}

bool Layout::is_derived(std::size_t index) const
{
    if (has_derived() && index == derived())
    {
        return true;
    }
    return variance_targeting && index == omega();
}

bool Layout::is_estimated(std::size_t index) const
{
    return !is_fixed(index) && !is_derived(index);
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
    if (index == skew() && has_skew())
    {
        if (dist::is_fernandez_steel(dist))
        {
            return ParameterKind::Positive;
        }
        return ParameterKind::Real;
    }
    if (index == ghlambda() && has_ghlambda())
    {
        return ParameterKind::Real;
    }
    return ParameterKind::Positive;
}

Layout layout_of(const Spec& spec)
{
    assert(!spec_problem(spec));
    Layout layout = base_layout(spec);
    layout.variance_targeting = spec.variance_targeting;
    std::vector<std::string> names = parameter_names(layout);
    for (const ParameterValue& entry : spec.fixed)
    {
        std::size_t index = *index_of(names, entry.name);
        layout.fixed.set(index);
        layout.given[index] = entry.value;
    }
    for (const ParameterValue& entry : spec.start)
    {
        std::size_t index = *index_of(names, entry.name);
        if (layout.is_estimated(index))
        {
            layout.started.set(index);
            layout.given[index] = entry.value;
        }
    }
    return layout;
}

std::optional<std::string> spec_problem(const Spec& spec)
{
    std::vector<std::string> names = parameter_names(base_layout(spec));
    std::optional<std::string> problem = unknown_name_problem(spec, names);
    if (!problem)
    {
        problem = derived_beta_problem(spec, names);
    }
    if (!problem)
    {
        problem = variance_targeting_problem(spec);
    }
    if (!problem)
    {
        problem = fixed_phi_problem(spec, names);
    }
    return problem;
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
    if (layout.has_skew())
    {
        names.emplace_back("skew");
    }
    if (layout.has_shape())
    {
        names.emplace_back("shape");
    }
    if (layout.has_ghlambda())
    {
        names.emplace_back("ghlambda");
    }
    return names;
}

}
