#pragma once

#include "core/spec.hpp"
#include "dist/densities.hpp"

#include <array>
#include <bitset>
#include <cstddef>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace ve::model
{

enum class ParameterKind
{
    Location,
    Real,
    Positive,
    Weight
};

struct Layout
{
    Model model = Model::SGarch;
    bool include_mean;
    int ar;
    int ma;
    int arch;
    int garch;
    int gamma = 0;
    int eta1 = 0;
    int eta2 = 0;
    bool free_power = false;
    double fixed_power = 2.0;
    Dist dist;
    std::bitset<kMaxParameterCount> fixed{};
    std::bitset<kMaxParameterCount> started{};
    // The fixed value or the user's start value; 0 elsewhere.
    std::array<double, kMaxParameterCount> given{};
    bool variance_targeting = false;
    int archm = 0;
    int mxreg = 0;
    int vxreg = 0;

    std::size_t mu() const { return 0; }
    std::size_t ar_begin() const;
    std::size_t ma_begin() const;
    bool has_archm() const { return archm != 0; }
    std::size_t archm_index() const;
    std::size_t mxreg_begin() const;
    std::size_t omega() const;
    std::size_t alpha_begin() const;
    std::size_t beta_begin() const;
    std::size_t gamma_begin() const;
    std::size_t eta1_begin() const;
    std::size_t eta2_begin() const;
    std::size_t power() const;
    std::size_t vxreg_begin() const;
    bool has_skew() const;
    std::size_t skew() const;
    bool has_shape() const;
    std::size_t shape() const;
    bool has_ghlambda() const;
    std::size_t ghlambda() const;
    std::size_t count() const;
    bool has_derived() const;
    // apARCH and the fGARCH submodels: the recursion in sigma^power.
    bool is_power_family() const;
    // The eta1 block, or the apARCH gamma that plays its part; empty in csGARCH, whose eta1 holds rho.
    std::size_t asymmetry_begin() const;
    std::size_t asymmetry_count() const;
    std::size_t derived() const;
    std::size_t estimated_count() const;
    ParameterKind kind(std::size_t index) const;
    bool is_fixed(std::size_t index) const;
    bool is_started(std::size_t index) const;
    bool is_derived(std::size_t index) const;
    bool is_estimated(std::size_t index) const;
};

template <class T>
T power_of(std::span<const T> theta, const Layout& layout)
{
    if (!layout.free_power)
    {
        return T{layout.fixed_power};
    }
    return theta[layout.power()];
}

template <class T>
dist::ShapeParameters<T> shape_parameters_of(std::span<const T> theta, const Layout& layout)
{
    dist::ShapeParameters<T> parameters;
    if (layout.has_skew())
    {
        parameters.skew = theta[layout.skew()];
    }
    if (layout.has_shape())
    {
        parameters.shape = theta[layout.shape()];
    }
    if (layout.has_ghlambda())
    {
        parameters.ghlambda = theta[layout.ghlambda()];
    }
    return parameters;
}

Layout layout_of(const Spec& spec);
std::vector<std::string> parameter_names(const Layout& layout);
std::optional<std::string> spec_problem(const Spec& spec);

}
