#pragma once

#include "core/spec.hpp"

#include <cstddef>
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

    std::size_t mu() const { return 0; }
    std::size_t ar_begin() const;
    std::size_t ma_begin() const;
    std::size_t omega() const;
    std::size_t alpha_begin() const;
    std::size_t beta_begin() const;
    std::size_t gamma_begin() const;
    std::size_t eta1_begin() const;
    std::size_t eta2_begin() const;
    std::size_t power() const;
    bool has_shape() const;
    std::size_t shape() const;
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

// 0 when the distribution has no shape.
template <class T>
T shape_of(std::span<const T> theta, const Layout& layout)
{
    if (!layout.has_shape())
    {
        return T{0.0};
    }
    return theta[layout.shape()];
}

Layout layout_of(const Spec& spec);
std::vector<std::string> parameter_names(const Layout& layout);

}
