#pragma once

#include "core/spec.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace ve::model
{

struct Layout
{
    bool include_mean;
    int ar;
    int ma;
    int arch;
    int garch;
    Dist dist;

    std::size_t mu() const { return 0; }
    std::size_t ar_begin() const;
    std::size_t ma_begin() const;
    std::size_t omega() const;
    std::size_t alpha_begin() const;
    std::size_t beta_begin() const;
    bool has_shape() const;
    std::size_t shape() const;
    std::size_t count() const;
};

Layout layout_of(const Spec& spec);
std::vector<std::string> parameter_names(const Layout& layout);

}
