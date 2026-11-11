#include "model/layout.hpp"

#include <string>

namespace ve::model
{

namespace
{

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

std::size_t Layout::shape() const
{
    return beta_begin() + static_cast<std::size_t>(garch);
}

std::size_t Layout::count() const
{
    return shape() + (has_shape() ? 1 : 0);
}

Layout layout_of(const Spec& spec)
{
    return Layout{spec.include_mean, spec.ar, spec.ma, spec.p, spec.q, spec.dist};
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
    if (layout.has_shape())
    {
        names.emplace_back("shape");
    }
    return names;
}

}
