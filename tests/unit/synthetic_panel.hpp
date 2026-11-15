#pragma once

#include "dcc/panel.hpp"
#include "simd/fixtures.hpp"

#include <cstddef>
#include <vector>

namespace ve::testing
{

// A standardized panel of m series from a fixed stream, with qbar, nbar and the deltas computed from z.
inline dcc::StandardizedPanel synthetic_panel(std::size_t m, std::size_t t)
{
    ve::test::Stream stream(100 + static_cast<unsigned>(m));
    dcc::StandardizedPanel panel;
    panel.t = t;
    panel.m = m;
    panel.z = stream.vector(m * t);
    for (double& z : panel.z)
    {
        z *= 1.7;
    }
    panel.sigma.assign(m * t, 1.0);
    dcc::refresh_moments(panel);
    return panel;
}

// Group labels 1..count that split m series into runs of nearly equal size, the larger runs last.
inline std::vector<int> groups_of(std::size_t m, std::size_t count)
{
    std::vector<int> groups(m);
    for (std::size_t i = 0; i < m; ++i)
    {
        groups[i] = static_cast<int>(i * count / m) + 1;
    }
    return groups;
}

}
