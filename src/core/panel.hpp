#pragma once

#include <cstddef>
#include <utility>
#include <vector>

namespace ve
{

// Borrowed, strided view of a T x N panel. Never owns or copies `data`.
// With negative strides `data` points at element (0, 0).
struct PanelView
{
    const double* data = nullptr;
    size_t t = 0;
    size_t n = 0;
    ptrdiff_t stride_t = 0;
    ptrdiff_t stride_n = 1;
    std::vector<size_t> starts;
    std::vector<size_t> lengths;

    bool has_ranges() const
    {
        return !starts.empty();
    }

    // Element (row, series); the caller guarantees row < t and series < n.
    double at(size_t row, size_t series) const
    {
        const ptrdiff_t offset = static_cast<ptrdiff_t>(row) * stride_t
            + static_cast<ptrdiff_t>(series) * stride_n;
        return data[offset];
    }

    // First row and row count used by `series`; the whole panel when rectangular.
    std::pair<size_t, size_t> series_range(size_t series) const
    {
        if (!has_ranges())
        {
            return {0, t};
        }
        return {starts[series], lengths[series]};
    }
};

}
