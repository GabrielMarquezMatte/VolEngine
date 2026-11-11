#include "capi/error.hpp"
#include "capi/handles.hpp"

#include <cstddef>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

using ve::capi::guard;
using ve::capi::set_error;

namespace
{

constexpr size_t kPtrMax = static_cast<size_t>(std::numeric_limits<ptrdiff_t>::max());

bool mul_overflows(size_t a, size_t b)
{
    return a != 0 && b > std::numeric_limits<size_t>::max() / a;
}

bool add_overflows(size_t a, size_t b)
{
    return b > std::numeric_limits<size_t>::max() - a;
}

size_t magnitude(ptrdiff_t value)
{
    if (value < 0)
    {
        return size_t{0} - static_cast<size_t>(value);
    }
    return static_cast<size_t>(value);
}

// |stride| * (count - 1) must fit in ptrdiff_t; contributes to the reachable offset sum.
bool reach_of(ptrdiff_t stride, size_t count, size_t& reach)
{
    reach = 0;
    if (count == 0)
    {
        return true;
    }
    const size_t span = magnitude(stride);
    if (mul_overflows(span, count - 1))
    {
        return false;
    }
    reach = span * (count - 1);
    return reach <= kPtrMax;
}

ve_status check_strides(size_t t, size_t n, ptrdiff_t stride_t, ptrdiff_t stride_n)
{
    const bool has_elements = t > 0 && n > 0;
    if (has_elements && ((stride_t == 0 && t > 1) || (stride_n == 0 && n > 1)))
    {
        return set_error(VE_INVALID_ARGUMENT, "a stride must not be 0 when its dimension is > 1");
    }
    size_t reach_t = 0;
    size_t reach_n = 0;
    if (!reach_of(stride_t, t, reach_t) || !reach_of(stride_n, n, reach_n)
        || add_overflows(reach_t, reach_n) || reach_t + reach_n > kPtrMax)
    {
        return set_error(VE_INVALID_ARGUMENT, "strides overflow the addressable offset range");
    }
    return VE_OK;
}

ve_status check_ranges(size_t t, const size_t* starts, const size_t* lengths, size_t n)
{
    for (size_t j = 0; j < n; ++j)
    {
        if (add_overflows(starts[j], lengths[j]) || starts[j] + lengths[j] > t)
        {
            return set_error(VE_INVALID_ARGUMENT, "a range exceeds the panel rows (start + length > t)");
        }
    }
    return VE_OK;
}

ve_status check_create(const double* data, size_t t, size_t n)
{
    if (mul_overflows(t, n))
    {
        return set_error(VE_INVALID_ARGUMENT, "t * n overflows size_t");
    }
    if (data == nullptr && t * n > 0)
    {
        return set_error(VE_INVALID_ARGUMENT, "data must not be null when t * n > 0");
    }
    if (n > kPtrMax)
    {
        return set_error(VE_INVALID_ARGUMENT, "n is too large");
    }
    return check_strides(t, n, static_cast<ptrdiff_t>(n), 1);
}

}

extern "C" ve_status ve_panel_create(const double* data, size_t t, size_t n, ve_panel** out)
{
    if (out != nullptr)
    {
        *out = nullptr;
    }
    return guard(
        [&]
        {
            if (out == nullptr)
            {
                return set_error(VE_INVALID_ARGUMENT, "out must not be null");
            }
            const ve_status status = check_create(data, t, n);
            if (status != VE_OK)
            {
                return status;
            }
            auto panel = std::make_unique<ve_panel>();
            panel->view.data = data;
            panel->view.t = t;
            panel->view.n = n;
            panel->view.stride_t = static_cast<ptrdiff_t>(n);
            panel->view.stride_n = 1;
            *out = panel.release();
            return VE_OK;
        });
}

extern "C" ve_status ve_panel_set_strides(ve_panel* p, ptrdiff_t stride_t, ptrdiff_t stride_n)
{
    return guard(
        [&]
        {
            if (p == nullptr)
            {
                return set_error(VE_INVALID_HANDLE, "p must not be null");
            }
            const ve_status status = check_strides(p->view.t, p->view.n, stride_t, stride_n);
            if (status != VE_OK)
            {
                return status;
            }
            p->view.stride_t = stride_t;
            p->view.stride_n = stride_n;
            return VE_OK;
        });
}

extern "C" ve_status ve_panel_set_ranges(ve_panel* p, const size_t* starts, const size_t* lengths)
{
    return guard(
        [&]
        {
            if (p == nullptr)
            {
                return set_error(VE_INVALID_HANDLE, "p must not be null");
            }
            if ((starts == nullptr) != (lengths == nullptr))
            {
                return set_error(VE_INVALID_ARGUMENT, "starts and lengths must both be null or both set");
            }
            if (starts == nullptr)
            {
                p->view.starts.clear();
                p->view.lengths.clear();
                return VE_OK;
            }
            const ve_status status = check_ranges(p->view.t, starts, lengths, p->view.n);
            if (status != VE_OK)
            {
                return status;
            }
            std::vector<size_t> new_starts(starts, starts + p->view.n);
            std::vector<size_t> new_lengths(lengths, lengths + p->view.n);
            p->view.starts = std::move(new_starts);
            p->view.lengths = std::move(new_lengths);
            return VE_OK;
        });
}

extern "C" void ve_panel_free(ve_panel* p)
{
    std::unique_ptr<ve_panel> owner(p);
}
