#pragma once

#include "capi/error.hpp"
#include "capi/handles.hpp"

#include <cstddef>
#include <limits>
#include <memory>
#include <vector>

namespace ve::capi::dcc
{

template <class Handle, class Output>
ve_status check(const Handle* handle, const Output* out)
{
    if (handle == nullptr)
    {
        return set_error(VE_INVALID_HANDLE, "handle must not be null");
    }
    if (out == nullptr)
    {
        return set_error(VE_INVALID_ARGUMENT, "output must not be null");
    }
    return VE_OK;
}

template <class Handle>
ve_status create(Handle** out)
{
    if (out == nullptr)
    {
        return set_error(VE_INVALID_ARGUMENT, "out must not be null");
    }
    *out = std::make_unique<Handle>().release();
    return VE_OK;
}

template <class Value>
bool valid_count(std::size_t count)
{
    return count > 0 && count <= std::vector<Value>().max_size()
        && count <= std::numeric_limits<std::ptrdiff_t>::max() / sizeof(Value);
}

inline ve_status fit_pointers(const ve_fit* const* fits, std::size_t count,
                              std::vector<const ve::FitResult*>& out)
{
    if (fits == nullptr || !valid_count<const ve_fit*>(count))
    {
        return set_error(VE_INVALID_ARGUMENT, "fits must not be null and count must be valid and positive");
    }
    for (std::size_t i = 0; i < count; ++i)
    {
        if (fits[i] == nullptr)
        {
            return set_error(VE_INVALID_HANDLE, "fits must not contain null handles");
        }
    }
    out.reserve(count);
    for (std::size_t i = 0; i < count; ++i)
    {
        out.push_back(&fits[i]->result);
    }
    return VE_OK;
}

}
