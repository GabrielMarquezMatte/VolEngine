#include "capi/error.hpp"
#include "capi/handles.hpp"

#include <memory>

using ve::capi::guard;
using ve::capi::set_error;

extern "C" ve_status ve_cancel_create(ve_cancel** out)
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
            *out = std::make_unique<ve_cancel>().release();
            return VE_OK;
        });
}

extern "C" ve_status ve_cancel_request(ve_cancel* c)
{
    return guard(
        [&]
        {
            if (c == nullptr)
            {
                return set_error(VE_INVALID_HANDLE, "c must not be null");
            }
            c->flag.request();
            return VE_OK;
        });
}

extern "C" void ve_cancel_free(ve_cancel* c)
{
    std::unique_ptr<ve_cancel> owner(c);
}
