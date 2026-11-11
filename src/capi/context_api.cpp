#include "capi/error.hpp"
#include "capi/handles.hpp"

#include <memory>

using ve::capi::guard;
using ve::capi::set_error;

extern "C" ve_status ve_context_create(ve_context** out)
{
    return guard(
        [&]
        {
            if (out == nullptr)
            {
                return set_error(VE_INVALID_ARGUMENT, "out must not be null");
            }
            *out = std::make_unique<ve_context>().release();
            return VE_OK;
        });
}

extern "C" ve_status ve_context_set_threads(ve_context* ctx, int32_t threads)
{
    return guard(
        [&]
        {
            if (ctx == nullptr)
            {
                return set_error(VE_INVALID_HANDLE, "ctx must not be null");
            }
            if (threads < 0)
            {
                return set_error(VE_INVALID_ARGUMENT, "threads must be >= 0");
            }
            ctx->context.set_threads(threads);
            return VE_OK;
        });
}

extern "C" ve_status ve_context_threads(const ve_context* ctx, int32_t* out)
{
    return guard(
        [&]
        {
            if (ctx == nullptr)
            {
                return set_error(VE_INVALID_HANDLE, "ctx must not be null");
            }
            if (out == nullptr)
            {
                return set_error(VE_INVALID_ARGUMENT, "out must not be null");
            }
            *out = ctx->context.threads();
            return VE_OK;
        });
}

extern "C" void ve_context_free(ve_context* ctx)
{
    std::unique_ptr<ve_context> owner(ctx);
}

extern "C" const char* ve_last_error(void)
{
    return ve::capi::last_error();
}
