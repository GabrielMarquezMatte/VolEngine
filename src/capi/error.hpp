#pragma once

#include <exception>
#include <new>
#include <string_view>
#include <volengine/volengine.h>

namespace ve::capi
{

ve_status set_error(ve_status status, std::string_view message);
void clear_error();
const char* last_error();

// The error of an accessor of the inference at the estimate (standard errors, t and p values, Nyblom) on a fit made
// with ve_fit_options_set_standard_errors(o, 0).
inline ve_status no_standard_errors()
{
    return set_error(VE_INVALID_ARGUMENT,
                     "standard errors were not computed: fit with ve_fit_options_set_standard_errors(o, 1)");
}

// The same for a DCC fit made with ve_dcc_options_set_standard_errors(o, 0).
inline ve_status no_dcc_standard_errors()
{
    return set_error(VE_INVALID_ARGUMENT,
                     "standard errors were not computed: fit with ve_dcc_options_set_standard_errors(o, 1) or call "
                     "ve_dcc_standard_errors");
}

template <typename F>
ve_status guard(F&& callable)
{
    clear_error();
    try
    {
        return callable();
    }
    catch (const std::bad_alloc&)
    {
        return set_error(VE_OUT_OF_MEMORY, "out of memory");
    }
    catch (const std::exception& error)
    {
        return set_error(VE_INTERNAL_ERROR, error.what());
    }
    catch (...)
    {
        return set_error(VE_INTERNAL_ERROR, "unknown error");
    }
}

}
