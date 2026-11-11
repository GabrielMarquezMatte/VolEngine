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
