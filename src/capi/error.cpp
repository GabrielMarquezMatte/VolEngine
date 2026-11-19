#include "capi/error.hpp"

#include <string>

namespace ve::capi
{

namespace
{

thread_local std::string last_error_message;
thread_local const char* fallback_message = nullptr;

}

// Runs inside the catch handlers of guard, where a bad_alloc from the assignment would escape the C interface: when
// the message cannot be stored, the status still reaches the caller with a message that needs no allocation.
ve_status set_error(ve_status status, std::string_view message) noexcept
{
    fallback_message = nullptr;
    try
    {
        last_error_message.assign(message);
    }
    catch (...)
    {
        last_error_message.clear();
        fallback_message = "out of memory while recording the error message";
    }
    return status;
}

void clear_error() noexcept
{
    fallback_message = nullptr;
    last_error_message.clear();
}

const char* last_error() noexcept
{
    if (fallback_message != nullptr)
    {
        return fallback_message;
    }
    return last_error_message.c_str();
}

}
