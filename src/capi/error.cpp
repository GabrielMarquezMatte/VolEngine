#include "capi/error.hpp"

#include <string>

namespace ve::capi
{

namespace
{

thread_local std::string last_error_message;

}

ve_status set_error(ve_status status, std::string_view message)
{
    last_error_message.assign(message);
    return status;
}

void clear_error()
{
    last_error_message.clear();
}

const char* last_error()
{
    return last_error_message.c_str();
}

}
