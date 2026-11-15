#ifndef VOLENGINE_CPP_ERROR_HPP
#define VOLENGINE_CPP_ERROR_HPP

#include <cstdint>
#include <expected>
#include <memory>
#include <string>
#include <utility>
#include <volengine/volengine.h>

namespace ve
{

enum class Status : std::int32_t
{
    ok = VE_OK,
    invalid_argument = VE_INVALID_ARGUMENT,
    invalid_handle = VE_INVALID_HANDLE,
    unsupported = VE_UNSUPPORTED,
    out_of_memory = VE_OUT_OF_MEMORY,
    internal_error = VE_INTERNAL_ERROR,
};

// A failed call: its status and the message of ve_last_error, copied when the call returned.
struct Error
{
    Status status;
    std::string message;
};

template <class T>
using Result = std::expected<T, Error>;

namespace detail
{

// The message of ve_last_error is valid only until the next call on this thread, so it is copied here.
inline Result<void> check(ve_status status)
{
    if (status == VE_OK)
    {
        return {};
    }
    const char* message = ve_last_error();
    return std::unexpected(Error{static_cast<Status>(status), message != nullptr ? message : ""});
}

// The same error, as the failure of a Result of another type.
template <class T>
std::unexpected<Error> failure(Result<T>& result)
{
    return std::unexpected(std::move(result.error()));
}

inline Result<void> check_abi_version(std::int32_t native, std::int32_t expected = VE_ABI_VERSION)
{
    if (native == expected)
    {
        return {};
    }
    return std::unexpected(Error{Status::unsupported,
        "the VolEngine library has ABI version " + std::to_string(native) + ", but these headers need version " +
            std::to_string(expected)});
}

template <class T, void (*Free)(T*)>
struct FreeWith
{
    void operator()(T* handle) const noexcept
    {
        Free(handle);
    }
};

template <class T, void (*Free)(T*)>
using Owned = std::unique_ptr<T, FreeWith<T, Free>>;

} // namespace detail

// Fails when the loaded library does not speak the ABI these headers were written for.
inline Result<void> check_abi()
{
    return detail::check_abi_version(ve_abi_version());
}

} // namespace ve

#endif
