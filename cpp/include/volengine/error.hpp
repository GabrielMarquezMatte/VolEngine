#ifndef VOLENGINE_CPP_ERROR_HPP
#define VOLENGINE_CPP_ERROR_HPP

#include <cstdint>
#include <expected>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
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

// `fill` writes one double.
template <class Fill>
Result<double> scalar(Fill fill)
{
    double value = 0.0;
    if (auto read = check(fill(&value)); !read)
    {
        return failure(read);
    }
    return value;
}

// `fill` writes `size` doubles.
template <class Fill>
Result<std::vector<double>> values(std::size_t size, Fill fill)
{
    std::vector<double> result(size);
    if (auto read = check(fill(result.data())); !read)
    {
        return failure(read);
    }
    return result;
}

// A count read by `function(handle, &count)`.
template <class Handle>
Result<std::size_t> count(const Handle* handle, ve_status (*function)(const Handle*, std::size_t*))
{
    std::size_t value = 0;
    if (auto read = check(function(handle, &value)); !read)
    {
        return failure(read);
    }
    return value;
}

// The names `function(handle, i, &name)` gives for i < size; they live as long as the handle.
template <class Handle>
Result<std::vector<std::string_view>> names(
    const Handle* handle, const Result<std::size_t>& size, ve_status (*function)(const Handle*, std::size_t, const char**))
{
    if (!size)
    {
        return std::unexpected(size.error());
    }
    std::vector<std::string_view> result;
    for (std::size_t i = 0; i < *size; ++i)
    {
        const char* name = nullptr;
        if (auto read = check(function(handle, i, &name)); !read)
        {
            return failure(read);
        }
        result.emplace_back(name);
    }
    return result;
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
