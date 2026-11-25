#pragma once

#include <atomic>
#include <exception>

namespace ve
{

// Cooperative cancellation: the owner calls request() from any thread, the long loops poll requested(). The flag
// never goes back to unrequested.
class CancelFlag
{
public:
    void request() noexcept
    {
        requested_.store(true, std::memory_order_relaxed);
    }

    bool requested() const noexcept
    {
        return requested_.load(std::memory_order_relaxed);
    }

private:
    std::atomic<bool> requested_{false};
};

struct Cancelled final : std::exception
{
    const char* what() const noexcept override
    {
        return "cancelled";
    }
};

// A null flag never cancels.
inline void throw_if_cancelled(const CancelFlag* flag)
{
    if (flag != nullptr && flag->requested())
    {
        throw Cancelled{};
    }
}

}
