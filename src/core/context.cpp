#include "core/context.hpp"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <thread>

namespace ve
{

namespace
{

int resolve_threads(int requested)
{
    if (requested > 0)
    {
        return requested;
    }
    unsigned ceiling = static_cast<unsigned>(Context::kMaxThreads);
    unsigned hardware = std::clamp(std::thread::hardware_concurrency(), 1u, ceiling);
    return static_cast<int>(hardware);
}

}

Context::Context() : threads_(0)
{
    set_threads(0);
}

void Context::set_threads(int threads)
{
    assert(threads >= 0 && threads <= kMaxThreads);
    int resolved = resolve_threads(threads);
    auto pool = std::make_unique<BS::light_thread_pool>(static_cast<std::size_t>(resolved));
    std::unique_lock lock(mutex_);
    pool_.swap(pool);
    threads_ = resolved;
}

int Context::threads() const
{
    std::shared_lock lock(mutex_);
    return threads_;
}

}
