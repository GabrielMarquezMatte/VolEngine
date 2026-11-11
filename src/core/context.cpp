#include "core/context.hpp"

#include <algorithm>
#include <cstddef>
#include <thread>
#include <utility>

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
    return static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
}

}

Context::Context() : threads_(0)
{
    set_threads(0);
}

void Context::set_threads(int threads)
{
    int resolved = resolve_threads(threads);
    auto pool = std::make_unique<BS::light_thread_pool>(static_cast<std::size_t>(resolved));
    pool_ = std::move(pool);
    threads_ = resolved;
}

BS::light_thread_pool& Context::pool()
{
    return *pool_;
}

int Context::threads() const
{
    return threads_;
}

}
