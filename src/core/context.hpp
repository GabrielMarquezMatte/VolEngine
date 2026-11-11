#pragma once

#include <BS_thread_pool.hpp>
#include <memory>
#include <mutex>
#include <shared_mutex>

namespace ve
{

class Context
{
public:
    static constexpr int kMaxThreads = 1024;

    Context();

    // threads in 0 .. kMaxThreads, 0 meaning the hardware concurrency capped at kMaxThreads. Waits for every
    // running with_pool call to return before it replaces the pool.
    void set_threads(int threads);
    int threads() const;

    // Calls work(pool, threads) while holding the pool shared, so set_threads cannot replace it meanwhile.
    template <class F>
    decltype(auto) with_pool(F&& work)
    {
        std::shared_lock lock(mutex_);
        return work(*pool_, threads_);
    }

private:
    mutable std::shared_mutex mutex_;
    int threads_;
    std::unique_ptr<BS::light_thread_pool> pool_;
};

}
