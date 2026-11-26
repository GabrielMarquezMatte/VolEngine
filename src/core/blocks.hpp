#pragma once

#include <BS_thread_pool.hpp>
#include <algorithm>
#include <cstddef>

namespace ve
{

// Calls work(first, last) on min(count, blocks_per_thread * threads) contiguous blocks of 0 .. count, spread over
// the pool, and returns when all have run. Rethrows the first exception a block raised, after every block of this
// call has finished; it does not wait for the blocks of other calls on the same pool. Call it inside
// Context::with_pool, with the pool and threads it hands over.
template <class F>
void run_in_blocks(BS::light_thread_pool& pool, int threads, std::size_t count, std::size_t blocks_per_thread,
                   F&& work)
{
    const std::size_t blocks = std::min(count, blocks_per_thread * static_cast<std::size_t>(threads));
    BS::multi_future<void> futures;
    try
    {
        futures = pool.submit_blocks(std::size_t{0}, count, work, blocks);
    }
    catch (...)
    {
        // Some blocks may be queued without a future to wait on: wait for the pool before `work` goes out of scope.
        pool.wait();
        throw;
    }
    futures.wait();
    futures.get();
}

}
