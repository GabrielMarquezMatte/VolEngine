#pragma once

#include <BS_thread_pool.hpp>
#include <memory>

namespace ve
{

class Context
{
public:
    Context();

    void set_threads(int threads);
    BS::light_thread_pool& pool();
    int threads() const;

private:
    int threads_;
    std::unique_ptr<BS::light_thread_pool> pool_;
};

}
