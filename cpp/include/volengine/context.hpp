#ifndef VOLENGINE_CPP_CONTEXT_HPP
#define VOLENGINE_CPP_CONTEXT_HPP

#include <utility>
#include <volengine/error.hpp>

namespace ve
{

// Owns the native thread pool. Fits, forecasts and simulations do not depend on it afterwards.
class Context
{
public:
    // threads in 1 .. VE_MAX_THREADS, or 0 for the hardware concurrency. Fails if the library's ABI differs.
    static Result<Context> create(int threads = 0)
    {
        if (auto abi = check_abi(); !abi)
        {
            return detail::failure(abi);
        }
        ve_context* raw = nullptr;
        if (auto created = detail::check(ve_context_create(&raw)); !created)
        {
            return detail::failure(created);
        }
        Context context{Handle{raw}};
        if (auto sized = context.set_threads(threads); !sized)
        {
            return detail::failure(sized);
        }
        return context;
    }

    Result<void> set_threads(int threads)
    {
        return detail::check(ve_context_set_threads(handle_.get(), threads));
    }

    Result<int> threads() const
    {
        std::int32_t count = 0;
        if (auto read = detail::check(ve_context_threads(handle_.get(), &count)); !read)
        {
            return detail::failure(read);
        }
        return count;
    }

    ve_context* native() const noexcept
    {
        return handle_.get();
    }

private:
    using Handle = detail::Owned<ve_context, ve_context_free>;

    explicit Context(Handle handle) : handle_(std::move(handle))
    {
    }

    Handle handle_;
};

} // namespace ve

#endif
