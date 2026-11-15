#ifndef VOLENGINE_CPP_CONTEXT_HPP
#define VOLENGINE_CPP_CONTEXT_HPP

#include <cstddef>
#include <span>
#include <utility>
#include <vector>
#include <volengine/error.hpp>
#include <volengine/fit.hpp>
#include <volengine/input.hpp>

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

    // One Fit per job of the batch, in job order. On any error no Fit is made.
    Result<std::vector<Fit>> fit(const Batch& batch) const
    {
        const auto jobs = batch.job_count();
        if (!jobs)
        {
            return std::unexpected(jobs.error());
        }
        std::vector<ve_fit*> raw(*jobs, nullptr);
        if (auto fitted = detail::check(ve_fit_batch(handle_.get(), batch.native(), raw.data())); !fitted)
        {
            return detail::failure(fitted);
        }
        std::vector<Fit> fits;
        fits.reserve(raw.size());
        for (ve_fit* handle : raw)
        {
            fits.emplace_back(handle);
        }
        return fits;
    }

    // Every spec on every series of the panel: the fit of spec s on series j is at index s * panel.cols() + j.
    Result<std::vector<Fit>> fit(std::span<const Spec> specs, const Panel& panel, const FitOptions& options = {}) const
    {
        std::vector<std::size_t> spec_index;
        std::vector<std::size_t> series_index;
        for (std::size_t s = 0; s < specs.size(); ++s)
        {
            for (std::size_t j = 0; j < panel.cols(); ++j)
            {
                spec_index.push_back(s);
                series_index.push_back(j);
            }
        }
        auto batch = Batch::create();
        if (!batch)
        {
            return detail::failure(batch);
        }
        auto described = batch->set_specs(specs)
                             .and_then([&] { return batch->set_panel(panel); })
                             .and_then([&] { return batch->set_jobs(spec_index, series_index); })
                             .and_then([&] { return batch->set_options(options); });
        if (!described)
        {
            return detail::failure(described);
        }
        return fit(*batch);
    }

    Result<Fit> fit(const Spec& spec, std::span<const double> series, const FitOptions& options = {}) const
    {
        auto panel = Panel::create(series, series.size(), 1);
        if (!panel)
        {
            return detail::failure(panel);
        }
        auto fits = fit(std::span<const Spec>(&spec, 1), *panel, options);
        if (!fits)
        {
            return detail::failure(fits);
        }
        return std::move(fits->front());
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
