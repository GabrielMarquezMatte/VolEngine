#ifndef VOLENGINE_CPP_CONTEXT_HPP
#define VOLENGINE_CPP_CONTEXT_HPP

#include <cstddef>
#include <span>
#include <utility>
#include <vector>
#include <volengine/dcc.hpp>
#include <volengine/error.hpp>
#include <volengine/fit.hpp>
#include <volengine/input.hpp>
#include <volengine/simulation.hpp>

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
        std::vector<Fit> fits;
        fits.reserve(raw.size()); // before the native call: nothing may throw while raw holds handles
        if (auto fitted = detail::check(ve_fit_batch_with_cancel(handle_.get(), batch.native(), nullptr,
                                                                 raw.data())); !fitted)
        {
            return detail::failure(fitted);
        }
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

    // Paths continuing after the last observation of the fit.
    Result<Simulation> simulate(const Fit& fit, const SimOptions& options) const
    {
        return simulate_with(options, [&](const ve_sim_options* native, ve_sim** out)
            { return ve_simulate_fit_with_cancel(handle_.get(), fit.native(), native, nullptr, out); });
    }

    // Paths of a spec with every parameter fixed, from its unconditional state.
    Result<Simulation> simulate(const Spec& spec, const SimOptions& options) const
    {
        auto native = spec.make_handle();
        if (!native)
        {
            return detail::failure(native);
        }
        return simulate_with(options, [&](const ve_sim_options* sim_options, ve_sim** out)
            { return ve_simulate_spec_with_cancel(handle_.get(), native->get(), sim_options, nullptr,
                                                  out); });
    }

    // Every DCC spec over the same univariate fits (at least two, finite, of one length >= 100), in spec order.
    Result<std::vector<DccFit>> fit_dcc(
        std::span<const DccSpec> specs, std::span<const Fit> univariate, const DccOptions& options = {}) const
    {
        auto batch = make_dcc_batch(specs, univariate, options);
        if (!batch)
        {
            return detail::failure(batch);
        }
        const auto count = detail::count(batch->get(), ve_dcc_batch_spec_count);
        if (!count)
        {
            return std::unexpected(count.error());
        }
        std::vector<ve_dcc_fit*> raw(*count, nullptr);
        std::vector<DccFit> fits;
        fits.reserve(raw.size()); // before the native call: nothing may throw while raw holds handles
        if (auto fitted = detail::check(ve_dcc_fit_batch_with_cancel(handle_.get(), batch->get(), nullptr,
                                                                     raw.data())); !fitted)
        {
            return detail::failure(fitted);
        }
        for (ve_dcc_fit* handle : raw)
        {
            fits.emplace_back(handle);
        }
        return fits;
    }

    Result<DccFit> fit_dcc(
        const DccSpec& spec, std::span<const Fit> univariate, const DccOptions& options = {}) const
    {
        auto fits = fit_dcc(std::span<const DccSpec>(&spec, 1), univariate, options);
        if (!fits)
        {
            return detail::failure(fits);
        }
        return std::move(fits->front());
    }

    // The same fit with its standard errors, computed from the univariate fits it came from (in the same order).
    Result<DccFit> standard_errors(const DccFit& fit, std::span<const Fit> univariate) const
    {
        const auto pointers = fit_pointers(univariate);
        ve_dcc_fit* raw = nullptr;
        if (auto computed = detail::check(
                ve_dcc_standard_errors_with_cancel(handle_.get(), fit.native(), pointers.data(),
                                                   pointers.size(), nullptr, &raw));
            !computed)
        {
            return detail::failure(computed);
        }
        return DccFit{raw};
    }

    // Paths of a DCC fit. The regressors of the horizon go in `future`, not in the options.
    Result<DccSimulation> simulate(
        const DccFit& fit, const SimOptions& options, const DccFuture* future = nullptr) const
    {
        auto native = options.make_handle();
        if (!native)
        {
            return detail::failure(native);
        }
        const ve_dcc_future* native_future = nullptr;
        if (future != nullptr)
        {
            native_future = future->native();
        }
        ve_dcc_sim* raw = nullptr;
        if (auto simulated = detail::check(
                ve_simulate_dcc_with_cancel(handle_.get(), fit.native(), native->get(), native_future,
                                            nullptr, &raw));
            !simulated)
        {
            return detail::failure(simulated);
        }
        return DccSimulation{raw};
    }

    ve_context* native() const noexcept
    {
        return handle_.get();
    }

private:
    using Handle = detail::Owned<ve_context, ve_context_free>;

    static std::vector<const ve_fit*> fit_pointers(std::span<const Fit> fits)
    {
        std::vector<const ve_fit*> pointers;
        for (const Fit& fit : fits)
        {
            pointers.push_back(fit.native());
        }
        return pointers;
    }

    static Result<detail::DccBatchHandle> make_dcc_batch(
        std::span<const DccSpec> specs, std::span<const Fit> univariate, const DccOptions& options)
    {
        ve_dcc_batch* raw = nullptr;
        if (auto created = detail::check(ve_dcc_batch_create(&raw)); !created)
        {
            return detail::failure(created);
        }
        detail::DccBatchHandle batch{raw};
        std::vector<detail::DccSpecHandle> handles;
        std::vector<const ve_dcc_spec*> pointers;
        for (const DccSpec& spec : specs)
        {
            auto handle = spec.make_handle();
            if (!handle)
            {
                return detail::failure(handle);
            }
            pointers.push_back(handle->get());
            handles.push_back(std::move(*handle));
        }
        const auto fits = fit_pointers(univariate);
        auto native_options = options.make_handle();
        if (!native_options)
        {
            return detail::failure(native_options);
        }
        auto described = detail::check(ve_dcc_batch_set_specs(raw, pointers.data(), pointers.size()))
                             .and_then([&] { return detail::check(ve_dcc_batch_set_fits(raw, fits.data(), fits.size())); })
                             .and_then([&] { return detail::check(ve_dcc_batch_set_options(raw, native_options->get())); });
        if (!described)
        {
            return detail::failure(described);
        }
        return batch;
    }

    template <class Run>
    static Result<Simulation> simulate_with(const SimOptions& options, Run run)
    {
        auto native = options.make_handle();
        if (!native)
        {
            return detail::failure(native);
        }
        ve_sim* raw = nullptr;
        if (auto simulated = detail::check(run(native->get(), &raw)); !simulated)
        {
            return detail::failure(simulated);
        }
        return Simulation{raw};
    }

    explicit Context(Handle handle) : handle_(std::move(handle))
    {
    }

    Handle handle_;
};

} // namespace ve

#endif
