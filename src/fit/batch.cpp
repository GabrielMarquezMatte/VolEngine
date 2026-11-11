#include "fit/batch.hpp"

#include "fit/fit.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace ve
{

namespace
{

constexpr std::size_t kBlocksPerThread = 8;

std::optional<std::string> check_job_indices(const Batch& batch)
{
    for (std::size_t job = 0; job < batch.jobs.size(); ++job)
    {
        if (batch.jobs[job].spec >= batch.specs.size())
        {
            return "job " + std::to_string(job) + ": spec index " + std::to_string(batch.jobs[job].spec)
                + " is out of range (n_specs = " + std::to_string(batch.specs.size()) + ")";
        }
        if (batch.jobs[job].series >= batch.panel->n)
        {
            return "job " + std::to_string(job) + ": series index " + std::to_string(batch.jobs[job].series)
                + " is out of range (n = " + std::to_string(batch.panel->n) + ")";
        }
    }
    return std::nullopt;
}

std::optional<std::string> check_series_values(const PanelView& panel, std::size_t series)
{
    const auto [start, length] = panel.series_range(series);
    for (std::size_t row = start; row < start + length; ++row)
    {
        if (!std::isfinite(panel.at(row, series)))
        {
            return "NaN or infinite value in series " + std::to_string(series) + " at row "
                + std::to_string(row);
        }
    }
    return std::nullopt;
}

std::optional<std::string> check_used_series(const Batch& batch)
{
    std::vector<bool> used(batch.panel->n, false);
    for (const FitJob& job : batch.jobs)
    {
        used[job.series] = true;
    }
    for (std::size_t series = 0; series < used.size(); ++series)
    {
        if (!used[series])
        {
            continue;
        }
        if (std::optional<std::string> message = check_series_values(*batch.panel, series))
        {
            return message;
        }
    }
    return std::nullopt;
}

FitResult run_job(const Batch& batch, const FitJob& job)
{
    const auto [start, length] = batch.panel->series_range(job.series);
    std::vector<double> series(length);
    for (std::size_t offset = 0; offset < length; ++offset)
    {
        series[offset] = batch.panel->at(start + offset, job.series);
    }
    return fit_one(series, batch.specs[job.spec], batch.config);
}

// Every job writes only its own slot of `results`.
void run_blocks(BS::light_thread_pool& pool, int threads, const Batch& batch, std::vector<FitResult>& results)
{
    const std::size_t max_blocks = kBlocksPerThread * static_cast<std::size_t>(threads);
    const std::size_t blocks = std::min(results.size(), max_blocks);
    auto run_block = [&](std::size_t first, std::size_t last)
    {
        for (std::size_t job = first; job < last; ++job)
        {
            results[job] = run_job(batch, batch.jobs[job]);
        }
    };
    try
    {
        auto futures = pool.submit_blocks(std::size_t{0}, results.size(), run_block, blocks);
        futures.wait();
        futures.get();
    }
    catch (...)
    {
        pool.wait();
        throw;
    }
}

}

std::optional<std::string> check_batch(const Batch& batch)
{
    if (batch.specs.empty())
    {
        return "batch has no specs; call ve_batch_set_specs";
    }
    if (batch.panel == nullptr)
    {
        return "batch has no panel; call ve_batch_set_panel";
    }
    if (batch.jobs.empty())
    {
        return "batch has no jobs; call ve_batch_set_jobs";
    }
    return check_job_indices(batch).or_else([&batch] { return check_used_series(batch); });
}

std::vector<FitResult> run_batch(Context& context, const Batch& batch)
{
    std::vector<FitResult> results(batch.jobs.size());
    context.with_pool([&](BS::light_thread_pool& pool, int threads)
                      { run_blocks(pool, threads, batch, results); });
    return results;
}

}
