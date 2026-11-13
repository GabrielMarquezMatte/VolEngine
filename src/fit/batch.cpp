#include "fit/batch.hpp"

#include "core/blocks.hpp"
#include "fit/fit.hpp"

#include <array>
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

std::optional<std::string> check_specs(const Batch& batch)
{
    for (std::size_t index = 0; index < batch.specs.size(); ++index)
    {
        if (std::optional<std::string> problem = model::spec_problem(batch.specs[index]))
        {
            return "spec " + std::to_string(index) + ": " + *problem;
        }
    }
    return std::nullopt;
}

struct RegressorKind
{
    const char* name;
    const std::vector<const PanelView*>* panels;
    int Spec::* count;
};

std::array<RegressorKind, 2> regressor_kinds(const Batch& batch)
{
    return {{{"mxreg", &batch.mean_regressors, &Spec::mxreg}, {"vxreg", &batch.variance_regressors, &Spec::vxreg}}};
}

const PanelView* regressor_of(const RegressorKind& kind, std::size_t job)
{
    if (kind.panels->empty())
    {
        return nullptr;
    }
    return (*kind.panels)[job];
}

std::optional<std::string> check_regressor_counts(const Batch& batch)
{
    for (const RegressorKind& kind : regressor_kinds(batch))
    {
        if (!kind.panels->empty() && kind.panels->size() != batch.jobs.size())
        {
            return std::string(kind.name) + " has " + std::to_string(kind.panels->size())
                + " entries but the batch has " + std::to_string(batch.jobs.size()) + " jobs";
        }
    }
    return std::nullopt;
}

std::optional<std::string> check_regressor_values(const PanelView& data, const PanelView& panel, std::size_t series)
{
    const auto [start, length] = data.series_range(series);
    for (std::size_t row = start; row < start + length; ++row)
    {
        for (std::size_t column = 0; column < panel.n; ++column)
        {
            if (!std::isfinite(panel.at(row, column)))
            {
                return "NaN or infinite value in column " + std::to_string(column) + " at row "
                    + std::to_string(row);
            }
        }
    }
    return std::nullopt;
}

std::optional<std::string> check_regressor_panel(const Batch& batch, const RegressorKind& kind, std::size_t job)
{
    const FitJob& fit_job = batch.jobs[job];
    const PanelView* panel = regressor_of(kind, job);
    const std::size_t count = static_cast<std::size_t>(batch.specs[fit_job.spec].*kind.count);
    const std::string name = kind.name;
    const std::string count_text = "the spec has " + name + " = " + std::to_string(count);
    if (panel == nullptr && count > 0)
    {
        return name + " is missing (" + count_text + ")";
    }
    if (panel == nullptr)
    {
        return std::nullopt;
    }
    if (count == 0)
    {
        return name + " is given but " + count_text;
    }
    if (panel->n != count)
    {
        return name + " panel has " + std::to_string(panel->n) + " columns, " + count_text;
    }
    if (panel->t != batch.panel->t)
    {
        return name + " panel has " + std::to_string(panel->t) + " rows, the data panel has "
            + std::to_string(batch.panel->t);
    }
    if (panel->has_ranges())
    {
        return name + " panel must not have ranges; the rows come from the data panel";
    }
    if (const auto problem = check_regressor_values(*batch.panel, *panel, fit_job.series))
    {
        return name + ": " + *problem;
    }
    return std::nullopt;
}

std::optional<std::string> check_regressors(const Batch& batch)
{
    if (const auto problem = check_regressor_counts(batch))
    {
        return problem;
    }
    for (std::size_t job = 0; job < batch.jobs.size(); ++job)
    {
        for (const RegressorKind& kind : regressor_kinds(batch))
        {
            if (const auto problem = check_regressor_panel(batch, kind, job))
            {
                return "job " + std::to_string(job) + ": " + *problem;
            }
        }
    }
    return std::nullopt;
}

// Rows start .. start + length - 1 of the panel, row-major; nothing without a panel.
std::vector<double> gather_rows(const PanelView* panel, std::size_t start, std::size_t length)
{
    std::vector<double> values;
    if (panel == nullptr)
    {
        return values;
    }
    values.reserve(length * panel->n);
    for (std::size_t row = start; row < start + length; ++row)
    {
        for (std::size_t column = 0; column < panel->n; ++column)
        {
            values.push_back(panel->at(row, column));
        }
    }
    return values;
}

FitResult run_job(const Batch& batch, std::size_t index)
{
    const FitJob& job = batch.jobs[index];
    const auto [start, length] = batch.panel->series_range(job.series);
    std::vector<double> series(length);
    for (std::size_t offset = 0; offset < length; ++offset)
    {
        series[offset] = batch.panel->at(start + offset, job.series);
    }
    const auto [mean_kind, variance_kind] = regressor_kinds(batch);
    const PanelView* mean_panel = regressor_of(mean_kind, index);
    const PanelView* variance_panel = regressor_of(variance_kind, index);
    const std::vector<double> mean = gather_rows(mean_panel, start, length);
    const std::vector<double> variance = gather_rows(variance_panel, start, length);
    const model::Data data{series, {mean, mean_panel == nullptr ? 0 : mean_panel->n},
                           {variance, variance_panel == nullptr ? 0 : variance_panel->n}};
    return fit_one(data, batch.specs[job.spec], batch.config);
}

// Every job writes only its own slot of `results`.
void run_blocks(BS::light_thread_pool& pool, int threads, const Batch& batch, std::vector<FitResult>& results)
{
    auto run_block = [&](std::size_t first, std::size_t last)
    {
        for (std::size_t job = first; job < last; ++job)
        {
            results[job] = run_job(batch, job);
        }
    };
    run_in_blocks(pool, threads, results.size(), kBlocksPerThread, run_block);
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
    return check_job_indices(batch)
        .or_else([&batch] { return check_specs(batch); })
        .or_else([&batch] { return check_used_series(batch); })
        .or_else([&batch] { return check_regressors(batch); });
}

std::vector<FitResult> run_batch(Context& context, const Batch& batch)
{
    std::vector<FitResult> results(batch.jobs.size());
    context.with_pool([&](BS::light_thread_pool& pool, int threads)
                      { run_blocks(pool, threads, batch, results); });
    return results;
}

}
