// Profile of the TCC grid (BM_TccGrid): runs the grid, or a slice of it, once through ve::run_batch and prints
// where the time goes, by spec, distribution, order and ARMA. See docs/performance/measuring.md.
//
// volengine_grid_profile [--series N] [--threads N] [--filter TEXT] [--standard-errors on|off] [--json PATH]

#include "core/context.hpp"
#include "fit/batch.hpp"
#include "grids.hpp"
#include "simulated_panel.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <format>
#include <fstream>
#include <functional>
#include <map>
#include <optional>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace
{

constexpr std::size_t kObservations = 5000;
constexpr std::size_t kMaxSeries = 20;
constexpr std::size_t kTopSpecs = 20;

struct Options
{
    std::size_t series = kMaxSeries;
    int threads = 0;
    std::string filter;
    bool standard_errors = true;
    std::string json;
};

struct Record
{
    std::string spec;
    std::size_t series = 0;
    ve::FitResult result;
};

struct Group
{
    std::string name;
    std::size_t fits = 0;
    double seconds = 0.0;
    double optimizer_seconds = 0.0;
    double iterations = 0.0;
    double evaluations = 0.0;
    std::size_t converged = 0;
    std::size_t max_iter = 0;
    std::size_t other = 0;
};

std::string_view value_after(std::span<char*> args, std::size_t& index)
{
    if (index + 1 >= args.size())
    {
        throw std::invalid_argument(std::string(args[index]) + " needs a value");
    }
    ++index;
    return args[index];
}

void apply_flag(Options& options, std::span<char*> args, std::size_t& index)
{
    const std::string_view flag = args[index];
    const std::string value(value_after(args, index));
    if (flag == "--series")
    {
        options.series = std::stoul(value);
        return;
    }
    if (flag == "--threads")
    {
        options.threads = std::stoi(value);
        return;
    }
    if (flag == "--filter")
    {
        options.filter = value;
        return;
    }
    if (flag == "--json")
    {
        options.json = value;
        return;
    }
    if (flag == "--standard-errors" && (value == "on" || value == "off"))
    {
        options.standard_errors = value == "on";
        return;
    }
    throw std::invalid_argument("unknown flag or value: " + std::string(flag) + " " + value);
}

Options parse(std::span<char*> args)
{
    Options options;
    for (std::size_t index = 1; index < args.size(); ++index)
    {
        apply_flag(options, args, index);
    }
    if (options.series == 0 || options.series > kMaxSeries)
    {
        throw std::invalid_argument("--series must be in 1..20");
    }
    if (!options.standard_errors)
    {
        throw std::invalid_argument("--standard-errors off needs ve_fit_options_set_standard_errors");
    }
    return options;
}

std::vector<ve::Spec> selected_specs(const std::string& filter)
{
    std::vector<ve::Spec> specs;
    for (const ve::Spec& spec : ve::bench::tcc_grid())
    {
        if (ve::bench::spec_label(spec).find(filter) != std::string::npos)
        {
            specs.push_back(spec);
        }
    }
    if (specs.empty())
    {
        throw std::invalid_argument("--filter " + filter + " selects no spec");
    }
    return specs;
}

// The grid's jobs, spec-major as in BM_TccGrid, on the sGARCH(1,1)-std (nu = 8) panel of BM_TccGrid.
std::vector<Record> run_grid(const Options& options, ve::Context& context, double& wall_seconds)
{
    const ve::Spec heavy_tails = ve::bench::make_spec(1, 1, 0, 0, ve::Dist::Std);
    const std::vector<double> data = ve::bench::simulate_panel(ve::bench::simulated_model_for(heavy_tails),
                                                               kObservations, options.series);
    const ve::PanelView panel{data.data(), kObservations, options.series,
                              static_cast<std::ptrdiff_t>(options.series), 1, {}, {}};
    ve::Batch batch;
    batch.specs = selected_specs(options.filter);
    batch.panel = &panel;
    for (std::size_t spec = 0; spec < batch.specs.size(); ++spec)
    {
        for (std::size_t series = 0; series < options.series; ++series)
        {
            batch.jobs.push_back({spec, series});
        }
    }
    const auto start = std::chrono::steady_clock::now();
    std::vector<ve::FitResult> results = ve::run_batch(context, batch);
    wall_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    std::vector<Record> records;
    for (std::size_t job = 0; job < results.size(); ++job)
    {
        const ve::FitJob& fit_job = batch.jobs[job];
        records.push_back({ve::bench::spec_label(batch.specs[fit_job.spec]), fit_job.series, std::move(results[job])});
    }
    return records;
}

void add(Group& group, const ve::FitResult& result)
{
    ++group.fits;
    group.seconds += result.work.seconds;
    group.optimizer_seconds += result.work.optimizer_seconds;
    group.iterations += result.work.iterations;
    group.evaluations += result.work.evaluations;
    group.converged += result.status == ve::FitStatus::Converged ? 1 : 0;
    group.max_iter += result.status == ve::FitStatus::MaxIter ? 1 : 0;
    group.other += result.status != ve::FitStatus::Converged && result.status != ve::FitStatus::MaxIter ? 1 : 0;
}

using KeyOf = std::function<std::string(const Record&)>;

std::vector<Group> group_by(const std::vector<Record>& records, const KeyOf& key_of)
{
    std::map<std::string, Group> groups;
    for (const Record& record : records)
    {
        const std::string key = key_of(record);
        groups[key].name = key;
        add(groups[key], record.result);
    }
    std::vector<Group> sorted;
    for (auto& [key, group] : groups)
    {
        sorted.push_back(std::move(group));
    }
    std::ranges::sort(sorted, std::greater<>{}, &Group::seconds);
    return sorted;
}

void print_table(const char* title, const std::vector<Group>& groups, double total_seconds, std::size_t rows)
{
    std::cout << std::format("\n{}", title) << "\n";
    std::cout << std::format("{:<28} {:>5} {:>9} {:>7} {:>9} {:>9} {:>9} {:>8} {:>10} {:>9} {:>8} {:>5}", "group", "fits",
                 "total_s", "share%", "mean_ms", "iter_mean", "eval_mean", "eval/it", "optimizer%", "converged",
                 "max_iter", "other") << "\n";
    for (const Group& group : std::span(groups).first(std::min(rows, groups.size())))
    {
        const double fits = static_cast<double>(group.fits);
        std::cout << std::format("{:<28} {:>5} {:>9.3f} {:>7.1f} {:>9.2f} {:>9.1f} {:>9.1f} {:>8.2f} {:>10.1f} {:>9} {:>8} {:>5}",
                     group.name, group.fits, group.seconds, 100.0 * group.seconds / total_seconds,
                     1000.0 * group.seconds / fits, group.iterations / fits, group.evaluations / fits,
                     group.evaluations / std::max(1.0, group.iterations),
                     100.0 * group.optimizer_seconds / group.seconds, group.converged, group.max_iter, group.other) << "\n";
    }
}

std::string order_of(const Record& record)
{
    return "p" + std::to_string(record.result.spec.p) + "q" + std::to_string(record.result.spec.q);
}

std::string arma_of(const Record& record)
{
    return "arma" + std::to_string(record.result.spec.ar) + std::to_string(record.result.spec.ma);
}

std::string dist_of(const Record& record)
{
    return ve::bench::kDistNames[static_cast<std::size_t>(record.result.spec.dist)];
}

void print_report(const std::vector<Record>& records, int threads, double wall_seconds)
{
    const Group all = group_by(records, [](const Record&) { return std::string("all"); }).front();
    std::cout << std::format("fits {}  threads {}  wall_s {:.3f}  fit_s_sum {:.3f}  pool_use {:.3f}", all.fits, threads,
                 wall_seconds, all.seconds, all.seconds / (threads * wall_seconds)) << "\n";
    print_table("By spec (top 20)", group_by(records, [](const Record& r) { return r.spec; }), all.seconds,
                kTopSpecs);
    print_table("By distribution", group_by(records, dist_of), all.seconds, records.size());
    print_table("By order", group_by(records, order_of), all.seconds, records.size());
    print_table("By ARMA", group_by(records, arma_of), all.seconds, records.size());
}

const char* status_name(ve::FitStatus status)
{
    switch (status)
    {
        case ve::FitStatus::Converged:
            return "converged";
        case ve::FitStatus::MaxIter:
            return "max_iter";
        case ve::FitStatus::Failed:
            return "failed";
        case ve::FitStatus::InsufficientData:
            return "insufficient_data";
    }
    return "unknown";
}

// One JSON object per fit, in job order; loglik is null when it is not finite.
void write_json(const std::string& path, const std::vector<Record>& records)
{
    std::ofstream out(path);
    out << "[\n";
    for (std::size_t index = 0; index < records.size(); ++index)
    {
        const Record& record = records[index];
        const ve::FitWork& work = record.result.work;
        const double loglik = record.result.loglik;
        const std::string loglik_text = std::isfinite(loglik) ? std::format("{:.17g}", loglik) : "null";
        out << std::format("  {{\"spec\": \"{}\", \"series\": {}, \"status\": \"{}\", \"seconds\": {:.9g}, "
                           "\"optimizer_seconds\": {:.9g}, \"iterations\": {}, \"evaluations\": {}, \"loglik\": {}}}{}\n",
                           record.spec, record.series, status_name(record.result.status), work.seconds,
                           work.optimizer_seconds, work.iterations, work.evaluations, loglik_text,
                           index + 1 < records.size() ? "," : "");
    }
    out << "]\n";
    if (!out)
    {
        throw std::runtime_error("could not write " + path);
    }
}

}

int main(int argc, char** argv)
{
    try
    {
        const Options options = parse(std::span(argv, static_cast<std::size_t>(argc)));
        ve::Context context;
        context.set_threads(options.threads);
        double wall_seconds = 0.0;
        const std::vector<Record> records = run_grid(options, context, wall_seconds);
        print_report(records, context.threads(), wall_seconds);
        if (!options.json.empty())
        {
            write_json(options.json, records);
        }
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "volengine_grid_profile: " << error.what() << "\n";
        return 2;
    }
}
