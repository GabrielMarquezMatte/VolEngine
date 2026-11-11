#pragma once

#include "core/context.hpp"
#include "core/panel.hpp"
#include "core/spec.hpp"
#include "fit/fit_result.hpp"

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

namespace ve
{

struct FitJob
{
    std::size_t spec = 0;
    std::size_t series = 0;
};

// Job i fits specs[jobs[i].spec] on series jobs[i].series of `panel`, which is borrowed.
struct Batch
{
    std::vector<Spec> specs;
    const PanelView* panel = nullptr;
    FitConfig config;
    std::vector<FitJob> jobs;
};

// The message of the first problem that makes the batch unrunnable, nothing when it can run.
std::optional<std::string> check_batch(const Batch& batch);

// Results are indexed like batch.jobs and do not depend on the thread count. Rethrows the first
// exception raised by a fit, after every started fit has finished. The batch must pass check_batch.
std::vector<FitResult> run_batch(Context& context, const Batch& batch);

}
