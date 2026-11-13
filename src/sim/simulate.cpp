#include "sim/simulate.hpp"

#include "core/blocks.hpp"
#include "forecast/step.hpp"
#include "sim/philox.hpp"
#include "sim/samplers.hpp"

#include <cmath>
#include <cstddef>
#include <vector>

namespace ve::sim
{

namespace
{

constexpr std::size_t kBlocksPerThread = 4;

model::RegressorMatrix row_of(const model::RegressorMatrix& matrix, std::size_t step)
{
    return {matrix.values.subspan(step * matrix.columns, matrix.columns), matrix.columns};
}

void drop_front(std::vector<double>& values, std::size_t keep)
{
    if (values.size() > keep)
    {
        values.erase(values.begin(), values.end() - static_cast<std::ptrdiff_t>(keep));
    }
}

// The rules read only the last L entries of the state, but the ARFIMA mean every deviation: once the state holds
// twice its start, its start-sized tail is all a step needs. Without this a step of eGARCH, apARCH and fGARCH,
// which transform the whole state, would cost O(h).
void trim(forecast::ForecastState& state, std::size_t keep, bool arfima)
{
    drop_front(state.residuals, keep);
    drop_front(state.sigma2, keep);
    drop_front(state.permanent, keep);
    if (!arfima)
    {
        drop_front(state.deviations, keep);
    }
}

// Path p writes only row p of out.
void simulate_path(const Start& start, const Request& request, const Sampler& sampler, std::size_t path,
                   Simulation& out)
{
    Stream stream(request.seed, path);
    forecast::ForecastState state = start.state;
    for (std::size_t i = 0; i < start.presample; ++i)
    {
        state.residuals[i] = start.presample_sigma * sampler.draw(stream);
    }
    bool component = start.layout.model == Model::CsGarch;
    std::size_t keep = start.state.sigma2.size();
    for (std::size_t h = 0; h < request.horizon; ++h)
    {
        forecast::FutureRegressors row{row_of(request.future.mean, h), row_of(request.future.variance, h)};
        forecast::detail::Inputs in{state, start.layout, start.theta, row, 1, start.moments};
        forecast::detail::Step step = forecast::detail::first_step(in);
        double sigma = std::sqrt(step.sigma2);
        double e = sigma * sampler.draw(stream);
        double r = step.mean + e;
        out.returns[path * request.horizon + h] = r;
        out.sigma[path * request.horizon + h] = sigma;
        state.residuals.push_back(e);
        state.sigma2.push_back(step.sigma2);
        if (component)
        {
            state.permanent.push_back(step.permanent);
        }
        state.deviations.push_back(r - step.level);
        if (state.sigma2.size() >= 2 * keep)
        {
            trim(state, keep, start.layout.has_arfima());
        }
    }
}

void run_blocks(BS::light_thread_pool& pool, int threads, const Start& start, const Request& request,
                Simulation& out)
{
    Sampler sampler(start.layout.dist, model::shape_parameters_of<double>(start.theta, start.layout));
    auto run_block = [&](std::size_t first, std::size_t last)
    {
        for (std::size_t path = first; path < last; ++path)
        {
            simulate_path(start, request, sampler, path, out);
        }
    };
    run_in_blocks(pool, threads, request.paths, kBlocksPerThread, run_block);
}

}

Simulation simulate(Context& context, const Start& start, const Request& request)
{
    Simulation out{request.paths, request.horizon, std::vector<double>(request.paths * request.horizon),
                   std::vector<double>(request.paths * request.horizon)};
    context.with_pool([&](BS::light_thread_pool& pool, int threads)
                      { run_blocks(pool, threads, start, request, out); });
    return out;
}

}
