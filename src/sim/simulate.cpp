#include "sim/simulate.hpp"

#include "forecast/step.hpp"
#include "sim/philox.hpp"
#include "sim/samplers.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace ve::sim
{

namespace
{

constexpr std::size_t kBlocksPerThread = 4;

model::RegressorMatrix row_of(const model::RegressorMatrix& matrix, std::size_t step)
{
    return {matrix.values.subspan(step * matrix.columns, matrix.columns), matrix.columns};
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
    }
}

void run_blocks(BS::light_thread_pool& pool, int threads, const Start& start, const Request& request,
                Simulation& out)
{
    Sampler sampler(start.layout.dist, model::shape_parameters_of<double>(start.theta, start.layout));
    std::size_t blocks = std::min(request.paths, kBlocksPerThread * static_cast<std::size_t>(threads));
    auto run_block = [&](std::size_t first, std::size_t last)
    {
        for (std::size_t path = first; path < last; ++path)
        {
            simulate_path(start, request, sampler, path, out);
        }
    };
    try
    {
        auto futures = pool.submit_blocks(std::size_t{0}, request.paths, run_block, blocks);
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

Simulation simulate(Context& context, const Start& start, const Request& request)
{
    Simulation out{request.paths, request.horizon, std::vector<double>(request.paths * request.horizon),
                   std::vector<double>(request.paths * request.horizon)};
    context.with_pool([&](BS::light_thread_pool& pool, int threads)
                      { run_blocks(pool, threads, start, request, out); });
    return out;
}

}
