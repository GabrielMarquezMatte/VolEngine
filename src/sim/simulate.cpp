#include "sim/simulate.hpp"

#include "core/blocks.hpp"
#include "sim/path.hpp"
#include "sim/philox.hpp"
#include "sim/samplers.hpp"

#include <cstddef>
#include <vector>

namespace ve::sim
{

namespace
{

constexpr std::size_t kBlocksPerThread = 4;

// Path p writes only row p of out: every step, or the running sum of the returns at each checkpoint.
void simulate_path(const Start& start, const Request& request, const Sampler& sampler, std::size_t path,
                   Simulation& out)
{
    Stream stream(request.seed, path);
    Path walker(start);
    walker.draw_presample(stream, sampler);
    const std::size_t count = request.checkpoints.size();
    double sum = 0.0;
    std::size_t next = 0;
    for (std::size_t h = 0; h < request.horizon; ++h)
    {
        Path::Draw draw = walker.step(future_row(request.future, h), sampler.draw(stream));
        if (count == 0)
        {
            out.returns[path * request.horizon + h] = draw.r;
            out.sigma[path * request.horizon + h] = draw.sigma;
            continue;
        }
        sum += draw.r;
        if (h + 1 == request.checkpoints[next])
        {
            out.cumulative[path * count + next] = sum;
            ++next;
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
            throw_if_cancelled(request.cancel);
            simulate_path(start, request, sampler, path, out);
        }
    };
    run_in_blocks(pool, threads, request.paths, kBlocksPerThread, run_block);
}

}

Simulation simulate(Context& context, const Start& start, const Request& request)
{
    Simulation out{request.paths, request.horizon, {}, {}, request.checkpoints, {}};
    if (request.checkpoints.empty())
    {
        out.returns.resize(request.paths * request.horizon);
        out.sigma.resize(out.returns.size());
    }
    out.cumulative.resize(request.paths * request.checkpoints.size());
    context.with_pool([&](BS::light_thread_pool& pool, int threads)
                      { run_blocks(pool, threads, start, request, out); });
    return out;
}

}
