#include "dcc/future/simulate.hpp"

#include "core/blocks.hpp"
#include "dcc/copula/margins.hpp"
#include "dcc/future/state.hpp"
#include "dcc/parameter_map.hpp"
#include "dist/cdf.hpp"
#include "linalg/cholesky.hpp"
#include "model/layout.hpp"
#include "sim/path.hpp"
#include "sim/philox.hpp"
#include "sim/samplers.hpp"
#include "sim/start.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <span>

namespace ve::dcc::future
{

namespace
{

constexpr std::size_t kBlocksPerThread = 4;
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

// How a copula turns the uniform of a series into its standardized shock: the quantile of the distribution of the
// fit, or of the standardized residuals of the sample (sorted) for the empirical margins.
struct Margin
{
    Dist dist = Dist::Norm;
    dist::ShapeParameters<double> parameters{};
    std::vector<double> sorted{};
};

double margin_quantile(const Margin& margin, double u)
{
    if (margin.sorted.empty())
    {
        return dist::quantile(margin.dist, margin.parameters, u);
    }
    const auto count = static_cast<double>(margin.sorted.size());
    const double rank = std::min(std::ceil(u * (count + 1.0)), count);
    return margin.sorted[static_cast<std::size_t>(rank) - 1];
}

// What every path reads: the start of each series, the margins of a copula, and the shock's distribution.
struct Setup
{
    const DccFitResult& fit;
    const DccRequest& request;
    std::vector<sim::Start> starts;
    std::vector<Margin> margins;
    double shape = 0.0;
};

Margin margin_of(const DccFitResult& fit, std::size_t series)
{
    const SeriesEnd& end = (*fit.series_end)[series];
    const model::Layout layout = model::layout_of(end.spec);
    Margin margin;
    if (fit.spec.copula == DccCopula::Parametric)
    {
        margin.dist = layout.dist;
        margin.parameters = model::shape_parameters_of<double>(end.estimate, layout);
        return margin;
    }
    margin.sorted.resize(fit.panel->t);
    for (std::size_t t = 0; t < margin.sorted.size(); ++t)
    {
        margin.sorted[t] = fit.panel->z[t * fit.series + series];
    }
    std::ranges::sort(margin.sorted);
    return margin;
}

std::vector<Margin> margins_of(const DccFitResult& fit)
{
    std::vector<Margin> margins;
    if (fit.spec.copula == DccCopula::None)
    {
        return margins;
    }
    for (std::size_t i = 0; i < fit.series; ++i)
    {
        margins.push_back(margin_of(fit, i));
    }
    return margins;
}

Setup setup_of(const DccFitResult& fit, const DccRequest& request)
{
    Setup setup{fit, request, {}, margins_of(fit), 0.0};
    for (const SeriesEnd& end : *fit.series_end)
    {
        setup.starts.push_back(sim::start_of(end.spec, end.estimate, end.state));
    }
    const DccLayout layout = layout_of(fit.spec, fit.series);
    if (layout.has_shape)
    {
        setup.shape = fit.end.theta[layout.shape()];
    }
    return setup;
}

// One path: its stream, the correlation stepper and a Path per series, with the buffers of a step.
class PathRun
{
public:
    PathRun(const Setup& setup, std::size_t path)
        : setup_(setup), stream_(setup.request.seed, path), path_(path),
          stepper_(setup.fit.end, setup.fit.spec, setup.fit.series), factor_(m() * m()), draws_(m()), shock_(m()),
          z_(m())
    {
        for (const sim::Start& start : setup.starts)
        {
            series_.emplace_back(start);
        }
    }

    void run(DccSimulation& out)
    {
        for (std::size_t h = 0; h < setup_.request.horizon; ++h)
        {
            step(h, out);
        }
    }

private:
    std::size_t m() const { return setup_.fit.series; }

    // eps = c L x, with x ~ N(0, I) drawn first and then the mixing variable.
    void draw_shock(std::span<const double> r)
    {
        const std::size_t size = m();
        for (double& x : draws_)
        {
            x = sim::normal(stream_);
        }
        double scale = 1.0;
        if (setup_.fit.spec.dist == DccDist::MvT)
        {
            scale = std::sqrt((setup_.shape - 2.0) / (2.0 * sim::gamma(stream_, 0.5 * setup_.shape)));
        }
        if (setup_.fit.spec.dist == DccDist::MvLaplace)
        {
            scale = std::sqrt(sim::gamma(stream_, 1.0));
        }
        std::ranges::copy(r, factor_.begin());
        const bool positive_definite = linalg::cholesky_factor(std::span<double>(factor_), size);
        for (std::size_t i = 0; i < size; ++i)
        {
            double sum = 0.0;
            for (std::size_t j = 0; j <= i; ++j)
            {
                sum += factor_[i * size + j] * draws_[j];
            }
            shock_[i] = positive_definite ? scale * sum : kNaN;
        }
    }

    // The standardized shock of each series: eps itself, or the quantile of its margin at the copula's uniform.
    void standardize()
    {
        if (setup_.margins.empty())
        {
            z_ = shock_;
            return;
        }
        for (std::size_t i = 0; i < m(); ++i)
        {
            double u = dist::cdf(Dist::Norm, {}, shock_[i]);
            if (setup_.fit.spec.dist == DccDist::MvT)
            {
                u = dist::cdf(Dist::Std, {.shape = setup_.shape}, shock_[i]);
            }
            u = std::clamp(u, copula::kUniformFloor, 1.0 - copula::kUniformFloor);
            z_[i] = margin_quantile(setup_.margins[i], u);
        }
    }

    void step(std::size_t h, DccSimulation& out)
    {
        const std::size_t size = m();
        const std::size_t cell = (path_ * setup_.request.horizon + h);
        const std::span<const double> r = stepper_.next();
        draw_shock(r);
        standardize();
        for (std::size_t i = 0; i < size; ++i)
        {
            const forecast::FutureRegressors row = sim::future_row(setup_.request.future[i], h);
            const sim::Path::Draw draw = series_[i].step(row, z_[i]);
            out.returns[cell * size + i] = draw.r;
            out.sigma[cell * size + i] = draw.sigma;
        }
        if (setup_.request.correlations)
        {
            store_correlations(r, cell, out);
        }
        stepper_.push(setup_.margins.empty() ? z_ : shock_);
    }

    void store_correlations(std::span<const double> r, std::size_t cell, DccSimulation& out) const
    {
        const std::size_t size = m();
        for (std::size_t k = 0; k < size * size; ++k)
        {
            const double sigma_row = out.sigma[cell * size + k / size];
            const double sigma_column = out.sigma[cell * size + k % size];
            out.r[cell * size * size + k] = r[k];
            out.h[cell * size * size + k] = sigma_row * r[k] * sigma_column;
        }
    }

    const Setup& setup_;
    sim::Stream stream_;
    std::size_t path_;
    CorrelationStepper stepper_;
    std::vector<sim::Path> series_;
    std::vector<double> factor_;
    std::vector<double> draws_;
    std::vector<double> shock_;
    std::vector<double> z_;
};

}

DccSimulation simulate_dcc(Context& context, const DccFitResult& fit, const DccRequest& request)
{
    const std::size_t m = fit.series;
    DccSimulation out{request.paths, request.horizon, m, {}, {}, {}, {}};
    out.returns.resize(request.paths * request.horizon * m);
    out.sigma.resize(out.returns.size());
    if (request.correlations)
    {
        out.r.resize(request.paths * request.horizon * m * m);
        out.h.resize(out.r.size());
    }
    const Setup setup = setup_of(fit, request);
    auto run_block = [&](std::size_t first, std::size_t last)
    {
        for (std::size_t path = first; path < last; ++path)
        {
            PathRun(setup, path).run(out);
        }
    };
    context.with_pool([&](BS::light_thread_pool& pool, int threads)
                      { run_in_blocks(pool, threads, request.paths, kBlocksPerThread, run_block); });
    return out;
}

}
