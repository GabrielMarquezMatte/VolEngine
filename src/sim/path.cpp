#include "sim/path.hpp"

#include "forecast/step.hpp"

#include <cmath>
#include <vector>

namespace ve::sim
{

namespace
{

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

model::RegressorMatrix row_of(const model::RegressorMatrix& matrix, std::size_t step)
{
    return {matrix.values.subspan(step * matrix.columns, matrix.columns), matrix.columns};
}

}

forecast::FutureRegressors future_row(const forecast::FutureRegressors& future, std::size_t step)
{
    return {row_of(future.mean, step), row_of(future.variance, step)};
}

Path::Path(const Start& start) : start_(start), state_(start.state), keep_(start.state.sigma2.size()) {}

void Path::draw_presample(Stream& stream, const Sampler& sampler)
{
    for (std::size_t i = 0; i < start_.presample; ++i)
    {
        state_.residuals[i] = start_.presample_sigma * sampler.draw(stream);
    }
}

Path::Draw Path::step(const forecast::FutureRegressors& row, double z)
{
    forecast::detail::Inputs in{state_, start_.layout, start_.theta, row, 1, start_.moments};
    forecast::detail::Step step = forecast::detail::first_step(in);
    double sigma = std::sqrt(step.sigma2);
    double e = sigma * z;
    double r = step.mean + e;
    state_.residuals.push_back(e);
    state_.sigma2.push_back(step.sigma2);
    if (start_.layout.model == Model::CsGarch)
    {
        state_.permanent.push_back(step.permanent);
    }
    state_.deviations.push_back(r - step.level);
    if (state_.sigma2.size() >= 2 * keep_)
    {
        trim(state_, keep_, start_.layout.has_arfima());
    }
    return {r, sigma};
}

}
