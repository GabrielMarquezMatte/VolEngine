#include "dcc/future/state.hpp"

#include "dcc/likelihood.hpp"
#include "simd/dcc_kernels.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iterator>
#include <utility>

namespace ve::dcc::future
{

DccEnd dcc_end(const StandardizedPanel& scores, const DccSpec& spec, std::span<const double> theta,
               std::size_t rows)
{
    detail::StageMoments moments = detail::stage_moments(scores, spec, Stage::Reported);
    detail::Weights<double> weights = detail::weights_of(split_parameters<double>(spec, theta), spec, moments);
    DccEnd end;
    if (!detail::in_domain(scores, spec, weights))
    {
        return end;
    }
    const std::size_t count = moments.qbar.size();
    const std::size_t score_lags = weights.a.size() / count;
    const std::size_t q_lags = weights.b.size() / count;
    assert(rows <= scores.t && rows >= std::max(score_lags, q_lags));
    end.theta.assign(theta.begin(), theta.end());
    end.qbar = moments.qbar;
    end.nbar = moments.nbar;
    detail::CorrelationRecursion<double> recursion(scores, std::move(weights), std::move(moments));
    std::vector<double> ring(q_lags * count);
    for (std::size_t t = 0; t < rows; ++t)
    {
        recursion.next();
        std::ranges::copy(recursion.q(), ring.begin() + static_cast<std::ptrdiff_t>((t % q_lags) * count));
    }
    for (std::size_t k = 0; k < q_lags; ++k)
    {
        const auto slot = ring.begin() + static_cast<std::ptrdiff_t>(((rows - 1 - k) % q_lags) * count);
        end.q.insert(end.q.end(), slot, slot + static_cast<std::ptrdiff_t>(count));
    }
    for (std::size_t k = 0; k < score_lags; ++k)
    {
        const auto row = scores.z.begin() + static_cast<std::ptrdiff_t>((rows - 1 - k) * scores.m);
        end.scores.insert(end.scores.end(), row, row + static_cast<std::ptrdiff_t>(scores.m));
    }
    return end;
}

namespace
{

detail::Weights<double> weights_at(const DccEnd& end, const DccSpec& spec)
{
    if (!end.static_r.empty())
    {
        return {};
    }
    detail::StageMoments moments{end.qbar, end.nbar, {}, {}};
    return detail::weights_of(split_parameters<double>(spec, end.theta), spec, moments);
}

}

CorrelationStepper::CorrelationStepper(const DccEnd& end, const DccSpec& spec, std::size_t m)
    : m_(m), count_(end.qbar.size()), constant_(!end.static_r.empty()), weights_(weights_at(end, spec)),
      q_ring_(end.q), score_ring_(end.scores), q_(count_), r_(constant_ ? end.static_r : std::vector<double>(m * m)),
      n_(m), scale_(m)
{
}

void CorrelationStepper::add_news(std::size_t lag)
{
    const std::span<const double> z(score_ring_.data() + (lag - 1) * m_, m_);
    const std::size_t offset = (lag - 1) * count_;
    simd::add_outer(q_, std::span<const double>(weights_.a).subspan(offset, count_), z);
    if (weights_.g.empty())
    {
        return;
    }
    for (std::size_t i = 0; i < m_; ++i)
    {
        n_[i] = z[i] < 0.0 ? z[i] : 0.0;
    }
    simd::add_outer(q_, std::span<const double>(weights_.g).subspan(offset, count_), n_);
}

void CorrelationStepper::correlation()
{
    simd::correlation_from_q(q_, m_, scale_, r_);
}

std::span<const double> CorrelationStepper::next()
{
    if (constant_)
    {
        return r_;
    }
    std::ranges::copy(weights_.intercept, q_.begin());
    for (std::size_t lag = 1; lag <= weights_.a.size() / count_; ++lag)
    {
        add_news(lag);
    }
    for (std::size_t lag = 1; lag <= weights_.b.size() / count_; ++lag)
    {
        const std::size_t offset = (lag - 1) * count_;
        simd::multiply_add(q_, std::span<const double>(weights_.b).subspan(offset, count_),
                           std::span<const double>(q_ring_).subspan(offset, count_));
    }
    correlation();
    return r_;
}

void CorrelationStepper::push(std::span<const double> scores)
{
    if (constant_)
    {
        return;
    }
    std::ranges::copy_backward(score_ring_.begin(), score_ring_.end() - static_cast<std::ptrdiff_t>(m_),
                               score_ring_.end());
    std::ranges::copy(scores, score_ring_.begin());
    std::ranges::copy_backward(q_ring_.begin(), q_ring_.end() - static_cast<std::ptrdiff_t>(count_), q_ring_.end());
    std::ranges::copy(q_, q_ring_.begin());
}

}
