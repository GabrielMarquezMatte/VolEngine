#include "dcc/likelihood/gradient.hpp"

#include "ad/dual.hpp"
#include "core/spec.hpp"
#include "dcc/density.hpp"
#include "dcc/likelihood.hpp"
#include "dcc/recursion.hpp"
#include "fit/optimizer.hpp"
#include "simd/dcc_kernels.hpp"
#include "simd/reduce.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numbers>
#include <vector>

namespace ve::dcc
{

namespace
{

constexpr double kInfinity = std::numeric_limits<double>::infinity();
constexpr std::size_t kBlock = simd::CorrelationBlock::kCapacity;

// d (weights) / d theta_p for every dynamic parameter p, packed like the weights: the intercept, then the a lags,
// the g lags and the b lags. A block that is zero for a parameter (the weight does not depend on it) is flagged and
// skipped by the recursion.
struct WeightTangents
{
    std::size_t count = 0;
    std::size_t a_lags = 0;
    std::size_t g_lags = 0;
    std::size_t b_lags = 0;
    std::vector<double> data;
    std::vector<char> nonzero;

    std::size_t blocks() const { return 1 + a_lags + g_lags + b_lags; }
    std::size_t g_block(std::size_t lag) const { return a_lags + lag; }
    std::size_t b_block(std::size_t lag) const { return a_lags + g_lags + lag; }

    std::span<const double> block(std::size_t p, std::size_t index) const
    {
        return std::span<const double>(data).subspan((p * blocks() + index) * count, count);
    }

    bool active(std::size_t p, std::size_t index) const { return nonzero[p * blocks() + index] != 0; }
};

template <std::size_t N>
void fill_block(WeightTangents& out, std::size_t p, std::size_t index, std::span<const ad::Dual<N>> weights)
{
    double* target = out.data.data() + (p * out.blocks() + index) * out.count;
    bool any = false;
    for (std::size_t k = 0; k < out.count; ++k)
    {
        target[k] = weights[k].grad[p];
        any = any || target[k] != 0.0;
    }
    out.nonzero[p * out.blocks() + index] = any ? 1 : 0;
}

// The weights with every dynamic parameter a Dual variable: their gradients are the derivatives of the weights,
// whichever the model, so no model has its own formulas here.
template <std::size_t N>
WeightTangents weight_tangents(const StandardizedPanel& panel, const DccSpec& spec, std::span<const double> theta,
                               std::size_t parameters)
{
    std::vector<ad::Dual<N>> variables(theta.size());
    for (std::size_t i = 0; i < theta.size(); ++i)
    {
        variables[i] = i < parameters ? ad::variable<N>(theta[i], i) : ad::constant<N>(theta[i]);
    }
    const detail::StageMoments moments = detail::stage_moments(panel, spec, Stage::Objective);
    const detail::Weights<ad::Dual<N>> weights = detail::weights_of(
        split_parameters(spec, std::span<const ad::Dual<N>>(variables)), spec, moments);
    WeightTangents out;
    out.count = moments.qbar.size();
    out.a_lags = weights.a.size() / out.count;
    out.g_lags = weights.g.size() / out.count;
    out.b_lags = weights.b.size() / out.count;
    out.data.assign(parameters * out.blocks() * out.count, 0.0);
    out.nonzero.assign(parameters * out.blocks(), 0);
    for (std::size_t p = 0; p < parameters; ++p)
    {
        fill_block<N>(out, p, 0, weights.intercept);
        for (std::size_t lag = 1; lag <= out.a_lags; ++lag)
        {
            fill_block<N>(out, p, lag, std::span<const ad::Dual<N>>(weights.a).subspan((lag - 1) * out.count, out.count));
        }
        for (std::size_t lag = 1; lag <= out.g_lags; ++lag)
        {
            fill_block<N>(out, p, out.g_block(lag),
                          std::span<const ad::Dual<N>>(weights.g).subspan((lag - 1) * out.count, out.count));
        }
        for (std::size_t lag = 1; lag <= out.b_lags; ++lag)
        {
            fill_block<N>(out, p, out.b_block(lag),
                          std::span<const ad::Dual<N>>(weights.b).subspan((lag - 1) * out.count, out.count));
        }
    }
    return out;
}

// dQ_t of every parameter, one after the other in the recursion's order:
//   dQ_t = dC + sum_j dA_j o ZZ_(t-j) + sum_j dG_j o NN_(t-j) + sum_k dB_k o Q_(t-k) + sum_k B_k o dQ_(t-k),
// the first four the direct dependence of the weights on the parameter, the last the recursion; a lag before the
// sample reads the presample moments and a zero dQ.
class TangentRecursion
{
public:
    TangentRecursion(const StandardizedPanel& panel, const WeightTangents& tangents, std::size_t parameters)
        : panel_(panel), tangents_(tangents), parameters_(parameters), count_(tangents.count),
          ring_(parameters * tangents.b_lags * tangents.count), current_(parameters * tangents.count),
          negative_(tangents.g_lags * panel.m)
    {
    }

    // The packed dQ_t of each parameter for t = recursion.time(), before recursion.advance().
    std::span<const double> step(const detail::CorrelationRecursion<double>& recursion)
    {
        const std::size_t t = recursion.time();
        fill_negative_scores(t);
        for (std::size_t p = 0; p < parameters_; ++p)
        {
            const std::span<double> dq = std::span<double>(current_).subspan(p * count_, count_);
            std::ranges::copy(tangents_.block(p, 0), dq.begin());
            add_direct_terms(p, recursion, dq);
            add_recursive_terms(p, recursion, dq);
        }
        keep(t);
        return current_;
    }

private:
    std::span<double> slot(std::size_t p, std::size_t t)
    {
        return std::span<double>(ring_).subspan((p * tangents_.b_lags + t % tangents_.b_lags) * count_, count_);
    }

    void fill_negative_scores(std::size_t t)
    {
        const std::size_t m = panel_.m;
        for (std::size_t lag = 1; lag <= tangents_.g_lags && lag <= t; ++lag)
        {
            for (std::size_t i = 0; i < m; ++i)
            {
                const double z = panel_.z[(t - lag) * m + i];
                negative_[(lag - 1) * m + i] = z < 0.0 ? z : 0.0;
            }
        }
    }

    // dq += w o (the scores of lag t - lag, or their presample moments).
    void add_news(std::span<double> dq, std::span<const double> w, std::size_t lag, bool negative,
                  const detail::CorrelationRecursion<double>& recursion)
    {
        const std::size_t t = recursion.time();
        const detail::StageMoments& moments = recursion.moments();
        if (t < lag)
        {
            simd::multiply_add(dq, w, negative ? moments.nn : moments.zz);
            return;
        }
        const std::size_t m = panel_.m;
        const std::span<const double> scores = negative
                                                   ? std::span<const double>(negative_).subspan((lag - 1) * m, m)
                                                   : std::span<const double>(panel_.z).subspan((t - lag) * m, m);
        simd::add_outer(dq, w, scores);
    }

    void add_direct_terms(std::size_t p, const detail::CorrelationRecursion<double>& recursion, std::span<double> dq)
    {
        for (std::size_t lag = 1; lag <= tangents_.a_lags; ++lag)
        {
            if (tangents_.active(p, lag))
            {
                add_news(dq, tangents_.block(p, lag), lag, false, recursion);
            }
        }
        for (std::size_t lag = 1; lag <= tangents_.g_lags; ++lag)
        {
            if (tangents_.active(p, tangents_.g_block(lag)))
            {
                add_news(dq, tangents_.block(p, tangents_.g_block(lag)), lag, true, recursion);
            }
        }
        for (std::size_t lag = 1; lag <= tangents_.b_lags; ++lag)
        {
            if (tangents_.active(p, tangents_.b_block(lag)))
            {
                simd::multiply_add(dq, tangents_.block(p, tangents_.b_block(lag)), recursion.lagged_q(lag));
            }
        }
    }

    void add_recursive_terms(std::size_t p, const detail::CorrelationRecursion<double>& recursion,
                             std::span<double> dq)
    {
        const std::size_t t = recursion.time();
        for (std::size_t lag = 1; lag <= tangents_.b_lags && lag <= t; ++lag)
        {
            const std::span<const double> b =
                std::span<const double>(recursion.weights().b).subspan((lag - 1) * count_, count_);
            simd::multiply_add(dq, b, slot(p, t - lag));
        }
    }

    // dQ_t takes the slot of dQ_(t - Q), which no later observation reads.
    void keep(std::size_t t)
    {
        for (std::size_t p = 0; p < parameters_; ++p)
        {
            const std::span<double> dq = std::span<double>(current_).subspan(p * count_, count_);
            std::ranges::copy(dq, slot(p, t).begin());
        }
    }

    const StandardizedPanel& panel_;
    const WeightTangents& tangents_;
    std::size_t parameters_;
    std::size_t count_;
    std::vector<double> ring_;
    std::vector<double> current_;
    std::vector<double> negative_;
};

struct DensityPartials
{
    double value;
    double d_log_det;
    double d_quadratic;
    double d_shape;
};

// log f(z; R) and its partials in log |R|, q = z' R^-1 z and the mvt shape, from the density of the likelihood
// evaluated with three Dual lanes.
class DensityDerivatives
{
public:
    DensityDerivatives(DccDist dist, std::size_t m, double shape)
        : density_{dist, static_cast<double>(m), ad::variable<3>(shape, 2), {}}
    {
        density_.constant = detail::density_constant(dist, density_.m, density_.shape);
    }

    DensityPartials at(double log_det, double quadratic) const
    {
        const ad::Dual<3> value =
            detail::log_density(density_, ad::variable<3>(log_det, 0), ad::variable<3>(quadratic, 1));
        return {value.value, value.grad[0], value.grad[1], value.grad[2]};
    }

private:
    detail::Density<ad::Dual<3>> density_;
};

// Takes the Q_t and dQ_t of kBlock consecutive observations, factors them together and pulls the density back, in
// the order of t: the value and the gradient accumulate in the order of objective<double>.
struct PassSetup
{
    const StandardizedPanel& panel;
    const DensityDerivatives& density;
    std::size_t parameters;
    // What the sum of the terms is taken from: 0, or the constant of the rmgarch mvnorm.
    double total;
};

class BlockPass
{
public:
    BlockPass(simd::CorrelationBlock& block, const PassSetup& setup, std::span<double> grad)
        : block_(block), panel_(setup.panel), density_(setup.density), grad_(grad), parameters_(setup.parameters),
          cells_(panel_.m * (panel_.m + 1) / 2), total_(setup.total), q_(kBlock * cells_),
          dq_(parameters_ * kBlock * cells_), g_(kBlock * cells_)
    {
    }

    bool failed() const { return failed_; }
    double total() const { return total_; }

    // q: the packed Q_t; dq: the packed dQ_t of each parameter, one after the other.
    void add(std::span<const double> q, std::span<const double> dq)
    {
        std::ranges::copy(q, q_.begin() + static_cast<std::ptrdiff_t>(filled_ * cells_));
        for (std::size_t p = 0; p < parameters_; ++p)
        {
            std::ranges::copy(dq.subspan(p * cells_, cells_),
                              dq_.begin() + static_cast<std::ptrdiff_t>((p * kBlock + filled_) * cells_));
        }
        if (++filled_ == kBlock)
        {
            flush();
        }
    }

    void finish() { flush(); }

private:
    void flush()
    {
        if (filled_ == 0 || failed_)
        {
            return;
        }
        std::array<double, kBlock> log_det;
        std::array<double, kBlock> quadratic;
        block_.factor(std::span<const double>(q_).first(filled_ * cells_),
                      std::span<const double>(panel_.z).subspan(first_ * panel_.m, filled_ * panel_.m),
                      std::span<double>(log_det).first(filled_), std::span<double>(quadratic).first(filled_));
        for (std::size_t lane = 0; lane < filled_; ++lane)
        {
            take_term(lane, density_.at(log_det[lane], quadratic[lane]), log_det[lane]);
        }
        if (!failed_)
        {
            pull_back();
        }
        first_ += filled_;
        filled_ = 0;
    }

    void take_term(std::size_t lane, const DensityPartials& partials, double log_det)
    {
        failed_ = failed_ || std::isnan(log_det) || !std::isfinite(partials.value);
        total_ = total_ - partials.value;
        d_log_det_[lane] = partials.d_log_det;
        d_quadratic_[lane] = partials.d_quadratic;
        d_shape_[lane] = partials.d_shape;
    }

    void pull_back()
    {
        block_.pull_back(std::span<const double>(d_log_det_).first(filled_),
                         std::span<const double>(d_quadratic_).first(filled_), std::span<double>(g_).first(filled_ * cells_));
        for (std::size_t lane = 0; lane < filled_; ++lane)
        {
            const std::span<const double> g = std::span<const double>(g_).subspan(lane * cells_, cells_);
            for (std::size_t p = 0; p < parameters_; ++p)
            {
                const std::span<const double> dq =
                    std::span<const double>(dq_).subspan((p * kBlock + lane) * cells_, cells_);
                grad_[p] -= simd::dot(g, dq);
            }
            if (grad_.size() > parameters_)
            {
                grad_[parameters_] -= d_shape_[lane];
            }
        }
    }

    simd::CorrelationBlock& block_;
    const StandardizedPanel& panel_;
    const DensityDerivatives& density_;
    std::span<double> grad_;
    std::size_t parameters_;
    std::size_t cells_;
    double total_;
    bool failed_ = false;
    std::size_t filled_ = 0;
    std::size_t first_ = 0;
    std::vector<double> q_;
    std::vector<double> dq_;
    std::vector<double> g_;
    std::array<double, kBlock> d_log_det_{};
    std::array<double, kBlock> d_quadratic_{};
    std::array<double, kBlock> d_shape_{};
};

}

bool has_fast_gradient(const DccSpec& spec)
{
    const bool model = spec.model == DccModel::Dcc || spec.model == DccModel::ADcc || spec.model == DccModel::FDcc;
    const bool dist = spec.dist == DccDist::MvNorm || spec.dist == DccDist::MvT || spec.dist == DccDist::MvLaplace;
    return model && dist;
}

double ObjectiveGradient::evaluate(const StandardizedPanel& panel, const DccSpec& spec, std::span<const double> theta,
                                   std::span<double> grad)
{
    std::ranges::fill(grad, 0.0);
    const std::size_t parameters = theta.size() - (spec.dist == DccDist::MvT ? 1 : 0);
    auto recursion = detail::recursion_of(panel, spec, split_parameters(spec, theta), Stage::Objective);
    if (!recursion)
    {
        return kInfinity;
    }
    const WeightTangents tangents = fit::with_dual_width(
        parameters, [&]<std::size_t N>() { return weight_tangents<N>(panel, spec, theta, parameters); });
    if (!block_ || block_m_ != panel.m)
    {
        block_.emplace(panel.m);
        block_m_ = panel.m;
    }
    const DensityDerivatives density(spec.dist, panel.m, spec.dist == DccDist::MvT ? theta.back() : 0.0);
    // The start of the sum, as objective<double> forms it.
    double constant = 0.0;
    if (spec.convention == Convention::Rmgarch && spec.dist == DccDist::MvNorm)
    {
        constant = -0.5 * static_cast<double>(panel.m * panel.t) * std::log(2.0 * std::numbers::pi);
    }
    TangentRecursion tangent_recursion(panel, tangents, parameters);
    BlockPass pass(*block_, {panel, density, parameters, constant}, grad);
    for (std::size_t t = 0; t < panel.t && !pass.failed(); ++t)
    {
        const std::span<const double> dq = tangent_recursion.step(*recursion);
        pass.add(recursion->advance(), dq);
    }
    pass.finish();
    if (pass.failed() || !std::isfinite(pass.total()))
    {
        std::ranges::fill(grad, 0.0);
        return kInfinity;
    }
    return pass.total();
}

}
