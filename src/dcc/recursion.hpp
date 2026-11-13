#pragma once

#include "ad/dual.hpp"
#include "dcc/panel.hpp"
#include "dcc/spec.hpp"
#include "linalg/cholesky.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <span>
#include <utility>
#include <vector>

namespace ve::dcc
{

// What the presample is for. Standard has one presample and ignores it. Rmgarch: Objective reads z_s = n_s = 0
// for s <= 0 and the FDCC uses Qbar = z'z/T; Reported reads z_s = n_s = (1, ..., 1) and Qbar = cov(z).
enum class Stage
{
    Objective,
    Reported
};

// Natural DCC parameters: a_1..a_P, b_1..b_Q, g_1..g_P (aDCC) and the mvt shape; the FDCC has a and b per group.
template <class T>
struct DccParameters
{
    std::span<const T> a;
    std::span<const T> b;
    std::span<const T> g;
    T shape;
};

namespace detail
{

// Entry (i, j), j <= i, of a symmetric matrix kept as its packed lower triangle.
inline std::size_t packed_index(std::size_t i, std::size_t j)
{
    return i * (i + 1) / 2 + j;
}

inline std::vector<double> packed(std::span<const double> full, std::size_t m)
{
    std::vector<double> out(m * (m + 1) / 2);
    for (std::size_t i = 0; i < m; ++i)
    {
        for (std::size_t j = 0; j <= i; ++j)
        {
            out[packed_index(i, j)] = full[i * m + j];
        }
    }
    return out;
}

inline bool is_rmgarch(const DccSpec& spec)
{
    return spec.convention == Convention::Rmgarch;
}

// The unconditional matrices of a stage, packed, and what the recursion reads for s <= 0: Q_s = qbar, z_s z_s' =
// zz and n_s n_s' = nn (nbar and nn are empty unless the model is the aDCC).
struct StageMoments
{
    std::vector<double> qbar;
    std::vector<double> nbar;
    std::vector<double> zz;
    std::vector<double> nn;
};

inline StageMoments stage_moments(const StandardizedPanel& panel, const DccSpec& spec, Stage stage)
{
    const bool rmgarch = is_rmgarch(spec);
    const bool raw_qbar = rmgarch && spec.model == DccModel::FDcc && stage == Stage::Objective;
    StageMoments moments{packed(panel.qbar, panel.m), {}, {}, {}};
    if (raw_qbar)
    {
        moments.qbar = packed(panel.qbar_raw, panel.m);
    }
    if (spec.model == DccModel::ADcc)
    {
        moments.nbar = packed(rmgarch ? panel.nbar_centered : panel.nbar, panel.m);
    }
    moments.zz = moments.qbar;
    moments.nn = moments.nbar;
    if (rmgarch)
    {
        const double fill = stage == Stage::Reported ? 1.0 : 0.0;
        std::ranges::fill(moments.zz, fill);
        std::ranges::fill(moments.nn, fill);
    }
    return moments;
}

// Q_t = C + sum_j A_j o (z z')_(t-j) + sum_j G_j o (n n')_(t-j) + sum_k B_k o Q_(t-k), each weight packed and the
// lags one after the other: the DCC repeats its scalar weights, the FDCC has the products of its group weights.
template <class T>
struct Weights
{
    std::vector<T> intercept;
    std::vector<T> a;
    std::vector<T> g;
    std::vector<T> b;
};

template <class T>
T sum_of(std::span<const T> x)
{
    T sum{0.0};
    for (const T& v : x)
    {
        sum = sum + v;
    }
    return sum;
}

template <class T>
std::vector<T> repeated(std::span<const T> weights, std::size_t count)
{
    std::vector<T> out;
    out.reserve(weights.size() * count);
    for (const T& w : weights)
    {
        out.insert(out.end(), count, w);
    }
    return out;
}

// C = (1 - sum a - sum b) Qbar - (sum g) Nbar.
template <class T>
Weights<T> dcc_weights(const DccParameters<T>& theta, const StageMoments& moments)
{
    const std::size_t count = moments.qbar.size();
    Weights<T> weights{std::vector<T>(count), repeated(theta.a, count), repeated(theta.g, count),
                       repeated(theta.b, count)};
    const T qbar_weight = 1.0 - sum_of(theta.a) - sum_of(theta.b);
    const T nbar_weight = sum_of(theta.g);
    for (std::size_t k = 0; k < count; ++k)
    {
        weights.intercept[k] = qbar_weight * moments.qbar[k];
    }
    for (std::size_t k = 0; k < moments.nbar.size(); ++k)
    {
        weights.intercept[k] = weights.intercept[k] - nbar_weight * moments.nbar[k];
    }
    return weights;
}

// alpha = S a and beta = S b with S the group selection; C = (11' - alpha alpha' - beta beta') o Qbar.
template <class T>
Weights<T> fdcc_weights(const DccParameters<T>& theta, std::span<const int> groups, const StageMoments& moments)
{
    const std::size_t count = moments.qbar.size();
    Weights<T> weights{std::vector<T>(count), std::vector<T>(count), {}, std::vector<T>(count)};
    for (std::size_t i = 0; i < groups.size(); ++i)
    {
        const auto gi = static_cast<std::size_t>(groups[i] - 1);
        for (std::size_t j = 0; j <= i; ++j)
        {
            const auto gj = static_cast<std::size_t>(groups[j] - 1);
            const std::size_t k = packed_index(i, j);
            weights.a[k] = theta.a[gi] * theta.a[gj];
            weights.b[k] = theta.b[gi] * theta.b[gj];
            weights.intercept[k] = (1.0 - weights.a[k] - weights.b[k]) * moments.qbar[k];
        }
    }
    return weights;
}

template <class T>
Weights<T> weights_of(const DccParameters<T>& theta, const DccSpec& spec, const StageMoments& moments)
{
    if (spec.model == DccModel::FDcc)
    {
        return fdcc_weights(theta, std::span<const int>(spec.groups), moments);
    }
    return dcc_weights(theta, moments);
}

// The aDCC needs a finite delta of its convention (NaN when Qbar is not positive definite); the FDCC intercept
// must be positive definite, since the reparametrisation does not impose it.
template <class T>
bool in_domain(const StandardizedPanel& panel, const DccSpec& spec, const Weights<T>& weights)
{
    if (spec.model == DccModel::ADcc)
    {
        return std::isfinite(is_rmgarch(spec) ? panel.delta_centered : panel.delta);
    }
    if (spec.model != DccModel::FDcc)
    {
        return true;
    }
    const std::size_t m = panel.m;
    std::vector<double> intercept(m * m);
    for (std::size_t i = 0; i < m; ++i)
    {
        for (std::size_t j = 0; j <= i; ++j)
        {
            intercept[i * m + j] = ad::value_of(weights.intercept[packed_index(i, j)]);
        }
    }
    return linalg::cholesky_factor(intercept, m);
}

// Q_t one observation after the other, from t = 0, and R_t = diag(Q_t)^-1/2 Q_t diag(Q_t)^-1/2 with an exact
// unit diagonal and exact symmetry.
template <class T>
class CorrelationRecursion
{
public:
    CorrelationRecursion(const StandardizedPanel& panel, Weights<T> weights, StageMoments moments)
        : panel_(panel), weights_(std::move(weights)), moments_(std::move(moments)), count_(moments_.qbar.size()),
          history_(weights_.b.size()), q_(count_), z_products_(count_), n_products_(count_), n_(panel.m),
          scale_(panel.m), r_(panel.m * panel.m)
    {
    }

    std::span<const T> next()
    {
        std::ranges::copy(weights_.intercept, q_.begin());
        for (std::size_t lag = 1; lag <= weights_.a.size() / count_; ++lag)
        {
            add_news(lag);
        }
        for (std::size_t lag = 1; lag <= weights_.b.size() / count_; ++lag)
        {
            add_lagged_q(lag);
        }
        store_q();
        correlation();
        ++t_;
        return r_;
    }

private:
    template <class U>
    void add_weighted(std::span<const T> weights, std::span<const U> x)
    {
        for (std::size_t k = 0; k < count_; ++k)
        {
            q_[k] = q_[k] + weights[k] * x[k];
        }
    }

    std::span<const T> lag_block(const std::vector<T>& weights, std::size_t lag) const
    {
        return std::span<const T>(weights).subspan((lag - 1) * count_, count_);
    }

    void fill_products(std::size_t s)
    {
        const std::size_t m = panel_.m;
        const std::span<const double> z(panel_.z.data() + s * m, m);
        for (std::size_t i = 0; i < m; ++i)
        {
            n_[i] = z[i] < 0.0 ? z[i] : 0.0;
            for (std::size_t j = 0; j <= i; ++j)
            {
                z_products_[packed_index(i, j)] = z[i] * z[j];
                n_products_[packed_index(i, j)] = n_[i] * n_[j];
            }
        }
    }

    void add_news(std::size_t lag)
    {
        const bool presample = t_ < lag;
        if (!presample)
        {
            fill_products(t_ - lag);
        }
        const std::vector<double>& zz = presample ? moments_.zz : z_products_;
        const std::vector<double>& nn = presample ? moments_.nn : n_products_;
        add_weighted(lag_block(weights_.a, lag), std::span<const double>(zz));
        if (!weights_.g.empty())
        {
            add_weighted(lag_block(weights_.g, lag), std::span<const double>(nn));
        }
    }

    void add_lagged_q(std::size_t lag)
    {
        if (t_ < lag)
        {
            add_weighted(lag_block(weights_.b, lag), std::span<const double>(moments_.qbar));
            return;
        }
        const std::size_t slots = weights_.b.size() / count_;
        const std::span<const T> previous = std::span<const T>(history_).subspan(((t_ - lag) % slots) * count_, count_);
        add_weighted(lag_block(weights_.b, lag), previous);
    }

    // Q_t takes the slot of Q_(t - Q), the one lag no later observation reads.
    void store_q()
    {
        const std::size_t slots = weights_.b.size() / count_;
        std::ranges::copy(q_, history_.begin() + static_cast<std::ptrdiff_t>((t_ % slots) * count_));
    }

    void correlation()
    {
        using std::sqrt;
        const std::size_t m = panel_.m;
        for (std::size_t i = 0; i < m; ++i)
        {
            scale_[i] = 1.0 / sqrt(q_[packed_index(i, i)]);
        }
        for (std::size_t i = 0; i < m; ++i)
        {
            r_[i * m + i] = T{1.0};
            for (std::size_t j = 0; j < i; ++j)
            {
                r_[i * m + j] = q_[packed_index(i, j)] * scale_[i] * scale_[j];
                r_[j * m + i] = r_[i * m + j];
            }
        }
    }

    const StandardizedPanel& panel_;
    Weights<T> weights_;
    StageMoments moments_;
    std::size_t count_;
    std::size_t t_ = 0;
    std::vector<T> history_;
    std::vector<T> q_;
    std::vector<double> z_products_;
    std::vector<double> n_products_;
    std::vector<double> n_;
    std::vector<T> scale_;
    std::vector<T> r_;
};

}

// Calls visit(t, R_t) for t = 0..T-1, R_t row-major m x m, under the convention of spec and the presample of
// stage. Returns false, without visiting, when theta is outside the domain: an aDCC whose delta (of the
// convention) is not finite, or an FDCC whose intercept is not positive definite.
template <class T, class Visit>
bool correlation_path(const StandardizedPanel& panel, const DccSpec& spec, const DccParameters<T>& theta, Stage stage,
                      Visit&& visit)
{
    detail::StageMoments moments = detail::stage_moments(panel, spec, stage);
    detail::Weights<T> weights = detail::weights_of(theta, spec, moments);
    if (!detail::in_domain(panel, spec, weights))
    {
        return false;
    }
    detail::CorrelationRecursion<T> recursion(panel, std::move(weights), std::move(moments));
    for (std::size_t t = 0; t < panel.t; ++t)
    {
        visit(t, recursion.next());
    }
    return true;
}

}
