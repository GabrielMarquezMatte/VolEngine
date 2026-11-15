#pragma once

// The views of a natural-space theta that the likelihood and its adjoint read: the ARMA, sGARCH, gjrGARCH and
// eGARCH parameters, and the iGARCH beta. Internal to model/.

#include "dist/moments.hpp"
#include "garch/egarch.hpp"
#include "garch/gjrgarch.hpp"
#include "garch/sgarch.hpp"
#include "mean/arma.hpp"
#include "model/layout.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <span>

namespace ve::model
{

template <class T>
std::span<const T> block(std::span<const T> theta, std::size_t begin, int count)
{
    return theta.subspan(begin, static_cast<std::size_t>(count));
}

template <class T>
mean::ArmaParameters<T> arma_parameters(std::span<const T> theta, const Layout& layout)
{
    T mu{0.0};
    if (layout.include_mean)
    {
        mu = theta[layout.mu()];
    }
    return {mu, block(theta, layout.ar_begin(), layout.ar), block(theta, layout.ma_begin(), layout.ma)};
}

template <class T>
garch::SGarchParameters<T> sgarch_parameters(std::span<const T> theta, const Layout& layout)
{
    return {theta[layout.omega()], block(theta, layout.alpha_begin(), layout.arch),
            block(theta, layout.beta_begin(), layout.garch)};
}

template <class T>
garch::EGarchParameters<T> egarch_parameters(std::span<const T> theta, const Layout& layout)
{
    garch::SGarchParameters<T> s = sgarch_parameters(theta, layout);
    return {s.omega, s.alpha, s.beta, block(theta, layout.gamma_begin(), layout.gamma),
            dist::abs_moment(layout.dist, shape_parameters_of(theta, layout), T{1.0})};
}

template <class T>
garch::GjrGarchParameters<T> gjrgarch_parameters(std::span<const T> theta, const Layout& layout)
{
    garch::SGarchParameters<T> s = sgarch_parameters(theta, layout);
    return {s.omega, s.alpha, s.beta, block(theta, layout.gamma_begin(), layout.gamma)};
}

// The sGARCH beta with beta_q = 1 - sum alpha - sum_{j<q} beta_j, written to storage; the stored beta_q is never
// read.
template <class T>
std::span<const T> igarch_beta(const garch::SGarchParameters<T>& p, std::array<T, kMaxOrder>& storage)
{
    std::ranges::copy(p.beta, storage.begin());
    T last{1.0};
    for (const T& alpha : p.alpha)
    {
        last = last - alpha;
    }
    for (const T& earlier : p.beta.first(p.beta.size() - 1))
    {
        last = last - earlier;
    }
    storage[p.beta.size() - 1] = last;
    return std::span<const T>(storage).first(p.beta.size());
}

}
