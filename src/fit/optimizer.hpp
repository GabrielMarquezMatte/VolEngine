#pragma once

#include "core/spec.hpp"
#include "fit/fit_result.hpp"
#include "optim/lbfgs.hpp"

#include <cassert>
#include <cstddef>

namespace ve::fit
{

// The likelihoods and parameter maps exist only for ad::Dual<8>, 16, 32 and kMaxParameterCount: body runs with the
// narrowest of them that holds k parameters. ad::variable writes grad[index] unchecked, so k must not exceed
// kMaxParameterCount.
template <class F>
decltype(auto) with_dual_width(std::size_t k, F&& body)
{
    assert(k <= kMaxParameterCount);
    if (k <= 8)
    {
        return body.template operator()<8>();
    }
    if (k <= 16)
    {
        return body.template operator()<16>();
    }
    if (k <= 32)
    {
        return body.template operator()<32>();
    }
    return body.template operator()<kMaxParameterCount>();
}

// A line search that fails, or a point that is not finite, is Failed.
inline FitStatus status_from(optim::Stop stop)
{
    switch (stop)
    {
        case optim::Stop::Converged:
            return FitStatus::Converged;
        case optim::Stop::MaxIter:
            return FitStatus::MaxIter;
        case optim::Stop::LineSearchFailed:
        case optim::Stop::NonFinite:
            return FitStatus::Failed;
    }
    return FitStatus::Failed;
}

}
