#pragma once

#include "optim/curvature_memory.hpp"
#include "optim/line_search.hpp"
#include "optim/vector_math.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <span>
#include <type_traits>
#include <vector>

namespace ve::optim
{

enum class Stop
{
    Converged,
    MaxIter,
    LineSearchFailed,
    NonFinite
};

struct LbfgsOptions
{
    int max_iter = 500;
    int memory = 10;
    double gtol = 1e-6;
    double ftol = 1e-12;
};

struct LbfgsResult
{
    Stop stop;
    int iterations;
    double f;
};

namespace detail
{

template <class F>
class Minimizer
{
public:
    Minimizer(F& objective, std::span<double> x, const LbfgsOptions& options)
        : objective_(objective), x_(x), options_(options), current_(x.size()),
          memory_(x.size(), static_cast<std::size_t>(options.memory)), search_(objective, x.size()),
          direction_(x.size()), step_(x.size()), gradient_change_(x.size())
    {
    }

    LbfgsResult run()
    {
        std::ranges::copy(x_, current_.x.begin());
        current_.f = objective_(std::span<const double>(current_.x), std::span<double>(current_.g));
        if (!std::isfinite(current_.f) || !all_finite(current_.g))
        {
            return finish(Stop::NonFinite, 0);
        }
        int iterations = 0;
        while (true)
        {
            if (gradient_small())
            {
                return finish(Stop::Converged, iterations);
            }
            if (iterations == options_.max_iter)
            {
                return finish(Stop::MaxIter, iterations);
            }
            Step step = iterate();
            if (step == Step::Failed)
            {
                return finish(Stop::LineSearchFailed, iterations);
            }
            ++iterations;
            if (step == Step::Flat)
            {
                return finish(Stop::Converged, iterations);
            }
        }
    }

private:
    enum class Step
    {
        Progressed,
        Flat,
        Failed
    };

    LbfgsResult finish(Stop stop, int iterations)
    {
        std::ranges::copy(current_.x, x_.begin());
        return {stop, iterations, current_.f};
    }

    double scale() const
    {
        return std::max(1.0, std::abs(current_.f));
    }

    bool gradient_small() const
    {
        return max_abs(current_.g) <= options_.gtol * scale();
    }

    bool search_along_quasi_newton_direction()
    {
        double initial_step = 1.0;
        if (!memory_.empty())
        {
            memory_.apply(current_.g, direction_);
        }
        if (memory_.empty() || dot(current_.g, direction_) >= 0.0)
        {
            initial_step = 1.0 / std::max(1.0, max_abs(current_.g));
            std::ranges::transform(current_.g, direction_.begin(), [](double value) { return -value; });
        }
        return search_.run(current_, direction_, initial_step);
    }

    Step iterate()
    {
        bool found = search_along_quasi_newton_direction();
        if (!found && !memory_.empty())
        {
            memory_.clear();
            found = search_along_quasi_newton_direction();
        }
        if (!found)
        {
            return Step::Failed;
        }
        return advance_to_accepted();
    }

    Step advance_to_accepted()
    {
        const Point& accepted = search_.accepted();
        for (std::size_t i = 0; i < x_.size(); ++i)
        {
            step_[i] = accepted.x[i] - current_.x[i];
            gradient_change_[i] = accepted.g[i] - current_.g[i];
        }
        memory_.push(step_, gradient_change_);
        double decrease = std::abs(current_.f - accepted.f);
        current_.x = accepted.x;
        current_.g = accepted.g;
        current_.f = accepted.f;
        if (search_.accepted_by_fallback())
        {
            return Step::Progressed;
        }
        return decrease <= options_.ftol * scale() ? Step::Flat : Step::Progressed;
    }

    F& objective_;
    std::span<double> x_;
    const LbfgsOptions& options_;
    Point current_;
    CurvatureMemory memory_;
    LineSearch<F> search_;
    std::vector<double> direction_;
    std::vector<double> step_;
    std::vector<double> gradient_change_;
};

}

// f(x, g) returns the value at x and fills g with the gradient. On return x holds the best finite point found.
template <class F>
LbfgsResult lbfgs_minimize(F&& f, std::span<double> x, const LbfgsOptions& options)
{
    assert(options.memory > 0);
    assert(options.max_iter >= 0);
    detail::Minimizer<std::remove_reference_t<F>> minimizer(f, x, options);
    return minimizer.run();
}

}
