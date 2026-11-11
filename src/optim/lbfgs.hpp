#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <numeric>
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

constexpr double kSufficientDecrease = 1e-4;
constexpr double kCurvatureDecrease = 0.9;
constexpr int kMaxLineSearchTrials = 30;
constexpr double kPositiveCurvatureTolerance = 1e-10;

inline double dot(std::span<const double> a, std::span<const double> b)
{
    return std::inner_product(a.begin(), a.end(), b.begin(), 0.0);
}

inline double max_abs(std::span<const double> values)
{
    double result = 0.0;
    for (double value : values)
    {
        result = std::max(result, std::abs(value));
    }
    return result;
}

inline bool all_finite(std::span<const double> values)
{
    return std::ranges::all_of(values, [](double value) { return std::isfinite(value); });
}

struct Point
{
    std::vector<double> x;
    std::vector<double> g;
    double f = 0.0;
    double alpha = 0.0;
    double slope = 0.0;

    explicit Point(std::size_t dimension) : x(dimension), g(dimension)
    {
    }
};

class CurvatureMemory
{
public:
    CurvatureMemory(std::size_t dimension, std::size_t capacity)
        : capacity_(capacity), s_(capacity, std::vector<double>(dimension)),
          y_(capacity, std::vector<double>(dimension)), rho_(capacity), weights_(capacity)
    {
        assert(capacity > 0);
    }

    bool empty() const
    {
        return count_ == 0;
    }

    void clear()
    {
        count_ = 0;
        oldest_ = 0;
    }

    void push(std::span<const double> s, std::span<const double> y)
    {
        double curvature = dot(s, y);
        double scale = std::sqrt(dot(s, s) * dot(y, y));
        if (!(curvature > kPositiveCurvatureTolerance * scale))
        {
            return;
        }
        std::size_t slot = (oldest_ + count_) % capacity_;
        if (count_ == capacity_)
        {
            slot = oldest_;
            oldest_ = (oldest_ + 1) % capacity_;
            --count_;
        }
        std::ranges::copy(s, s_[slot].begin());
        std::ranges::copy(y, y_[slot].begin());
        rho_[slot] = 1.0 / curvature;
        gamma_ = curvature / dot(y, y);
        ++count_;
    }

    void apply(std::span<const double> g, std::span<double> direction)
    {
        std::ranges::copy(g, direction.begin());
        for (std::size_t back = 0; back < count_; ++back)
        {
            std::size_t slot = newest_slot(back);
            weights_[slot] = rho_[slot] * dot(s_[slot], direction);
            add_scaled(-weights_[slot], y_[slot], direction);
        }
        for (double& value : direction)
        {
            value *= gamma_;
        }
        for (std::size_t forward = 0; forward < count_; ++forward)
        {
            std::size_t slot = (oldest_ + forward) % capacity_;
            double correction = weights_[slot] - rho_[slot] * dot(y_[slot], direction);
            add_scaled(correction, s_[slot], direction);
        }
        for (double& value : direction)
        {
            value = -value;
        }
    }

private:
    std::size_t newest_slot(std::size_t back) const
    {
        return (oldest_ + count_ - 1 - back) % capacity_;
    }

    static void add_scaled(double factor, std::span<const double> source, std::span<double> target)
    {
        for (std::size_t i = 0; i < target.size(); ++i)
        {
            target[i] += factor * source[i];
        }
    }

    std::size_t capacity_;
    std::size_t count_ = 0;
    std::size_t oldest_ = 0;
    double gamma_ = 1.0;
    std::vector<std::vector<double>> s_;
    std::vector<std::vector<double>> y_;
    std::vector<double> rho_;
    std::vector<double> weights_;
};

template <class F>
class LineSearch
{
public:
    LineSearch(F& objective, std::size_t dimension)
        : objective_(objective), trial_(dimension), best_(dimension), accepted_(dimension)
    {
    }

    const Point& accepted() const
    {
        return accepted_;
    }

    bool run(const Point& origin, std::span<const double> direction, double initial_step)
    {
        origin_ = &origin;
        direction_ = direction;
        slope0_ = dot(origin.g, direction);
        trials_used_ = 0;
        best_ = origin;
        best_.alpha = 0.0;
        best_.slope = slope0_;
        return bracket(initial_step);
    }

private:
    bool exhausted() const
    {
        return trials_used_ >= kMaxLineSearchTrials;
    }

    bool evaluate(double alpha)
    {
        trial_.alpha = alpha;
        for (std::size_t i = 0; i < trial_.x.size(); ++i)
        {
            trial_.x[i] = origin_->x[i] + alpha * direction_[i];
        }
        trial_.f = objective_(std::span<const double>(trial_.x), std::span<double>(trial_.g));
        ++trials_used_;
        if (!std::isfinite(trial_.f) || !all_finite(trial_.g))
        {
            return false;
        }
        trial_.slope = dot(trial_.g, direction_);
        return true;
    }

    bool sufficient_decrease() const
    {
        return trial_.f <= origin_->f + kSufficientDecrease * trial_.alpha * slope0_;
    }

    bool curvature_satisfied() const
    {
        return std::abs(trial_.slope) <= -kCurvatureDecrease * slope0_;
    }

    bool trial_is_too_far()
    {
        return !sufficient_decrease() || trial_.f >= best_.f;
    }

    bool accept_best_so_far()
    {
        if (best_.alpha <= 0.0)
        {
            return false;
        }
        accepted_ = best_;
        return true;
    }

    bool bracket(double initial_step)
    {
        double alpha = initial_step;
        while (!exhausted())
        {
            bool finite = evaluate(alpha);
            if (!finite || trial_is_too_far())
            {
                return zoom(alpha);
            }
            if (curvature_satisfied())
            {
                accepted_ = trial_;
                return true;
            }
            double previous_alpha = best_.alpha;
            best_ = trial_;
            if (trial_.slope >= 0.0)
            {
                return zoom(previous_alpha);
            }
            alpha *= 2.0;
        }
        return accept_best_so_far();
    }

    bool zoom(double far_alpha)
    {
        while (!exhausted())
        {
            double alpha = 0.5 * (best_.alpha + far_alpha);
            bool finite = evaluate(alpha);
            if (!finite || trial_is_too_far())
            {
                far_alpha = alpha;
                continue;
            }
            if (curvature_satisfied())
            {
                accepted_ = trial_;
                return true;
            }
            if (trial_.slope * (far_alpha - best_.alpha) >= 0.0)
            {
                far_alpha = best_.alpha;
            }
            best_ = trial_;
        }
        return accept_best_so_far();
    }

    F& objective_;
    const Point* origin_ = nullptr;
    std::span<const double> direction_{};
    double slope0_ = 0.0;
    int trials_used_ = 0;
    Point trial_;
    Point best_;
    Point accepted_;
};

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
    detail::Minimizer<std::remove_reference_t<F>> minimizer(f, x, options);
    return minimizer.run();
}

}
