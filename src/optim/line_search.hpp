#pragma once

#include "optim/vector_math.hpp"

#include <cmath>
#include <cstddef>
#include <span>
#include <vector>

namespace ve::optim::detail
{

constexpr double kSufficientDecrease = 1e-4;
constexpr double kCurvatureDecrease = 0.9;
constexpr int kMaxLineSearchTrials = 30;

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

    bool accepted_by_fallback() const
    {
        return accepted_by_fallback_;
    }

    bool run(const Point& origin, std::span<const double> direction, double initial_step)
    {
        origin_ = &origin;
        direction_ = direction;
        slope0_ = dot(origin.g, direction);
        trials_used_ = 0;
        accepted_by_fallback_ = false;
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
        accepted_by_fallback_ = true;
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
    bool accepted_by_fallback_ = false;
    Point trial_;
    Point best_;
    Point accepted_;
};

}
