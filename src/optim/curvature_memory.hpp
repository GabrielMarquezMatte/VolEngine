#pragma once

#include "optim/vector_math.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <span>
#include <vector>

namespace ve::optim::detail
{

constexpr double kPositiveCurvatureTolerance = 1e-10;

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

}
