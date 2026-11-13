#include "dcc/panel.hpp"

#include "dcc/spec.hpp"
#include "fit/fit.hpp"
#include "linalg/symmetric_eigen.hpp"
#include "simulated_garch.hpp"

#include <bit>
#include <cmath>
#include <cstddef>
#include <gtest/gtest.h>
#include <optional>
#include <string>
#include <vector>

namespace
{

using ve::FitResult;

constexpr std::size_t kRows = 5;
constexpr std::size_t kSeries = 3;

// z is given column by column; sigma = 1 so the residuals are z.
FitResult fit_with_z(const std::vector<double>& z)
{
    FitResult fit;
    fit.residuals = z;
    fit.sigma.assign(z.size(), 1.0);
    fit.nobs = z.size();
    return fit;
}

std::vector<const FitResult*> pointers(const std::vector<FitResult>& fits)
{
    std::vector<const FitResult*> out;
    for (const FitResult& fit : fits)
    {
        out.push_back(&fit);
    }
    return out;
}

double covariance(const std::vector<double>& a, const std::vector<double>& b, bool centred, double divisor)
{
    double mean_a = 0.0;
    double mean_b = 0.0;
    if (centred)
    {
        for (std::size_t t = 0; t < a.size(); ++t)
        {
            mean_a += a[t];
            mean_b += b[t];
        }
        mean_a /= static_cast<double>(a.size());
        mean_b /= static_cast<double>(a.size());
    }
    double sum = 0.0;
    for (std::size_t t = 0; t < a.size(); ++t)
    {
        sum += (a[t] - mean_a) * (b[t] - mean_b);
    }
    return sum / divisor;
}

TEST(DccPanel, CovariancesLikeR)
{
    const std::vector<std::vector<double>> columns{{0.5, -1.2, 0.3, 2.0, -0.7},
                                                  {-0.4, 0.9, -1.5, 0.2, 1.1},
                                                  {1.3, 0.0, -0.6, -0.8, 0.4}};
    std::vector<FitResult> fits;
    std::vector<std::vector<double>> negative(kSeries);
    for (std::size_t i = 0; i < kSeries; ++i)
    {
        fits.push_back(fit_with_z(columns[i]));
        for (const double z : columns[i])
        {
            negative[i].push_back(z < 0.0 ? z : 0.0);
        }
    }
    const std::vector<const FitResult*> ptrs = pointers(fits);
    const ve::dcc::StandardizedPanel panel = ve::dcc::standardize(ptrs);
    ASSERT_EQ(panel.t, kRows);
    ASSERT_EQ(panel.m, kSeries);
    const double count = static_cast<double>(kRows);
    for (std::size_t i = 0; i < kSeries; ++i)
    {
        for (std::size_t j = 0; j < kSeries; ++j)
        {
            const std::size_t k = i * kSeries + j;
            EXPECT_NEAR(panel.qbar[k], covariance(columns[i], columns[j], true, count - 1.0), 1e-15);
            EXPECT_NEAR(panel.qbar_raw[k], covariance(columns[i], columns[j], false, count), 1e-15);
            EXPECT_NEAR(panel.nbar[k], covariance(negative[i], negative[j], false, count), 1e-15);
            EXPECT_NEAR(panel.nbar_centered[k], covariance(negative[i], negative[j], true, count - 1.0), 1e-15);
        }
    }
    EXPECT_EQ(panel.delta, ve::linalg::largest_relative_eigenvalue(panel.qbar, panel.nbar, kSeries));
    EXPECT_EQ(panel.delta_centered, ve::linalg::largest_relative_eigenvalue(panel.qbar, panel.nbar_centered, kSeries));
    EXPECT_GT(panel.delta, 0.0);
}

std::vector<FitResult> two_simulated_fits(std::size_t count)
{
    std::vector<FitResult> fits;
    for (const std::uint64_t seed : {11U, 12U})
    {
        fits.push_back(ve::fit_one(ve::testing::simulate_garch_series(seed, count), ve::Spec{}, ve::FitConfig{}));
    }
    return fits;
}

TEST(DccPanel, ZFromTheFits)
{
    const std::vector<FitResult> fits = two_simulated_fits(300);
    const std::vector<const FitResult*> ptrs = pointers(fits);
    ASSERT_FALSE(ve::dcc::panel_problem(ptrs).has_value());
    const ve::dcc::StandardizedPanel panel = ve::dcc::standardize(ptrs);
    ASSERT_EQ(panel.t, 300U);
    for (std::size_t i = 0; i < 2; ++i)
    {
        for (std::size_t t = 0; t < panel.t; ++t)
        {
            const double expected = fits[i].residuals[t] / fits[i].sigma[t];
            EXPECT_EQ(std::bit_cast<std::uint64_t>(panel.z[t * 2 + i]), std::bit_cast<std::uint64_t>(expected));
            EXPECT_EQ(panel.sigma[t * 2 + i], fits[i].sigma[t]);
        }
    }
}

bool mentions(const std::optional<std::string>& problem, const std::string& word)
{
    return problem.has_value() && problem->find(word) != std::string::npos;
}

TEST(DccPanel, Problems)
{
    std::vector<FitResult> fits = two_simulated_fits(120);
    std::vector<const FitResult*> ptrs = pointers(fits);
    EXPECT_FALSE(ve::dcc::panel_problem(ptrs).has_value());

    EXPECT_TRUE(ve::dcc::panel_problem(std::vector<const FitResult*>{&fits[0]}).has_value());

    const FitResult longer = two_simulated_fits(130)[0];
    ptrs[1] = &longer;
    EXPECT_TRUE(mentions(ve::dcc::panel_problem(ptrs), "series 1"));

    const FitResult empty;
    ptrs[1] = &empty;
    EXPECT_TRUE(mentions(ve::dcc::panel_problem(ptrs), "finite"));

    const std::vector<FitResult> short_fits = two_simulated_fits(99);
    EXPECT_TRUE(mentions(ve::dcc::panel_problem(pointers(short_fits)), "100"));
}

}
