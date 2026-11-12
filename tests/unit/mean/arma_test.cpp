#include "mean/arma.hpp"

#include <cmath>
#include <cstddef>
#include <gtest/gtest.h>
#include <vector>

namespace
{

TEST(Arma, ResidualsHandCase)
{
    std::vector<double> r{0.2, 0.4, -0.1};
    std::vector<double> ar{0.5};
    std::vector<double> ma{0.3};
    std::vector<double> residuals(3);
    ve::mean::arma_residuals<double>(r, {0.1, ar, ma}, residuals);
    EXPECT_NEAR(residuals[0], 0.1, 1e-14);
    EXPECT_NEAR(residuals[1], 0.3 - 0.5 * 0.1 - 0.3 * 0.1, 1e-14);
    EXPECT_NEAR(residuals[1], 0.22, 1e-14);
    EXPECT_NEAR(residuals[2], -0.416, 1e-14);
}

TEST(Arma, ResidualsBeforeTheArOrderAreCenteredReturns)
{
    std::vector<double> r{1.0, 2.0, 3.0};
    std::vector<double> ar{0.5, 0.25};
    std::vector<double> ma{0.4};
    std::vector<double> residuals(3);
    ve::mean::arma_residuals<double>(r, {0.0, ar, ma}, residuals);
    EXPECT_EQ(residuals[0], 1.0);
    EXPECT_EQ(residuals[1], 2.0);
    EXPECT_EQ(residuals[2], 3.0 - 0.5 * 2.0 - 0.25 * 1.0 - 0.4 * 2.0);
}

TEST(Arma, MaLagsBeforeTheSampleAreZero)
{
    std::vector<double> r{1.0, 2.0, 3.0};
    std::vector<double> ma{0.5, 0.25};
    std::vector<double> residuals(3);
    ve::mean::arma_residuals<double>(r, {0.0, {}, ma}, residuals);
    EXPECT_EQ(residuals[0], 1.0);
    EXPECT_EQ(residuals[1], 1.5);
    EXPECT_EQ(residuals[2], 3.0 - 0.5 * 1.5 - 0.25 * 1.0);
}

TEST(Arma, WithoutLagsResidualsAreCenteredReturns)
{
    std::vector<double> r{0.3, -0.1};
    std::vector<double> residuals(2);
    ve::mean::arma_residuals<double>(r, {0.1, {}, {}}, residuals);
    EXPECT_EQ(residuals[0], 0.3 - 0.1);
    EXPECT_EQ(residuals[1], -0.1 - 0.1);
}

TEST(Arma, DeviationsReduceToToday)
{
    std::vector<double> r(30);
    for (std::size_t t = 0; t < r.size(); ++t)
    {
        r[t] = 0.4 * std::sin(0.9 * static_cast<double>(t)) + 0.1 * std::cos(2.1 * static_cast<double>(t));
    }
    const double mu = 0.07;
    std::vector<double> d(r.size());
    for (std::size_t t = 0; t < r.size(); ++t)
    {
        d[t] = r[t] - mu;
    }
    std::vector<double> ar_all{0.5, -0.2};
    std::vector<double> ma_all{0.3, 0.15};
    for (std::ptrdiff_t m = 0; m <= 2; ++m)
    {
        for (std::ptrdiff_t n = 0; n <= 2; ++n)
        {
            std::vector<double> ar(ar_all.begin(), ar_all.begin() + m);
            std::vector<double> ma(ma_all.begin(), ma_all.begin() + n);
            std::vector<double> today(r.size());
            std::vector<double> deviations(r.size());
            ve::mean::arma_residuals<double>(r, {mu, ar, ma}, today);
            ve::mean::arma_residuals_of_deviations<double>(d, {mu, ar, ma}, deviations);
            EXPECT_EQ(deviations, today) << "ARMA(" << m << "," << n << ")";
        }
    }
}

}
