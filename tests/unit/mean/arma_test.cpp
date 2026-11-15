#include "ad/dual.hpp"
#include "mean/arma.hpp"
#include "simulated_garch.hpp"

#include <algorithm>
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

// L = sum_t w_t e_t through the recursion in Dual<8> (lanes mu, ar..., ma...) against the reverse sweep.
TEST(Arma, BackwardMatchesDual)
{
    using Dual8 = ve::ad::Dual<8>;
    const std::vector<double> r = ve::testing::simulate_garch_series(3, 200);
    struct Orders
    {
        std::size_t ar;
        std::size_t ma;
    };
    for (Orders orders : {Orders{0, 0}, Orders{1, 1}, Orders{2, 2}, Orders{3, 1}})
    {
        SCOPED_TRACE(::testing::Message() << "ARMA(" << orders.ar << ", " << orders.ma << ")");
        std::vector<double> values{0.04, 0.3, -0.1, 0.05, -0.2, 0.1};
        std::vector<double> ar_values(values.begin() + 1, values.begin() + 1 + static_cast<long>(orders.ar));
        std::vector<double> ma_values(values.begin() + 4, values.begin() + 4 + static_cast<long>(orders.ma));
        Dual8 mu = ve::ad::variable<8>(values[0], 0);
        std::vector<Dual8> ar;
        std::vector<Dual8> ma;
        for (std::size_t i = 0; i < orders.ar; ++i)
        {
            ar.push_back(ve::ad::variable<8>(ar_values[i], 1 + i));
        }
        for (std::size_t j = 0; j < orders.ma; ++j)
        {
            ma.push_back(ve::ad::variable<8>(ma_values[j], 1 + orders.ar + j));
        }
        std::vector<Dual8> dual_residuals(r.size());
        ve::mean::arma_residuals<Dual8>(r, {mu, ar, ma}, dual_residuals);
        Dual8 expected = ve::ad::constant<8>(0.0);
        std::vector<double> weights(r.size());
        for (std::size_t t = 0; t < r.size(); ++t)
        {
            weights[t] = std::sin(static_cast<double>(t));
            expected = expected + weights[t] * dual_residuals[t];
        }
        std::vector<double> residuals(r.size());
        ve::mean::ArmaParameters<double> p{values[0], ar_values, ma_values};
        ve::mean::arma_residuals<double>(r, p, residuals);
        std::vector<double> ar_gradient(orders.ar);
        std::vector<double> ma_gradient(orders.ma);
        ve::mean::ArmaGradient gradient{0.0, ar_gradient, ma_gradient};
        ve::mean::arma_residuals_backward(r, p, residuals, weights, gradient);
        std::vector<double> actual{gradient.mu};
        actual.insert(actual.end(), ar_gradient.begin(), ar_gradient.end());
        actual.insert(actual.end(), ma_gradient.begin(), ma_gradient.end());
        double scale = 1.0;
        for (std::size_t i = 0; i < actual.size(); ++i)
        {
            scale = std::max(scale, std::abs(expected.grad[i]));
        }
        for (std::size_t i = 0; i < actual.size(); ++i)
        {
            EXPECT_NEAR(actual[i], expected.grad[i], 1e-12 * scale) << "parameter " << i;
        }
    }
}

}
