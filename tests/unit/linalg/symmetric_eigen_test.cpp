#include "linalg/symmetric_eigen.hpp"

#include <cmath>
#include <gtest/gtest.h>
#include <vector>

namespace
{

TEST(SymmetricEigen, MatchesKnownSpectrum)
{
    const std::vector<double> identity{1.0, 0.0, 0.0, 1.0};
    const std::vector<double> n{2.0, 1.0, 1.0, 2.0};
    EXPECT_NEAR(ve::linalg::largest_relative_eigenvalue(identity, n, 2), 3.0, 1e-14);

    const std::vector<double> q{4.0, 0.0, 0.0, 1.0};
    const std::vector<double> ones{1.0, 0.0, 0.0, 1.0};
    EXPECT_NEAR(ve::linalg::largest_relative_eigenvalue(q, ones, 2), 1.0, 1e-14);

    const std::vector<double> zero(4, 0.0);
    EXPECT_EQ(ve::linalg::largest_relative_eigenvalue(q, zero, 2), 0.0);
}

TEST(SymmetricEigen, NonSpdQGivesNaN)
{
    const std::vector<double> singular{1.0, 1.0, 1.0, 1.0};
    const std::vector<double> n{1.0, 0.0, 0.0, 1.0};
    EXPECT_TRUE(std::isnan(ve::linalg::largest_relative_eigenvalue(singular, n, 2)));
}

TEST(SymmetricEigen, GeneralQMatchesClosedForm)
{
    // Q = [[2,1],[1,2]], N = Q: Q^-1/2 Q Q^-1/2 = I, so 1; N = 3 Q gives 3.
    const std::vector<double> q{2.0, 1.0, 1.0, 2.0};
    const std::vector<double> n{6.0, 3.0, 3.0, 6.0};
    EXPECT_NEAR(ve::linalg::largest_relative_eigenvalue(q, n, 2), 3.0, 1e-13);
}

}
