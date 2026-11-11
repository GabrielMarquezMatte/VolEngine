#include "ad/dual.hpp"
#include "ad/special.hpp"

#include <cmath>
#include <gtest/gtest.h>

namespace
{

TEST(Special, DigammaKnownValues)
{
    EXPECT_NEAR(ve::ad::digamma(1.0), -0.57721566490153286, 1e-14);
    EXPECT_NEAR(ve::ad::digamma(0.5), -1.9635100260214235, 1e-13);
    EXPECT_NEAR(ve::ad::digamma(10.0), 2.2517525890667211, 1e-14);
    EXPECT_NEAR(ve::ad::digamma(1e-3), -1000.5755719318103, 1e-9);
}

TEST(Special, LgammaDualDerivative)
{
    for (double x : {0.3, 1.0, 4.5, 50.0})
    {
        auto f = ve::ad::lgamma(ve::ad::variable<1>(x, 0));
        EXPECT_NEAR(f.value, std::lgamma(x), 1e-14);
        EXPECT_NEAR(f.grad[0], ve::ad::digamma(x), 1e-14);
    }
}

}
