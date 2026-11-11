#include "model/layout.hpp"

#include <gtest/gtest.h>
#include <string>
#include <vector>

namespace
{

using Names = std::vector<std::string>;

TEST(Layout, PlanOneShape)
{
    auto l = ve::model::layout_of({});
    EXPECT_EQ(l.count(), 4u);
    EXPECT_EQ(ve::model::parameter_names(l), (Names{"mu", "omega", "alpha1", "beta1"}));
}

TEST(Layout, FullShapeFollowsRugarchOrder)
{
    ve::Spec s;
    s.p = 2;
    s.q = 3;
    s.ar = 1;
    s.ma = 2;
    s.dist = ve::Dist::Std;
    auto l = ve::model::layout_of(s);
    EXPECT_EQ(ve::model::parameter_names(l),
              (Names{"mu", "ar1", "ma1", "ma2", "omega", "alpha1", "alpha2", "beta1", "beta2",
                     "beta3", "shape"}));
    EXPECT_EQ(l.count(), 11u);
    EXPECT_EQ(l.mu(), 0u);
    EXPECT_EQ(l.ar_begin(), 1u);
    EXPECT_EQ(l.ma_begin(), 2u);
    EXPECT_EQ(l.omega(), 4u);
    EXPECT_EQ(l.alpha_begin(), 5u);
    EXPECT_EQ(l.beta_begin(), 7u);
    EXPECT_EQ(l.shape(), 10u);
    EXPECT_TRUE(l.has_shape());
}

TEST(Layout, NoMeanNoShape)
{
    ve::Spec s;
    s.include_mean = false;
    auto l = ve::model::layout_of(s);
    EXPECT_FALSE(l.has_shape());
    EXPECT_EQ(ve::model::parameter_names(l), (Names{"omega", "alpha1", "beta1"}));
    s.ar = 1;
    EXPECT_EQ(ve::model::parameter_names(ve::model::layout_of(s)),
              (Names{"ar1", "omega", "alpha1", "beta1"}));
}

TEST(Layout, GedHasShape)
{
    ve::Spec s;
    s.dist = ve::Dist::Ged;
    EXPECT_EQ(ve::model::parameter_names(ve::model::layout_of(s)).back(), "shape");
}

TEST(Layout, LargestShapeFitsDual32)
{
    ve::Spec s;
    s.p = s.q = s.ar = s.ma = ve::kMaxOrder;
    s.dist = ve::Dist::Ged;
    auto l = ve::model::layout_of(s);
    EXPECT_EQ(l.count(), 27u);
    EXPECT_EQ(ve::model::parameter_names(l).size(), 27u);
}

}
