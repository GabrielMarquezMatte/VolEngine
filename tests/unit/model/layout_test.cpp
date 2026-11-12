#include "model/layout.hpp"

#include <gtest/gtest.h>
#include <string>
#include <utility>
#include <vector>

namespace
{

using Names = std::vector<std::string>;
using ve::Model;
using ve::model::ParameterKind;

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

ve::Spec spec_of(Model model, int p = 1, int q = 1)
{
    ve::Spec s;
    s.model = model;
    s.p = p;
    s.q = q;
    return s;
}

using NamesCase = std::pair<Model, Names>;

class LayoutNames : public testing::TestWithParam<NamesCase>
{
};

TEST_P(LayoutNames, FollowRugarchOrder)
{
    auto [model, tail] = GetParam();
    Names expected{"mu", "omega", "alpha1", "beta1"};
    expected.insert(expected.end(), tail.begin(), tail.end());
    auto l = ve::model::layout_of(spec_of(model));
    EXPECT_EQ(ve::model::parameter_names(l), expected);
    EXPECT_EQ(l.count(), expected.size());
}

INSTANTIATE_TEST_SUITE_P(
    AllModels, LayoutNames,
    testing::Values(NamesCase{Model::SGarch, {}}, NamesCase{Model::IGarch, {}}, NamesCase{Model::FGarchGarch, {}},
                    NamesCase{Model::EGarch, {"gamma1"}}, NamesCase{Model::GjrGarch, {"gamma1"}},
                    NamesCase{Model::ApArch, {"gamma1", "delta"}}, NamesCase{Model::CsGarch, {"eta11", "eta21"}},
                    NamesCase{Model::FGarchTGarch, {"eta11"}}, NamesCase{Model::FGarchGjrGarch, {"eta11"}},
                    NamesCase{Model::FGarchAvGarch, {"eta11", "eta21"}}, NamesCase{Model::FGarchNGarch, {"lambda"}},
                    NamesCase{Model::FGarchNaGarch, {"eta21"}}, NamesCase{Model::FGarchApArch, {"eta11", "lambda"}},
                    NamesCase{Model::FGarchAllGarch, {"eta11", "eta21", "lambda"}}));

TEST(Layout, HigherOrdersRepeatPerLag)
{
    ve::Spec all = spec_of(Model::FGarchAllGarch, 2, 1);
    all.ar = 1;
    all.dist = ve::Dist::Std;
    EXPECT_EQ(ve::model::parameter_names(ve::model::layout_of(all)),
              (Names{"mu", "ar1", "omega", "alpha1", "alpha2", "beta1", "eta11", "eta12", "eta21", "eta22", "lambda",
                     "shape"}));
    EXPECT_EQ(ve::model::parameter_names(ve::model::layout_of(spec_of(Model::CsGarch, 2, 2))),
              (Names{"mu", "omega", "alpha1", "alpha2", "beta1", "beta2", "eta11", "eta21"}));
}

TEST(Layout, BlockOffsetsChain)
{
    ve::Spec all = spec_of(Model::FGarchAllGarch, 2, 1);
    all.dist = ve::Dist::Std;
    auto l = ve::model::layout_of(all);
    EXPECT_EQ(l.gamma_begin(), 5u);
    EXPECT_EQ(l.eta1_begin(), 5u);
    EXPECT_EQ(l.eta2_begin(), 7u);
    EXPECT_EQ(l.power(), 9u);
    EXPECT_EQ(l.shape(), 10u);
    EXPECT_EQ(l.count(), 11u);
    auto apa = ve::model::layout_of(spec_of(Model::ApArch, 2, 1));
    EXPECT_EQ(apa.gamma_begin(), 5u);
    EXPECT_EQ(apa.power(), 7u);
}

TEST(Layout, IGarchDerivesLastBeta)
{
    auto l = ve::model::layout_of(spec_of(Model::IGarch, 1, 2));
    EXPECT_TRUE(l.has_derived());
    EXPECT_EQ(l.derived(), l.beta_begin() + 1);
    EXPECT_EQ(l.estimated_count(), l.count() - 1);
    EXPECT_EQ(l.count(), 5u);
    for (Model model : {Model::SGarch, Model::EGarch, Model::CsGarch, Model::FGarchGarch})
    {
        auto other = ve::model::layout_of(spec_of(model, 1, 2));
        EXPECT_FALSE(other.has_derived());
        EXPECT_EQ(other.estimated_count(), other.count());
    }
}

TEST(Layout, WidestFitsWidestDual)
{
    ve::Spec s = spec_of(Model::FGarchAllGarch, ve::kMaxOrder, ve::kMaxOrder);
    s.ar = s.ma = ve::kMaxOrder;
    s.dist = ve::Dist::Ghyp;
    auto l = ve::model::layout_of(s);
    EXPECT_EQ(l.count(), ve::kMaxParameterCount);
    EXPECT_EQ(l.count(), 42u);
    EXPECT_EQ(ve::model::parameter_names(l).size(), 42u);
}

TEST(Layout, DistributionBlocksFollowRugarchOrder)
{
    for (ve::Dist dist : {ve::Dist::Sstd, ve::Dist::Sged, ve::Dist::Nig, ve::Dist::Jsu})
    {
        ve::Spec s;
        s.dist = dist;
        auto l = ve::model::layout_of(s);
        EXPECT_EQ(ve::model::parameter_names(l),
                  (Names{"mu", "omega", "alpha1", "beta1", "skew", "shape"}));
        EXPECT_EQ(l.skew(), 4u);
        EXPECT_EQ(l.shape(), 5u);
        EXPECT_FALSE(l.has_ghlambda());
        EXPECT_EQ(l.count(), 6u);
    }
    ve::Spec g;
    g.dist = ve::Dist::Ghyp;
    auto l = ve::model::layout_of(g);
    EXPECT_EQ(ve::model::parameter_names(l),
              (Names{"mu", "omega", "alpha1", "beta1", "skew", "shape", "ghlambda"}));
    EXPECT_EQ(l.ghlambda(), 6u);
    EXPECT_EQ(l.count(), 7u);
    for (ve::Dist dist : {ve::Dist::Norm, ve::Dist::Std, ve::Dist::Ged})
    {
        ve::Spec s;
        s.dist = dist;
        EXPECT_FALSE(ve::model::layout_of(s).has_skew());
        EXPECT_FALSE(ve::model::layout_of(s).has_ghlambda());
    }
}

TEST(Layout, DistributionKinds)
{
    auto kind_of = [](ve::Dist dist, std::size_t offset)
    {
        ve::Spec s;
        s.dist = dist;
        auto l = ve::model::layout_of(s);
        return l.kind(l.skew() + offset);
    };
    EXPECT_EQ(kind_of(ve::Dist::Sstd, 0), ParameterKind::Positive);
    EXPECT_EQ(kind_of(ve::Dist::Sged, 0), ParameterKind::Positive);
    for (ve::Dist dist : {ve::Dist::Nig, ve::Dist::Ghyp, ve::Dist::Jsu})
    {
        EXPECT_EQ(kind_of(dist, 0), ParameterKind::Real);
    }
    for (ve::Dist dist : {ve::Dist::Sstd, ve::Dist::Sged, ve::Dist::Nig, ve::Dist::Ghyp, ve::Dist::Jsu})
    {
        EXPECT_EQ(kind_of(dist, 1), ParameterKind::Positive);
    }
    EXPECT_EQ(kind_of(ve::Dist::Ghyp, 2), ParameterKind::Real);
    EXPECT_EQ(kind_of(ve::Dist::Std, 0), ParameterKind::Positive);
}

TEST(Layout, ShapeParametersOf)
{
    ve::Spec s;
    s.dist = ve::Dist::Ghyp;
    auto l = ve::model::layout_of(s);
    std::vector<double> theta{0.0, 0.1, 0.1, 0.8, -0.3, 1.5, 2.0};
    auto p = ve::model::shape_parameters_of<double>(theta, l);
    EXPECT_EQ(p.skew, -0.3);
    EXPECT_EQ(p.shape, 1.5);
    EXPECT_EQ(p.ghlambda, 2.0);
    auto norm = ve::model::layout_of({});
    auto q = ve::model::shape_parameters_of<double>(std::span<const double>(theta).first(4), norm);
    EXPECT_EQ(q.skew, 0.0);
    EXPECT_EQ(q.shape, 0.0);
    EXPECT_EQ(q.ghlambda, 0.0);
}

TEST(Layout, ParameterKinds)
{
    ve::Spec s = spec_of(Model::SGarch);
    s.ar = 1;
    s.dist = ve::Dist::Std;
    auto sg = ve::model::layout_of(s);
    EXPECT_EQ(sg.kind(sg.mu()), ParameterKind::Location);
    EXPECT_EQ(sg.kind(sg.ar_begin()), ParameterKind::Real);
    EXPECT_EQ(sg.kind(sg.omega()), ParameterKind::Positive);
    EXPECT_EQ(sg.kind(sg.alpha_begin()), ParameterKind::Weight);
    EXPECT_EQ(sg.kind(sg.beta_begin()), ParameterKind::Weight);
    EXPECT_EQ(sg.kind(sg.shape()), ParameterKind::Positive);

    auto eg = ve::model::layout_of(spec_of(Model::EGarch));
    EXPECT_EQ(eg.kind(eg.omega()), ParameterKind::Real);
    EXPECT_EQ(eg.kind(eg.alpha_begin()), ParameterKind::Real);
    EXPECT_EQ(eg.kind(eg.beta_begin()), ParameterKind::Weight);
    EXPECT_EQ(eg.kind(eg.gamma_begin()), ParameterKind::Real);

    auto gjr = ve::model::layout_of(spec_of(Model::GjrGarch));
    EXPECT_EQ(gjr.kind(gjr.alpha_begin()), ParameterKind::Weight);
    EXPECT_EQ(gjr.kind(gjr.gamma_begin()), ParameterKind::Real);

    auto cs = ve::model::layout_of(spec_of(Model::CsGarch));
    EXPECT_EQ(cs.kind(cs.eta1_begin()), ParameterKind::Positive);
    EXPECT_EQ(cs.kind(cs.eta2_begin()), ParameterKind::Weight);

    auto all = ve::model::layout_of(spec_of(Model::FGarchAllGarch));
    EXPECT_EQ(all.kind(all.eta1_begin()), ParameterKind::Real);
    EXPECT_EQ(all.kind(all.eta2_begin()), ParameterKind::Real);
    EXPECT_EQ(all.kind(all.power()), ParameterKind::Positive);

    auto ap = ve::model::layout_of(spec_of(Model::ApArch));
    EXPECT_EQ(ap.kind(ap.gamma_begin()), ParameterKind::Real);
    EXPECT_EQ(ap.kind(ap.power()), ParameterKind::Positive);
}

TEST(Layout, FixedPowers)
{
    for (Model model : {Model::FGarchTGarch, Model::FGarchAvGarch})
    {
        auto l = ve::model::layout_of(spec_of(model));
        EXPECT_FALSE(l.free_power);
        EXPECT_EQ(l.fixed_power, 1.0);
    }
    for (Model model : {Model::FGarchGarch, Model::FGarchNaGarch, Model::FGarchGjrGarch, Model::SGarch,
                        Model::CsGarch})
    {
        auto l = ve::model::layout_of(spec_of(model));
        EXPECT_FALSE(l.free_power);
        EXPECT_EQ(l.fixed_power, 2.0);
    }
    for (Model model : {Model::FGarchNGarch, Model::FGarchApArch, Model::FGarchAllGarch, Model::ApArch})
    {
        EXPECT_TRUE(ve::model::layout_of(spec_of(model)).free_power);
    }
}


ve::Spec role_spec(Model model, int p, int q, ve::Dist dist = ve::Dist::Norm)
{
    ve::Spec spec;
    spec.model = model;
    spec.p = p;
    spec.q = q;
    spec.dist = dist;
    return spec;
}

bool mentions(const std::optional<std::string>& problem, const std::string& text)
{
    return problem.has_value() && problem->find(text) != std::string::npos;
}

TEST(Layout, ResolvesFixedAndStartByName)
{
    auto spec = role_spec(Model::SGarch, 1, 1, ve::Dist::Std);
    spec.fixed = {{"shape", 5.0}};
    spec.start = {{"alpha1", 0.1}};
    auto l = ve::model::layout_of(spec);
    EXPECT_TRUE(l.is_fixed(l.shape()));
    EXPECT_EQ(l.given[l.shape()], 5.0);
    EXPECT_TRUE(l.is_started(l.alpha_begin()));
    EXPECT_EQ(l.given[l.alpha_begin()], 0.1);
    EXPECT_FALSE(l.is_started(l.shape()));
    EXPECT_EQ(l.estimated_count(), 4u);
}

TEST(Layout, DerivedRoles)
{
    auto igarch = ve::model::layout_of(role_spec(Model::IGarch, 1, 2));
    EXPECT_TRUE(igarch.is_derived(igarch.derived()));
    EXPECT_FALSE(igarch.is_estimated(igarch.derived()));

    auto spec = role_spec(Model::SGarch, 1, 1);
    spec.variance_targeting = true;
    auto l = ve::model::layout_of(spec);
    EXPECT_TRUE(l.is_derived(l.omega()));
    EXPECT_EQ(l.estimated_count(), l.count() - 1);

    spec.start = {{"omega", 0.1}};
    l = ve::model::layout_of(spec);
    EXPECT_FALSE(l.is_started(l.omega()));
}

TEST(Layout, SpecProblems)
{
    auto unknown = role_spec(Model::SGarch, 1, 1);
    unknown.fixed = {{"shape", 5.0}};
    EXPECT_TRUE(mentions(ve::model::spec_problem(unknown), "shape"));
    EXPECT_TRUE(mentions(ve::model::spec_problem(unknown), "mu omega alpha1 beta1"));

    auto unknown_start = role_spec(Model::SGarch, 1, 1);
    unknown_start.start = {{"beta2", 0.1}};
    EXPECT_TRUE(mentions(ve::model::spec_problem(unknown_start), "beta2"));

    auto igarch = role_spec(Model::IGarch, 1, 1);
    igarch.fixed = {{"beta1", 0.9}};
    EXPECT_TRUE(mentions(ve::model::spec_problem(igarch), "derived"));
    EXPECT_TRUE(mentions(ve::model::spec_problem(igarch), "beta1"));

    auto omega = role_spec(Model::SGarch, 1, 1);
    omega.variance_targeting = true;
    omega.fixed = {{"omega", 0.1}};
    EXPECT_TRUE(mentions(ve::model::spec_problem(omega), "derived"));
    EXPECT_TRUE(mentions(ve::model::spec_problem(omega), "omega"));

    for (Model model : {Model::IGarch, Model::CsGarch})
    {
        auto targeted = role_spec(model, 1, 1);
        targeted.variance_targeting = true;
        EXPECT_TRUE(mentions(ve::model::spec_problem(targeted), "variance targeting"));
    }

    auto phi = role_spec(Model::CsGarch, 1, 1);
    phi.fixed = {{"eta21", 0.03}};
    EXPECT_TRUE(mentions(ve::model::spec_problem(phi), "eta21"));
    phi.fixed.push_back({"beta1", 0.9});
    EXPECT_FALSE(ve::model::spec_problem(phi).has_value());

    EXPECT_FALSE(ve::model::spec_problem(role_spec(Model::SGarch, 1, 1)).has_value());
}

TEST(Layout, PlainSpecUnchanged)
{
    for (int model = 0; model <= static_cast<int>(Model::FGarchAllGarch); ++model)
    {
        auto l = ve::model::layout_of(role_spec(static_cast<Model>(model), 2, 2, ve::Dist::Sged));
        EXPECT_TRUE(l.fixed.none());
        EXPECT_TRUE(l.started.none());
        EXPECT_FALSE(l.variance_targeting);
        EXPECT_EQ(l.estimated_count(), l.count() - (l.has_derived() ? 1 : 0));
    }
}

}
