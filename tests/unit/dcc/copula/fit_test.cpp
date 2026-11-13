#include "dcc/copula/fit.hpp"

#include "../dcc3_panel.hpp"
#include "copula3_panel.hpp"
#include "core/context.hpp"
#include "core/spec.hpp"
#include "dcc/copula/scores.hpp"
#include "dcc/fit.hpp"
#include "dcc/likelihood.hpp"
#include "dcc/parameter_map.hpp"
#include "dcc/spec.hpp"
#include "fit/fit_result.hpp"

#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <span>
#include <string>
#include <vector>

namespace
{

using ve::FitConfig;
using ve::FitResult;
using ve::FitStatus;
using ve::dcc::Convention;
using ve::dcc::Criterion;
using ve::dcc::DccCopula;
using ve::dcc::DccDist;
using ve::dcc::DccFitResult;
using ve::dcc::DccModel;
using ve::dcc::DccOptions;
using ve::dcc::DccSpec;
using ve::dcc::copula::CopulaInputs;

std::vector<const FitResult*> pointers_of(const std::vector<FitResult>& fits)
{
    std::vector<const FitResult*> pointers;
    for (const FitResult& fit : fits)
    {
        pointers.push_back(&fit);
    }
    return pointers;
}

DccSpec copula_spec(DccModel model, DccDist dist, DccCopula copula, Convention convention = Convention::Standard)
{
    DccSpec spec;
    spec.model = model;
    spec.dist = dist;
    spec.copula = copula;
    spec.convention = convention;
    return spec;
}

CopulaInputs inputs_of(const std::vector<FitResult>& fits, const std::vector<DccSpec>& specs)
{
    return ve::dcc::copula::copula_inputs(pointers_of(fits), specs);
}

// rmgarch's dynamic cases on copula3 (plan 4b notes, section 7): theta in the layout order, llh, and R21, R31, R32
// at t = 1, 1000 and 2000.
struct Oracle
{
    const char* id;
    DccModel model;
    DccDist dist;
    DccCopula copula;
    std::vector<double> theta;
    double llh;
    std::array<std::array<double, 3>, 3> r;
};

const std::vector<Oracle>& oracles()
{
    static const std::vector<Oracle> cases{
        {"dcc_mvnorm_param", DccModel::Dcc, DccDist::MvNorm, DccCopula::Parametric,
         {0.0473819588956065, 0.9175354409917849}, -7999.2557546252,
         {{{0.513710062071992, 0.398508626536416, 0.406936683559792},
           {0.542055905211159, 0.243814997355676, 0.142280592260793},
           {0.155972386680157, 0.088123875634007, 0.207603432323858}}}},
        {"dcc_mvnorm_emp", DccModel::Dcc, DccDist::MvNorm, DccCopula::Empirical,
         {0.0469471761609685, 0.9179675558919811}, -8000.3387982860,
         {{{0.513374041590776, 0.399416241144631, 0.408805414514274},
           {0.542471816068691, 0.244683660349199, 0.149864669396081},
           {0.154850759329173, 0.0880884649009998, 0.21091235366846}}}},
        {"dcc_mvt_param", DccModel::Dcc, DccDist::MvT, DccCopula::Parametric,
         {0.0444504443322488, 0.9194035843020166, 5.8841836499319067}, -7916.6409789490,
         {{{0.518184331109798, 0.412627644979759, 0.438797045747512},
           {0.538995915675457, 0.293665921136297, 0.168695842937856},
           {0.144506744820049, 0.144239270083496, 0.239873143273593}}}},
        {"dcc_mvt_emp", DccModel::Dcc, DccDist::MvT, DccCopula::Empirical,
         {0.0451190098670297, 0.9172060812961198, 6.3934861824246285}, -7922.6218623579,
         {{{0.516726508249743, 0.414282949503144, 0.445589776318746},
           {0.535304409700781, 0.289680200120334, 0.182618695813444},
           {0.149082826076376, 0.144239174414481, 0.24885537152601}}}},
        {"adcc_mvnorm_param", DccModel::ADcc, DccDist::MvNorm, DccCopula::Parametric,
         {0.0453138390863231, 0.9178306244174149, 0.0048824360257134}, -7999.0810297674,
         {{{0.5138191159492, 0.398532802736662, 0.406978936073836},
           {0.543515030706206, 0.255728591665038, 0.15239787292293},
           {0.170666902988211, 0.102095366922369, 0.213960418915813}}}},
        {"adcc_mvt_param", DccModel::ADcc, DccDist::MvT, DccCopula::Parametric,
         {0.0444504473500459, 0.919403559315639, 4.75615550848946e-08, 5.88418954448619}, -7916.6409793054,
         {{{0.518184334381777, 0.412627632339022, 0.438796992462346},
           {0.538995932542449, 0.293665925607916, 0.168695845279408},
           {0.144506866673249, 0.144239252917887, 0.239873140428432}}}},
    };
    return cases;
}

TEST(CopulaFit, MatchesRmgarchAtItsPoint)
{
    const std::vector<FitResult> fits = ve::testing::copula3_fits();
    for (const Oracle& oracle : oracles())
    {
        SCOPED_TRACE(oracle.id);
        const DccSpec spec = copula_spec(oracle.model, oracle.dist, oracle.copula, Convention::Rmgarch);
        const CopulaInputs inputs = inputs_of(fits, {spec});
        std::vector<double> r(ve::testing::kCopula3Rows * 9);
        const double llh = ve::dcc::copula::copula_loglik(inputs, spec, oracle.theta, r);
        EXPECT_NEAR(llh, oracle.llh, 1e-10 * std::abs(oracle.llh));
        const std::array<std::size_t, 3> times{0, 999, 1999};
        for (std::size_t k = 0; k < 3; ++k)
        {
            const double* rt = r.data() + times[k] * 9;
            EXPECT_NEAR(rt[3], oracle.r[k][0], 1e-10) << "t " << times[k];
            EXPECT_NEAR(rt[6], oracle.r[k][1], 1e-10) << "t " << times[k];
            EXPECT_NEAR(rt[7], oracle.r[k][2], 1e-10) << "t " << times[k];
        }
    }
}

TEST(CopulaFit, FixedShape)
{
    const std::vector<FitResult> fits = ve::testing::copula3_fits();
    DccSpec spec = copula_spec(DccModel::Dcc, DccDist::MvT, DccCopula::Parametric, Convention::Rmgarch);
    spec.fixed = {{"mshape", 8.0}};
    const CopulaInputs inputs = inputs_of(fits, {spec});
    const std::vector<double> theta{0.0463737883675772, 0.9188279296008739, 8.0};
    EXPECT_NEAR(ve::dcc::copula::copula_loglik(inputs, spec, theta), -7921.3163263539, 1e-10 * 7921.3163263539);
    const DccFitResult fit = ve::dcc::copula::fit_copula(inputs, spec, FitConfig{});
    EXPECT_EQ(fit.names, (std::vector<std::string>{"dcca1", "dccb1"}));
    EXPECT_EQ(fit.status, FitStatus::Converged);
    EXPECT_NEAR(fit.estimate[0], 0.0463737883675772, 1e-3 * 0.0463737883675772);
    EXPECT_NEAR(fit.estimate[1], 0.9188279296008739, 1e-3 * 0.9188279296008739);
}

TEST(CopulaFit, MvnormIsDccOnScores)
{
    const std::vector<FitResult> fits = ve::testing::copula3_fits();
    for (const DccCopula copula : {DccCopula::Parametric, DccCopula::Empirical})
    {
        const DccSpec spec = copula_spec(DccModel::Dcc, DccDist::MvNorm, copula);
        const CopulaInputs inputs = inputs_of(fits, {spec});
        const DccFitResult fit = ve::dcc::copula::fit_copula(inputs, spec, FitConfig{});
        ASSERT_EQ(fit.status, FitStatus::Converged);
        const ve::dcc::copula::CopulaPanel scores =
            ve::dcc::copula::copula_panel(inputs.uniforms_of(spec), *inputs.univariate, DccDist::MvNorm, 0.0);
        const DccFitResult direct = ve::dcc::fit_dcc(scores.panel, spec, FitConfig{});
        EXPECT_EQ(fit.estimate, direct.estimate);
        double log_sigma = 0.0;
        for (double s : scores.panel->sigma)
        {
            log_sigma += std::log(s);
        }
        const double expected = ve::testing::kCopula3UnivariateLoglik +
                                ve::dcc::reported_loglik(*scores.panel, spec, direct.estimate) + log_sigma -
                                scores.marginal_log_density;
        EXPECT_NEAR(fit.loglik, expected, 1e-9 * std::abs(expected));
        EXPECT_EQ(fit.panel->sigma, inputs.univariate->sigma);
    }
}

double fixed_shape_loglik(const CopulaInputs& inputs, DccSpec spec, double nu)
{
    spec.fixed = {{"mshape", nu}};
    return ve::dcc::copula::fit_copula(inputs, spec, FitConfig{}).loglik;
}

TEST(CopulaFit, ProfileFindsTheShape)
{
    const std::vector<FitResult> fits = ve::testing::copula3_fits();
    const DccSpec spec = copula_spec(DccModel::Dcc, DccDist::MvT, DccCopula::Parametric);
    const CopulaInputs inputs = inputs_of(fits, {spec});
    const DccFitResult fit = ve::dcc::copula::fit_copula(inputs, spec, FitConfig{});
    ASSERT_EQ(fit.status, FitStatus::Converged);
    ASSERT_EQ(fit.names, (std::vector<std::string>{"dcca1", "dccb1", "mshape"}));
    const double nu = fit.estimate[2];
    EXPECT_NEAR(nu, 5.8841836499319067, 0.05 * 5.88);
    EXPECT_LE(fixed_shape_loglik(inputs, spec, nu - 1e-2), fit.loglik + 1e-8);
    EXPECT_LE(fixed_shape_loglik(inputs, spec, nu + 1e-2), fit.loglik + 1e-8);
}

TEST(CopulaFit, ShapeAtTheCeiling)
{
    // Review Focus 5: dcc3's normal margins push the copula shape to its 50 ceiling.
    const std::vector<FitResult> fits = ve::testing::dcc3_fits();
    const DccSpec spec = copula_spec(DccModel::Dcc, DccDist::MvT, DccCopula::Parametric);
    const DccFitResult fit = ve::dcc::copula::fit_copula(inputs_of(fits, {spec}), spec, FitConfig{});
    EXPECT_EQ(fit.status, FitStatus::Converged);
    EXPECT_LE(fit.estimate[2], 50.0);
    EXPECT_GT(fit.estimate[2], 45.0);
    EXPECT_TRUE(std::isfinite(fit.loglik));
}

TEST(CopulaFit, FixedShapeOutsideTheBoundsFailsItsSpecOnly)
{
    // Final review 1-2: a fixed copula shape outside (4, 50) fails that spec, static or dynamic, like the plain DCC,
    // and never throws out of the batch.
    const std::vector<FitResult> fits = ve::testing::copula3_fits();
    const std::vector<const FitResult*> pointers = pointers_of(fits);
    std::vector<DccSpec> specs{copula_spec(DccModel::Dcc, DccDist::MvNorm, DccCopula::None)};
    for (const double nu : {0.0, 3.0, 60.0})
    {
        for (const DccModel model : {DccModel::Dcc, DccModel::Static})
        {
            DccSpec spec = copula_spec(model, DccDist::MvT, DccCopula::Parametric);
            spec.fixed = {{"mshape", nu}};
            specs.push_back(spec);
        }
    }
    const DccOptions options{FitConfig{}, false};
    ASSERT_FALSE(ve::dcc::check_dcc_batch(pointers, specs, options).has_value());
    ve::Context context;
    const std::vector<DccFitResult> results = ve::dcc::fit_dcc_batch(context, pointers, specs, options);
    EXPECT_EQ(results[0].status, FitStatus::Converged);
    for (std::size_t i = 1; i < results.size(); ++i)
    {
        SCOPED_TRACE(i);
        EXPECT_EQ(results[i].status, FitStatus::Failed);
        EXPECT_TRUE(std::isnan(results[i].loglik));
    }
}

TEST(CopulaFit, CriteriaUseTheDccK)
{
    const std::vector<FitResult> fits = ve::testing::copula3_fits();
    const std::vector<const FitResult*> pointers = pointers_of(fits);
    const std::vector<DccSpec> specs{copula_spec(DccModel::Dcc, DccDist::MvNorm, DccCopula::Parametric)};
    const DccOptions options{FitConfig{}, false};
    ASSERT_FALSE(ve::dcc::check_dcc_batch(pointers, specs, options).has_value());
    ve::Context context;
    const DccFitResult fit = ve::dcc::fit_dcc_batch(context, pointers, specs, options)[0];
    // The univariate fits are fixed, so k = 0 + 2 + 3; estimated ones would add 5 x 3.
    const double k = 2.0 + 3.0;
    const double t = static_cast<double>(ve::testing::kCopula3Rows);
    EXPECT_NEAR(ve::dcc::dcc_criterion(fit, Criterion::Bic, false), -2.0 * fit.loglik + k * std::log(t), 1e-9);
}

bool same_bits(std::span<const double> a, std::span<const double> b)
{
    if (a.size() != b.size())
    {
        return false;
    }
    for (std::size_t i = 0; i < a.size(); ++i)
    {
        if (std::bit_cast<uint64_t>(a[i]) != std::bit_cast<uint64_t>(b[i]))
        {
            return false;
        }
    }
    return true;
}

void expect_same(const DccFitResult& actual, const DccFitResult& expected)
{
    EXPECT_EQ(actual.status, expected.status);
    EXPECT_TRUE(same_bits(actual.estimate, expected.estimate));
    EXPECT_EQ(std::bit_cast<uint64_t>(actual.loglik), std::bit_cast<uint64_t>(expected.loglik));
    ASSERT_EQ(actual.r == nullptr, expected.r == nullptr);
    if (actual.r != nullptr)
    {
        EXPECT_TRUE(same_bits(*actual.r, *expected.r));
    }
}

std::vector<DccFitResult> run_batch(int threads, const std::vector<FitResult>& fits, const std::vector<DccSpec>& specs)
{
    const std::vector<const FitResult*> pointers = pointers_of(fits);
    const DccOptions options{FitConfig{}, false};
    EXPECT_FALSE(ve::dcc::check_dcc_batch(pointers, specs, options).has_value());
    ve::Context context;
    context.set_threads(threads);
    return ve::dcc::fit_dcc_batch(context, pointers, specs, options);
}

TEST(DccBatch, MixedBatchSameAsAlone)
{
    // Review Focus 4.
    const std::vector<FitResult> fits = ve::testing::copula3_fits();
    std::vector<DccSpec> plain;
    for (const DccModel model : {DccModel::Dcc, DccModel::ADcc, DccModel::FDcc})
    {
        for (const DccDist dist : {DccDist::MvNorm, DccDist::MvT, DccDist::MvLaplace})
        {
            DccSpec spec = copula_spec(model, dist, DccCopula::None);
            if (model == DccModel::FDcc)
            {
                spec.groups = {1, 1, 2};
            }
            plain.push_back(spec);
        }
    }
    std::vector<DccSpec> specs = plain;
    for (const DccModel model : {DccModel::Dcc, DccModel::ADcc})
    {
        for (const DccDist dist : {DccDist::MvNorm, DccDist::MvT})
        {
            for (const DccCopula copula : {DccCopula::Parametric, DccCopula::Empirical})
            {
                specs.push_back(copula_spec(model, dist, copula));
            }
        }
    }
    for (const ve::dcc::StaticMethod method : {ve::dcc::StaticMethod::Kendall, ve::dcc::StaticMethod::Ml})
    {
        for (const DccDist dist : {DccDist::MvNorm, DccDist::MvT})
        {
            DccSpec spec = copula_spec(DccModel::Static, dist, DccCopula::Parametric);
            spec.static_method = method;
            specs.push_back(spec);
        }
    }
    std::vector<DccFitResult> alone;
    for (const DccSpec& spec : specs)
    {
        alone.push_back(run_batch(1, fits, {spec})[0]);
    }
    const std::vector<DccFitResult> plain_only = run_batch(1, fits, plain);
    for (std::size_t i = 0; i < plain.size(); ++i)
    {
        SCOPED_TRACE(i);
        expect_same(plain_only[i], alone[i]);
    }
    for (const int threads : {1, 3, 16})
    {
        SCOPED_TRACE(threads);
        const std::vector<DccFitResult> results = run_batch(threads, fits, specs);
        ASSERT_EQ(results.size(), specs.size());
        for (std::size_t i = 0; i < specs.size(); ++i)
        {
            SCOPED_TRACE(i);
            expect_same(results[i], alone[i]);
        }
    }
}

}
