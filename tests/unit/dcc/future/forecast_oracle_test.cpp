#include "dcc/future/forecast.hpp"

#include "core/context.hpp"
#include "dcc/fit.hpp"
#include "dcc/spec.hpp"
#include "dcc/dcc3_panel.hpp"
#include "fit/fit.hpp"
#include "fit/fit_result.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <gtest/gtest.h>
#include <string>
#include <utility>
#include <vector>

namespace
{

using ve::FitConfig;
using ve::FitResult;
using ve::dcc::Convention;
using ve::dcc::DccDist;
using ve::dcc::DccFitResult;
using ve::dcc::DccModel;
using ve::dcc::DccOptions;
using ve::dcc::DccSpec;

constexpr std::size_t kSeries = 3;

// rmgarch 1.4.3's dccforecast (n.roll = 0) on dcc3 (tests/reference/dcc_forecast_rmgarch.R): the DCC estimates it
// found, and R, H and sigma at some of the steps h = 1..ahead.
struct Step
{
    std::size_t h;
    std::array<double, 9> r;
    std::array<double, 9> covariance;
    std::array<double, 3> sigma;
};

struct Oracle
{
    const char* id;
    DccModel model;
    int order;
    DccDist dist;
    std::vector<std::pair<std::string, double>> parameters;
    std::vector<Step> steps;
};

// The sGARCH(1,1)-norm estimates (mu, omega, alpha1, beta1) of its multifit, which the DCC stage started from.
constexpr std::array<std::array<double, 4>, kSeries> kUnivariate{{
    {-0.008920131111225086, 0.019478881309471997, 0.07282124034137887, 0.9065714955528129},
    {-0.02778242404762519, 0.061441088509364916, 0.084080533476537, 0.8462651839828331},
    {-0.009972193379566455, 0.025655072442689313, 0.050370071713741006, 0.9338721906860878},
}};

std::vector<FitResult> univariate_fits()
{
    const std::vector<std::vector<double>> columns = ve::testing::read_dcc3();
    std::vector<FitResult> fits;
    for (std::size_t i = 0; i < kSeries; ++i)
    {
        const std::array<double, 4>& c = kUnivariate[i];
        ve::Spec spec;
        spec.fixed = {{"mu", c[0]}, {"omega", c[1]}, {"alpha1", c[2]}, {"beta1", c[3]}};
        fits.push_back(ve::fit_one(columns[i], spec, FitConfig{}));
    }
    return fits;
}

const std::vector<Oracle>& oracles()
{
    static const std::vector<Oracle> cases{
    {"dcc_mvnorm", DccModel::Dcc, 1, DccDist::MvNorm, {{"dcca1", 0.04316409231673219}, {"dccb1", 0.914269645079578}}, {
        {1, {0.9999999999999998, 0.6795950881128057, 0.47693709938012757, 0.6795950881128056, 0.9999999999999998, 0.5835730145898013, 0.47693709938012757, 0.5835730145898013, 0.9999999999999998},
         {1.0583942053725393, 0.8532924804971495, 0.8259397407259402, 0.8532924804971493, 1.4895254261157844, 1.1988992592650698, 0.8259397407259402, 1.1988992592650698, 2.8335250760716146},
         {1.0287828757189434, 1.2204611530547724, 1.6833077781771266}},
        {5, {0.9999999999999998, 0.6468648298594859, 0.4475969968031149, 0.6468648298594859, 0.9999999999999998, 0.5447115103526826, 0.4475969968031149, 0.5447115103526826, 0.9999999999999998},
         {1.0493516577432143, 0.7662411909054879, 0.7616385835693397, 0.766241190905488, 1.3371584039301232, 1.0463072337830726, 0.7616385835693397, 1.0463072337830726, 2.7593226927204793},
         {1.0243786691176335, 1.1563556563316164, 1.6611209145394803}},
        {10, {0.9999999999999999, 0.6132015310781206, 0.41742049676814463, 0.6132015310781205, 1.0, 0.504742182335091, 0.41742049676814463, 0.504742182335091, 0.9999999999999999},
         {1.0390578186853268, 0.684510957927556, 0.6956484606328269, 0.684510957927556, 1.199263549722033, 0.9036975786201169, 0.6956484606328269, 0.9036975786201168, 2.672959638655696},
         {1.01934185565262, 1.0951089213964211, 1.6349188477278302}},
    }},
    {"dcc21_mvnorm", DccModel::Dcc, 2, DccDist::MvNorm, {{"dcca1", 0.04060235353492948}, {"dcca2", 0.002921158960317734}, {"dccb1", 0.9135314210061152}}, {
        {1, {1.0, 0.6801978077915407, 0.47853895354800635, 0.6801978077915408, 0.9999999999999998, 0.5850161062508648, 0.47853895354800635, 0.5850161062508649, 1.0},
         {1.0583942053725395, 0.8540492490181528, 0.8287137648432045, 0.854049249018153, 1.4895254261157844, 1.2018639637326276, 0.8287137648432046, 1.201863963732628, 2.833525076071615},
         {1.0287828757189437, 1.2204611530547724, 1.6833077781771268}},
        {5, {1.0, 0.6470981053715998, 0.44869673189900355, 0.6470981053716, 0.9999999999999998, 0.5455988086578509, 0.44869673189900355, 0.545598808657851, 1.0},
         {1.0493516577432145, 0.7665175164961837, 0.7635099113188925, 0.7665175164961839, 1.3371584039301232, 1.0480115976850213, 0.7635099113188925, 1.0480115976850215, 2.75932269272048},
         {1.0243786691176338, 1.1563556563316164, 1.6611209145394805}},
        {10, {1.0, 0.6131149593282672, 0.41805801002503884, 0.6131149593282673, 1.0, 0.5051294484278962, 0.41805801002503884, 0.5051294484278962, 1.0},
         {1.039057818685327, 0.6844143187177396, 0.6967109027486912, 0.6844143187177397, 1.199263549722033, 0.9043909453380133, 0.6967109027486912, 0.9043909453380136, 2.6729596386556964},
         {1.0193418556526201, 1.0951089213964211, 1.6349188477278302}},
    }},
    {"dcc_mvt", DccModel::Dcc, 1, DccDist::MvT, {{"dcca1", 0.04306513249227842}, {"dccb1", 0.9135411753302598}, {"mshape", 49.999999327155265}}, {
        {1, {0.9999999999999998, 0.6777913790675022, 0.4758364918794121, 0.6777913790675022, 1.0, 0.582011620919586, 0.4758364918794122, 0.582011620919586, 0.9999999999999998},
         {1.0583942053725393, 0.8510277622961452, 0.8240337546431561, 0.8510277622961452, 1.4895254261157846, 1.1956915137596371, 0.8240337546431562, 1.1956915137596371, 2.8335250760716146},
         {1.0287828757189434, 1.2204611530547724, 1.6833077781771266}},
        {5, {0.9999999999999998, 0.644759835849805, 0.44614236458261564, 0.644759835849805, 1.0, 0.5426980528674507, 0.4461423645826157, 0.5426980528674507, 0.9999999999999998},
         {1.0493516577432143, 0.7637477285276103, 0.759163356899035, 0.7637477285276103, 1.3371584039301234, 1.0424396908880291, 0.759163356899035, 1.0424396908880291, 2.7593226927204793},
         {1.0243786691176335, 1.1563556563316164, 1.6611209145394803}},
        {10, {0.9999999999999999, 0.6109176216576865, 0.4157194742682505, 0.6109176216576865, 1.0, 0.5024196383256809, 0.4157194742682505, 0.5024196383256809, 0.9999999999999999},
         {1.0390578186853268, 0.6819614518582342, 0.6928136365340706, 0.6819614518582342, 1.199263549722033, 0.8995392628086017, 0.6928136365340707, 0.8995392628086019, 2.672959638655696},
         {1.01934185565262, 1.0951089213964211, 1.6349188477278302}},
    }},
    {"adcc_mvnorm", DccModel::ADcc, 1, DccDist::MvNorm, {{"dcca1", 0.04243294015912483}, {"dccb1", 0.9139819885859802}, {"dccg1", 0.0024125702667934694}}, {
        {1, {0.9999999999999998, 0.6788794589408066, 0.4764444746472641, 0.6788794589408066, 0.9999999999999998, 0.5825046075474282, 0.4764444746472641, 0.5825046075474282, 1.0},
         {1.0583942053725393, 0.8523939439980307, 0.8250866337970281, 0.8523939439980307, 1.4895254261157844, 1.1967043112814058, 0.8250866337970281, 1.1967043112814058, 2.833525076071615},
         {1.0287828757189434, 1.2204611530547724, 1.6833077781771268}},
    }},
    {"fdcc_mvnorm", DccModel::FDcc, 1, DccDist::MvNorm, {{"fdcca1", 0.19005046485228047}, {"fdcca2", 0.2410866956437819}, {"fdccb1", 0.9646299453228471}, {"fdccb2", 0.942962587187677}}, {
        {1, {1.0, 0.6804184809641098, 0.48196931996297504, 0.6804184809641098, 1.0, 0.5916391057050421, 0.48196931996297504, 0.5916391057050421, 1.0000000000000002},
         {1.0583942053725395, 0.8543263239442291, 0.8346543300688005, 0.8543263239442291, 1.4895254261157846, 1.2154703316441178, 0.8346543300688005, 1.2154703316441178, 2.833525076071616},
         {1.0287828757189437, 1.2204611530547724, 1.683307778177127}},
    }},
    };
    return cases;
}

TEST(DccForecastOracle, MatchesDccforecast)
{
    const std::vector<FitResult> fits = univariate_fits();
    std::vector<const FitResult*> pointers;
    for (const FitResult& fit : fits)
    {
        pointers.push_back(&fit);
    }
    ve::Context context;
    for (const Oracle& oracle : oracles())
    {
        SCOPED_TRACE(oracle.id);
        DccSpec spec;
        spec.model = oracle.model;
        spec.p = oracle.order;
        spec.dist = oracle.dist;
        spec.convention = Convention::Rmgarch;
        if (oracle.model == DccModel::FDcc)
        {
            spec.groups = {1, 1, 2};
        }
        for (const auto& [name, value] : oracle.parameters)
        {
            spec.fixed.push_back({name, value});
        }
        const std::vector<DccSpec> specs{spec};
        const DccOptions options{FitConfig{}, false};
        ASSERT_FALSE(ve::dcc::check_dcc_batch(pointers, specs, options).has_value());
        const std::vector<DccFitResult> results = ve::dcc::fit_dcc_batch(context, pointers, specs, options);
        const std::size_t horizon = oracle.steps.back().h;
        const ve::dcc::future::SeriesRegressors none(kSeries);
        ASSERT_FALSE(ve::dcc::future::forecast_problem(results[0], horizon, none).has_value());
        const ve::dcc::future::DccForecast forecast = ve::dcc::future::forecast_dcc(results[0], horizon, none);
        for (const Step& step : oracle.steps)
        {
            SCOPED_TRACE(step.h);
            const std::size_t offset = (step.h - 1) * 9;
            for (std::size_t e = 0; e < 9; ++e)
            {
                EXPECT_NEAR(forecast.r[offset + e], step.r[e], 1e-8) << e;
                EXPECT_NEAR(forecast.h[offset + e], step.covariance[e], 1e-8) << e;
            }
            for (std::size_t i = 0; i < kSeries; ++i)
            {
                const double sigma = forecast.sigma[(step.h - 1) * kSeries + i];
                EXPECT_NEAR(sigma, step.sigma[i], 1e-8 * step.sigma[i]) << i;
            }
        }
    }
}

}
