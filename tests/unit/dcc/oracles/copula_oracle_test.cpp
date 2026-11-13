#include "../copula/copula3_panel.hpp"
#include "core/context.hpp"
#include "dcc/copula/fit.hpp"
#include "dcc/fit.hpp"
#include "dcc/spec.hpp"
#include "dcc/standard_errors.hpp"
#include "fit/fit_result.hpp"

#include <cmath>
#include <cstddef>
#include <expected>
#include <gtest/gtest.h>
#include <string>
#include <utility>
#include <vector>

namespace
{

using ve::FitConfig;
using ve::FitResult;
using ve::FitStatus;
using ve::dcc::Convention;
using ve::dcc::DccCopula;
using ve::dcc::DccDist;
using ve::dcc::DccFitResult;
using ve::dcc::DccModel;
using ve::dcc::DccOptions;
using ve::dcc::DccSpec;
using ve::dcc::StaticMethod;

// rmgarch 1.4.3's cgarchfit on copula3, copied from tests/reference/fixtures/rmgarch_copula.json: the [Joint]
// estimates and robust errors in table order, and likelihood(fit).
struct Oracle
{
    const char* id;
    DccModel model;
    DccDist dist;
    DccCopula copula;
    StaticMethod method;
    bool fixed_shape;
    std::vector<double> estimate;
    std::vector<double> robust;
    double llh;
};

const std::vector<Oracle> kOracles{
    {"static_mvnorm_kendall_param", DccModel::Static, DccDist::MvNorm, DccCopula::Parametric, StaticMethod::Kendall, false, {}, {}, -8117.18691650443},
    {"static_mvnorm_ml_param", DccModel::Static, DccDist::MvNorm, DccCopula::Parametric, StaticMethod::Ml, false, {0.5986723962798544, 0.4493667562788657, 0.26583249745036047}, {0.028647345451266668, 0.028057928426636, 0.026965886681679326}, -8117.186856431025},
    {"static_mvnorm_ml_emp", DccModel::Static, DccDist::MvNorm, DccCopula::Empirical, StaticMethod::Ml, false, {0.5972262542776872, 0.45016498200549426, 0.26783342956413914}, {0.02911070966577508, 0.027342343591526458, 0.02687689042190415}, -8115.748072645789},
    {"static_mvt_kendall_param", DccModel::Static, DccDist::MvT, DccCopula::Parametric, StaticMethod::Kendall, false, {4.779357880806603}, {0.3902939004743996}, -8012.184133365942},
    {"static_mvt_kendall_emp", DccModel::Static, DccDist::MvT, DccCopula::Empirical, StaticMethod::Kendall, false, {4.95613482263086}, {0.392780601957729}, -8015.15113396422},
    {"static_mvt_ml_param", DccModel::Static, DccDist::MvT, DccCopula::Parametric, StaticMethod::Ml, false, {0.594217920400736, 0.43693669887169084, 0.27781002657349496, 4.790827028666584}, {0.027200410037139588, 0.026900527736183726, 0.025353814473267743, 0.406471562914237}, -8012.073664539513},
    {"dcc_mvnorm_param", DccModel::Dcc, DccDist::MvNorm, DccCopula::Parametric, StaticMethod::Default, false, {0.04738195889560655, 0.9175354409917849}, {0.006882604761282959, 0.01506025946251671}, -7999.255754625151},
    {"dcc_mvnorm_emp", DccModel::Dcc, DccDist::MvNorm, DccCopula::Empirical, StaticMethod::Default, false, {0.04694717616096846, 0.9179675558919811}, {0.007041902130832463, 0.01539842075082857}, -8000.3387982860495},
    {"dcc_mvt_param", DccModel::Dcc, DccDist::MvT, DccCopula::Parametric, StaticMethod::Default, false, {0.044450444332248755, 0.9194035843020166, 5.884183649931907}, {0.005821997841870897, 0.012185085306975704, 0.4971728250148147}, -7916.640978948953},
    {"dcc_mvt_emp", DccModel::Dcc, DccDist::MvT, DccCopula::Empirical, StaticMethod::Default, false, {0.04511900986702971, 0.9172060812961198, 6.3934861824246285}, {0.006298715708712233, 0.012873964209987345, 0.5173895861989676}, -7922.621862357918},
    {"adcc_mvnorm_param", DccModel::ADcc, DccDist::MvNorm, DccCopula::Parametric, StaticMethod::Default, false, {0.04531383908632312, 0.9178306244174149, 0.004882436025713399}, {0.008003332651169768, 0.014907340070356906, 0.009754484376388954}, -7999.081029767403},
    {"adcc_mvt_param", DccModel::ADcc, DccDist::MvT, DccCopula::Parametric, StaticMethod::Default, false, {0.04445044735004595, 0.9194035593156392, 4.756155508489455e-08, 5.884189544486187}, {0.006920098488837153, 0.01218639325288598, 0.009131826616165227, 0.49737239024369195}, -7916.640979305362},
    {"dcc_mvt_param_shape8", DccModel::Dcc, DccDist::MvT, DccCopula::Parametric, StaticMethod::Default, true, {0.04637378836757718, 0.9188279296008739}, {0.006086169918291443, 0.012812149542380044}, -7921.316326353931},
};

DccSpec spec_of(const Oracle& oracle)
{
    DccSpec spec;
    spec.model = oracle.model;
    spec.dist = oracle.dist;
    spec.copula = oracle.copula;
    spec.static_method = oracle.method;
    spec.convention = Convention::Rmgarch;
    if (oracle.fixed_shape)
    {
        spec.fixed = {{"mshape", 8.0}};
    }
    return spec;
}

std::vector<const FitResult*> pointers_of(const std::vector<FitResult>& fits)
{
    std::vector<const FitResult*> pointers;
    for (const FitResult& fit : fits)
    {
        pointers.push_back(&fit);
    }
    return pointers;
}

// An estimate at most rmgarch's boundary noise from zero (aDCC-t's g of 4.8e-8) is compared absolutely.
void expect_estimate(double actual, double expected, const std::string& name)
{
    if (std::abs(expected) < 1e-6)
    {
        EXPECT_NEAR(actual, expected, 1e-6) << name;
        return;
    }
    EXPECT_NEAR(actual, expected, 1e-3 * std::abs(expected)) << name;
}

}

TEST(CopulaOracle, RmgarchEstimates)
{
    const std::vector<FitResult> fits = ve::testing::copula3_fits();
    const std::vector<const FitResult*> pointers = pointers_of(fits);
    for (const Oracle& oracle : kOracles)
    {
        SCOPED_TRACE(oracle.id);
        const DccSpec spec = spec_of(oracle);
        const std::vector<DccSpec> specs{spec};
        const ve::dcc::copula::CopulaInputs inputs = ve::dcc::copula::copula_inputs(pointers, specs);
        // As the plan 4a oracle: a tight tolerance, so that the comparison measures the estimate, not the stop.
        FitConfig config;
        config.tol = 1e-10;
        config.max_iter = 2000;
        const DccFitResult fit = ve::dcc::copula::fit_copula(inputs, spec, config);
        ASSERT_EQ(fit.status, FitStatus::Converged);
        ASSERT_EQ(fit.estimate.size(), oracle.estimate.size());
        // The objective is -sum c_t: no worse than at rmgarch's point.
        EXPECT_GE(fit.loglik, oracle.llh - 1e-7);
        EXPECT_NEAR(fit.loglik, oracle.llh, 1e-7 * std::abs(oracle.llh));
        for (std::size_t k = 0; k < oracle.estimate.size(); ++k)
        {
            expect_estimate(fit.estimate[k], oracle.estimate[k], fit.names[k]);
        }
    }
}

TEST(CopulaOracle, RmgarchRobustErrors)
{
    const std::vector<FitResult> fits = ve::testing::copula3_estimated_fits();
    const std::vector<const FitResult*> pointers = pointers_of(fits);
    for (const Oracle& oracle : kOracles)
    {
        if (oracle.estimate.empty())
        {
            continue;
        }
        SCOPED_TRACE(oracle.id);
        const std::vector<DccSpec> specs{spec_of(oracle)};
        ve::Context context;
        const DccOptions options{FitConfig{}, false};
        ASSERT_FALSE(ve::dcc::check_dcc_batch(pointers, specs, options).has_value());
        DccFitResult fit = ve::dcc::fit_dcc_batch(context, pointers, specs, options)[0];
        fit.estimate = oracle.estimate;
        const std::expected<ve::dcc::DccStandardErrors, std::string> errors = ve::dcc::dcc_standard_errors(fit, pointers);
        ASSERT_TRUE(errors.has_value()) << errors.error();
        for (std::size_t k = 0; k < oracle.robust.size(); ++k)
        {
            EXPECT_NEAR(errors->robust[k], oracle.robust[k], 5e-3 * oracle.robust[k]) << fit.names[k];
        }
    }
}
