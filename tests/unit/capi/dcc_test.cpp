#include "capi/handles.hpp"
#include "capi_scope.hpp"
#include "core/context.hpp"
#include "dcc/dcc3_panel.hpp"
#include "dcc/fit.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <memory>
#include <string>
#include <thread>
#include <vector>
#include <volengine/volengine.h>

namespace
{

using ve::dcc::DccDist;
using ve::dcc::DccFitResult;
using ve::dcc::DccModel;
using ve::dcc::DccSpec;
using ve::testing::FitSet;

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr std::size_t kSeries = ve::testing::kDcc3Series;
constexpr std::size_t kRows = ve::testing::kDcc3Rows;

template <typename Handle, void (*Free)(Handle*)>
struct Release
{
    void operator()(Handle* handle) const
    {
        Free(handle);
    }
};

using DccSpecPtr = std::unique_ptr<ve_dcc_spec, Release<ve_dcc_spec, ve_dcc_spec_free>>;
using DccOptionsPtr = std::unique_ptr<ve_dcc_options, Release<ve_dcc_options, ve_dcc_options_free>>;
using DccBatchPtr = std::unique_ptr<ve_dcc_batch, Release<ve_dcc_batch, ve_dcc_batch_free>>;
using DccFitPtr = std::unique_ptr<ve_dcc_fit, Release<ve_dcc_fit, ve_dcc_fit_free>>;

bool mentions(const std::string& word)
{
    return std::string(ve_last_error()).find(word) != std::string::npos;
}

bool same_bits(const std::vector<double>& a, const std::vector<double>& b)
{
    if (a.size() != b.size())
    {
        return false;
    }
    for (std::size_t i = 0; i < a.size(); ++i)
    {
        if (std::bit_cast<std::uint64_t>(a[i]) != std::bit_cast<std::uint64_t>(b[i]))
        {
            return false;
        }
    }
    return true;
}

// The columns of dcc3, T x 3 in C order, kept alive for the borrowed panel.
const std::vector<double>& dcc3_rows()
{
    static const std::vector<double> rows = []
    {
        const std::vector<std::vector<double>> columns = ve::testing::read_dcc3();
        std::vector<double> flat(kRows * kSeries);
        for (std::size_t t = 0; t < kRows; ++t)
        {
            for (std::size_t i = 0; i < kSeries; ++i)
            {
                flat[t * kSeries + i] = columns[i][t];
            }
        }
        return flat;
    }();
    return rows;
}

// The univariate sGARCH(1,1) fits of the three series, estimated through the C API; with `regressors`, each has
// one mean regressor.
FitSet univariate_fits(bool regressors)
{
    ve::testing::ContextPtr context = ve::testing::make_context(1);
    ve::testing::SpecPtr spec = ve::testing::make_spec(1);
    EXPECT_EQ(ve_spec_set_mxreg(spec.get(), regressors ? 1 : 0), VE_OK);
    ve::testing::PanelPtr panel = ve::testing::make_panel(dcc3_rows().data(), kRows, kSeries);
    std::vector<double> x(kRows);
    for (std::size_t t = 0; t < kRows; ++t)
    {
        x[t] = std::sin(0.3 * static_cast<double>(t));
    }
    ve::testing::PanelPtr mean = ve::testing::make_panel(x.data(), kRows, 1);
    ve::testing::BatchInputs inputs;
    inputs.specs = {spec.get()};
    inputs.panel = panel.get();
    inputs.job_spec = {0, 0, 0};
    inputs.job_series = {0, 1, 2};
    if (regressors)
    {
        inputs.mxreg = {mean.get(), mean.get(), mean.get()};
    }
    ve::testing::BatchPtr batch = ve::testing::build_batch(inputs);
    return ve::testing::fit_all(context.get(), batch.get());
}

std::vector<const ve_fit*> handles_of(const FitSet& fits)
{
    std::vector<const ve_fit*> handles;
    for (std::size_t i = 0; i < fits.size(); ++i)
    {
        handles.push_back(fits[i]);
    }
    return handles;
}

std::vector<const ve::FitResult*> results_of(const FitSet& fits)
{
    std::vector<const ve::FitResult*> results;
    for (std::size_t i = 0; i < fits.size(); ++i)
    {
        results.push_back(&fits[i]->result);
    }
    return results;
}

DccSpecPtr make_dcc_spec(int32_t model, int32_t dist)
{
    ve_dcc_spec* raw = nullptr;
    EXPECT_EQ(ve_dcc_spec_create(model, &raw), VE_OK) << ve_last_error();
    DccSpecPtr spec(raw);
    EXPECT_EQ(ve_dcc_spec_set_dist(raw, dist), VE_OK);
    if (model == VE_DCC_MODEL_FDCC)
    {
        const int32_t groups[] = {1, 1, 2};
        EXPECT_EQ(ve_dcc_spec_set_groups(raw, groups, 3), VE_OK) << ve_last_error();
    }
    return spec;
}

DccOptionsPtr make_dcc_options(bool standard_errors)
{
    ve_dcc_options* raw = nullptr;
    EXPECT_EQ(ve_dcc_options_create(&raw), VE_OK);
    DccOptionsPtr options(raw);
    EXPECT_EQ(ve_dcc_options_set_standard_errors(raw, standard_errors ? 1 : 0), VE_OK);
    return options;
}

struct DccInputs
{
    std::vector<const ve_dcc_spec*> specs;
    std::vector<const ve_fit*> fits;
    const ve_dcc_options* options = nullptr;
};

DccBatchPtr build_dcc_batch(const DccInputs& inputs)
{
    ve_dcc_batch* raw = nullptr;
    EXPECT_EQ(ve_dcc_batch_create(&raw), VE_OK);
    DccBatchPtr batch(raw);
    if (!inputs.specs.empty())
    {
        EXPECT_EQ(ve_dcc_batch_set_specs(raw, inputs.specs.data(), inputs.specs.size()), VE_OK) << ve_last_error();
    }
    if (!inputs.fits.empty())
    {
        EXPECT_EQ(ve_dcc_batch_set_fits(raw, inputs.fits.data(), inputs.fits.size()), VE_OK) << ve_last_error();
    }
    if (inputs.options != nullptr)
    {
        EXPECT_EQ(ve_dcc_batch_set_options(raw, inputs.options), VE_OK);
    }
    return batch;
}

class DccFits
{
public:
    explicit DccFits(std::size_t count) : fits_(count, nullptr)
    {
    }

    DccFits(const DccFits&) = delete;
    DccFits& operator=(const DccFits&) = delete;
    DccFits(DccFits&&) = delete;
    DccFits& operator=(DccFits&&) = delete;

    ~DccFits()
    {
        for (ve_dcc_fit* fit : fits_)
        {
            ve_dcc_fit_free(fit);
        }
    }

    ve_dcc_fit** out()
    {
        return fits_.data();
    }

    ve_dcc_fit* operator[](std::size_t index) const
    {
        return fits_.at(index);
    }

    std::size_t size() const
    {
        return fits_.size();
    }

private:
    std::vector<ve_dcc_fit*> fits_;
};

std::size_t spec_count(const ve_dcc_batch* batch)
{
    std::size_t count = 0;
    EXPECT_EQ(ve_dcc_batch_spec_count(batch, &count), VE_OK);
    return count;
}

// The three specs of the tests: DCC with the normal, aDCC with the t, FDCC with the Laplace.
std::vector<DccSpecPtr> three_specs()
{
    std::vector<DccSpecPtr> specs;
    specs.push_back(make_dcc_spec(VE_DCC_MODEL_DCC, VE_DCC_DIST_MVNORM));
    specs.push_back(make_dcc_spec(VE_DCC_MODEL_ADCC, VE_DCC_DIST_MVT));
    specs.push_back(make_dcc_spec(VE_DCC_MODEL_FDCC, VE_DCC_DIST_MVLAPLACE));
    return specs;
}

std::vector<const ve_dcc_spec*> raw_specs(const std::vector<DccSpecPtr>& specs)
{
    std::vector<const ve_dcc_spec*> raw;
    for (const DccSpecPtr& spec : specs)
    {
        raw.push_back(spec.get());
    }
    return raw;
}

std::vector<DccSpec> core_specs()
{
    std::vector<DccSpec> specs(3);
    specs[0].model = DccModel::Dcc;
    specs[0].dist = DccDist::MvNorm;
    specs[1].model = DccModel::ADcc;
    specs[1].dist = DccDist::MvT;
    specs[2].model = DccModel::FDcc;
    specs[2].dist = DccDist::MvLaplace;
    specs[2].groups = {1, 1, 2};
    return specs;
}

using Reader = ve_status (*)(const ve_dcc_fit*, int32_t, double*);
using View = ve_status (*)(const ve_dcc_fit*, const double**, std::size_t*);

std::vector<double> read_table(const ve_dcc_fit* fit, Reader reader, int32_t kind)
{
    std::size_t count = 0;
    EXPECT_EQ(ve_dcc_fit_param_count(fit, &count), VE_OK);
    std::vector<double> values(count);
    EXPECT_EQ(reader(fit, kind, values.data()), VE_OK) << ve_last_error();
    return values;
}

std::vector<double> read_estimates(const ve_dcc_fit* fit)
{
    std::size_t count = 0;
    EXPECT_EQ(ve_dcc_fit_param_count(fit, &count), VE_OK);
    std::vector<double> values(count);
    EXPECT_EQ(ve_dcc_fit_estimates(fit, values.data()), VE_OK);
    return values;
}

std::vector<double> read_view(const ve_dcc_fit* fit, View view, const double** pointer = nullptr)
{
    const double* data = nullptr;
    std::size_t len = 0;
    EXPECT_EQ(view(fit, &data, &len), VE_OK) << ve_last_error();
    if (pointer != nullptr)
    {
        *pointer = data;
    }
    return {data, data + len};
}

double read_loglik(const ve_dcc_fit* fit)
{
    double value = 0.0;
    EXPECT_EQ(ve_dcc_fit_loglik(fit, &value), VE_OK);
    return value;
}

double read_criterion(const ve_dcc_fit* fit, int32_t which, int32_t scale)
{
    double value = 0.0;
    EXPECT_EQ(ve_dcc_fit_criterion(fit, which, scale, &value), VE_OK) << ve_last_error();
    return value;
}

int32_t read_status(const ve_dcc_fit* fit)
{
    int32_t status = -1;
    EXPECT_EQ(ve_dcc_fit_status(fit, &status), VE_OK);
    return status;
}

// What a handle reports, to compare with the core and across the lifetime of the inputs.
struct Report
{
    int32_t status = -1;
    std::vector<double> estimate;
    std::vector<double> se_hessian;
    std::vector<double> se_robust;
    double loglik = 0.0;
    std::vector<double> correlation;
    std::vector<double> covariance;
};

Report report_of(const ve_dcc_fit* fit)
{
    Report report;
    report.status = read_status(fit);
    report.estimate = read_estimates(fit);
    report.se_hessian = read_table(fit, ve_dcc_fit_std_errors, VE_SE_HESSIAN);
    report.se_robust = read_table(fit, ve_dcc_fit_std_errors, VE_SE_ROBUST);
    report.loglik = read_loglik(fit);
    report.correlation = read_view(fit, ve_dcc_fit_correlation);
    report.covariance = read_view(fit, ve_dcc_fit_covariance);
    return report;
}

void expect_same_report(const Report& actual, const Report& expected)
{
    EXPECT_EQ(actual.status, expected.status);
    EXPECT_TRUE(same_bits(actual.estimate, expected.estimate));
    EXPECT_TRUE(same_bits(actual.se_hessian, expected.se_hessian));
    EXPECT_TRUE(same_bits(actual.se_robust, expected.se_robust));
    EXPECT_EQ(std::bit_cast<std::uint64_t>(actual.loglik), std::bit_cast<std::uint64_t>(expected.loglik));
    EXPECT_TRUE(same_bits(actual.correlation, expected.correlation));
    EXPECT_TRUE(same_bits(actual.covariance, expected.covariance));
}

Report report_of_core(const DccFitResult& result)
{
    Report report;
    report.status = static_cast<int32_t>(result.status);
    report.estimate = result.estimate;
    report.se_hessian = result.se_hessian;
    report.se_robust = result.se_robust;
    report.loglik = result.loglik;
    if (result.r != nullptr)
    {
        report.correlation = *result.r;
    }
    report.covariance = ve::dcc::covariance_path(result);
    return report;
}

std::vector<DccFitResult> core_fits(const FitSet& fits, bool standard_errors)
{
    ve::Context context;
    ve::dcc::DccOptions options;
    options.standard_errors = standard_errors;
    const std::vector<const ve::FitResult*> results = results_of(fits);
    const std::vector<DccSpec> specs = core_specs();
    EXPECT_FALSE(ve::dcc::check_dcc_batch(results, specs, options).has_value());
    return ve::dcc::fit_dcc_batch(context, results, specs, options);
}

}

TEST(CapiDcc, MatchesCore)
{
    FitSet univariate = univariate_fits(false);
    const std::vector<DccSpecPtr> specs = three_specs();
    DccOptionsPtr options = make_dcc_options(true);
    DccBatchPtr batch = build_dcc_batch({raw_specs(specs), handles_of(univariate), options.get()});
    ve::testing::ContextPtr context = ve::testing::make_context(3);
    ASSERT_EQ(spec_count(batch.get()), 3U);
    DccFits fits(3);
    ASSERT_EQ(ve_dcc_fit_batch(context.get(), batch.get(), fits.out()), VE_OK) << ve_last_error();
    const std::vector<DccFitResult> expected = core_fits(univariate, true);
    for (std::size_t i = 0; i < fits.size(); ++i)
    {
        SCOPED_TRACE(i);
        const ve_dcc_fit* fit = fits[i];
        ASSERT_NE(fit, nullptr);
        expect_same_report(report_of(fit), report_of_core(expected[i]));
        std::size_t nobs = 0;
        std::size_t series = 0;
        EXPECT_EQ(ve_dcc_fit_nobs(fit, &nobs), VE_OK);
        EXPECT_EQ(ve_dcc_fit_series(fit, &series), VE_OK);
        EXPECT_EQ(nobs, kRows);
        EXPECT_EQ(series, kSeries);
        EXPECT_EQ(read_status(fit), VE_FIT_CONVERGED);
        EXPECT_EQ(std::bit_cast<std::uint64_t>(read_criterion(fit, VE_DCC_CRITERION_BIC, VE_IC_PER_OBS)),
                  std::bit_cast<std::uint64_t>(ve::dcc::dcc_criterion(expected[i], ve::dcc::Criterion::Bic, true)));
        EXPECT_EQ(std::bit_cast<std::uint64_t>(read_criterion(fit, VE_DCC_CRITERION_AIC, VE_IC_TOTAL)),
                  std::bit_cast<std::uint64_t>(ve::dcc::dcc_criterion(expected[i], ve::dcc::Criterion::Aic, false)));
        EXPECT_TRUE(std::isfinite(read_criterion(fit, VE_DCC_CRITERION_HQIC, VE_IC_PER_OBS)));
        EXPECT_TRUE(std::isfinite(read_criterion(fit, VE_DCC_CRITERION_SIC, VE_IC_PER_OBS)));
    }
}

TEST(CapiDcc, TablesAreConsistent)
{
    FitSet univariate = univariate_fits(false);
    const std::vector<DccSpecPtr> specs = three_specs();
    DccBatchPtr batch = build_dcc_batch({raw_specs(specs), handles_of(univariate), nullptr});
    ve::testing::ContextPtr context = ve::testing::make_context(1);
    DccFits fits(3);
    ASSERT_EQ(ve_dcc_fit_batch(context.get(), batch.get(), fits.out()), VE_OK) << ve_last_error();
    const std::vector<DccFitResult> expected = core_fits(univariate, true);
    for (std::size_t i = 0; i < fits.size(); ++i)
    {
        SCOPED_TRACE(i);
        std::size_t count = 0;
        ASSERT_EQ(ve_dcc_fit_param_count(fits[i], &count), VE_OK);
        ASSERT_EQ(count, expected[i].names.size());
        for (std::size_t p = 0; p < count; ++p)
        {
            const char* name = nullptr;
            ASSERT_EQ(ve_dcc_fit_param_name(fits[i], p, &name), VE_OK);
            EXPECT_EQ(std::string(name), expected[i].names[p]);
        }
        const std::vector<double> estimate = read_estimates(fits[i]);
        for (const int32_t kind : {VE_SE_HESSIAN, VE_SE_ROBUST})
        {
            const std::vector<double> se = read_table(fits[i], ve_dcc_fit_std_errors, kind);
            const std::vector<double> t = read_table(fits[i], ve_dcc_fit_t_values, kind);
            const std::vector<double> p = read_table(fits[i], ve_dcc_fit_p_values, kind);
            ASSERT_EQ(se.size(), count);
            for (std::size_t k = 0; k < count; ++k)
            {
                EXPECT_TRUE(std::isfinite(se[k]));
                EXPECT_DOUBLE_EQ(t[k], estimate[k] / se[k]);
                EXPECT_NEAR(p[k], std::erfc(std::fabs(t[k]) / std::sqrt(2.0)), 1e-15);
            }
        }
    }
}

TEST(CapiDcc, ViewsAreZeroCopyAndStable)
{
    FitSet univariate = univariate_fits(false);
    const std::vector<DccSpecPtr> specs = three_specs();
    DccBatchPtr batch = build_dcc_batch({raw_specs(specs), handles_of(univariate), nullptr});
    ve::testing::ContextPtr context = ve::testing::make_context(1);
    DccFits fits(3);
    ASSERT_EQ(ve_dcc_fit_batch(context.get(), batch.get(), fits.out()), VE_OK) << ve_last_error();
    const double* r = nullptr;
    const double* h = nullptr;
    const std::vector<double> correlation = read_view(fits[0], ve_dcc_fit_correlation, &r);
    const std::vector<double> covariance = read_view(fits[0], ve_dcc_fit_covariance, &h);
    EXPECT_EQ(correlation.size(), kRows * kSeries * kSeries);
    EXPECT_EQ(covariance.size(), kRows * kSeries * kSeries);
    EXPECT_EQ(r, fits[0]->result.r->data());
    const double* again = nullptr;
    read_view(fits[0], ve_dcc_fit_covariance, &again);
    EXPECT_EQ(again, h);
    read_view(fits[0], ve_dcc_fit_correlation, &again);
    EXPECT_EQ(again, r);
    for (std::size_t i = 0; i < kSeries; ++i)
    {
        const double* data = nullptr;
        std::size_t len = 0;
        EXPECT_EQ(ve_fit_sigma(univariate[i], &data, &len), VE_OK);
        for (std::size_t t : {std::size_t{0}, std::size_t{700}, kRows - 1})
        {
            EXPECT_DOUBLE_EQ(covariance[t * 9 + i * 4], data[t] * data[t] * correlation[t * 9 + i * 4]);
        }
    }
}

TEST(CapiDcc, OutlivesInputs)
{
    std::vector<DccFitResult> expected;
    DccFits fits(3);
    {
        FitSet univariate = univariate_fits(false);
        expected = core_fits(univariate, true);
        std::vector<DccSpecPtr> specs = three_specs();
        DccOptionsPtr options = make_dcc_options(true);
        DccBatchPtr batch = build_dcc_batch({raw_specs(specs), handles_of(univariate), options.get()});
        ve::testing::ContextPtr context = ve::testing::make_context(2);
        ASSERT_EQ(ve_dcc_fit_batch(context.get(), batch.get(), fits.out()), VE_OK) << ve_last_error();
        EXPECT_EQ(read_status(fits[1]), VE_FIT_CONVERGED);
        specs.clear();
        options.reset();
        batch.reset();
        context.reset();
    }
    for (std::size_t i = 0; i < fits.size(); ++i)
    {
        SCOPED_TRACE(i);
        expect_same_report(report_of(fits[i]), report_of_core(expected[i]));
        EXPECT_TRUE(std::isfinite(read_criterion(fits[i], VE_DCC_CRITERION_BIC, VE_IC_PER_OBS)));
    }
}

TEST(CapiDcc, StandardErrorsLater)
{
    FitSet univariate = univariate_fits(false);
    const std::vector<DccSpecPtr> specs = three_specs();
    DccOptionsPtr off = make_dcc_options(false);
    DccOptionsPtr on = make_dcc_options(true);
    DccBatchPtr without = build_dcc_batch({raw_specs(specs), handles_of(univariate), off.get()});
    DccBatchPtr with = build_dcc_batch({raw_specs(specs), handles_of(univariate), on.get()});
    ve::testing::ContextPtr context = ve::testing::make_context(2);
    DccFits plain(3);
    DccFits full(3);
    ASSERT_EQ(ve_dcc_fit_batch(context.get(), without.get(), plain.out()), VE_OK) << ve_last_error();
    ASSERT_EQ(ve_dcc_fit_batch(context.get(), with.get(), full.out()), VE_OK) << ve_last_error();
    const std::vector<const ve_fit*> fits = handles_of(univariate);
    for (std::size_t i = 0; i < plain.size(); ++i)
    {
        SCOPED_TRACE(i);
        for (const int32_t kind : {VE_SE_HESSIAN, VE_SE_ROBUST})
        {
            for (const double value : read_table(plain[i], ve_dcc_fit_std_errors, kind))
            {
                EXPECT_TRUE(std::isnan(value));
            }
            for (const double value : read_table(plain[i], ve_dcc_fit_t_values, kind))
            {
                EXPECT_TRUE(std::isnan(value));
            }
            for (const double value : read_table(plain[i], ve_dcc_fit_p_values, kind))
            {
                EXPECT_TRUE(std::isnan(value));
            }
        }
        ve_dcc_fit* later = nullptr;
        ASSERT_EQ(ve_dcc_fit_standard_errors(context.get(), plain[i], fits.data(), fits.size(), &later), VE_OK)
            << ve_last_error();
        DccFitPtr owner(later);
        ASSERT_NE(later, nullptr);
        EXPECT_NE(later, plain[i]);
        expect_same_report(report_of(later), report_of(full[i]));
        EXPECT_TRUE(std::isnan(read_table(plain[i], ve_dcc_fit_std_errors, VE_SE_ROBUST).front()));
    }
}

TEST(CapiDcc, ANonConvergedSpecKeepsItsOwnStatus)
{
    FitSet univariate = univariate_fits(false);
    std::vector<DccSpecPtr> specs;
    specs.push_back(make_dcc_spec(VE_DCC_MODEL_DCC, VE_DCC_DIST_MVNORM));
    specs.push_back(make_dcc_spec(VE_DCC_MODEL_DCC, VE_DCC_DIST_MVNORM));
    ASSERT_EQ(ve_dcc_spec_set_fixed(specs[1].get(), "dccb1", 1.2), VE_OK);
    DccBatchPtr batch = build_dcc_batch({raw_specs(specs), handles_of(univariate), nullptr});
    ve::testing::ContextPtr context = ve::testing::make_context(2);
    DccFits fits(2);
    ASSERT_EQ(ve_dcc_fit_batch(context.get(), batch.get(), fits.out()), VE_OK) << ve_last_error();
    EXPECT_EQ(read_status(fits[0]), VE_FIT_CONVERGED);
    EXPECT_EQ(read_status(fits[1]), VE_FIT_FAILED);
    const double* data = &kNaN;
    EXPECT_TRUE(read_view(fits[1], ve_dcc_fit_correlation, &data).empty());
    EXPECT_EQ(data, nullptr);
    EXPECT_TRUE(read_view(fits[1], ve_dcc_fit_covariance, &data).empty());
    EXPECT_EQ(data, nullptr);
    EXPECT_TRUE(std::isnan(read_loglik(fits[1])));
    EXPECT_TRUE(std::isnan(read_criterion(fits[1], VE_DCC_CRITERION_AIC, VE_IC_PER_OBS)));
    for (const double value : read_table(fits[1], ve_dcc_fit_std_errors, VE_SE_ROBUST))
    {
        EXPECT_TRUE(std::isnan(value));
    }
}

TEST(CapiDcc, SpecSettersRejectBadArguments)
{
    ve_dcc_spec* spec = reinterpret_cast<ve_dcc_spec*>(&spec);
    EXPECT_EQ(ve_dcc_spec_create(4, &spec), VE_INVALID_ARGUMENT);
    EXPECT_EQ(spec, nullptr);
    EXPECT_EQ(ve_dcc_spec_create(-1, &spec), VE_INVALID_ARGUMENT);
    EXPECT_EQ(ve_dcc_spec_create(VE_DCC_MODEL_DCC, nullptr), VE_INVALID_ARGUMENT);
    EXPECT_TRUE(mentions("out"));
    DccSpecPtr dcc = make_dcc_spec(VE_DCC_MODEL_DCC, VE_DCC_DIST_MVNORM);
    ve_dcc_spec* raw = dcc.get();
    EXPECT_EQ(ve_dcc_spec_set_order(nullptr, 1, 1), VE_INVALID_HANDLE);
    EXPECT_EQ(ve_dcc_spec_set_order(raw, 0, 1), VE_INVALID_ARGUMENT);
    EXPECT_EQ(ve_dcc_spec_set_order(raw, 1, VE_MAX_ORDER + 1), VE_INVALID_ARGUMENT);
    EXPECT_EQ(ve_dcc_spec_set_order(raw, 2, 3), VE_OK);
    EXPECT_EQ(ve_dcc_spec_set_dist(nullptr, VE_DCC_DIST_MVT), VE_INVALID_HANDLE);
    EXPECT_EQ(ve_dcc_spec_set_dist(raw, 3), VE_INVALID_ARGUMENT);
    EXPECT_EQ(ve_dcc_spec_set_dist(raw, -1), VE_INVALID_ARGUMENT);
    EXPECT_EQ(ve_dcc_spec_set_fixed(nullptr, "dcca1", 0.1), VE_INVALID_HANDLE);
    EXPECT_EQ(ve_dcc_spec_set_fixed(raw, nullptr, 0.1), VE_INVALID_ARGUMENT);
    EXPECT_EQ(ve_dcc_spec_set_fixed(raw, "", 0.1), VE_INVALID_ARGUMENT);
    EXPECT_EQ(ve_dcc_spec_set_fixed(raw, "dcca1", kNaN), VE_INVALID_ARGUMENT);
    EXPECT_EQ(ve_dcc_spec_set_start(raw, "dcca1", std::numeric_limits<double>::infinity()), VE_INVALID_ARGUMENT);
    EXPECT_EQ(ve_dcc_spec_set_start(nullptr, "dcca1", 0.1), VE_INVALID_HANDLE);
    const int32_t groups[] = {1, 1, 2};
    EXPECT_EQ(ve_dcc_spec_set_groups(raw, groups, 3), VE_INVALID_ARGUMENT);
    EXPECT_TRUE(mentions("FDCC"));
    ve_dcc_spec_free(nullptr);
    DccSpecPtr fdcc = make_dcc_spec(VE_DCC_MODEL_FDCC, VE_DCC_DIST_MVNORM);
    EXPECT_EQ(ve_dcc_spec_set_groups(nullptr, groups, 3), VE_INVALID_HANDLE);
    EXPECT_EQ(ve_dcc_spec_set_groups(fdcc.get(), nullptr, 3), VE_INVALID_ARGUMENT);
    EXPECT_EQ(ve_dcc_spec_set_groups(fdcc.get(), groups, 0), VE_INVALID_ARGUMENT);
    ve_dcc_options* options = reinterpret_cast<ve_dcc_options*>(&options);
    EXPECT_EQ(ve_dcc_options_create(nullptr), VE_INVALID_ARGUMENT);
    ASSERT_EQ(ve_dcc_options_create(&options), VE_OK);
    DccOptionsPtr owner(options);
    EXPECT_EQ(ve_dcc_options_set_standard_errors(options, 2), VE_INVALID_ARGUMENT);
    EXPECT_EQ(ve_dcc_options_set_standard_errors(nullptr, 1), VE_INVALID_HANDLE);
    ve_dcc_options_free(nullptr);
}

TEST(CapiDcc, BatchSettersRejectBadArguments)
{
    ve_dcc_batch* batch = nullptr;
    EXPECT_EQ(ve_dcc_batch_create(nullptr), VE_INVALID_ARGUMENT);
    ASSERT_EQ(ve_dcc_batch_create(&batch), VE_OK);
    DccBatchPtr owner(batch);
    DccSpecPtr spec = make_dcc_spec(VE_DCC_MODEL_DCC, VE_DCC_DIST_MVNORM);
    const ve_dcc_spec* specs[] = {spec.get(), nullptr};
    EXPECT_EQ(ve_dcc_batch_set_specs(nullptr, specs, 1), VE_INVALID_HANDLE);
    EXPECT_EQ(ve_dcc_batch_set_specs(batch, specs, 0), VE_INVALID_ARGUMENT);
    EXPECT_EQ(ve_dcc_batch_set_specs(batch, nullptr, 1), VE_INVALID_ARGUMENT);
    EXPECT_EQ(ve_dcc_batch_set_specs(batch, specs, 2), VE_INVALID_HANDLE);
    EXPECT_EQ(spec_count(batch), 0U);
    FitSet univariate = univariate_fits(false);
    const std::vector<const ve_fit*> fits = handles_of(univariate);
    const ve_fit* with_null[] = {fits[0], nullptr};
    EXPECT_EQ(ve_dcc_batch_set_fits(nullptr, fits.data(), 3), VE_INVALID_HANDLE);
    EXPECT_EQ(ve_dcc_batch_set_fits(batch, fits.data(), 0), VE_INVALID_ARGUMENT);
    EXPECT_EQ(ve_dcc_batch_set_fits(batch, nullptr, 3), VE_INVALID_ARGUMENT);
    EXPECT_EQ(ve_dcc_batch_set_fits(batch, with_null, 2), VE_INVALID_HANDLE);
    EXPECT_EQ(ve_dcc_batch_set_options(batch, nullptr), VE_INVALID_HANDLE);
    EXPECT_EQ(ve_dcc_batch_set_options(nullptr, nullptr), VE_INVALID_HANDLE);
    std::size_t count = 7;
    EXPECT_EQ(ve_dcc_batch_spec_count(nullptr, &count), VE_INVALID_HANDLE);
    EXPECT_EQ(ve_dcc_batch_spec_count(batch, nullptr), VE_INVALID_ARGUMENT);
    ve_dcc_batch_free(nullptr);
}

namespace
{

struct Attempt
{
    ve_status status;
    ve_dcc_fit* fit;
};

// Runs the batch with a fit slot set to a sentinel, to see that a failure leaves it NULL.
Attempt attempt(const DccInputs& inputs)
{
    DccBatchPtr batch = build_dcc_batch(inputs);
    ve::testing::ContextPtr context = ve::testing::make_context(1);
    const std::size_t count = std::max(std::size_t{1}, inputs.specs.size());
    std::vector<ve_dcc_fit*> slots(count, reinterpret_cast<ve_dcc_fit*>(&context));
    const ve_status status = ve_dcc_fit_batch(context.get(), batch.get(), slots.data());
    return {status, slots.front()};
}

DccInputs one_dcc(const DccSpecPtr& spec, const FitSet& univariate)
{
    return {{spec.get()}, handles_of(univariate), nullptr};
}

}

TEST(CapiDcc, FitBatchRejectsInvalidArguments)
{
    FitSet univariate = univariate_fits(false);
    DccSpecPtr dcc = make_dcc_spec(VE_DCC_MODEL_DCC, VE_DCC_DIST_MVNORM);
    DccBatchPtr batch = build_dcc_batch(one_dcc(dcc, univariate));
    ve::testing::ContextPtr context = ve::testing::make_context(1);
    ve_dcc_fit* out = nullptr;
    EXPECT_EQ(ve_dcc_fit_batch(nullptr, batch.get(), &out), VE_INVALID_HANDLE);
    EXPECT_EQ(ve_dcc_fit_batch(context.get(), nullptr, &out), VE_INVALID_HANDLE);
    EXPECT_EQ(ve_dcc_fit_batch(context.get(), batch.get(), nullptr), VE_INVALID_ARGUMENT);
    DccBatchPtr empty = build_dcc_batch({});
    EXPECT_EQ(ve_dcc_fit_batch(context.get(), empty.get(), &out), VE_INVALID_ARGUMENT);
    EXPECT_EQ(out, nullptr);
    DccBatchPtr no_specs = build_dcc_batch({{}, handles_of(univariate), nullptr});
    EXPECT_EQ(ve_dcc_fit_batch(context.get(), no_specs.get(), &out), VE_INVALID_ARGUMENT);
    EXPECT_TRUE(mentions("spec"));
}

TEST(CapiDcc, FitBatchNamesTheSeriesThatCannotFormAPanel)
{
    FitSet univariate = univariate_fits(false);
    DccSpecPtr dcc = make_dcc_spec(VE_DCC_MODEL_DCC, VE_DCC_DIST_MVNORM);
    const std::vector<const ve_fit*> fits = handles_of(univariate);
    Attempt one = attempt({{dcc.get()}, {fits[0]}, nullptr});
    EXPECT_EQ(one.status, VE_INVALID_ARGUMENT);
    EXPECT_EQ(one.fit, nullptr);
    EXPECT_TRUE(mentions("at least 2 series"));
    auto no_estimate = std::make_unique<ve_fit>();
    Attempt failed = attempt({{dcc.get()}, {fits[0], no_estimate.get(), fits[2]}, nullptr});
    EXPECT_EQ(failed.status, VE_INVALID_ARGUMENT);
    EXPECT_EQ(failed.fit, nullptr);
    EXPECT_TRUE(mentions("series 2"));
    auto shorter = std::make_unique<ve_fit>();
    shorter->result = univariate[2]->result;
    shorter->result.sigma.pop_back();
    Attempt sizes = attempt({{dcc.get()}, {fits[0], fits[1], shorter.get()}, nullptr});
    EXPECT_EQ(sizes.status, VE_INVALID_ARGUMENT);
    EXPECT_EQ(sizes.fit, nullptr);
    EXPECT_TRUE(mentions("series 3"));
}

TEST(CapiDcc, FitBatchRejectsSpecsThatDoNotFit)
{
    FitSet univariate = univariate_fits(false);
    DccSpecPtr dcc = make_dcc_spec(VE_DCC_MODEL_DCC, VE_DCC_DIST_MVNORM);
    DccSpecPtr unlabeled = make_dcc_spec(VE_DCC_MODEL_FDCC, VE_DCC_DIST_MVNORM);
    const int32_t skipped[] = {1, 3, 3};
    ASSERT_EQ(ve_dcc_spec_set_groups(unlabeled.get(), skipped, 3), VE_OK);
    Attempt bad_groups = attempt({{dcc.get(), unlabeled.get()}, handles_of(univariate), nullptr});
    EXPECT_EQ(bad_groups.status, VE_INVALID_ARGUMENT);
    EXPECT_EQ(bad_groups.fit, nullptr);
    EXPECT_TRUE(mentions("spec 2")) << ve_last_error();
    ve_dcc_spec* raw = nullptr;
    ASSERT_EQ(ve_dcc_spec_create(VE_DCC_MODEL_FDCC, &raw), VE_OK);
    DccSpecPtr without_groups(raw);
    EXPECT_EQ(attempt(one_dcc(without_groups, univariate)).status, VE_INVALID_ARGUMENT);
    const int32_t short_groups[] = {1, 2};
    ASSERT_EQ(ve_dcc_spec_set_groups(raw, short_groups, 2), VE_OK);
    EXPECT_EQ(attempt(one_dcc(without_groups, univariate)).status, VE_INVALID_ARGUMENT);
    DccSpecPtr unknown = make_dcc_spec(VE_DCC_MODEL_DCC, VE_DCC_DIST_MVNORM);
    ASSERT_EQ(ve_dcc_spec_set_fixed(unknown.get(), "dccz9", 0.1), VE_OK);
    Attempt name = attempt(one_dcc(unknown, univariate));
    EXPECT_EQ(name.status, VE_INVALID_ARGUMENT);
    EXPECT_TRUE(mentions("dccz9"));
}

TEST(CapiDcc, StandardErrorsNeedFitsWithoutRegressors)
{
    FitSet with_regressors = univariate_fits(true);
    DccSpecPtr dcc = make_dcc_spec(VE_DCC_MODEL_DCC, VE_DCC_DIST_MVNORM);
    DccOptionsPtr on = make_dcc_options(true);
    DccOptionsPtr off = make_dcc_options(false);
    Attempt blocked = attempt({{dcc.get()}, handles_of(with_regressors), on.get()});
    EXPECT_EQ(blocked.status, VE_UNSUPPORTED);
    EXPECT_EQ(blocked.fit, nullptr);
    EXPECT_TRUE(mentions("regressors"));
    DccBatchPtr batch = build_dcc_batch({{dcc.get()}, handles_of(with_regressors), off.get()});
    ve::testing::ContextPtr context = ve::testing::make_context(1);
    DccFits fits(1);
    ASSERT_EQ(ve_dcc_fit_batch(context.get(), batch.get(), fits.out()), VE_OK) << ve_last_error();
    EXPECT_EQ(read_status(fits[0]), VE_FIT_CONVERGED);
    const std::vector<const ve_fit*> handles = handles_of(with_regressors);
    ve_dcc_fit* later = reinterpret_cast<ve_dcc_fit*>(&later);
    EXPECT_EQ(ve_dcc_fit_standard_errors(context.get(), fits[0], handles.data(), handles.size(), &later),
              VE_UNSUPPORTED);
    EXPECT_EQ(later, nullptr);
    EXPECT_TRUE(mentions("regressors"));
}

TEST(CapiDcc, StandardErrorsRejectOtherFitsAndBadArguments)
{
    FitSet univariate = univariate_fits(false);
    DccSpecPtr dcc = make_dcc_spec(VE_DCC_MODEL_DCC, VE_DCC_DIST_MVNORM);
    DccOptionsPtr off = make_dcc_options(false);
    DccBatchPtr batch = build_dcc_batch({{dcc.get()}, handles_of(univariate), off.get()});
    ve::testing::ContextPtr context = ve::testing::make_context(1);
    DccFits fits(1);
    ASSERT_EQ(ve_dcc_fit_batch(context.get(), batch.get(), fits.out()), VE_OK) << ve_last_error();
    const std::vector<const ve_fit*> handles = handles_of(univariate);
    ve_dcc_fit* out = reinterpret_cast<ve_dcc_fit*>(&out);
    const std::vector<const ve_fit*> reordered{handles[1], handles[0], handles[2]};
    EXPECT_EQ(ve_dcc_fit_standard_errors(context.get(), fits[0], reordered.data(), 3, &out), VE_INVALID_ARGUMENT);
    EXPECT_EQ(out, nullptr);
    EXPECT_TRUE(mentions("estimates differ"));
    out = reinterpret_cast<ve_dcc_fit*>(&out);
    EXPECT_EQ(ve_dcc_fit_standard_errors(context.get(), fits[0], handles.data(), 2, &out), VE_INVALID_ARGUMENT);
    EXPECT_EQ(out, nullptr);
    EXPECT_TRUE(mentions("3 series"));
    EXPECT_EQ(ve_dcc_fit_standard_errors(nullptr, fits[0], handles.data(), 3, &out), VE_INVALID_HANDLE);
    EXPECT_EQ(ve_dcc_fit_standard_errors(context.get(), nullptr, handles.data(), 3, &out), VE_INVALID_HANDLE);
    EXPECT_EQ(ve_dcc_fit_standard_errors(context.get(), fits[0], nullptr, 3, &out), VE_INVALID_ARGUMENT);
    EXPECT_EQ(ve_dcc_fit_standard_errors(context.get(), fits[0], handles.data(), 0, &out), VE_INVALID_ARGUMENT);
    EXPECT_EQ(ve_dcc_fit_standard_errors(context.get(), fits[0], handles.data(), 3, nullptr), VE_INVALID_ARGUMENT);
    const std::vector<const ve_fit*> with_null{handles[0], nullptr, handles[2]};
    EXPECT_EQ(ve_dcc_fit_standard_errors(context.get(), fits[0], with_null.data(), 3, &out), VE_INVALID_HANDLE);
}

TEST(CapiDcc, AccessorsRejectBadArguments)
{
    FitSet univariate = univariate_fits(false);
    DccSpecPtr dcc = make_dcc_spec(VE_DCC_MODEL_DCC, VE_DCC_DIST_MVNORM);
    DccBatchPtr batch = build_dcc_batch(one_dcc(dcc, univariate));
    ve::testing::ContextPtr context = ve::testing::make_context(1);
    DccFits fits(1);
    ASSERT_EQ(ve_dcc_fit_batch(context.get(), batch.get(), fits.out()), VE_OK) << ve_last_error();
    const ve_dcc_fit* fit = fits[0];
    double value = 0.0;
    double table[2] = {0.0, 0.0};
    std::size_t count = 0;
    const char* name = nullptr;
    const double* data = nullptr;
    EXPECT_EQ(ve_dcc_fit_status(nullptr, nullptr), VE_INVALID_HANDLE);
    EXPECT_EQ(ve_dcc_fit_status(fit, nullptr), VE_INVALID_ARGUMENT);
    EXPECT_EQ(ve_dcc_fit_nobs(nullptr, &count), VE_INVALID_HANDLE);
    EXPECT_EQ(ve_dcc_fit_series(nullptr, &count), VE_INVALID_HANDLE);
    EXPECT_EQ(ve_dcc_fit_param_count(nullptr, &count), VE_INVALID_HANDLE);
    EXPECT_EQ(ve_dcc_fit_param_count(fit, nullptr), VE_INVALID_ARGUMENT);
    EXPECT_EQ(ve_dcc_fit_param_name(nullptr, 0, &name), VE_INVALID_HANDLE);
    EXPECT_EQ(ve_dcc_fit_param_name(fit, 2, &name), VE_INVALID_ARGUMENT);
    EXPECT_EQ(ve_dcc_fit_param_name(fit, 0, nullptr), VE_INVALID_ARGUMENT);
    EXPECT_EQ(ve_dcc_fit_estimates(nullptr, table), VE_INVALID_HANDLE);
    EXPECT_EQ(ve_dcc_fit_estimates(fit, nullptr), VE_INVALID_ARGUMENT);
    EXPECT_EQ(ve_dcc_fit_std_errors(fit, 2, table), VE_INVALID_ARGUMENT);
    EXPECT_EQ(ve_dcc_fit_t_values(fit, -1, table), VE_INVALID_ARGUMENT);
    EXPECT_EQ(ve_dcc_fit_p_values(fit, VE_SE_ROBUST, nullptr), VE_INVALID_ARGUMENT);
    EXPECT_EQ(ve_dcc_fit_std_errors(nullptr, VE_SE_ROBUST, table), VE_INVALID_HANDLE);
    EXPECT_EQ(ve_dcc_fit_loglik(nullptr, &value), VE_INVALID_HANDLE);
    EXPECT_EQ(ve_dcc_fit_loglik(fit, nullptr), VE_INVALID_ARGUMENT);
    EXPECT_EQ(ve_dcc_fit_criterion(fit, 4, VE_IC_TOTAL, &value), VE_INVALID_ARGUMENT);
    EXPECT_EQ(ve_dcc_fit_criterion(fit, VE_DCC_CRITERION_AIC, 2, &value), VE_INVALID_ARGUMENT);
    EXPECT_EQ(ve_dcc_fit_criterion(fit, VE_DCC_CRITERION_AIC, VE_IC_TOTAL, nullptr), VE_INVALID_ARGUMENT);
    EXPECT_EQ(ve_dcc_fit_criterion(nullptr, VE_DCC_CRITERION_AIC, VE_IC_TOTAL, &value), VE_INVALID_HANDLE);
    EXPECT_EQ(ve_dcc_fit_correlation(nullptr, &data, &count), VE_INVALID_HANDLE);
    EXPECT_EQ(ve_dcc_fit_correlation(fit, nullptr, &count), VE_INVALID_ARGUMENT);
    EXPECT_EQ(ve_dcc_fit_correlation(fit, &data, nullptr), VE_INVALID_ARGUMENT);
    EXPECT_EQ(ve_dcc_fit_covariance(nullptr, &data, &count), VE_INVALID_HANDLE);
    EXPECT_EQ(ve_dcc_fit_covariance(fit, nullptr, &count), VE_INVALID_ARGUMENT);
    EXPECT_EQ(ve_dcc_fit_covariance(fit, &data, nullptr), VE_INVALID_ARGUMENT);
    ve_dcc_fit_free(nullptr);
}

TEST(CapiDcc, CovarianceIsComputedOnceAcrossThreads)
{
    FitSet univariate = univariate_fits(false);
    DccSpecPtr dcc = make_dcc_spec(VE_DCC_MODEL_DCC, VE_DCC_DIST_MVNORM);
    DccBatchPtr batch = build_dcc_batch(one_dcc(dcc, univariate));
    ve::testing::ContextPtr context = ve::testing::make_context(1);
    DccFits fits(1);
    ASSERT_EQ(ve_dcc_fit_batch(context.get(), batch.get(), fits.out()), VE_OK) << ve_last_error();
    std::vector<const double*> pointers(4, nullptr);
    std::vector<std::thread> threads;
    for (std::size_t i = 0; i < pointers.size(); ++i)
    {
        threads.emplace_back(
            [&, i]
            {
                std::size_t len = 0;
                ve_dcc_fit_covariance(fits[0], &pointers[i], &len);
            });
    }
    for (std::thread& thread : threads)
    {
        thread.join();
    }
    for (const double* pointer : pointers)
    {
        EXPECT_NE(pointer, nullptr);
        EXPECT_EQ(pointer, pointers[0]);
    }
}

TEST(CapiDcc, BatchOwnsCopiesBeforeFitting)
{
    DccBatchPtr batch;
    Report expected;
    {
        FitSet univariate = univariate_fits(false);
        DccSpecPtr spec = make_dcc_spec(VE_DCC_MODEL_DCC, VE_DCC_DIST_MVNORM);
        DccOptionsPtr options = make_dcc_options(false);
        batch = build_dcc_batch({{spec.get()}, handles_of(univariate), options.get()});
        expected = report_of_core(core_fits(univariate, false).front());
        ASSERT_EQ(ve_dcc_spec_set_fixed(spec.get(), "dccb1", 1.2), VE_OK);
        univariate[0]->result.estimate[0] += 1.0;
    }
    ve::testing::ContextPtr context = ve::testing::make_context(1);
    DccFits fits(1);
    ASSERT_EQ(ve_dcc_fit_batch(context.get(), batch.get(), fits.out()), VE_OK) << ve_last_error();
    expect_same_report(report_of(fits[0]), expected);
}

TEST(CapiDcc, CountsRejectOverflowBeforeReadingInputs)
{
    DccSpecPtr spec = make_dcc_spec(VE_DCC_MODEL_FDCC, VE_DCC_DIST_MVNORM);
    DccBatchPtr batch = build_dcc_batch({});
    const std::size_t huge = std::numeric_limits<std::size_t>::max();
    const ve_dcc_spec* specs[] = {spec.get()};
    ve_fit fit;
    const ve_fit* fits[] = {&fit};
    const int32_t groups[] = {1};
    EXPECT_EQ(ve_dcc_spec_set_groups(spec.get(), groups, huge), VE_INVALID_ARGUMENT);
    EXPECT_TRUE(mentions("count"));
    EXPECT_EQ(ve_dcc_batch_set_specs(batch.get(), specs, huge), VE_INVALID_ARGUMENT);
    EXPECT_TRUE(mentions("count"));
    EXPECT_EQ(ve_dcc_batch_set_fits(batch.get(), fits, huge), VE_INVALID_ARGUMENT);
    EXPECT_TRUE(mentions("count"));
}

TEST(CapiDcc, InvalidSpecsTakePrecedenceOverUnsupportedStandardErrors)
{
    FitSet univariate = univariate_fits(true);
    DccSpecPtr spec = make_dcc_spec(VE_DCC_MODEL_DCC, VE_DCC_DIST_MVNORM);
    ASSERT_EQ(ve_dcc_spec_set_fixed(spec.get(), "dccz9", 0.1), VE_OK);
    Attempt invalid = attempt(one_dcc(spec, univariate));
    EXPECT_EQ(invalid.status, VE_INVALID_ARGUMENT);
    EXPECT_EQ(invalid.fit, nullptr);
    EXPECT_TRUE(mentions("dccz9"));
}

namespace
{

DccSpecPtr make_copula_spec(int32_t model, int32_t dist, int32_t transformation, int32_t method)
{
    DccSpecPtr spec = make_dcc_spec(model, dist);
    EXPECT_EQ(ve_dcc_spec_set_copula(spec.get(), transformation), VE_OK) << ve_last_error();
    EXPECT_EQ(ve_dcc_spec_set_static_method(spec.get(), method), VE_OK) << ve_last_error();
    return spec;
}

DccSpec core_copula(DccModel model, DccDist dist, ve::dcc::DccCopula copula, ve::dcc::StaticMethod method)
{
    DccSpec spec;
    spec.model = model;
    spec.dist = dist;
    spec.copula = copula;
    spec.static_method = method;
    return spec;
}

}

TEST(CapiDcc, CopulaMatchesCore)
{
    using ve::dcc::DccCopula;
    using ve::dcc::StaticMethod;
    FitSet univariate = univariate_fits(false);
    std::vector<DccSpecPtr> specs;
    specs.push_back(make_dcc_spec(VE_DCC_MODEL_DCC, VE_DCC_DIST_MVNORM));
    specs.push_back(make_copula_spec(VE_DCC_MODEL_DCC, VE_DCC_DIST_MVT, VE_DCC_COPULA_PARAMETRIC, VE_DCC_STATIC_DEFAULT));
    specs.push_back(make_copula_spec(VE_DCC_MODEL_ADCC, VE_DCC_DIST_MVNORM, VE_DCC_COPULA_EMPIRICAL, VE_DCC_STATIC_DEFAULT));
    specs.push_back(make_copula_spec(VE_DCC_MODEL_STATIC, VE_DCC_DIST_MVT, VE_DCC_COPULA_PARAMETRIC, VE_DCC_STATIC_ML));
    DccOptionsPtr options = make_dcc_options(true);
    DccBatchPtr batch = build_dcc_batch({raw_specs(specs), handles_of(univariate), options.get()});
    ve::testing::ContextPtr context = ve::testing::make_context(3);
    DccFits fits(4);
    ASSERT_EQ(ve_dcc_fit_batch(context.get(), batch.get(), fits.out()), VE_OK) << ve_last_error();

    const std::vector<DccSpec> core{
        core_copula(DccModel::Dcc, DccDist::MvNorm, DccCopula::None, StaticMethod::Default),
        core_copula(DccModel::Dcc, DccDist::MvT, DccCopula::Parametric, StaticMethod::Default),
        core_copula(DccModel::ADcc, DccDist::MvNorm, DccCopula::Empirical, StaticMethod::Default),
        core_copula(DccModel::Static, DccDist::MvT, DccCopula::Parametric, StaticMethod::Ml),
    };
    ve::Context core_context;
    const std::vector<const ve::FitResult*> results = results_of(univariate);
    const std::vector<DccFitResult> expected =
        ve::dcc::fit_dcc_batch(core_context, results, core, ve::dcc::DccOptions{ve::FitConfig{}, true});
    for (std::size_t i = 0; i < fits.size(); ++i)
    {
        SCOPED_TRACE(i);
        expect_same_report(report_of(fits[i]), report_of_core(expected[i]));
        EXPECT_EQ(std::bit_cast<std::uint64_t>(read_criterion(fits[i], VE_DCC_CRITERION_BIC, VE_IC_TOTAL)),
                  std::bit_cast<std::uint64_t>(ve::dcc::dcc_criterion(expected[i], ve::dcc::Criterion::Bic, false)));
    }
    std::size_t count = 0;
    ASSERT_EQ(ve_dcc_fit_param_count(fits[3], &count), VE_OK);
    EXPECT_EQ(count, 4U);
}

TEST(CapiDcc, CopulaInvalidCombinations)
{
    // Review Focus 2: the setters check ranges, the batch the combinations, before any fit.
    DccSpecPtr dcc = make_dcc_spec(VE_DCC_MODEL_DCC, VE_DCC_DIST_MVNORM);
    EXPECT_EQ(ve_dcc_spec_set_copula(nullptr, VE_DCC_COPULA_PARAMETRIC), VE_INVALID_HANDLE);
    EXPECT_EQ(ve_dcc_spec_set_copula(dcc.get(), 3), VE_INVALID_ARGUMENT);
    EXPECT_EQ(ve_dcc_spec_set_copula(dcc.get(), -1), VE_INVALID_ARGUMENT);
    EXPECT_EQ(ve_dcc_spec_set_static_method(nullptr, VE_DCC_STATIC_ML), VE_INVALID_HANDLE);
    EXPECT_EQ(ve_dcc_spec_set_static_method(dcc.get(), 3), VE_INVALID_ARGUMENT);
    EXPECT_TRUE(mentions("static_method"));

    FitSet univariate = univariate_fits(false);
    struct Case
    {
        int32_t model;
        int32_t dist;
        int32_t copula;
        int32_t method;
        const char* words;
    };
    const std::vector<Case> cases{
        {VE_DCC_MODEL_STATIC, VE_DCC_DIST_MVNORM, VE_DCC_COPULA_NONE, VE_DCC_STATIC_DEFAULT, "needs a copula"},
        {VE_DCC_MODEL_FDCC, VE_DCC_DIST_MVNORM, VE_DCC_COPULA_PARAMETRIC, VE_DCC_STATIC_DEFAULT, "fdcc model has no copula"},
        {VE_DCC_MODEL_DCC, VE_DCC_DIST_MVLAPLACE, VE_DCC_COPULA_EMPIRICAL, VE_DCC_STATIC_DEFAULT, "mvlaplace copula"},
        {VE_DCC_MODEL_DCC, VE_DCC_DIST_MVT, VE_DCC_COPULA_PARAMETRIC, VE_DCC_STATIC_KENDALL, "static_method"},
    };
    for (const Case& one : cases)
    {
        SCOPED_TRACE(one.words);
        DccSpecPtr spec = make_copula_spec(one.model, one.dist, one.copula, one.method);
        if (one.model == VE_DCC_MODEL_FDCC)
        {
            const int32_t groups[] = {1, 1, 2};
            ASSERT_EQ(ve_dcc_spec_set_groups(spec.get(), groups, 3), VE_OK);
        }
        const Attempt result = attempt(one_dcc(spec, univariate));
        EXPECT_EQ(result.status, VE_INVALID_ARGUMENT);
        EXPECT_EQ(result.fit, nullptr);
        EXPECT_TRUE(mentions(one.words)) << ve_last_error();
    }
    DccSpecPtr started = make_copula_spec(VE_DCC_MODEL_DCC, VE_DCC_DIST_MVT, VE_DCC_COPULA_PARAMETRIC, VE_DCC_STATIC_DEFAULT);
    ASSERT_EQ(ve_dcc_spec_set_start(started.get(), "mshape", 8.0), VE_OK);
    EXPECT_EQ(attempt(one_dcc(started, univariate)).status, VE_INVALID_ARGUMENT);
    EXPECT_TRUE(mentions("mshape takes no start"));
}
