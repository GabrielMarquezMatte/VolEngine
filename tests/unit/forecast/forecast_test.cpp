#include "fit/fit.hpp"
#include "forecast/forecast.hpp"
#include "forecast/step.hpp"
#include "forecast/variance_forecast.hpp"
#include "model/layout.hpp"
#include "model/likelihood.hpp"
#include "model/parameter_map.hpp"
#include "simulated_garch.hpp"

#include <cmath>
#include <cstddef>
#include <gtest/gtest.h>
#include <numbers>
#include <span>
#include <utility>
#include <vector>

namespace
{

using ve::Dist;
using ve::Model;
using ve::forecast::Forecast;

constexpr std::size_t kObservations = 600;

std::vector<double> columns(std::size_t rows, std::size_t count, double base)
{
    std::vector<double> x(rows * count);
    for (std::size_t t = 0; t < rows; ++t)
    {
        for (std::size_t k = 0; k < count; ++k)
        {
            double s = std::sin(0.37 * static_cast<double>((k + 1) * t) + static_cast<double>(k));
            x[t * count + k] = base + 0.5 * s * s;
        }
    }
    return x;
}

// The fit of spec on the first n points, the forecast of h = 1, and the filter at the estimate on n + 1 points. The
// point n is chosen so that the presample (the mean of e^2, of e0^2 = (r - mu)^2 with ARCH-in-mean) and the
// variance target stay those of the n points, and v_n is the mean of v: the filter at n then reads only the past.
void expect_first_step_matches_filter(const ve::Spec& spec, double tolerance = 1e-12)
{
    std::vector<double> r = ve::testing::simulate_garch_series(11, kObservations);
    std::size_t k_mean = static_cast<std::size_t>(spec.mxreg);
    std::size_t k_variance = static_cast<std::size_t>(spec.vxreg);
    std::vector<double> x = columns(kObservations + 1, k_mean, -0.25);
    std::vector<double> v = columns(kObservations + 1, k_variance, 0.5);
    std::span<const double> v_sample(v.data(), kObservations * k_variance);
    for (std::size_t k = 0; k < k_variance; ++k)
    {
        double sum = 0.0;
        for (std::size_t t = 0; t < kObservations; ++t)
        {
            sum += v_sample[t * k_variance + k];
        }
        v[kObservations * k_variance + k] = sum / static_cast<double>(kObservations);
    }
    ve::model::Data sample{r, {std::span<const double>(x.data(), kObservations * k_mean), k_mean}, {v_sample, k_variance}};
    ve::FitResult fit = ve::fit_one(sample, spec, ve::FitConfig{});
    ASSERT_FALSE(fit.sigma.empty());
    ve::forecast::FutureRegressors future{
        {std::span<const double>(x).last(k_mean), k_mean},
        {std::span<const double>(v).last(k_variance), k_variance}};
    Forecast forecast = ve::forecast::forecast(fit, 1, future);
    double scale = 0.0;
    std::vector<double> extended = r;
    if (spec.archm != 0)
    {
        double mu = fit.estimate[0];
        for (double value : r)
        {
            scale += (value - mu) * (value - mu) / static_cast<double>(r.size());
        }
        extended.push_back(mu + std::sqrt(scale));
    }
    ve::model::Layout layout = ve::model::layout_of(spec);
    double power = 2.0;
    if (layout.is_power_family())
    {
        power = ve::model::power_of<double>(fit.estimate, layout);
    }
    if (spec.archm == 0)
    {
        for (double e : fit.residuals)
        {
            scale += std::pow(std::abs(e), power) / static_cast<double>(r.size());
        }
        extended.push_back(forecast.mean[0] + std::pow(scale, 1.0 / power));
    }
    std::vector<double> residuals(extended.size());
    std::vector<double> sigma2(extended.size());
    double ll = ve::model::loglik<double>(ve::model::Data{extended, {x, k_mean}, {v, k_variance}}, fit.estimate,
                                          layout, ve::VarInit::Sample, {.residuals = residuals, .sigma2 = sigma2});
    ASSERT_TRUE(std::isfinite(ll));
    double variance = forecast.sigma[0] * forecast.sigma[0];
    EXPECT_NEAR(variance, sigma2.back(), tolerance * sigma2.back());
    double mean = extended.back() - residuals.back();
    EXPECT_NEAR(forecast.mean[0], mean, 1e-12 * (1.0 + std::abs(mean)));
}

TEST(Forecast, FirstStepMatchesExtendedFilter)
{
    struct Variant
    {
        Model model;
        Dist dist;
    };
    for (Variant variant : {Variant{Model::SGarch, Dist::Norm}, Variant{Model::IGarch, Dist::Norm},
                            Variant{Model::GjrGarch, Dist::Norm}, Variant{Model::GjrGarch, Dist::Sstd},
                            Variant{Model::CsGarch, Dist::Norm}, Variant{Model::EGarch, Dist::Norm},
                            Variant{Model::EGarch, Dist::Std}, Variant{Model::ApArch, Dist::Ged}})
    {
        for (int arma : {0, 1})
        {
            for (int mxreg : {0, 2})
            {
                for (int vxreg : {0, 1})
                {
                    for (bool targeting : {false, true})
                    {
                        ve::Spec spec{.model = variant.model, .ar = arma, .ma = arma, .dist = variant.dist,
                                      .variance_targeting = targeting, .mxreg = mxreg, .vxreg = vxreg};
                        if (ve::model::spec_problem(spec))
                        {
                            continue;
                        }
                        SCOPED_TRACE(::testing::Message() << "model " << static_cast<int>(variant.model) << " dist "
                                                          << static_cast<int>(variant.dist) << " arma " << arma
                                                          << " mxreg " << mxreg << " vxreg " << vxreg << " vt "
                                                          << targeting);
                        expect_first_step_matches_filter(spec);
                    }
                }
            }
        }
    }
    for (int archm : {1, 2})
    {
        SCOPED_TRACE(::testing::Message() << "archm " << archm);
        expect_first_step_matches_filter(ve::Spec{.model = Model::GjrGarch, .archm = archm});
    }
    for (int arma : {0, 1})
    {
        for (int mxreg : {0, 2})
        {
            SCOPED_TRACE(::testing::Message() << "arfima, arma " << arma << " mxreg " << mxreg);
            expect_first_step_matches_filter(ve::Spec{.ar = arma, .ma = arma, .mxreg = mxreg, .arfima = true});
        }
    }
}

// The filter of the fit takes |z - eta2| as sqrt((z - eta2)^2 + 1e-6), rugarch's ugarchforecast as |z - eta2|:
// the first step agrees only to that smoothing, which weighs most when the last z sits near eta2 (2.8e-5 relative
// on ALLGARCH here). FGarchFirstStepUsesExactAbs pins the exact step.
TEST(Forecast, FGarchFirstStepNearExtendedFilter)
{
    for (Model model : {Model::FGarchGarch, Model::FGarchTGarch, Model::FGarchAvGarch, Model::FGarchNGarch,
                        Model::FGarchNaGarch, Model::FGarchApArch, Model::FGarchGjrGarch, Model::FGarchAllGarch})
    {
        SCOPED_TRACE(::testing::Message() << "model " << static_cast<int>(model));
        expect_first_step_matches_filter(ve::Spec{.model = model}, 1e-4);
    }
}

// A fit at a point given by hand: spec, estimate in Layout order and the final state; nothing else is read.
ve::FitResult at_point(const ve::Spec& spec, std::vector<double> estimate, ve::forecast::ForecastState state)
{
    ve::FitResult fit;
    fit.status = ve::FitStatus::Converged;
    fit.spec = spec;
    fit.estimate = std::move(estimate);
    fit.nobs = 100;
    fit.state = std::move(state);
    return fit;
}

TEST(Forecast, SGarchClosedForm)
{
    const double omega = 0.02;
    const double alpha = 0.08;
    const double beta = 0.9;
    const double persistence = alpha + beta;
    ve::FitResult fit = at_point(ve::Spec{}, {0.05, omega, alpha, beta}, {{0.4}, {1.3}, {}, {0.4}});
    Forecast forecast = ve::forecast::forecast(fit, 50);
    double first = omega + alpha * 0.16 + beta * 1.3;
    for (std::size_t h = 1; h <= 50; ++h)
    {
        double power = std::pow(persistence, static_cast<double>(h - 1));
        double expected = omega * (1.0 - power) / (1.0 - persistence) + power * first;
        double sigma = forecast.sigma[h - 1];
        EXPECT_NEAR(sigma * sigma, expected, 1e-13 * expected) << "h = " << h;
        EXPECT_EQ(forecast.mean[h - 1], 0.05) << "h = " << h;
    }
}

TEST(Forecast, LongHorizonReachesUnconditional)
{
    const std::size_t horizon = 10000;
    ve::FitResult sgarch = at_point(ve::Spec{}, {0.05, 0.02, 0.08, 0.9}, {{0.4}, {1.3}, {}, {0.4}});
    Forecast s = ve::forecast::forecast(sgarch, horizon);
    EXPECT_NEAR(s.sigma.back() * s.sigma.back(), 0.02 / (1.0 - 0.98), 1e-10);

    ve::Spec gjr_spec{.model = Model::GjrGarch, .dist = Dist::Sstd};
    ve::model::Layout gjr_layout = ve::model::layout_of(gjr_spec);
    std::vector<double> theta(gjr_layout.count());
    theta[gjr_layout.mu()] = 0.05;
    theta[gjr_layout.omega()] = 0.02;
    theta[gjr_layout.alpha_begin()] = 0.03;
    theta[gjr_layout.beta_begin()] = 0.88;
    theta[gjr_layout.gamma_begin()] = 0.1;
    theta[gjr_layout.skew()] = 0.85;
    theta[gjr_layout.shape()] = 6.0;
    double persistence = ve::model::persistence<double>(theta, gjr_layout);
    Forecast g = ve::forecast::forecast(at_point(gjr_spec, theta, {{-0.4}, {1.3}, {}, {-0.4}}), horizon);
    EXPECT_NEAR(g.sigma.back() * g.sigma.back(), 0.02 / (1.0 - persistence), 1e-10 * 0.02 / (1.0 - persistence));

    ve::Spec cs_spec{.model = Model::CsGarch};
    ve::model::Layout cs_layout = ve::model::layout_of(cs_spec);
    std::vector<double> cs_theta(cs_layout.count());
    cs_theta[cs_layout.mu()] = 0.05;
    cs_theta[cs_layout.omega()] = 0.01;
    cs_theta[cs_layout.alpha_begin()] = 0.05;
    cs_theta[cs_layout.beta_begin()] = 0.85;
    cs_theta[cs_layout.eta1_begin()] = 0.97;
    cs_theta[cs_layout.eta2_begin()] = 0.03;
    Forecast c = ve::forecast::forecast(at_point(cs_spec, cs_theta, {{0.4}, {1.3}, {0.9}, {0.4}}), horizon);
    EXPECT_NEAR(c.sigma.back() * c.sigma.back(), 0.01 / (1.0 - 0.97), 1e-10 * 0.01 / 0.03);

    ve::Spec arma_spec{.ar = 1, .ma = 1};
    ve::FitResult arma = at_point(arma_spec, {0.05, 0.5, 0.2, 0.02, 0.08, 0.9}, {{0.4}, {1.3}, {}, {0.7}});
    Forecast a = ve::forecast::forecast(arma, horizon);
    EXPECT_NEAR(a.mean.back(), 0.05, 1e-12);
    for (std::size_t h = 0; h < horizon; ++h)
    {
        ASSERT_TRUE(std::isfinite(s.sigma[h]) && std::isfinite(g.sigma[h]) && std::isfinite(c.sigma[h]));
    }
}

TEST(Forecast, IGarchGrowsByOmega)
{
    const std::size_t horizon = 10000;
    ve::Spec spec{.model = Model::IGarch};
    ve::model::Layout layout = ve::model::layout_of(spec);
    std::vector<double> theta(layout.count());
    theta[layout.mu()] = 0.05;
    theta[layout.omega()] = 0.02;
    theta[layout.alpha_begin()] = 0.08;
    theta[layout.beta_begin()] = 0.0;  // derived: 1 - alpha, never read
    Forecast forecast = ve::forecast::forecast(at_point(spec, theta, {{0.4}, {1.3}, {}, {0.4}}), horizon);
    EXPECT_NEAR(forecast.sigma[0] * forecast.sigma[0], 0.02 + 0.08 * 0.16 + 0.92 * 1.3, 1e-14);
    for (std::size_t h = 1; h < horizon; ++h)
    {
        double now = forecast.sigma[h] * forecast.sigma[h];
        double before = forecast.sigma[h - 1] * forecast.sigma[h - 1];
        ASSERT_NEAR(now - before, 0.02, 1e-12 * now) << "h = " << h;
    }
}

TEST(Forecast, ArmaMeanByHand)
{
    const double mu = 0.05;
    const double phi1 = 0.4;
    const double phi2 = -0.2;
    const double theta1 = 0.3;
    ve::Spec spec{.ar = 2, .ma = 1};
    std::vector<double> d{0.6, -0.3};
    std::vector<double> e{0.2, 0.5};
    ve::FitResult fit = at_point(spec, {mu, phi1, phi2, theta1, 0.02, 0.08, 0.9}, {e, {1.0, 1.2}, {}, d});
    Forecast forecast = ve::forecast::forecast(fit, 3);
    double d1 = phi1 * d[1] + phi2 * d[0] + theta1 * e[1];
    double d2 = phi1 * d1 + phi2 * d[1];
    double d3 = phi1 * d2 + phi2 * d1;
    EXPECT_NEAR(forecast.mean[0], mu + d1, 1e-14);
    EXPECT_NEAR(forecast.mean[1], mu + d2, 1e-14);
    EXPECT_NEAR(forecast.mean[2], mu + d3, 1e-14);
}

TEST(Forecast, CsGarchClosedForm)
{
    ve::Spec spec{.model = Model::CsGarch, .p = 2, .q = 2, .vxreg = 1};
    ve::model::Layout layout = ve::model::layout_of(spec);
    std::vector<double> theta(layout.count());
    theta[layout.mu()] = 0.05;
    theta[layout.omega()] = 0.01;
    theta[layout.alpha_begin()] = 0.03;
    theta[layout.alpha_begin() + 1] = 0.02;
    theta[layout.beta_begin()] = 0.5;
    theta[layout.beta_begin() + 1] = 0.3;
    theta[layout.eta1_begin()] = 0.97;
    theta[layout.eta2_begin()] = 0.03;
    theta[layout.vxreg_begin()] = 0.004;
    const double rho = 0.97;
    const double sum = 0.03 + 0.02 + 0.5 + 0.3;
    std::vector<double> v{0.6, 0.7, 0.8, 0.9, 1.0};
    ve::forecast::ForecastState state{{0.3, -0.5}, {1.1, 1.3}, {0.8, 0.9}, {0.3, -0.5}};
    Forecast forecast = ve::forecast::forecast(at_point(spec, theta, state), 5, {{}, {v, 1}});
    for (std::size_t h = 2; h <= 5; ++h)
    {
        double power = std::pow(rho, static_cast<double>(h));
        double q = (1.0 - power) / (1.0 - rho) * (0.01 + 0.004 * v[h - 1]) + power * 0.9;
        double expected = q + std::pow(sum, static_cast<double>(h)) * (1.3 - 0.9);
        double sigma = forecast.sigma[h - 1];
        EXPECT_NEAR(sigma * sigma, expected, 1e-13 * expected) << "h = " << h;
    }
}

// Review Focus 5: the forecast of the series x 100 at the scaled point is the forecast x 100.
TEST(Forecast, UnitEquivariance)
{
    ve::Spec spec{.ar = 1, .ma = 1};
    ve::FitResult plain = at_point(spec, {0.05, 0.5, 0.2, 0.02, 0.08, 0.9}, {{0.4}, {1.3}, {}, {0.7}});
    ve::FitResult scaled = at_point(spec, {5.0, 0.5, 0.2, 200.0, 0.08, 0.9}, {{40.0}, {13000.0}, {}, {70.0}});
    Forecast a = ve::forecast::forecast(plain, 30);
    Forecast b = ve::forecast::forecast(scaled, 30);
    for (std::size_t h = 0; h < 30; ++h)
    {
        EXPECT_NEAR(b.sigma[h], 100.0 * a.sigma[h], 1e-12 * b.sigma[h]) << "h = " << h;
        EXPECT_NEAR(b.mean[h], 100.0 * a.mean[h], 1e-12 * std::abs(b.mean[h])) << "h = " << h;
    }
}

TEST(Forecast, FGarchFirstStepUsesExactAbs)
{
    ve::Spec spec{.model = Model::FGarchAllGarch};
    ve::model::Layout layout = ve::model::layout_of(spec);
    std::vector<double> theta(layout.count());
    theta[layout.mu()] = 0.05;
    theta[layout.omega()] = 0.02;
    theta[layout.alpha_begin()] = 0.06;
    theta[layout.beta_begin()] = 0.9;
    theta[layout.eta1_begin()] = 0.3;
    theta[layout.eta2_begin()] = 0.2;
    theta[layout.power()] = 1.5;
    const double e = 0.25;
    const double variance = 1.3;
    Forecast forecast = ve::forecast::forecast(at_point(spec, theta, {{e}, {variance}, {}, {e}}), 1);
    double sigma = std::sqrt(variance);
    double shift = e - 0.2 * sigma;
    double exact = 0.02 + 0.06 * std::pow(std::abs(shift) - 0.3 * shift, 1.5) + 0.9 * std::pow(sigma, 1.5);
    double smoothed = 0.02 + 0.06 * std::pow(std::sqrt(shift * shift + 1e-6 * variance) - 0.3 * shift, 1.5) +
                      0.9 * std::pow(sigma, 1.5);
    double got = std::pow(forecast.sigma[0], 1.5);
    EXPECT_NEAR(got, exact, 1e-13 * exact);
    EXPECT_GT(std::abs(got - smoothed), 1e-12 * exact);
}

TEST(Forecast, EGarchClosedForm)
{
    ve::Spec spec{.model = Model::EGarch};
    ve::model::Layout layout = ve::model::layout_of(spec);
    std::vector<double> theta(layout.count());
    theta[layout.mu()] = 0.05;
    theta[layout.omega()] = 0.01;
    theta[layout.alpha_begin()] = -0.06;
    theta[layout.beta_begin()] = 0.97;
    theta[layout.gamma_begin()] = 0.12;
    const double e = -0.6;
    const double variance = 1.3;
    Forecast forecast = ve::forecast::forecast(at_point(spec, theta, {{e}, {variance}, {}, {e}}), 40);
    double z = e / std::sqrt(variance);
    double first = 0.01 - 0.06 * z + 0.12 * (std::abs(z) - std::sqrt(2.0 / std::numbers::pi)) +
                   0.97 * std::log(variance);
    for (std::size_t h = 1; h <= 40; ++h)
    {
        double power = std::pow(0.97, static_cast<double>(h - 1));
        double expected = 0.01 * (1.0 - power) / (1.0 - 0.97) + power * first;
        double log_variance = 2.0 * std::log(forecast.sigma[h - 1]);
        EXPECT_NEAR(log_variance, expected, 1e-13 * std::abs(expected)) << "h = " << h;
    }
}

TEST(Forecast, PowerLongHorizon)
{
    struct Case
    {
        Model model;
        std::vector<double> asymmetry;
    };
    for (const Case& c : {Case{Model::ApArch, {0.3}}, Case{Model::FGarchAllGarch, {0.3, 0.2}}})
    {
        ve::Spec spec{.model = c.model};
        ve::model::Layout layout = ve::model::layout_of(spec);
        std::vector<double> theta(layout.count());
        theta[layout.mu()] = 0.05;
        theta[layout.omega()] = 0.02;
        theta[layout.alpha_begin()] = 0.06;
        theta[layout.beta_begin()] = 0.9;
        theta[layout.asymmetry_begin()] = c.asymmetry[0];
        if (c.asymmetry.size() > 1)
        {
            theta[layout.eta2_begin()] = c.asymmetry[1];
        }
        theta[layout.power()] = 1.5;
        double kappa = ve::model::shock_moment<double>(theta, layout, 0);
        double level = 0.02 / (1.0 - 0.06 * kappa - 0.9);
        Forecast forecast = ve::forecast::forecast(at_point(spec, theta, {{0.25}, {1.3}, {}, {0.25}}), 10000);
        EXPECT_NEAR(std::pow(forecast.sigma.back(), 1.5), level, 1e-10 * level)
            << "model " << static_cast<int>(c.model);
    }
}

TEST(Forecast, ArfimaMeanByHand)
{
    const double mu = 0.05;
    const double phi = 0.4;
    const double theta1 = 0.3;
    const double d = 0.3;
    ve::Spec spec{.ar = 1, .ma = 1, .arfima = true};
    std::vector<double> deviations{0.6, -0.3, 0.2, 0.5, -0.1};
    const double e = 0.15;
    ve::FitResult fit = at_point(spec, {mu, phi, theta1, d, 0.02, 0.08, 0.9}, {{e}, {1.2}, {}, deviations});
    Forecast forecast = ve::forecast::forecast(fit, 3);
    std::vector<double> pi(8, 1.0);
    for (std::size_t k = 1; k < pi.size(); ++k)
    {
        pi[k] = pi[k - 1] * (static_cast<double>(k) - 1.0 - d) / static_cast<double>(k);
    }
    std::vector<double> series = deviations;
    double w = 0.0;
    for (std::size_t k = 0; k <= 4; ++k)
    {
        w += pi[k] * series[4 - k];
    }
    for (std::size_t h = 1; h <= 3; ++h)
    {
        std::size_t t = 4 + h;
        double f = 0.0;
        for (std::size_t k = 1; k <= t; ++k)
        {
            f -= pi[k] * series[t - k];
        }
        double next = f + phi * w + (h == 1 ? theta1 * e : 0.0);
        EXPECT_NEAR(forecast.mean[h - 1], mu + next, 1e-14) << "h = " << h;
        series.push_back(next);
        w = next - f;
    }
}

TEST(Forecast, ArfimaZeroIsPlain)
{
    std::vector<double> deviations{0.6, -0.3, 0.2, 0.5, -0.1};
    ve::FitResult arfima = at_point(ve::Spec{.ar = 1, .ma = 1, .arfima = true},
                                    {0.05, 0.4, 0.3, 0.0, 0.02, 0.08, 0.9}, {{0.15}, {1.2}, {}, deviations});
    ve::FitResult plain =
        at_point(ve::Spec{.ar = 1, .ma = 1}, {0.05, 0.4, 0.3, 0.02, 0.08, 0.9}, {{0.15}, {1.2}, {}, {-0.1}});
    Forecast a = ve::forecast::forecast(arfima, 20);
    Forecast b = ve::forecast::forecast(plain, 20);
    EXPECT_EQ(a.mean, b.mean);
    EXPECT_EQ(a.sigma, b.sigma);
}


// first_step is the forecast of one step from the state of the fit, with its level (the mean before the ARMA part)
// and the csGARCH q of that step.
void expect_first_step_is_forecast_of_one(const ve::Spec& spec)
{
    std::vector<double> r = ve::testing::simulate_garch_series(5, kObservations);
    std::size_t k_mean = static_cast<std::size_t>(spec.mxreg);
    std::size_t k_variance = static_cast<std::size_t>(spec.vxreg);
    std::vector<double> x = columns(kObservations + 1, k_mean, -0.25);
    std::vector<double> v = columns(kObservations + 1, k_variance, 0.5);
    ve::model::Data sample{r,
                           {std::span<const double>(x.data(), kObservations * k_mean), k_mean},
                           {std::span<const double>(v.data(), kObservations * k_variance), k_variance}};
    ve::FitResult fit = ve::fit_one(sample, spec, ve::FitConfig{});
    ASSERT_FALSE(fit.sigma.empty());
    ve::forecast::FutureRegressors future{{std::span<const double>(x).last(k_mean), k_mean},
                                          {std::span<const double>(v).last(k_variance), k_variance}};
    Forecast forecast = ve::forecast::forecast(fit, 1, future);
    ve::model::Layout layout = ve::model::layout_of(spec);
    ve::forecast::detail::Moments moments = ve::forecast::detail::moments_of(layout, fit.estimate);
    ve::forecast::detail::Inputs in{fit.state, layout, fit.estimate, future, 1, moments};
    ve::forecast::detail::Step step = ve::forecast::detail::first_step(in);
    EXPECT_EQ(std::sqrt(step.sigma2), forecast.sigma[0]);
    EXPECT_EQ(step.mean, forecast.mean[0]);
    double level = layout.include_mean ? fit.estimate[layout.mu()] : 0.0;
    for (std::size_t k = 0; k < k_mean; ++k)
    {
        level += fit.estimate[layout.mxreg_begin() + k] * future.mean.at(0, k);
    }
    if (spec.archm != 0)
    {
        level += fit.estimate[layout.archm_index()] * std::pow(forecast.sigma[0], spec.archm);
    }
    EXPECT_NEAR(step.level, level, 1e-14 * (1.0 + std::abs(level)));
    if (spec.model != Model::CsGarch)
    {
        EXPECT_EQ(step.permanent, 0.0);
        return;
    }
    const ve::forecast::ForecastState& s = fit.state;
    double e = s.residuals.back();
    double c = 0.0;
    for (std::size_t k = 0; k < k_variance; ++k)
    {
        c += fit.estimate[layout.vxreg_begin() + k] * future.variance.at(0, k);
    }
    double q = fit.estimate[layout.omega()] + c + fit.estimate[layout.eta1_begin()] * s.permanent.back() +
               fit.estimate[layout.eta2_begin()] * (e * e - s.sigma2.back());
    EXPECT_NEAR(step.permanent, q, 1e-15 * q);
}

TEST(Forecast, FirstStepIsForecastOfOne)
{
    for (Model model : {Model::SGarch, Model::IGarch, Model::GjrGarch, Model::EGarch, Model::ApArch,
                        Model::FGarchAllGarch, Model::CsGarch})
    {
        Dist dist = model == Model::GjrGarch ? Dist::Sstd : Dist::Norm;
        SCOPED_TRACE(::testing::Message() << "model " << static_cast<int>(model));
        expect_first_step_is_forecast_of_one(
            ve::Spec{.model = model, .ar = 1, .ma = 1, .dist = dist, .mxreg = 1, .vxreg = 1});
    }
    SCOPED_TRACE("archm and arfima");
    expect_first_step_is_forecast_of_one(ve::Spec{.model = Model::GjrGarch, .archm = 1});
    expect_first_step_is_forecast_of_one(ve::Spec{.ar = 1, .ma = 1, .arfima = true});
}

}
