#pragma once

#include "core/spec.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numbers>
#include <random>
#include <vector>

namespace ve::testing
{

// std::normal_distribution differs across standard libraries, so a test drawing from it sees other values under
// MSVC than under GCC. Box-Muller on the engine's raw output gives the same draws everywhere, up to the last bits
// of std::log and std::cos. Called as draw(engine), like std::normal_distribution.
struct PortableNormal
{
    double deviation = 1.0;

    double operator()(std::mt19937_64& engine) const
    {
        auto uniform = [&engine] { return (static_cast<double>(engine() >> 11) + 0.5) * 0x1.0p-53; };
        const double radius = std::sqrt(-2.0 * std::log(uniform()));
        return deviation * radius * std::cos(2.0 * std::numbers::pi * uniform());
    }
};

// sGARCH(1,1)-norm with mu = 0.05, omega = 0.01, alpha = 0.08, beta = 0.9, the same series on every platform.
inline std::vector<double> simulate_garch_series(std::uint64_t seed, std::size_t count)
{
    std::mt19937_64 engine(seed);
    const PortableNormal shock;
    const double mu = 0.05;
    const double omega = 0.01;
    const double alpha = 0.08;
    const double beta = 0.9;
    double sigma2 = omega / (1.0 - alpha - beta);
    std::vector<double> r(count);
    for (double& value : r)
    {
        const double residual = std::sqrt(sigma2) * shock(engine);
        value = mu + residual;
        sigma2 = omega + alpha * residual * residual + beta * sigma2;
    }
    return r;
}

// ARMA(m, n)-sGARCH(p, q) with standardized (mean 0, variance 1) innovations. Lags before the first
// observation take the unconditional value: zero for the mean terms, omega / (1 - persistence) for the
// variance terms.
struct SimulatedModel
{
    double mu = 0.05;
    std::vector<double> ar;
    std::vector<double> ma;
    double omega = 0.01;
    std::vector<double> alpha{0.08};
    std::vector<double> beta{0.9};
    ve::Dist dist = ve::Dist::Norm;
    double shape = 0.0;
};

inline double normal_cdf(double x)
{
    return 0.5 * std::erfc(-x / std::numbers::sqrt2);
}

// Johnson SU in the Rigby and Stasinopoulos parametrization: z = c (sqrt(w) sinh(omega) + sinh((N + nu) / tau)).
struct JohnsonConstants
{
    double w;
    double omega;
    double c;
};

inline JohnsonConstants johnson_constants(double nu, double tau)
{
    const double w = std::exp(1.0 / (tau * tau));
    const double omega = -nu / tau;
    return {w, omega, 1.0 / std::sqrt(0.5 * (w - 1.0) * (w * std::cosh(2.0 * omega) + 1.0))};
}

// z has mean 0, so E|z| = 2 E[z; N > n0] where z(n0) = 0, and E[exp(a N); N > n0] = exp(a^2 / 2) Phi(a - n0).
inline double johnson_abs_moment(double nu, double tau)
{
    const JohnsonConstants k = johnson_constants(nu, tau);
    const double root = std::sqrt(k.w);
    const double zero = tau * std::asinh(-root * std::sinh(k.omega)) - nu;
    const double upper = std::exp(nu / tau) * normal_cdf(1.0 / tau - zero);
    const double lower = std::exp(-nu / tau) * normal_cdf(-1.0 / tau - zero);
    return 2.0 * k.c * root * (std::sinh(k.omega) * normal_cdf(-zero) + 0.5 * (upper - lower));
}

// E|z| of the standardized shock; NaN for the distributions whose E|z| has no closed form here (the helper
// simulates eGARCH only under norm, std, ged and jsu).
inline double expected_abs_shock(ve::Dist dist, double skew, double shape)
{
    if (dist == ve::Dist::Std)
    {
        return std::sqrt(shape - 2.0) * std::exp(std::lgamma(0.5 * (shape - 1.0)) - std::lgamma(0.5 * shape)) /
               std::sqrt(std::numbers::pi);
    }
    if (dist == ve::Dist::Ged)
    {
        const double lambda =
            std::sqrt(std::pow(2.0, -2.0 / shape) * std::tgamma(1.0 / shape) / std::tgamma(3.0 / shape));
        return lambda * std::pow(2.0, 1.0 / shape) * std::tgamma(2.0 / shape) / std::tgamma(1.0 / shape);
    }
    if (dist == ve::Dist::Jsu)
    {
        return johnson_abs_moment(skew, shape);
    }
    if (dist == ve::Dist::Norm)
    {
        return std::sqrt(2.0 / std::numbers::pi);
    }
    return std::numeric_limits<double>::quiet_NaN();
}

// Draws from the engine's raw output only, so the series is the same with every standard library. Each draw
// is its own statement: the evaluation order of operands is unspecified, and compilers consume the engine in
// different orders.
class PortableShocks
{
public:
    explicit PortableShocks(std::uint64_t seed) : engine_(seed)
    {
    }

    double uniform()
    {
        return (static_cast<double>(engine_() >> 11) + 0.5) * 0x1.0p-53;
    }

    double normal()
    {
        return PortableNormal{}(engine_);
    }

    // Marsaglia and Tsang (2000), unit scale; a shape below one is boosted by U^(1 / shape).
    double gamma(double shape)
    {
        if (shape < 1.0)
        {
            const double boosted = gamma(shape + 1.0);
            return boosted * std::pow(uniform(), 1.0 / shape);
        }
        const double d = shape - 1.0 / 3.0;
        const double c = 1.0 / std::sqrt(9.0 * d);
        while (true)
        {
            const double x = normal();
            const double v = std::pow(1.0 + c * x, 3.0);
            if (v > 0.0 && std::log(uniform()) < 0.5 * x * x + d - d * v + d * std::log(v))
            {
                return d * v;
            }
        }
    }

    double standardized(ve::Dist dist, double shape)
    {
        if (dist == ve::Dist::Std)
        {
            const double numerator = normal();
            return numerator * std::sqrt((shape - 2.0) / (2.0 * gamma(0.5 * shape)));
        }
        if (dist == ve::Dist::Ged)
        {
            const double lambda =
                std::sqrt(std::pow(2.0, -2.0 / shape) * std::tgamma(1.0 / shape) / std::tgamma(3.0 / shape));
            const double sign = uniform() < 0.5 ? -1.0 : 1.0;
            return sign * lambda * std::pow(2.0 * gamma(1.0 / shape), 1.0 / shape);
        }
        return normal();
    }

    // skew and shape as in the library's parameter vector; ghyp is not simulated.
    double draw(ve::Dist dist, double skew, double shape)
    {
        if (dist == ve::Dist::Sstd)
        {
            return fernandez_steel(ve::Dist::Std, skew, shape);
        }
        if (dist == ve::Dist::Sged)
        {
            return fernandez_steel(ve::Dist::Ged, skew, shape);
        }
        if (dist == ve::Dist::Nig)
        {
            return normal_inverse_gaussian(skew, shape);
        }
        if (dist == ve::Dist::Jsu)
        {
            return johnson_su(skew, shape);
        }
        return standardized(dist, shape);
    }

private:
    // y = xi |x| with probability xi^2 / (1 + xi^2), else -|x| / xi, for x from the standardized kernel; then
    // z = (y - mu_xi) / sigma_xi.
    double fernandez_steel(ve::Dist kernel, double xi, double shape)
    {
        const double m1 = expected_abs_shock(kernel, 0.0, shape);
        const double mean = m1 * (xi - 1.0 / xi);
        const double sd = std::sqrt((1.0 - m1 * m1) * (xi * xi + 1.0 / (xi * xi)) + 2.0 * m1 * m1 - 1.0);
        const bool right = uniform() < xi * xi / (1.0 + xi * xi);
        const double magnitude = std::abs(standardized(kernel, shape));
        double y = -magnitude / xi;
        if (right)
        {
            y = xi * magnitude;
        }
        return (y - mean) / sd;
    }

    // Michael, Schucany and Haas (1976). The smaller root is m^2 / (larger root), which avoids the cancellation
    // of the textbook form.
    double inverse_gaussian(double mean, double shape)
    {
        const double n = normal();
        const double y = n * n;
        const double root = std::sqrt(4.0 * mean * shape * y + mean * mean * y * y);
        const double larger = mean + mean * mean * y / (2.0 * shape) + mean / (2.0 * shape) * root;
        const double smaller = mean * mean / larger;
        if (uniform() <= mean / (mean + smaller))
        {
            return smaller;
        }
        return larger;
    }

    // z = mu + beta W + sqrt(W) N, W inverse Gaussian with mean delta / sqrt(alpha^2 - beta^2) = delta^2 / zeta
    // and shape delta^2. At lambda = -1/2, K_{1/2} / K_{-1/2} = 1 and K_{3/2} / K_{-1/2} = 1 + 1 / zeta.
    double normal_inverse_gaussian(double rho, double zeta)
    {
        const double first_ratio = 1.0;
        const double second_ratio = 1.0 + 1.0 / zeta;
        const double skew_weight = rho * rho / (1.0 - rho * rho);
        const double delta =
            1.0 / std::sqrt(first_ratio / zeta + skew_weight * (second_ratio - first_ratio * first_ratio));
        const double alpha = zeta / (delta * std::sqrt(1.0 - rho * rho));
        const double beta = rho * alpha;
        const double mu = -beta * delta * delta * first_ratio / zeta;
        const double mixing = inverse_gaussian(delta * delta / zeta, delta * delta);
        const double n = normal();
        return mu + beta * mixing + std::sqrt(mixing) * n;
    }

    double johnson_su(double nu, double tau)
    {
        const JohnsonConstants k = johnson_constants(nu, tau);
        const double n = normal();
        return k.c * (std::sqrt(k.w) * std::sinh(k.omega) + std::sinh((n + nu) / tau));
    }

    std::mt19937_64 engine_;
};

inline double lagged(const std::vector<double>& values, std::size_t t, std::size_t lag, double presample)
{
    if (t < lag)
    {
        return presample;
    }
    return values[t - lag];
}

inline double simulated_variance(const SimulatedModel& model, const std::vector<double>& squared_residuals,
                                 const std::vector<double>& sigma2, std::size_t t)
{
    double persistence = 0.0;
    for (double weight : model.alpha)
    {
        persistence += weight;
    }
    for (double weight : model.beta)
    {
        persistence += weight;
    }
    const double unconditional = model.omega / (1.0 - persistence);
    double variance = model.omega;
    for (std::size_t i = 0; i < model.alpha.size(); ++i)
    {
        variance += model.alpha[i] * lagged(squared_residuals, t, i + 1, unconditional);
    }
    for (std::size_t j = 0; j < model.beta.size(); ++j)
    {
        variance += model.beta[j] * lagged(sigma2, t, j + 1, unconditional);
    }
    return variance;
}

inline double simulated_mean(const SimulatedModel& model, const std::vector<double>& centered,
                             const std::vector<double>& residuals, std::size_t t)
{
    double mean = model.mu;
    for (std::size_t i = 0; i < model.ar.size(); ++i)
    {
        mean += model.ar[i] * lagged(centered, t, i + 1, 0.0);
    }
    for (std::size_t j = 0; j < model.ma.size(); ++j)
    {
        mean += model.ma[j] * lagged(residuals, t, j + 1, 0.0);
    }
    return mean;
}

inline std::vector<double> simulate_arma_garch(const SimulatedModel& model, std::uint64_t seed,
                                               std::size_t count)
{
    PortableShocks shocks(seed);
    std::vector<double> r(count);
    std::vector<double> centered(count);
    std::vector<double> residuals(count);
    std::vector<double> squared_residuals(count);
    std::vector<double> sigma2(count);
    for (std::size_t t = 0; t < count; ++t)
    {
        sigma2[t] = simulated_variance(model, squared_residuals, sigma2, t);
        residuals[t] = std::sqrt(sigma2[t]) * shocks.standardized(model.dist, model.shape);
        squared_residuals[t] = residuals[t] * residuals[t];
        r[t] = simulated_mean(model, centered, residuals, t) + residuals[t];
        centered[t] = r[t] - model.mu;
    }
    return r;
}

// A GARCH variant of order (1, 1) with mean 0.05, written apart from the library. alpha, beta, gamma and shape
// are the model's own parameters; power is the apARCH delta or the fGARCH lambda; eta1 and eta2 belong to
// fGARCH-ALLGARCH, rho and phi to csGARCH; skew and shape belong to the distribution.
struct VariantProcess
{
    const char* name;
    ve::Model model;
    double omega;
    double alpha;
    double beta;
    double gamma = 0.0;
    double eta1 = 0.0;
    double eta2 = 0.0;
    double power = 2.0;
    double rho = 0.0;
    double phi = 0.0;
    ve::Dist dist = ve::Dist::Norm;
    double skew = 0.0;
    double shape = 0.0;
};

inline std::vector<VariantProcess> variant_processes()
{
    using ve::Dist;
    using ve::Model;
    return {
        {.name = "egarch11_std", .model = Model::EGarch, .omega = -0.01, .alpha = -0.08, .beta = 0.97,
         .gamma = 0.15, .dist = Dist::Std, .shape = 6.0},
        {.name = "gjrgarch11_norm", .model = Model::GjrGarch, .omega = 0.01, .alpha = 0.03, .beta = 0.9,
         .gamma = 0.09},
        {.name = "aparch11_ged", .model = Model::ApArch, .omega = 0.02, .alpha = 0.08, .beta = 0.9, .gamma = 0.4,
         .power = 1.3, .dist = Dist::Ged, .shape = 1.5},
        {.name = "igarch11_norm", .model = Model::IGarch, .omega = 0.01, .alpha = 0.1, .beta = 0.9},
        {.name = "csgarch11_std", .model = Model::CsGarch, .omega = 0.01, .alpha = 0.06, .beta = 0.85,
         .rho = 0.99, .phi = 0.04, .dist = Dist::Std, .shape = 7.0},
        {.name = "fgarch_allgarch11_norm", .model = Model::FGarchAllGarch, .omega = 0.02, .alpha = 0.08,
         .beta = 0.88, .eta1 = 0.3, .eta2 = 0.2, .power = 1.5},
    };
}

// permanent is the csGARCH long-run component q.
struct VariantState
{
    double sigma2 = 1.0;
    double permanent = 1.0;
};

// The variance after the shock z drawn at the current state.
inline VariantState next_variant_state(const VariantProcess& process, const VariantState& state, double z,
                                       double mean_abs)
{
    const double residual = std::sqrt(state.sigma2) * z;
    const double square = residual * residual;
    const double sigma_power = std::pow(state.sigma2, 0.5 * process.power);
    if (process.model == ve::Model::EGarch)
    {
        const double news = process.alpha * z + process.gamma * (std::abs(z) - mean_abs);
        return {std::exp(process.omega + news + process.beta * std::log(state.sigma2)), 0.0};
    }
    if (process.model == ve::Model::GjrGarch)
    {
        const double leverage = residual < 0.0 ? process.gamma : 0.0;
        return {process.omega + (process.alpha + leverage) * square + process.beta * state.sigma2, 0.0};
    }
    if (process.model == ve::Model::ApArch)
    {
        const double news = std::pow(std::abs(residual) - process.gamma * residual, process.power);
        const double next = process.omega + process.alpha * news + process.beta * sigma_power;
        return {std::pow(next, 2.0 / process.power), 0.0};
    }
    if (process.model == ve::Model::FGarchAllGarch)
    {
        const double shifted = z - process.eta2;
        const double news = std::pow(std::abs(shifted) - process.eta1 * shifted, process.power);
        const double next = process.omega + process.alpha * sigma_power * news + process.beta * sigma_power;
        return {std::pow(next, 2.0 / process.power), 0.0};
    }
    if (process.model == ve::Model::CsGarch)
    {
        const double permanent = process.omega + process.rho * state.permanent + process.phi * (square - state.sigma2);
        const double transitory =
            process.alpha * (square - state.permanent) + process.beta * (state.sigma2 - state.permanent);
        return {permanent + transitory, permanent};
    }
    return {process.omega + process.alpha * square + process.beta * state.sigma2, 0.0};
}

// Starts at sigma^2 = 1 (and q = 1) and drops the first 1000 observations.
inline std::vector<double> simulate_variant(const VariantProcess& process, std::uint64_t seed, std::size_t count)
{
    constexpr std::size_t kBurnIn = 1000;
    const double mu = 0.05;
    const double mean_abs = expected_abs_shock(process.dist, process.skew, process.shape);
    PortableShocks shocks(seed);
    VariantState state;
    std::vector<double> r(count);
    for (std::size_t t = 0; t < kBurnIn + count; ++t)
    {
        const double z = shocks.draw(process.dist, process.skew, process.shape);
        if (t >= kBurnIn)
        {
            r[t - kBurnIn] = mu + std::sqrt(state.sigma2) * z;
        }
        state = next_variant_state(process, state, z, mean_abs);
    }
    return r;
}

inline std::vector<VariantProcess> skewed_processes()
{
    using ve::Dist;
    using ve::Model;
    return {
        {.name = "gjrgarch11_sstd", .model = Model::GjrGarch, .omega = 0.01, .alpha = 0.03, .beta = 0.9,
         .gamma = 0.09, .dist = Dist::Sstd, .skew = 0.85, .shape = 6.0},
        {.name = "aparch11_sged", .model = Model::ApArch, .omega = 0.02, .alpha = 0.08, .beta = 0.9, .gamma = 0.4,
         .power = 1.3, .dist = Dist::Sged, .skew = 0.9, .shape = 1.4},
        {.name = "egarch11_jsu", .model = Model::EGarch, .omega = -0.01, .alpha = -0.08, .beta = 0.97,
         .gamma = 0.15, .dist = Dist::Jsu, .skew = -0.5, .shape = 2.0},
        {.name = "sgarch11_nig", .model = Model::SGarch, .omega = 0.02, .alpha = 0.08, .beta = 0.9,
         .dist = Dist::Nig, .skew = -0.3, .shape = 1.5},
    };
}

// Series index of skewed_processes(), drawn with seed 41 + index.
inline std::vector<double> simulate_skewed(std::size_t index, std::size_t count)
{
    return simulate_variant(skewed_processes()[index], 41 + index, count);
}

// A fit of a skewed series: each under its own distribution, and the nig series under ghyp as well, where the
// true ghlambda is -1/2.
struct SkewedFit
{
    const char* name;
    std::size_t series;
    ve::Dist dist;
};

inline std::vector<SkewedFit> skewed_fits()
{
    using ve::Dist;
    return {{.name = "gjrgarch11_sstd", .series = 0, .dist = Dist::Sstd},
            {.name = "aparch11_sged", .series = 1, .dist = Dist::Sged},
            {.name = "egarch11_jsu", .series = 2, .dist = Dist::Jsu},
            {.name = "sgarch11_nig", .series = 3, .dist = Dist::Nig},
            {.name = "sgarch11_nig_ghyp", .series = 3, .dist = Dist::Ghyp}};
}

// A series with mean regressors (x1, x2), a variance regressor v1 and ARCH-in-mean, written apart from the library:
// r_t = mu + sum_k delta_k x_t,k + xi sigma_t^archm + d_t, with d_t = ar d_(t-1) + e_t, and the variance of
// `variance` with its omega shifted by zeta v_t. Regressors: x1 = sin(t / 40), x2 ~ N(0, 1), v1 = |cos(t / 25)|.
struct RegressorProcess
{
    const char* name;
    VariantProcess variance;
    double mu = 0.05;
    double ar = 0.0;
    std::vector<double> delta{};
    double zeta = 0.0;
    bool has_vxreg = false;
    int archm = 0;
    double xi = 0.0;
};

inline std::vector<RegressorProcess> regressor_processes()
{
    using ve::Dist;
    using ve::Model;
    return {
        {.name = "arx_sgarch11_norm",
         .variance = {.name = "", .model = Model::SGarch, .omega = 0.02, .alpha = 0.08, .beta = 0.9},
         .mu = 0.05,
         .ar = 0.3,
         .delta = {0.3, -0.2}},
        {.name = "sgarch11_vx_std",
         .variance = {.name = "", .model = Model::SGarch, .omega = 0.01, .alpha = 0.08, .beta = 0.88,
                      .dist = Dist::Std, .shape = 6.0},
         .zeta = 0.05,
         .has_vxreg = true},
        {.name = "archm_gjrgarch11_norm",
         .variance = {.name = "", .model = Model::GjrGarch, .omega = 0.02, .alpha = 0.03, .beta = 0.9,
                      .gamma = 0.09},
         .mu = 0.02,
         .archm = 1,
         .xi = 0.1},
        {.name = "archm2_egarch11_std",
         .variance = {.name = "", .model = Model::EGarch, .omega = -0.01, .alpha = -0.08, .beta = 0.97,
                      .gamma = 0.15, .dist = Dist::Std, .shape = 6.0},
         .mu = 0.02,
         .archm = 2,
         .xi = 0.05},
    };
}

// Row-major count x columns matrices.
struct RegressorSeries
{
    std::vector<double> r;
    std::vector<double> x;
    std::vector<double> v;
};

struct RegressorRow
{
    double x1;
    double x2;
    double v1;
};

inline RegressorRow regressor_row(double t, PortableShocks& shocks)
{
    return {std::sin(t / 40.0), shocks.normal(), std::abs(std::cos(t / 25.0))};
}

// Starts at sigma^2 = 1 and drops 1000 observations (t <= 0), whose regressors follow the same rule.
inline RegressorSeries simulate_regressor_process(const RegressorProcess& process, std::uint64_t seed,
                                                  std::size_t count)
{
    constexpr std::size_t kBurnIn = 1000;
    const VariantProcess& base = process.variance;
    const double mean_abs = expected_abs_shock(base.dist, base.skew, base.shape);
    PortableShocks shocks(seed);
    PortableShocks regressors(seed + 1000);
    VariantState state;
    double z = 0.0;
    double deviation = 0.0;
    RegressorSeries series;
    for (std::size_t index = 0; index < kBurnIn + count; ++index)
    {
        const RegressorRow row = regressor_row(static_cast<double>(index) + 1.0 - static_cast<double>(kBurnIn),
                                               regressors);
        if (index > 0)
        {
            VariantProcess shifted = base;
            shifted.omega += process.zeta * row.v1;
            state = next_variant_state(shifted, state, z, mean_abs);
        }
        z = shocks.draw(base.dist, base.skew, base.shape);
        deviation = process.ar * deviation + std::sqrt(state.sigma2) * z;
        double mean = process.mu + process.xi * std::pow(state.sigma2, 0.5 * process.archm);
        if (!process.delta.empty())
        {
            mean += process.delta[0] * row.x1 + process.delta[1] * row.x2;
        }
        if (index < kBurnIn)
        {
            continue;
        }
        series.r.push_back(mean + deviation);
        if (!process.delta.empty())
        {
            series.x.insert(series.x.end(), {row.x1, row.x2});
        }
        if (process.has_vxreg)
        {
            series.v.push_back(row.v1);
        }
    }
    return series;
}

// C-order count x n matrix whose column j is the series simulated with seed j + 1.
inline std::vector<double> simulate_garch_matrix(std::size_t count, std::size_t n)
{
    std::vector<double> matrix(count * n);
    for (std::size_t j = 0; j < n; ++j)
    {
        const std::vector<double> series = simulate_garch_series(j + 1, count);
        for (std::size_t t = 0; t < count; ++t)
        {
            matrix[t * n + j] = series[t];
        }
    }
    return matrix;
}

}
