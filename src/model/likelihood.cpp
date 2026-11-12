#include "model/likelihood.hpp"

#include "dist/densities.hpp"
#include "dist/moments.hpp"
#include "garch/csgarch.hpp"
#include "garch/egarch.hpp"
#include "garch/gjrgarch.hpp"
#include "garch/power_garch.hpp"
#include "garch/sgarch.hpp"
#include "linalg/least_squares.hpp"
#include "mean/arma.hpp"
#include "model/likelihood_terms.hpp"
#include "model/parameter_map.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <variant>
#include <vector>

namespace ve::model
{

namespace
{

// arch backcast: exponentially weighted mean of the first min(75, T) squared residuals, decay 0.94.
constexpr std::size_t kBackcastWindow = 75;
constexpr double kBackcastDecay = 0.94;

double sample_mean(std::span<const double> r)
{
    double sum = 0.0;
    for (double value : r)
    {
        sum += value;
    }
    return sum / static_cast<double>(r.size());
}

double backcast_around(std::span<const double> r, double center)
{
    double weight = 1.0;
    double weight_sum = 0.0;
    double weighted_square_sum = 0.0;
    for (double value : r.first(std::min(kBackcastWindow, r.size())))
    {
        double residual = value - center;
        weight_sum += weight;
        weighted_square_sum += weight * residual * residual;
        weight *= kBackcastDecay;
    }
    return weighted_square_sum / weight_sum;
}

// The residuals of the least squares of r on [1 (with a mean), X].
std::vector<double> ols_residuals(const Data& data, bool include_mean)
{
    std::size_t rows = data.r.size();
    std::size_t columns = data.mean.columns + (include_mean ? 1 : 0);
    std::vector<double> a;
    a.reserve(rows * columns);
    for (std::size_t t = 0; t < rows; ++t)
    {
        if (include_mean)
        {
            a.push_back(1.0);
        }
        for (std::size_t k = 0; k < data.mean.columns; ++k)
        {
            a.push_back(data.mean.at(t, k));
        }
    }
    std::vector<double> b = linalg::least_squares(a, columns, data.r);
    std::vector<double> residuals(data.r.begin(), data.r.end());
    for (std::size_t t = 0; t < rows; ++t)
    {
        for (std::size_t c = 0; c < columns; ++c)
        {
            residuals[t] -= b[c] * a[t * columns + c];
        }
    }
    return residuals;
}

// arch computes the backcast once, from the residuals at its starting mean, and keeps it fixed during the
// optimization: it depends on no parameter. Without regressors that mean is the sample mean, or zero without a
// mean; with them, the least squares of r on [1 (with a mean), X].
double backcast(const Data& data, bool include_mean)
{
    if (data.mean.columns > 0)
    {
        return backcast_around(ols_residuals(data, include_mean), 0.0);
    }
    return backcast_around(data.r, include_mean ? sample_mean(data.r) : 0.0);
}

template <class T>
std::span<const T> block(std::span<const T> theta, std::size_t begin, int count)
{
    return theta.subspan(begin, static_cast<std::size_t>(count));
}

template <class T>
mean::ArmaParameters<T> arma_parameters(std::span<const T> theta, const Layout& layout)
{
    T mu{0.0};
    if (layout.include_mean)
    {
        mu = theta[layout.mu()];
    }
    return {mu, block(theta, layout.ar_begin(), layout.ar), block(theta, layout.ma_begin(), layout.ma)};
}

template <class T>
garch::SGarchParameters<T> sgarch_parameters(std::span<const T> theta, const Layout& layout)
{
    return {theta[layout.omega()], block(theta, layout.alpha_begin(), layout.arch),
            block(theta, layout.beta_begin(), layout.garch)};
}

template <class T>
garch::EGarchParameters<T> egarch_parameters(std::span<const T> theta, const Layout& layout)
{
    garch::SGarchParameters<T> s = sgarch_parameters(theta, layout);
    return {s.omega, s.alpha, s.beta, block(theta, layout.gamma_begin(), layout.gamma),
            dist::abs_moment(layout.dist, shape_parameters_of(theta, layout), T{1.0})};
}

template <class T>
garch::GjrGarchParameters<T> gjrgarch_parameters(std::span<const T> theta, const Layout& layout)
{
    garch::SGarchParameters<T> s = sgarch_parameters(theta, layout);
    return {s.omega, s.alpha, s.beta, block(theta, layout.gamma_begin(), layout.gamma)};
}

template <class T>
garch::CsGarchParameters<T> csgarch_parameters(std::span<const T> theta, const Layout& layout)
{
    garch::SGarchParameters<T> s = sgarch_parameters(theta, layout);
    return {s.omega, s.alpha, s.beta, theta[layout.eta1_begin()], theta[layout.eta2_begin()]};
}

// rugarch's fGARCH takes |z - eta2| as sqrt((z - eta2)^2 + 1e-6) in every submodel; its apARCH is exact.
constexpr double kFGarchAbsSmoothing = 1e-6;

double abs_smoothing_of(const Layout& layout)
{
    if (layout.model == Model::ApArch)
    {
        return 0.0;
    }
    return kFGarchAbsSmoothing;
}

// moments must outlive the result, whose moments span views it; shock_moments fills it.
template <class T>
garch::PowerGarchParameters<T> power_garch_parameters(std::span<const T> theta, const Layout& layout,
                                                      std::span<const T> moments)
{
    garch::SGarchParameters<T> s = sgarch_parameters(theta, layout);
    return {s.omega,
            s.alpha,
            s.beta,
            theta.subspan(layout.asymmetry_begin(), layout.asymmetry_count()),
            block(theta, layout.eta2_begin(), layout.eta2),
            power_of(theta, layout),
            moments.first(s.alpha.size()),
            abs_smoothing_of(layout),
            layout.free_power};
}

// False when a kappa is not finite (std with shape <= power), which can come back as NaN as well as +inf.
template <class T>
bool shock_moments(std::span<const T> theta, const Layout& layout, std::span<T> moments)
{
    for (std::size_t i = 0; i < static_cast<std::size_t>(layout.arch); ++i)
    {
        moments[i] = shock_moment(theta, layout, i);
        if (!std::isfinite(ad::value_of(moments[i])))
        {
            return false;
        }
    }
    return true;
}

template <class T>
T mean_square(std::span<const T> residuals)
{
    T sum{0.0};
    for (const T& e : residuals)
    {
        sum = sum + e * e;
    }
    return sum / static_cast<double>(residuals.size());
}

template <class T>
T mean_power(std::span<const T> residuals, const T& power, bool free_power)
{
    using std::abs;
    T sum{0.0};
    for (const T& e : residuals)
    {
        sum = sum + garch::raise(abs(e), power, free_power);
    }
    return sum / static_cast<double>(residuals.size());
}

// (mean |e|^power)^(2 / power): the variance whose sigma^power is the sample mean of |e|^power.
template <class T>
T mean_power_variance(std::span<const T> residuals, const T& power, bool free_power)
{
    return garch::raise(mean_power(residuals, power, free_power), 2.0 / power, free_power);
}

// rugarch holds sigma2_t for t < max(m, n, p, q), the ARMA orders included, at the sample mean of the squared
// residuals taken at the current mean parameters; in apARCH and fGARCH it holds sigma^power at the sample mean of
// |e|^power instead.
template <class T>
garch::Presample<T> presample(const Data& data, std::span<const T> residuals, std::span<const T> theta,
                              const Layout& layout, VarInit var_init)
{
    if (var_init == VarInit::Backcast)
    {
        return {T{backcast(data, layout.include_mean)}, 0};
    }
    std::size_t held = static_cast<std::size_t>(std::max({layout.ar, layout.ma, layout.arch, layout.garch}));
    if (layout.is_power_family())
    {
        return {mean_power_variance(residuals, power_of(theta, layout), layout.free_power), held};
    }
    return {mean_square(residuals), held};
}

// What the variance recursion starts from: the presample and c_t = sum_k zeta_k v_t,k (empty without vxreg).
template <class T>
struct Start
{
    garch::Presample<T> presample;
    std::span<const T> intercept_shift;
};

template <class T, class Parameters>
Parameters shifted(Parameters p, const Start<T>& start)
{
    p.intercept_shift = start.intercept_shift;
    return p;
}

// The sGARCH parameters with beta_q = 1 - sum alpha - sum_{j<q} beta_j, held in the beta of the caller; the stored
// beta_q is never read.
template <class T>
garch::SGarchParameters<T> igarch_parameters(std::span<const T> theta, const Layout& layout, const Start<T>& start,
                                             std::array<T, kMaxOrder>& beta)
{
    garch::SGarchParameters<T> p = sgarch_parameters(theta, layout);
    p.intercept_shift = start.intercept_shift;
    std::ranges::copy(p.beta, beta.begin());
    T last{1.0};
    for (const T& alpha : p.alpha)
    {
        last = last - alpha;
    }
    for (const T& earlier : p.beta.first(p.beta.size() - 1))
    {
        last = last - earlier;
    }
    beta[p.beta.size() - 1] = last;
    p.beta = std::span<const T>(beta).first(p.beta.size());
    return p;
}

template <class T>
void igarch_variance(std::span<const T> residuals, std::span<const T> theta, const Layout& layout,
                     const Start<T>& start, std::span<T> sigma2)
{
    std::array<T, kMaxOrder> beta{};
    garch::sgarch_variance<T>(residuals, igarch_parameters(theta, layout, start, beta), start.presample, sigma2);
}

template <class T>
void csgarch_from_residuals(std::span<const T> residuals, garch::CsGarchParameters<T> p, const Start<T>& start,
                            std::span<T> sigma2)
{
    p.intercept_shift = start.intercept_shift;
    std::vector<T> squared_residuals(residuals.size());
    std::vector<T> permanent(residuals.size());
    std::ranges::transform(residuals, squared_residuals.begin(), [](const T& e) { return e * e; });
    garch::csgarch_variance<T>(squared_residuals, p, start.presample, sigma2, permanent);
}

// False when a shock moment is not finite.
template <class T>
bool power_variance(std::span<const T> residuals, std::span<const T> theta, const Layout& layout,
                    const Start<T>& start, std::span<T> sigma2)
{
    std::array<T, kMaxOrder> moments{};
    garch::PowerGarchParameters<T> p = power_garch_parameters<T>(theta, layout, moments);
    p.intercept_shift = start.intercept_shift;
    if (!shock_moments(theta, layout, std::span<T>(moments)))
    {
        return false;
    }
    garch::power_garch_variance<T>(residuals, p, start.presample, sigma2);
    return true;
}

// False when the point is outside the domain of the recursion.
template <class T>
bool fill_variance(std::span<const T> residuals, std::span<const T> theta, const Layout& layout,
                   const Start<T>& start, std::span<T> sigma2)
{
    switch (layout.model)
    {
        case Model::SGarch:
            garch::sgarch_variance<T>(residuals, shifted(sgarch_parameters(theta, layout), start), start.presample,
                                      sigma2);
            return true;
        case Model::IGarch:
            igarch_variance<T>(residuals, theta, layout, start, sigma2);
            return true;
        case Model::EGarch:
            garch::egarch_variance<T>(residuals, shifted(egarch_parameters(theta, layout), start), start.presample,
                                      sigma2);
            return true;
        case Model::GjrGarch:
            garch::gjrgarch_variance<T>(residuals, shifted(gjrgarch_parameters(theta, layout), start),
                                        start.presample, sigma2);
            return true;
        case Model::CsGarch:
            csgarch_from_residuals<T>(residuals, csgarch_parameters(theta, layout), start, sigma2);
            return true;
        case Model::ApArch:
        case Model::FGarchGarch:
        case Model::FGarchTGarch:
        case Model::FGarchAvGarch:
        case Model::FGarchNGarch:
        case Model::FGarchNaGarch:
        case Model::FGarchApArch:
        case Model::FGarchGjrGarch:
        case Model::FGarchAllGarch:
            return power_variance<T>(residuals, theta, layout, start, sigma2);
    }
    return false;
}

template <class T>
std::span<T> provided_or(std::span<T> provided, std::vector<T>& storage, std::size_t size)
{
    if (!provided.empty())
    {
        return provided;
    }
    storage.resize(size);
    return storage;
}

template <class T>
dist::ShapeParameters<double> value_of(const dist::ShapeParameters<T>& parameters)
{
    return {ad::value_of(parameters.skew), ad::value_of(parameters.shape), ad::value_of(parameters.ghlambda)};
}

// norm and std read z^2 = e^2 / sigma2, which needs neither sqrt nor a second division.
template <class T>
T sum_of_alternative(std::span<const T> residuals, std::span<const T> sigma2, const dist::LogDensity<T>& density,
                     std::span<T> terms)
{
    using std::abs;
    using std::sqrt;
    switch (density.dist)
    {
        case Dist::Norm:
            return sum_of_terms<T>(residuals, sigma2,
                                   [&density](const T& e, const T& variance)
                                   { return dist::normal_log_density_at(density, e * e / variance); },
                                   terms);
        case Dist::Std:
            return sum_of_terms<T>(residuals, sigma2,
                                   [&density](const T& e, const T& variance)
                                   { return dist::student_log_density_at(density, e * e / variance); },
                                   terms);
        case Dist::Ged:
            return sum_of_terms<T>(residuals, sigma2,
                                   [&density](const T& e, const T& variance)
                                   { return dist::ged_log_density_at(density, abs(e) / sqrt(variance)); },
                                   terms);
        case Dist::Sstd:
        case Dist::Sged:
        case Dist::Nig:
        case Dist::Ghyp:
        case Dist::Jsu:
            break;
    }
    return T{kInvalidLoglik};
}

// The skewed, hyperbolic and Johnson densities read z = e / sqrt(sigma2).
template <class T, class Alternative>
T sum_of_alternative(std::span<const T> residuals, std::span<const T> sigma2, const Alternative& density,
                     std::span<T> terms)
{
    using std::sqrt;
    return sum_of_terms<T>(residuals, sigma2,
                           [&density](const T& e, const T& variance)
                           { return dist::log_density(density, e / sqrt(variance)); },
                           terms);
}

template <class T>
T sum_of_density_terms(std::span<const T> residuals, std::span<const T> sigma2,
                       const dist::Density<T>& density, std::span<T> terms)
{
    return std::visit([&](const auto& alternative)
                      { return sum_of_alternative<T>(residuals, sigma2, alternative, terms); }, density);
}

// r_t - mu - sum_k delta_k x_t,k.
template <class T>
T deviation_at(const Data& data, std::span<const T> theta, const Layout& layout, const T& mu, std::size_t t)
{
    std::span<const T> delta = block(theta, layout.mxreg_begin(), layout.mxreg);
    T deviation = data.r[t] - mu;
    for (std::size_t k = 0; k < delta.size(); ++k)
    {
        deviation = deviation - delta[k] * data.mean.at(t, k);
    }
    return deviation;
}

// e_t of the mean rule over d_t = r_t - mu - sum_k delta_k x_t,k, without the ARCH-in-mean term. Without
// regressors it is the ARMA recursion on r.
template <class T>
void mean_residuals(const Data& data, std::span<const T> theta, const Layout& layout, std::span<T> residuals)
{
    mean::ArmaParameters<T> p = arma_parameters(theta, layout);
    if (layout.mxreg == 0)
    {
        mean::arma_residuals(data.r, p, residuals);
        return;
    }
    std::vector<T> d(data.r.size());
    for (std::size_t t = 0; t < d.size(); ++t)
    {
        d[t] = deviation_at<T>(data, theta, layout, p.mu, t);
    }
    mean::arma_residuals_of_deviations<T>(d, p, residuals);
}

// ARCH-in-mean: the mean and the variance advance together, since m_t reads sigma_t and sigma2_t reads e_{t-1}.
// d holds r_t - m_t, archm included, and the AR part of the ARMA acts on it.
template <class T>
struct JointMean
{
    const Data& data;
    std::span<const T> theta;
    const Layout& layout;
    mean::ArmaParameters<T> arma;
    std::vector<T> d;
    std::span<T> residuals;

    // xi sigma_t^p: xi sqrt(sigma2) for p = 1, xi sigma2 for p = 2.
    T archm_term(const T& sigma2) const
    {
        using std::sqrt;
        const T& xi = theta[layout.archm_index()];
        if (layout.archm == 1)
        {
            return xi * sqrt(sigma2);
        }
        return xi * sigma2;
    }

    void step(const T& sigma2, std::size_t t)
    {
        d[t] = deviation_at<T>(data, theta, layout, arma.mu, t) - archm_term(sigma2);
        residuals[t] = mean::arma_step<T>(d, residuals, arma, t);
    }
};

// For each t, the variance from the residuals before t (false when it is not positive and finite), then the mean.
template <class T, class VarianceAt>
bool joint_loop(JointMean<T>& mean, std::span<T> sigma2, VarianceAt variance_at)
{
    for (std::size_t t = 0; t < sigma2.size(); ++t)
    {
        T variance = variance_at(std::span<const T>(mean.residuals), t);
        if (!is_valid_variance(variance))
        {
            return false;
        }
        sigma2[t] = variance;
        mean.step(variance, t);
    }
    return true;
}

// sGARCH and iGARCH keep sigma2 itself as their state.
template <class T>
bool joint_sgarch(JointMean<T>& mean, const garch::SGarchParameters<T>& p, const garch::Presample<T>& presample,
                  std::span<T> sigma2)
{
    return joint_loop<T>(mean, sigma2,
                         [&](std::span<const T> e, std::size_t t)
                         {
                             if (t < presample.held)
                             {
                                 return presample.value;
                             }
                             return garch::sgarch_step<T>(e, p, presample, sigma2, t);
                         });
}

template <class T>
bool joint_igarch(JointMean<T>& mean, std::span<const T> theta, const Layout& layout, const Start<T>& start,
                  std::span<T> sigma2)
{
    std::array<T, kMaxOrder> beta{};
    return joint_sgarch<T>(mean, igarch_parameters(theta, layout, start, beta), start.presample, sigma2);
}

template <class T>
bool joint_gjrgarch(JointMean<T>& mean, const garch::GjrGarchParameters<T>& p,
                    const garch::Presample<T>& presample, std::span<T> sigma2)
{
    return joint_loop<T>(mean, sigma2,
                         [&](std::span<const T> e, std::size_t t)
                         {
                             if (t < presample.held)
                             {
                                 return presample.value;
                             }
                             return garch::gjrgarch_step<T>(e, p, presample, sigma2, t);
                         });
}

// log sigma2 is the state; sigma2 = exp of it.
template <class T>
bool joint_egarch(JointMean<T>& mean, const garch::EGarchParameters<T>& p, const garch::Presample<T>& presample,
                  std::span<T> sigma2)
{
    using std::exp;
    using std::log;
    T log_presample = log(presample.value);
    std::vector<T> log_sigma2(sigma2.size());
    return joint_loop<T>(mean, sigma2,
                         [&](std::span<const T> e, std::size_t t)
                         {
                             log_sigma2[t] = log_presample;
                             if (t >= presample.held)
                             {
                                 log_sigma2[t] = garch::egarch_step<T>(e, p, log_presample, log_sigma2, t);
                             }
                             return exp(log_sigma2[t]);
                         });
}

// q and e^2 are the state beside sigma2; e^2_{t-1} enters the step as soon as e_{t-1} exists.
template <class T>
bool joint_csgarch(JointMean<T>& mean, const garch::CsGarchParameters<T>& p, const garch::Presample<T>& presample,
                   std::span<T> sigma2)
{
    std::vector<T> permanent(sigma2.size());
    std::vector<T> squared_residuals(sigma2.size());
    T long_run = p.omega / (1.0 - p.rho);
    return joint_loop<T>(mean, sigma2,
                         [&](std::span<const T> e, std::size_t t)
                         {
                             if (t > 0)
                             {
                                 squared_residuals[t - 1] = e[t - 1] * e[t - 1];
                             }
                             if (t < presample.held)
                             {
                                 permanent[t] = long_run;
                                 return presample.value + long_run;
                             }
                             garch::CsGarchStep<T> step =
                                 garch::csgarch_step<T>(squared_residuals, p, presample, sigma2, permanent, t);
                             permanent[t] = step.permanent;
                             return step.variance;
                         });
}

// sigma^power is the state; sigma2 = (sigma^power)^(2 / power). False when a shock moment is not finite.
template <class T>
bool joint_power(JointMean<T>& mean, std::span<const T> theta, const Layout& layout, const Start<T>& start,
                 std::span<T> sigma2)
{
    std::array<T, kMaxOrder> moments{};
    garch::PowerGarchParameters<T> p = power_garch_parameters<T>(theta, layout, moments);
    p.intercept_shift = start.intercept_shift;
    if (!shock_moments(theta, layout, std::span<T>(moments)))
    {
        return false;
    }
    T presample_power = garch::raise(start.presample.value, 0.5 * p.power, p.free_power);
    T exponent = 2.0 / p.power;
    std::vector<T> sigma_power(sigma2.size());
    return joint_loop<T>(mean, sigma2,
                         [&](std::span<const T> e, std::size_t t)
                         {
                             sigma_power[t] = presample_power;
                             if (t >= start.presample.held)
                             {
                                 sigma_power[t] = garch::power_garch_step<T>(e, p, presample_power, sigma_power, t);
                             }
                             return garch::raise(sigma_power[t], exponent, p.free_power);
                         });
}

// The residuals and the variance of ARCH-in-mean; false when the point is outside the domain of the recursion.
template <class T>
bool joint_series(JointMean<T>& mean, const Layout& layout, const Start<T>& start, std::span<T> sigma2)
{
    std::span<const T> theta = mean.theta;
    switch (layout.model)
    {
        case Model::SGarch:
            return joint_sgarch<T>(mean, shifted(sgarch_parameters(theta, layout), start), start.presample, sigma2);
        case Model::IGarch:
            return joint_igarch<T>(mean, theta, layout, start, sigma2);
        case Model::EGarch:
            return joint_egarch<T>(mean, shifted(egarch_parameters(theta, layout), start), start.presample, sigma2);
        case Model::GjrGarch:
            return joint_gjrgarch<T>(mean, shifted(gjrgarch_parameters(theta, layout), start), start.presample,
                                     sigma2);
        case Model::CsGarch:
            return joint_csgarch<T>(mean, shifted(csgarch_parameters(theta, layout), start), start.presample,
                                    sigma2);
        case Model::ApArch:
        case Model::FGarchGarch:
        case Model::FGarchTGarch:
        case Model::FGarchAvGarch:
        case Model::FGarchNGarch:
        case Model::FGarchNaGarch:
        case Model::FGarchApArch:
        case Model::FGarchGjrGarch:
        case Model::FGarchAllGarch:
            return joint_power<T>(mean, theta, layout, start, sigma2);
    }
    return false;
}

template <class T>
struct Evaluation
{
    const Data& data;
    std::span<const T> e0;
    std::span<T> residuals;
    std::span<T> sigma2;
    std::span<T> terms;
    VarInit var_init;
    const dist::ShapeParameters<T>& shape;
    std::span<const T> intercept_shift;
};

// c_t = sum_k zeta_k v_t,k, or an empty span without vxreg.
template <class T>
std::span<const T> variance_shift(const Data& data, std::span<const T> theta, const Layout& layout,
                                  std::vector<T>& storage)
{
    if (layout.vxreg == 0)
    {
        return {};
    }
    std::span<const T> zeta = block(theta, layout.vxreg_begin(), layout.vxreg);
    storage.resize(data.r.size());
    for (std::size_t t = 0; t < storage.size(); ++t)
    {
        T shift{0.0};
        for (std::size_t k = 0; k < zeta.size(); ++k)
        {
            shift = shift + zeta[k] * data.variance.at(t, k);
        }
        storage[t] = shift;
    }
    return storage;
}

// sum_k zeta_k mean_t v_t,k.
template <class T>
T mean_shift(const RegressorMatrix& variance, std::span<const T> theta, const Layout& layout)
{
    std::span<const T> zeta = block(theta, layout.vxreg_begin(), layout.vxreg);
    std::size_t rows = variance.values.size() / variance.columns;
    T total{0.0};
    for (std::size_t k = 0; k < variance.columns; ++k)
    {
        double sum = 0.0;
        for (std::size_t t = 0; t < rows; ++t)
        {
            sum += variance.at(t, k);
        }
        total = total + zeta[k] * (sum / static_cast<double>(rows));
    }
    return total;
}

// Without ARCH-in-mean the residuals are e0 and the variance a recursion over them; with it the two are stepped
// together, and e0 only feeds the presample and the target.
template <class T>
bool fill_series(const Evaluation<T>& e, std::span<const T> theta, const Layout& layout, const Start<T>& start)
{
    if (!layout.has_archm())
    {
        return fill_variance<T>(e.e0, theta, layout, start, e.sigma2);
    }
    JointMean<T> mean{e.data, theta, layout, arma_parameters(theta, layout), std::vector<T>(e.sigma2.size()),
                      e.residuals};
    return joint_series<T>(mean, layout, start, e.sigma2);
}

template <class T>
T evaluate(const Evaluation<T>& e, std::span<const T> theta, const Layout& layout)
{
    Start<T> start{presample<T>(e.data, e.e0, theta, layout, e.var_init), e.intercept_shift};
    if (!fill_series<T>(e, theta, layout, start))
    {
        return T{kInvalidLoglik};
    }
    return sum_of_density_terms<T>(e.residuals, e.sigma2, dist::make_density(layout.dist, e.shape), e.terms);
}

}

namespace
{

template <class T>
T variance_target(std::span<const T> residuals, std::span<const T> theta, const Layout& layout)
{
    using std::log;
    T slack = T{1.0} - persistence<T>(theta, layout);
    if (layout.model == Model::EGarch)
    {
        return log(mean_square(residuals)) * slack;
    }
    if (layout.is_power_family())
    {
        return mean_power(residuals, power_of(theta, layout), layout.free_power) * slack;
    }
    return mean_square(residuals) * slack;
}

}

template <class T>
T targeted_omega(std::span<const T> residuals, std::span<const T> theta, const Layout& layout,
                 const RegressorMatrix& variance)
{
    if (variance.columns == 0)
    {
        return variance_target<T>(residuals, theta, layout);
    }
    return variance_target<T>(residuals, theta, layout) - mean_shift<T>(variance, theta, layout);
}

template <class T>
T loglik(const Data& data, std::span<const T> theta, const Layout& layout, VarInit var_init,
         const LoglikOutputs<T>& out)
{
    std::span<const double> r = data.r;
    assert(data.mean.columns == static_cast<std::size_t>(layout.mxreg));
    assert(data.mean.values.size() == r.size() * data.mean.columns);
    assert(data.variance.columns == static_cast<std::size_t>(layout.vxreg));
    assert(data.variance.values.size() == r.size() * data.variance.columns);
    assert(theta.size() == layout.count());
    assert(out.terms.empty() || out.terms.size() == r.size());
    assert(out.residuals.empty() || out.residuals.size() == r.size());
    assert(out.sigma2.empty() || out.sigma2.size() == r.size());
    dist::ShapeParameters<T> shape = shape_parameters_of(theta, layout);
    if (!dist::in_domain(layout.dist, value_of(shape)))
    {
        return T{kInvalidLoglik};
    }
    std::vector<T> residual_storage;
    std::vector<T> variance_storage;
    std::span<T> residuals = provided_or(out.residuals, residual_storage, r.size());
    std::span<T> sigma2 = provided_or(out.sigma2, variance_storage, r.size());
    std::vector<T> e0_storage;
    std::span<T> e0 = residuals;
    if (layout.has_archm())
    {
        e0_storage.resize(r.size());
        e0 = e0_storage;
    }
    mean_residuals<T>(data, theta, layout, e0);
    std::vector<T> shift_storage;
    Evaluation<T> evaluation{data,  e0,        residuals, sigma2,
                             out.terms, var_init, shape,     variance_shift<T>(data, theta, layout, shift_storage)};
    if (!layout.variance_targeting)
    {
        return evaluate<T>(evaluation, theta, layout);
    }
    std::array<T, kMaxParameterCount> targeted;
    std::ranges::copy(theta, targeted.begin());
    targeted[layout.omega()] = targeted_omega<T>(e0, theta, layout, data.variance);
    return evaluate<T>(evaluation, std::span<const T>(targeted).first(theta.size()), layout);
}

template double targeted_omega<double>(std::span<const double>, std::span<const double>, const Layout&,
                                    const RegressorMatrix&);
template ad::Dual<8> targeted_omega<ad::Dual<8>>(std::span<const ad::Dual<8>>, std::span<const ad::Dual<8>>,
                                                 const Layout&, const RegressorMatrix&);
template ad::Dual<16> targeted_omega<ad::Dual<16>>(std::span<const ad::Dual<16>>, std::span<const ad::Dual<16>>,
                                                   const Layout&, const RegressorMatrix&);
template ad::Dual<32> targeted_omega<ad::Dual<32>>(std::span<const ad::Dual<32>>, std::span<const ad::Dual<32>>,
                                                   const Layout&, const RegressorMatrix&);
template ad::Dual<kMaxParameterCount> targeted_omega<ad::Dual<kMaxParameterCount>>(std::span<const ad::Dual<kMaxParameterCount>>, std::span<const ad::Dual<kMaxParameterCount>>,
                                                   const Layout&, const RegressorMatrix&);

double targeted_omega_at(const Data& data, std::span<const double> theta, const Layout& layout)
{
    std::vector<double> e0(data.r.size());
    mean_residuals<double>(data, theta, layout, e0);
    return targeted_omega<double>(e0, theta, layout, data.variance);
}

template double loglik<double>(const Data&, std::span<const double>, const Layout&, VarInit,
                               const LoglikOutputs<double>&);
template ad::Dual<8> loglik<ad::Dual<8>>(const Data&, std::span<const ad::Dual<8>>, const Layout&,
                                         VarInit, const LoglikOutputs<ad::Dual<8>>&);
template ad::Dual<16> loglik<ad::Dual<16>>(const Data&, std::span<const ad::Dual<16>>,
                                           const Layout&, VarInit, const LoglikOutputs<ad::Dual<16>>&);
template ad::Dual<32> loglik<ad::Dual<32>>(const Data&, std::span<const ad::Dual<32>>,
                                           const Layout&, VarInit, const LoglikOutputs<ad::Dual<32>>&);
template ad::Dual<kMaxParameterCount> loglik<ad::Dual<kMaxParameterCount>>(const Data&, std::span<const ad::Dual<kMaxParameterCount>>,
                                           const Layout&, VarInit, const LoglikOutputs<ad::Dual<kMaxParameterCount>>&);
}
