#ifndef VOLENGINE_CPP_RISK_MEASURES_HPP
#define VOLENGINE_CPP_RISK_MEASURES_HPP

#include <cstdint>
#include <limits>
#include <span>
#include <volengine/error.hpp>
#include <volengine/input.hpp>

// VaR and ES at a confidence in (0, 1) (0.99: the 1% tail), both positive as losses, in units of the return.
namespace ve::risk
{

struct RiskMeasure
{
    double var = 0.0;
    double es = 0.0;
};

enum class ReturnKind : std::int32_t
{
    log = VE_RETURNS_LOG,
    simple = VE_RETURNS_SIMPLE,
};

// A standardized density with the fit's parameters; the ones it lacks are not read.
struct Distribution
{
    Dist dist = Dist::norm;
    double skew = std::numeric_limits<double>::quiet_NaN();
    double shape = std::numeric_limits<double>::quiet_NaN();
    double ghlambda = std::numeric_limits<double>::quiet_NaN();
};

// Over the rows of a T x N panel (dates or simulated paths): row t is the portfolio's simple return
// sum_j weights[j] s_tj, s = exp(r) - 1 for log returns; VaR is minus its numpy.quantile at 1 - confidence and ES minus
// the mean at or below it. One weight per column, not normalized. The cumulative sums of a simulation at checkpoint k
// are the panel Panel::create_strided(cumulative.subspan(k * m), paths, m, K * m, 1).
inline Result<RiskMeasure> empirical(const Panel& returns, std::span<const double> weights, ReturnKind kind,
                                     double confidence)
{
    if (weights.size() != returns.cols())
    {
        return std::unexpected(Error{Status::invalid_argument, "weights needs one value per column"});
    }
    RiskMeasure measure;
    if (auto computed = detail::check(ve_risk_empirical(returns.native(), weights.data(), static_cast<std::int32_t>(kind),
                                                        confidence, &measure.var, &measure.es));
        !computed)
    {
        return detail::failure(computed);
    }
    return measure;
}

// The return mean + sigma z, z the standardized density: VaR = -(mean + sigma q), ES = -(mean + sigma E[z | z <= q]),
// q its quantile at 1 - confidence.
inline Result<RiskMeasure> parametric(const Distribution& distribution, double mean, double sigma, double confidence)
{
    RiskMeasure measure;
    if (auto computed = detail::check(ve_risk_parametric(static_cast<std::int32_t>(distribution.dist), distribution.skew,
                                                         distribution.shape, distribution.ghlambda, mean, sigma,
                                                         confidence, &measure.var, &measure.es));
        !computed)
    {
        return detail::failure(computed);
    }
    return measure;
}

}

#endif
