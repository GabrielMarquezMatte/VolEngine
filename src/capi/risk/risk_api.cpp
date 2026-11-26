#include "capi/error.hpp"
#include "capi/handles.hpp"
#include "dist/densities.hpp"
#include "risk/measures.hpp"

#include <cmath>
#include <cstddef>
#include <span>

using ve::capi::guard;
using ve::capi::set_error;

namespace
{

bool valid_confidence(double confidence)
{
    return confidence > 0.0 && confidence < 1.0;
}

bool all_finite(const ve::PanelView& view)
{
    for (std::size_t row = 0; row < view.t; ++row)
    {
        for (std::size_t series = 0; series < view.n; ++series)
        {
            if (!std::isfinite(view.at(row, series)))
            {
                return false;
            }
        }
    }
    return true;
}

ve_status check_outputs(const double* var, const double* es, double confidence)
{
    if (var == nullptr || es == nullptr)
    {
        return set_error(VE_INVALID_ARGUMENT, "var and es must not be null");
    }
    if (!valid_confidence(confidence))
    {
        return set_error(VE_INVALID_ARGUMENT, "confidence must be in (0, 1)");
    }
    return VE_OK;
}

ve_status check_panel(const ve::PanelView& view, const double* weights)
{
    if (view.has_ranges())
    {
        return set_error(VE_INVALID_ARGUMENT, "the risk measures need a rectangular panel (no ranges)");
    }
    if (view.t == 0 || view.n == 0)
    {
        return set_error(VE_INVALID_ARGUMENT, "the risk measures need at least 1 row and 1 series");
    }
    if (weights == nullptr)
    {
        return set_error(VE_INVALID_ARGUMENT, "weights must not be null");
    }
    for (std::size_t series = 0; series < view.n; ++series)
    {
        if (!std::isfinite(weights[series]))
        {
            return set_error(VE_INVALID_ARGUMENT, "the weights must be finite");
        }
    }
    if (!all_finite(view))
    {
        return set_error(VE_INVALID_ARGUMENT, "the panel has a value that is not finite");
    }
    return VE_OK;
}

ve_status check_distribution(int32_t dist, const ve::dist::ShapeParameters<double>& parameters)
{
    if (dist < VE_DIST_NORM || dist > VE_DIST_JSU)
    {
        return set_error(VE_INVALID_ARGUMENT, "dist must be one of VE_DIST_NORM .. VE_DIST_JSU");
    }
    if (!ve::dist::in_domain(static_cast<ve::Dist>(dist), parameters))
    {
        return set_error(VE_INVALID_ARGUMENT, "the distribution parameters are outside the density's domain");
    }
    return VE_OK;
}

}

extern "C" ve_status ve_risk_empirical(const ve_panel* returns, const double* weights, int32_t kind, double confidence,
                                       double* var, double* es)
{
    return guard(
        [&]
        {
            if (returns == nullptr)
            {
                return set_error(VE_INVALID_HANDLE, "returns must not be null");
            }
            if (const ve_status status = check_outputs(var, es, confidence); status != VE_OK)
            {
                return status;
            }
            if (kind != VE_RETURNS_LOG && kind != VE_RETURNS_SIMPLE)
            {
                return set_error(VE_INVALID_ARGUMENT, "kind must be VE_RETURNS_LOG or VE_RETURNS_SIMPLE");
            }
            if (const ve_status status = check_panel(returns->view, weights); status != VE_OK)
            {
                return status;
            }
            const auto return_kind = kind == VE_RETURNS_LOG ? ve::risk::ReturnKind::log : ve::risk::ReturnKind::simple;
            const auto measure = ve::risk::empirical(returns->view, std::span(weights, returns->view.n), return_kind,
                                                     confidence);
            if (!std::isfinite(measure.var))
            {
                return set_error(VE_INVALID_ARGUMENT, "a portfolio return is not finite (an input overflows)");
            }
            *var = measure.var;
            *es = measure.es;
            return VE_OK;
        });
}

extern "C" ve_status ve_risk_parametric(int32_t dist, double skew, double shape, double ghlambda, double mean,
                                        double sigma, double confidence, double* var, double* es)
{
    return guard(
        [&]
        {
            if (const ve_status status = check_outputs(var, es, confidence); status != VE_OK)
            {
                return status;
            }
            const ve::dist::ShapeParameters<double> parameters{.skew = skew, .shape = shape, .ghlambda = ghlambda};
            if (const ve_status status = check_distribution(dist, parameters); status != VE_OK)
            {
                return status;
            }
            if (!std::isfinite(mean) || !std::isfinite(sigma) || sigma < 0.0)
            {
                return set_error(VE_INVALID_ARGUMENT, "mean must be finite and sigma finite and >= 0");
            }
            const auto measure = ve::risk::parametric(static_cast<ve::Dist>(dist), parameters,
                                                      ve::risk::Location{.mean = mean, .sigma = sigma}, confidence);
            *var = measure.var;
            *es = measure.es;
            return VE_OK;
        });
}
