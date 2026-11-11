#include "capi/error.hpp"
#include "capi/handles.hpp"

#include <cmath>
#include <memory>
#include <string>

static_assert(VE_MAX_ORDER == ve::kMaxOrder, "the C header and the engine disagree on the order limit");

using ve::capi::guard;
using ve::capi::set_error;

namespace
{

ve_status bad_value(const char* name, int32_t value)
{
    return set_error(VE_INVALID_ARGUMENT,
        std::string(name) + " has unknown value " + std::to_string(value));
}

static_assert(VE_MODEL_SGARCH == static_cast<int32_t>(ve::Model::SGarch));
static_assert(VE_MODEL_EGARCH == static_cast<int32_t>(ve::Model::EGarch));
static_assert(VE_MODEL_GJRGARCH == static_cast<int32_t>(ve::Model::GjrGarch));
static_assert(VE_MODEL_APARCH == static_cast<int32_t>(ve::Model::ApArch));
static_assert(VE_MODEL_IGARCH == static_cast<int32_t>(ve::Model::IGarch));
static_assert(VE_MODEL_CSGARCH == static_cast<int32_t>(ve::Model::CsGarch));
static_assert(VE_MODEL_FGARCH_GARCH == static_cast<int32_t>(ve::Model::FGarchGarch));
static_assert(VE_MODEL_FGARCH_TGARCH == static_cast<int32_t>(ve::Model::FGarchTGarch));
static_assert(VE_MODEL_FGARCH_AVGARCH == static_cast<int32_t>(ve::Model::FGarchAvGarch));
static_assert(VE_MODEL_FGARCH_NGARCH == static_cast<int32_t>(ve::Model::FGarchNGarch));
static_assert(VE_MODEL_FGARCH_NAGARCH == static_cast<int32_t>(ve::Model::FGarchNaGarch));
static_assert(VE_MODEL_FGARCH_APARCH == static_cast<int32_t>(ve::Model::FGarchApArch));
static_assert(VE_MODEL_FGARCH_GJRGARCH == static_cast<int32_t>(ve::Model::FGarchGjrGarch));
static_assert(VE_MODEL_FGARCH_ALLGARCH == static_cast<int32_t>(ve::Model::FGarchAllGarch));

ve_status to_model(int32_t value, ve::Model& model)
{
    if (value < VE_MODEL_SGARCH || value > VE_MODEL_FGARCH_ALLGARCH)
    {
        return bad_value("model", value);
    }
    model = static_cast<ve::Model>(value);
    return VE_OK;
}

static_assert(VE_DIST_NORM == static_cast<int32_t>(ve::Dist::Norm));
static_assert(VE_DIST_STD == static_cast<int32_t>(ve::Dist::Std));
static_assert(VE_DIST_GED == static_cast<int32_t>(ve::Dist::Ged));

ve_status to_dist(int32_t value, ve::Dist& dist)
{
    if (value != VE_DIST_NORM && value != VE_DIST_STD && value != VE_DIST_GED)
    {
        return bad_value("dist", value);
    }
    dist = static_cast<ve::Dist>(value);
    return VE_OK;
}

bool in_range(int32_t value, int32_t low)
{
    return value >= low && value <= VE_MAX_ORDER;
}

ve_status to_var_init(int32_t value, ve::VarInit& var_init)
{
    if (value == VE_VARINIT_SAMPLE)
    {
        var_init = ve::VarInit::Sample;
        return VE_OK;
    }
    if (value == VE_VARINIT_BACKCAST)
    {
        var_init = ve::VarInit::Backcast;
        return VE_OK;
    }
    return bad_value("var_init", value);
}

}

extern "C" ve_status ve_spec_create(int32_t model, ve_spec** out)
{
    if (out != nullptr)
    {
        *out = nullptr;
    }
    return guard(
        [&]
        {
            if (out == nullptr)
            {
                return set_error(VE_INVALID_ARGUMENT, "out must not be null");
            }
            auto spec = std::make_unique<ve_spec>();
            const ve_status status = to_model(model, spec->spec.model);
            if (status != VE_OK)
            {
                return status;
            }
            *out = spec.release();
            return VE_OK;
        });
}

extern "C" ve_status ve_spec_set_order(ve_spec* s, int32_t p, int32_t q)
{
    return guard(
        [&]
        {
            if (s == nullptr)
            {
                return set_error(VE_INVALID_HANDLE, "s must not be null");
            }
            if (!in_range(p, 1) || !in_range(q, 1))
            {
                return set_error(VE_INVALID_ARGUMENT, "p and q must be in 1 .. VE_MAX_ORDER (6)");
            }
            s->spec.p = p;
            s->spec.q = q;
            return VE_OK;
        });
}

extern "C" ve_status ve_spec_set_arma(ve_spec* s, int32_t m, int32_t n)
{
    return guard(
        [&]
        {
            if (s == nullptr)
            {
                return set_error(VE_INVALID_HANDLE, "s must not be null");
            }
            if (!in_range(m, 0) || !in_range(n, 0))
            {
                return set_error(VE_INVALID_ARGUMENT, "m and n must be in 0 .. VE_MAX_ORDER (6)");
            }
            s->spec.ar = m;
            s->spec.ma = n;
            return VE_OK;
        });
}

extern "C" ve_status ve_spec_set_dist(ve_spec* s, int32_t dist)
{
    return guard(
        [&]
        {
            if (s == nullptr)
            {
                return set_error(VE_INVALID_HANDLE, "s must not be null");
            }
            ve::Dist parsed = s->spec.dist;
            const ve_status status = to_dist(dist, parsed);
            if (status != VE_OK)
            {
                return status;
            }
            s->spec.dist = parsed;
            return VE_OK;
        });
}

extern "C" ve_status ve_spec_set_include_mean(ve_spec* s, int32_t include_mean)
{
    return guard(
        [&]
        {
            if (s == nullptr)
            {
                return set_error(VE_INVALID_HANDLE, "s must not be null");
            }
            if (include_mean != 0 && include_mean != 1)
            {
                return bad_value("include_mean", include_mean);
            }
            s->spec.include_mean = include_mean == 1;
            return VE_OK;
        });
}

extern "C" void ve_spec_free(ve_spec* s)
{
    std::unique_ptr<ve_spec> owner(s);
}

extern "C" ve_status ve_fit_options_create(ve_fit_options** out)
{
    if (out != nullptr)
    {
        *out = nullptr;
    }
    return guard(
        [&]
        {
            if (out == nullptr)
            {
                return set_error(VE_INVALID_ARGUMENT, "out must not be null");
            }
            *out = std::make_unique<ve_fit_options>().release();
            return VE_OK;
        });
}

extern "C" ve_status ve_fit_options_set_max_iter(ve_fit_options* o, int32_t max_iter)
{
    return guard(
        [&]
        {
            if (o == nullptr)
            {
                return set_error(VE_INVALID_HANDLE, "o must not be null");
            }
            if (max_iter < 1)
            {
                return set_error(VE_INVALID_ARGUMENT, "max_iter must be >= 1");
            }
            o->config.max_iter = max_iter;
            return VE_OK;
        });
}

extern "C" ve_status ve_fit_options_set_tolerance(ve_fit_options* o, double tol)
{
    return guard(
        [&]
        {
            if (o == nullptr)
            {
                return set_error(VE_INVALID_HANDLE, "o must not be null");
            }
            if (!std::isfinite(tol) || tol <= 0.0)
            {
                return set_error(VE_INVALID_ARGUMENT, "tol must be finite and > 0");
            }
            o->config.tol = tol;
            return VE_OK;
        });
}

extern "C" ve_status ve_fit_options_set_var_init(ve_fit_options* o, int32_t var_init)
{
    return guard(
        [&]
        {
            if (o == nullptr)
            {
                return set_error(VE_INVALID_HANDLE, "o must not be null");
            }
            ve::VarInit parsed = o->config.var_init;
            const ve_status status = to_var_init(var_init, parsed);
            if (status != VE_OK)
            {
                return status;
            }
            o->config.var_init = parsed;
            return VE_OK;
        });
}

extern "C" void ve_fit_options_free(ve_fit_options* o)
{
    std::unique_ptr<ve_fit_options> owner(o);
}
