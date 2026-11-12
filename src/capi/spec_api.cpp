#include "capi/error.hpp"
#include "capi/handles.hpp"
#include "model/layout.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <optional>
#include <string>
#include <vector>

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
static_assert(VE_DIST_SSTD == static_cast<int32_t>(ve::Dist::Sstd));
static_assert(VE_DIST_SGED == static_cast<int32_t>(ve::Dist::Sged));
static_assert(VE_DIST_NIG == static_cast<int32_t>(ve::Dist::Nig));
static_assert(VE_DIST_GHYP == static_cast<int32_t>(ve::Dist::Ghyp));
static_assert(VE_DIST_JSU == static_cast<int32_t>(ve::Dist::Jsu));

ve_status to_dist(int32_t value, ve::Dist& dist)
{
    if (value < VE_DIST_NORM || value > VE_DIST_JSU)
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

void remove_name(std::vector<ve::ParameterValue>& values, const std::string& name)
{
    std::erase_if(values, [&name](const ve::ParameterValue& entry) { return entry.name == name; });
}

void put_value(std::vector<ve::ParameterValue>& values, const std::string& name, double value)
{
    const auto found = std::ranges::find(values, name, &ve::ParameterValue::name);
    if (found == values.end())
    {
        values.push_back(ve::ParameterValue{name, value});
        return;
    }
    found->value = value;
}

ve_status check_name_and_value(const char* name, double value)
{
    if (name == nullptr || name[0] == '\0')
    {
        return set_error(VE_INVALID_ARGUMENT, "name must not be null or empty");
    }
    if (!std::isfinite(value))
    {
        return set_error(VE_INVALID_ARGUMENT, "value must be finite");
    }
    return VE_OK;
}

// Fixing a name removes its start value. A start value never unfixes: on a fixed name it stays in the
// list and the core ignores it, so the setters give the same fit in either order.
ve_status set_named_value(ve_spec* s, const char* name, double value, bool as_fixed)
{
    return guard(
        [&]
        {
            if (s == nullptr)
            {
                return set_error(VE_INVALID_HANDLE, "s must not be null");
            }
            const ve_status status = check_name_and_value(name, value);
            if (status != VE_OK)
            {
                return status;
            }
            if (as_fixed)
            {
                remove_name(s->spec.start, name);
            }
            put_value(as_fixed ? s->spec.fixed : s->spec.start, name, value);
            return VE_OK;
        });
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

extern "C" ve_status ve_spec_set_fixed(ve_spec* s, const char* name, double value)
{
    return set_named_value(s, name, value, true);
}

extern "C" ve_status ve_spec_set_start(ve_spec* s, const char* name, double value)
{
    return set_named_value(s, name, value, false);
}

extern "C" ve_status ve_spec_set_variance_targeting(ve_spec* s, int32_t on)
{
    return guard(
        [&]
        {
            if (s == nullptr)
            {
                return set_error(VE_INVALID_HANDLE, "s must not be null");
            }
            if (on != 0 && on != 1)
            {
                return bad_value("variance_targeting", on);
            }
            s->spec.variance_targeting = on == 1;
            return VE_OK;
        });
}

extern "C" ve_status ve_spec_validate(const ve_spec* s)
{
    return guard(
        [&]
        {
            if (s == nullptr)
            {
                return set_error(VE_INVALID_HANDLE, "s must not be null");
            }
            const std::optional<std::string> problem = ve::model::spec_problem(s->spec);
            if (problem.has_value())
            {
                return set_error(VE_INVALID_ARGUMENT, *problem);
            }
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
