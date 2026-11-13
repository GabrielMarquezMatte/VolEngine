#include "capi/dcc/boundary.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <utility>

using ve::capi::guard;
using ve::capi::set_error;
using ve::capi::dcc::check;

extern "C" ve_status ve_dcc_spec_create(int32_t model, ve_dcc_spec** out)
{
    return guard(
        [&]
        {
            if (out != nullptr)
            {
                *out = nullptr;
            }
            if (out == nullptr)
            {
                return set_error(VE_INVALID_ARGUMENT, "out must not be null");
            }
            if (model < VE_DCC_MODEL_DCC || model > VE_DCC_MODEL_FDCC)
            {
                return set_error(VE_INVALID_ARGUMENT, "unknown DCC model");
            }
            auto spec = std::make_unique<ve_dcc_spec>();
            spec->spec.model = static_cast<ve::dcc::DccModel>(model);
            *out = spec.release();
            return VE_OK;
        });
}

extern "C" ve_status ve_dcc_spec_set_order(ve_dcc_spec* s, int32_t p, int32_t q)
{
    return guard(
        [&]
        {
            if (s == nullptr)
            {
                return set_error(VE_INVALID_HANDLE, "s must not be null");
            }
            if (p < 1 || p > VE_MAX_ORDER || q < 1 || q > VE_MAX_ORDER)
            {
                return set_error(VE_INVALID_ARGUMENT, "orders must be in 1..VE_MAX_ORDER");
            }
            s->spec.p = p;
            s->spec.q = q;
            return VE_OK;
        });
}

extern "C" ve_status ve_dcc_spec_set_dist(ve_dcc_spec* s, int32_t dist)
{
    return guard(
        [&]
        {
            if (s == nullptr)
            {
                return set_error(VE_INVALID_HANDLE, "s must not be null");
            }
            if (dist < VE_DCC_DIST_MVNORM || dist > VE_DCC_DIST_MVLAPLACE)
            {
                return set_error(VE_INVALID_ARGUMENT, "unknown DCC distribution");
            }
            s->spec.dist = static_cast<ve::dcc::DccDist>(dist);
            return VE_OK;
        });
}

extern "C" ve_status ve_dcc_spec_set_groups(ve_dcc_spec* s, const int32_t* groups, size_t count)
{
    return guard(
        [&]
        {
            if (s == nullptr)
            {
                return set_error(VE_INVALID_HANDLE, "s must not be null");
            }
            if (s->spec.model != ve::dcc::DccModel::FDcc)
            {
                return set_error(VE_INVALID_ARGUMENT, "groups are only supported for FDCC");
            }
            if (groups == nullptr || !ve::capi::dcc::valid_count<int>(count))
            {
                return set_error(VE_INVALID_ARGUMENT, "groups must not be null and count must be valid and positive");
            }
            std::vector<int> copy(groups, groups + count);
            s->spec.groups = std::move(copy);
            return VE_OK;
        });
}

namespace
{

ve_status put_parameter(ve_dcc_spec* s, const char* name, double value, bool fixed)
{
    const ve_status status = check(s, name);
    if (status != VE_OK)
    {
        return status;
    }
    if (name[0] == '\0' || !std::isfinite(value))
    {
        return set_error(VE_INVALID_ARGUMENT, "name must not be empty and value must be finite");
    }
    auto& values = fixed ? s->spec.fixed : s->spec.start;
    const auto found = std::ranges::find(values, name, &ve::ParameterValue::name);
    if (found != values.end())
    {
        found->value = value;
        return VE_OK;
    }
    values.push_back(ve::ParameterValue{name, value});
    return VE_OK;
}

}

extern "C" ve_status ve_dcc_spec_set_fixed(ve_dcc_spec* s, const char* name, double value)
{
    return guard(
        [&]
        {
            return put_parameter(s, name, value, true);
        });
}

extern "C" ve_status ve_dcc_spec_set_start(ve_dcc_spec* s, const char* name, double value)
{
    return guard(
        [&]
        {
            return put_parameter(s, name, value, false);
        });
}

extern "C" ve_status ve_dcc_options_create(ve_dcc_options** out)
{
    return guard(
        [&]
        {
            return ve::capi::dcc::create(out);
        });
}

extern "C" ve_status ve_dcc_options_set_standard_errors(ve_dcc_options* o, int32_t on)
{
    return guard(
        [&]
        {
            if (o == nullptr)
            {
                return set_error(VE_INVALID_HANDLE, "o must not be null");
            }
            if (on != 0 && on != 1)
            {
                return set_error(VE_INVALID_ARGUMENT, "standard_errors must be 0 or 1");
            }
            o->options.standard_errors = on != 0;
            return VE_OK;
        });
}

extern "C" void ve_dcc_spec_free(ve_dcc_spec* handle)
{
    std::unique_ptr<ve_dcc_spec> owner(handle);
}

extern "C" void ve_dcc_options_free(ve_dcc_options* handle)
{
    std::unique_ptr<ve_dcc_options> owner(handle);
}
