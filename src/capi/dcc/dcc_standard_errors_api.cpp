#include "capi/dcc/boundary.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <utility>

using ve::capi::guard;
using ve::capi::set_error;
using ve::capi::dcc::check;

#include "dcc/standard_errors.hpp"

extern "C" ve_status ve_dcc_fit_standard_errors(ve_context* ctx, const ve_dcc_fit* f, const ve_fit* const* fits, size_t m, ve_dcc_fit** out)
{
    return guard(
        [&]
        {
            if (out != nullptr)
            {
                *out = nullptr;
            }
            if (ctx == nullptr)
            {
                return set_error(VE_INVALID_HANDLE, "ctx must not be null");
            }
            const ve_status status = check(f, out);
            if (status != VE_OK)
            {
                return status;
            }
            if (m != f->result.series)
            {
                return set_error(VE_INVALID_ARGUMENT, "the DCC fit has " + std::to_string(f->result.series)
                    + " series, got " + std::to_string(m) + " univariate fits");
            }
            std::vector<const ve::FitResult*> pointers;
            const ve_status pointer_status = ve::capi::dcc::fit_pointers(fits, m, pointers);
            if (pointer_status != VE_OK)
            {
                return pointer_status;
            }
            if (const auto problem = ve::dcc::fits_problem(f->result, pointers))
            {
                return set_error(VE_INVALID_ARGUMENT, *problem);
            }
            if (const auto problem = ve::dcc::standard_errors_problem(pointers))
            {
                return set_error(VE_UNSUPPORTED, *problem);
            }
            const auto errors = ve::dcc::dcc_standard_errors(f->result, pointers);
            if (!errors)
            {
                return set_error(VE_INVALID_ARGUMENT, errors.error());
            }
            auto handle = std::make_unique<ve_dcc_fit>();
            handle->result = f->result;
            handle->result.se_hessian = errors->hessian;
            handle->result.se_robust = errors->robust;
            *out = handle.release();
            return VE_OK;
        });
}
