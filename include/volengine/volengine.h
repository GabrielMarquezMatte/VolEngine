#ifndef VOLENGINE_VOLENGINE_H
#define VOLENGINE_VOLENGINE_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#if defined(VE_BUILDING)
#define VE_API __declspec(dllexport)
#else
#define VE_API __declspec(dllimport)
#endif
#else
#define VE_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C"
{
#endif

#define VE_ABI_VERSION 0

VE_API int32_t ve_abi_version(void);

typedef int32_t ve_status;
#define VE_OK 0
#define VE_INVALID_ARGUMENT 1
#define VE_INVALID_HANDLE 2
#define VE_UNSUPPORTED 3
#define VE_OUT_OF_MEMORY 4
#define VE_INTERNAL_ERROR 5

typedef struct ve_context ve_context;

VE_API const char* ve_last_error(void);

VE_API ve_status ve_context_create(ve_context** out);
VE_API ve_status ve_context_set_threads(ve_context* ctx, int32_t threads);
VE_API ve_status ve_context_threads(const ve_context* ctx, int32_t* out);
VE_API void ve_context_free(ve_context* ctx);

#define VE_MODEL_SGARCH 0
#define VE_DIST_NORM 0
#define VE_VARINIT_SAMPLE 0
#define VE_VARINIT_BACKCAST 1

typedef struct ve_spec ve_spec;
typedef struct ve_fit_options ve_fit_options;
typedef struct ve_panel ve_panel;

/* On any failure of a *_create function, *out is set to NULL (when out is not NULL). */
VE_API ve_status ve_spec_create(int32_t model, ve_spec** out);
/* p, q < 1 is VE_INVALID_ARGUMENT; only (1, 1) is supported in this version (else VE_UNSUPPORTED). */
VE_API ve_status ve_spec_set_order(ve_spec* s, int32_t p, int32_t q);
VE_API ve_status ve_spec_set_dist(ve_spec* s, int32_t dist);
VE_API ve_status ve_spec_set_include_mean(ve_spec* s, int32_t include_mean); /* 0 or 1 */
VE_API void ve_spec_free(ve_spec* s);

VE_API ve_status ve_fit_options_create(ve_fit_options** out);
VE_API ve_status ve_fit_options_set_max_iter(ve_fit_options* o, int32_t max_iter); /* >= 1 */
VE_API ve_status ve_fit_options_set_tolerance(ve_fit_options* o, double tol);      /* finite, > 0 */
VE_API ve_status ve_fit_options_set_var_init(ve_fit_options* o, int32_t var_init);
VE_API void ve_fit_options_free(ve_fit_options* o);

/* Borrowed panel: `data` is not copied and must stay alive until the end of every call that uses
   the panel. `data` may be NULL only when t * n == 0. Default layout is C order
   (stride_t = n, stride_n = 1). */
VE_API ve_status ve_panel_create(const double* data, size_t t, size_t n, ve_panel** out);
/* Element (t, j) is data[t * stride_t + j * stride_n]; strides are in elements and may be negative.
   With a negative stride, `data` must point at element (0, 0) (numpy convention), so the
   remaining elements lie at negative offsets from it. A stride of 0 is rejected when the
   corresponding dimension is greater than 1. */
VE_API ve_status ve_panel_set_strides(ve_panel* p, ptrdiff_t stride_t, ptrdiff_t stride_n);
/* Optional: series j uses rows starts[j] .. starts[j] + lengths[j] - 1. Copies n entries from each
   array. Both NULL clears the ranges (rectangular panel); exactly one NULL is invalid. */
VE_API ve_status ve_panel_set_ranges(ve_panel* p, const size_t* starts, const size_t* lengths);
VE_API void ve_panel_free(ve_panel* p);

#ifdef __cplusplus
}
#endif

#endif
