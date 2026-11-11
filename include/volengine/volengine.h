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

#define VE_FIT_CONVERGED 0
#define VE_FIT_MAX_ITER 1
#define VE_FIT_FAILED 2
#define VE_FIT_INSUFFICIENT_DATA 3
#define VE_SE_HESSIAN 0
#define VE_SE_ROBUST 1
#define VE_IC_PER_OBS 0
#define VE_IC_TOTAL 1

typedef struct ve_batch ve_batch;
typedef struct ve_fit ve_fit;

/* Description of a batch. Copies specs, options and jobs; the panel stays borrowed and must stay
   alive (with its data) until ve_fit_batch returns. */
VE_API ve_status ve_batch_create(ve_batch** out);
/* n_specs must be >= 1 and no entry may be NULL (VE_INVALID_HANDLE). */
VE_API ve_status ve_batch_set_specs(ve_batch* b, const ve_spec* const* specs, size_t n_specs);
VE_API ve_status ve_batch_set_panel(ve_batch* b, const ve_panel* panel);
/* Job i fits specs[job_spec[i]] on series job_series[i]; n_jobs must be >= 1. The indices are
   checked by ve_fit_batch, so specs and panel may be set in any order. */
VE_API ve_status ve_batch_set_jobs(ve_batch* b, const size_t* job_spec, const size_t* job_series, size_t n_jobs);
VE_API ve_status ve_batch_set_options(ve_batch* b, const ve_fit_options* options); /* not called = defaults */
VE_API ve_status ve_batch_job_count(const ve_batch* b, size_t* out);
VE_API void ve_batch_free(ve_batch* b);

/* fits_out has ve_batch_job_count elements. On success it receives one new handle per job; on any
   error every element is set to NULL and no handle is created. Missing specs, panel or jobs, a job
   index out of range, or a NaN or infinite value in a used series is VE_INVALID_ARGUMENT. A fit that
   does not converge is not an error: see ve_fit_status. */
VE_API ve_status ve_fit_batch(ve_context* ctx, const ve_batch* batch, ve_fit** fits_out);

/* VE_FIT_FAILED means "did not converge to a usable optimum": the values of such a fit are those of
   the best finite point the optimizer reached, or NaN (empty series) when it reached none (invalid
   start, constant series). VE_FIT_INSUFFICIENT_DATA always has NaN values and empty series. Check
   the status before using the values below. */
VE_API ve_status ve_fit_status(const ve_fit* f, int32_t* out);
VE_API ve_status ve_fit_nobs(const ve_fit* f, size_t* out);
VE_API ve_status ve_fit_param_count(const ve_fit* f, size_t* out);
/* *out is a static NUL-terminated string; i >= param_count is VE_INVALID_ARGUMENT. */
VE_API ve_status ve_fit_param_name(const ve_fit* f, size_t i, const char** out);
/* Each fills `out` with param_count elements. The values are NaN when the fit has no finite point
   (see ve_fit_status). The t value is estimate / standard error and the p value is
   2 (1 - Phi(|t|)); both are NaN when the standard error is NaN. */
VE_API ve_status ve_fit_estimates(const ve_fit* f, double* out);
VE_API ve_status ve_fit_std_errors(const ve_fit* f, int32_t se_kind, double* out);
VE_API ve_status ve_fit_t_values(const ve_fit* f, int32_t se_kind, double* out);
VE_API ve_status ve_fit_p_values(const ve_fit* f, int32_t se_kind, double* out);
/* NaN with VE_OK when the fit has no finite log-likelihood (see ve_fit_status); the criteria below
   behave the same. */
VE_API ve_status ve_loglik(const ve_fit* f, double* out);
VE_API ve_status ve_aic(const ve_fit* f, int32_t scale, double* out);
VE_API ve_status ve_bic(const ve_fit* f, int32_t scale, double* out);
VE_API ve_status ve_hqic(const ve_fit* f, int32_t scale, double* out);
VE_API ve_status ve_sic(const ve_fit* f, int32_t scale, double* out);
/* Views stay valid until the handle is freed and are read-only. *len is nobs, or 0 when the fit has
   no series (no finite point, see ve_fit_status; then *data is NULL). sigma is the conditional
   standard deviation. */
VE_API ve_status ve_fit_sigma(const ve_fit* f, const double** data, size_t* len);
VE_API ve_status ve_fit_residuals(const ve_fit* f, const double** data, size_t* len);
VE_API ve_status ve_fit_std_residuals(const ve_fit* f, const double** data, size_t* len);
VE_API ve_status ve_fit_fitted(const ve_fit* f, const double** data, size_t* len);
VE_API void ve_fit_free(ve_fit* f);

#ifdef __cplusplus
}
#endif

#endif
