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

/* Message of the last failed call on the calling thread ("" after a success). The pointer stays valid
   until the next VolEngine call on that thread. */
VE_API const char* ve_last_error(void);

#define VE_MAX_THREADS 1024

/* The ve_context_* functions may be called concurrently from several threads on the same context,
   except ve_context_free; ve_fit_batch may also run concurrently. */
VE_API ve_status ve_context_create(ve_context** out);
/* threads in 1 .. VE_MAX_THREADS sets the pool size; 0 means std::thread::hardware_concurrency(),
   capped at VE_MAX_THREADS; anything else is VE_INVALID_ARGUMENT. Waits for the ve_fit_batch calls
   running on ctx to finish before it replaces the pool. */
VE_API ve_status ve_context_set_threads(ve_context* ctx, int32_t threads);
VE_API ve_status ve_context_threads(const ve_context* ctx, int32_t* out);
VE_API void ve_context_free(ve_context* ctx);

/* Parameter names and order for every model: mu, ar1..arm, ma1..man, omega, alpha1..alphap,
   beta1..betaq, gamma1..gammap, eta11..eta1p, eta21..eta2p, delta or lambda, skew, shape, ghlambda. A
   model only has the blocks listed next to it; mu is absent without a mean and the last three are the
   distribution parameters (see VE_DIST_*).
   Only the iGARCH beta_q is derived (1 - sum alpha - the other betas): it has a name and a value,
   it is not estimated, its standard error is NaN, and the information criteria count the estimated
   parameters only. Each fGARCH submodel is a model value.
   ARFIMA (ve_spec_set_arfima) adds arfima after ma, ARCH-in-mean (ve_spec_set_archm) adds archm after ma (a spec
   has one or the other), and the regressors add mxreg1..mxregK after them and vxreg1..vxregJ after the variance
   block (before skew), as in rugarch. */
#define VE_MODEL_SGARCH 0
#define VE_MODEL_EGARCH 1 /* gamma1..gammap */
#define VE_MODEL_GJRGARCH 2 /* gamma1..gammap */
#define VE_MODEL_APARCH 3 /* gamma1..gammap, delta */
#define VE_MODEL_IGARCH 4 /* alpha and beta only; beta_q is derived */
#define VE_MODEL_CSGARCH 5 /* eta11 (rho), eta21 (phi) */
#define VE_MODEL_FGARCH_GARCH 6 /* no extra block, lambda = 2 fixed */
#define VE_MODEL_FGARCH_TGARCH 7 /* eta11..eta1p, lambda = 1 fixed */
#define VE_MODEL_FGARCH_AVGARCH 8 /* eta11..eta1p, eta21..eta2p, lambda = 1 fixed */
#define VE_MODEL_FGARCH_NGARCH 9 /* lambda free */
#define VE_MODEL_FGARCH_NAGARCH 10 /* eta21..eta2p, lambda = 2 fixed */
#define VE_MODEL_FGARCH_APARCH 11 /* eta11..eta1p, lambda free */
#define VE_MODEL_FGARCH_GJRGARCH 12 /* eta11..eta1p, lambda = 2 fixed */
#define VE_MODEL_FGARCH_ALLGARCH 13 /* eta11..eta1p, eta21..eta2p, lambda free */
/* A model value outside VE_MODEL_SGARCH .. VE_MODEL_FGARCH_ALLGARCH makes ve_spec_create return
   VE_INVALID_ARGUMENT. */
/* Every distribution is standardized to zero mean and unit variance (as in rugarch and arch). The
   distribution parameters come last in the parameter vector, in the order skew, shape, ghlambda;
   a distribution only has the ones listed next to it. */
#define VE_DIST_NORM 0
#define VE_DIST_STD 1 /* shape: nu > 2 */
#define VE_DIST_GED 2 /* shape: nu > 0 */
/* Fernandez-Steel skewing of the std and ged kernels (with the same nu), standardized as in rugarch;
   xi = 1 is the symmetric kernel. */
#define VE_DIST_SSTD 3 /* skew: xi > 0, shape: nu > 2 */
#define VE_DIST_SGED 4 /* skew: xi > 0, shape: nu > 0 */
/* sstd versus arch's SkewStudent (Hansen): arch's skewness lambda in (-1, 1) relates to xi by
   xi = sqrt((1 + lambda) / (1 - lambda)), and the standard error of lambda follows by the delta method,
   d xi / d lambda = 1 / (xi (1 - lambda)^2). VolEngine only reports xi; the other way,
   lambda = (xi^2 - 1) / (xi^2 + 1) and SE(lambda) = SE(xi) 4 xi / (xi^2 + 1)^2. */
/* Generalized hyperbolic in the standardized (rho, zeta) parametrization; nig is ghyp with
   lambda = -1/2 fixed. Past zeta = 1e6 double precision no longer standardizes the density (by then a
   normal to an excess kurtosis of order 1 / zeta, 1.5e-5 for nig): the log-likelihood there is -inf, as
   outside the domain. Near-normal data drives zeta up, where rho and zeta are barely identified: the
   Hessian is then often not positive definite, and every standard error is NaN. */
#define VE_DIST_NIG 5 /* skew: rho in (-1, 1), shape: zeta in (0, 1e6] */
#define VE_DIST_GHYP 6 /* skew: rho in (-1, 1), shape: zeta in (0, 1e6], ghlambda: lambda in R */
/* Johnson SU as in rugarch (Rigby and Stasinopoulos). */
#define VE_DIST_JSU 7 /* skew: nu in R, shape: tau > 0 */
#define VE_MAX_ORDER 6
#define VE_MAX_REGRESSORS 8
#define VE_VARINIT_SAMPLE 0
#define VE_VARINIT_BACKCAST 1

typedef struct ve_spec ve_spec;
typedef struct ve_fit_options ve_fit_options;
typedef struct ve_panel ve_panel;

/* On any failure of a *_create function, *out is set to NULL (when out is not NULL). */
VE_API ve_status ve_spec_create(int32_t model, ve_spec** out);
/* p is the ARCH order (number of alpha terms, lagged squared residuals) and q the GARCH order (number
   of beta terms, lagged variances), as in arch's p and q and in rugarch's
   garchOrder = c(alpha order, beta order); rugarch calls those orders q and p, the other way round.
   p, q outside 1 .. VE_MAX_ORDER is VE_INVALID_ARGUMENT and leaves the spec unchanged. */
VE_API ve_status ve_spec_set_order(ve_spec* s, int32_t p, int32_t q);
/* ARMA(m, n) conditional mean; m, n in 0 .. VE_MAX_ORDER, else VE_INVALID_ARGUMENT and the spec is
   unchanged. The default is ARMA(0, 0). */
VE_API ve_status ve_spec_set_arma(ve_spec* s, int32_t m, int32_t n);
/* dist in VE_DIST_NORM .. VE_DIST_JSU, else VE_INVALID_ARGUMENT and the spec is unchanged. */
VE_API ve_status ve_spec_set_dist(ve_spec* s, int32_t dist);
VE_API ve_status ve_spec_set_include_mean(ve_spec* s, int32_t include_mean); /* 0 or 1 */
/* Mean regressors (K columns) and variance regressors (J columns), 0 .. VE_MAX_REGRESSORS, default 0; else
   VE_INVALID_ARGUMENT and the spec is unchanged. The data come with ve_batch_set_regressors. */
VE_API ve_status ve_spec_set_mxreg(ve_spec* s, int32_t count);
VE_API ve_status ve_spec_set_vxreg(ve_spec* s, int32_t count);
/* ARCH-in-mean: 0 none (default), 1 adds xi sigma_t to the mean, 2 adds xi sigma_t^2. Else
   VE_INVALID_ARGUMENT and the spec is unchanged. */
VE_API ve_status ve_spec_set_archm(ve_spec* s, int32_t power);
/* ARFIMA mean (rugarch's arfima = TRUE), 0 or 1 (default 0); other values are VE_INVALID_ARGUMENT and leave the
   spec unchanged. Cannot be combined with ve_spec_set_archm: ve_spec_validate and the batch reject the spec.
   The parameter arfima is d in the open interval (1e-8, 0.5) (rugarch: [1e-8, 0.5]); see ve_batch_set_regressors
   for where it enters the mean. */
VE_API ve_status ve_spec_set_arfima(ve_spec* s, int32_t on);
/* Parameter roles. Every parameter is estimated, fixed or derived. A fixed parameter (set_fixed) keeps
   the given value, is not optimized, has NaN standard errors and does not count in the criteria. A
   derived parameter is a function of the others that the model defines: the iGARCH beta_q and, under
   variance targeting, omega. It has a name and a value, NaN standard errors and does not count in the
   criteria. The criteria count the estimated parameters only (none when everything is fixed).
   Names are those of ve_fit_param_name (rugarch's: mu, ar1, ma1, omega, alpha1, beta1, gamma1, eta11,
   eta21, delta, skew, shape, ghlambda, ...). Fixing a name removes its start value, and a start value
   given for a name that is fixed is ignored: fixed wins in either order of the setters, which give the
   same fit. Setting a name again replaces its value. Names are resolved when the spec is used
   (ve_spec_validate, ve_fit_batch), not by the setters. NULL or empty name, or a value that is NaN or
   infinite, is VE_INVALID_ARGUMENT and leaves the spec unchanged.
   Fixing the parameters together can leave no admissible point (sGARCH with alpha1 = 0.6 and beta1 = 0.5,
   persistence >= 1): that fit ends VE_FIT_FAILED with NaN values, and the rest of the batch is
   unaffected. A spec with no estimated parameter does not run the optimizer: the fit evaluates the
   model at the fixed point (like rugarch's ugarchfilter) and is VE_FIT_CONVERGED when the
   log-likelihood is finite, VE_FIT_FAILED when it is not, with NaN standard errors. */
VE_API ve_status ve_spec_set_fixed(ve_spec* s, const char* name, double value);
/* Replaces the default start value of the parameter. A start value for a derived parameter is
   ignored, and so is one for a fixed parameter (it never unfixes it), so all the estimates of a
   previous fit can be passed back as a warm start next to the fixed values. When the starting point
   leaves the model's domain, the fit ends VE_FIT_FAILED with NaN. The same name rules as
   ve_spec_set_fixed apply. */
VE_API ve_status ve_spec_set_start(ve_spec* s, const char* name, double value);
/* Variance targeting, 0 or 1 (default 0); other values are VE_INVALID_ARGUMENT. omega becomes
   derived and is computed inside the likelihood from the residuals e_t of the current mean parameters
   over the whole sample (for any var_init), with P the persistence of the model:
     sGARCH, gjrGARCH:    omega = mean(e_t^2) (1 - P), P = sum(alpha) + sum(beta) (+ pi sum(gamma) in
                          gjrGARCH, pi = P(z < 0));
     eGARCH:              omega = log(mean(e_t^2)) (1 - sum(beta));
     apARCH, fGARCH:      omega = mean(|e_t|^delta) (1 - P), P = sum(alpha_i kappa_i) + sum(beta), with
                          delta the (free or fixed) power and kappa_i the shock moment of the submodel.
   The estimate of omega is that value at the estimates; its standard error is NaN. The standard errors
   of the other parameters come from the likelihood in which omega is a function of them (as in
   rugarch), not from the two-step covariance of Francq, Horvath and Zakoian (2011). */
VE_API ve_status ve_spec_set_variance_targeting(ve_spec* s, int32_t on);
/* VE_OK when the spec can be fitted, else VE_INVALID_ARGUMENT with the reason in ve_last_error().
   Rejected: a fixed or start name the spec does not have (the message gives the name and the names
   the spec has); fixing the iGARCH beta_q (derived); fixing omega under variance targeting (derived);
   variance targeting in iGARCH (omega would be 0) or csGARCH (rugarch ignores it silently); a csGARCH
   with eta21 (phi) fixed and some beta free (it needs sum(beta) >= phi, which the weight map cannot
   express; fixing phi together with every beta is accepted). ve_fit_batch applies the same check to every
   spec, before any fit, and prefixes the message with "spec <i>: ". */
VE_API ve_status ve_spec_validate(const ve_spec* s);
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
   alive (with its data) during every ve_fit_batch call on a batch that references it. */
VE_API ve_status ve_batch_create(ve_batch** out);
/* n_specs must be >= 1 and no entry may be NULL (VE_INVALID_HANDLE). */
VE_API ve_status ve_batch_set_specs(ve_batch* b, const ve_spec* const* specs, size_t n_specs);
VE_API ve_status ve_batch_set_panel(ve_batch* b, const ve_panel* panel); /* NULL panel: VE_INVALID_HANDLE */
/* Job i fits specs[job_spec[i]] on series job_series[i]; n_jobs must be >= 1. The indices are
   checked by ve_fit_batch, so specs and panel may be set in any order. */
VE_API ve_status ve_batch_set_jobs(ve_batch* b, const size_t* job_spec, const size_t* job_series,
                                    size_t n_jobs);
/* Not called = defaults; a NULL options is VE_INVALID_HANDLE. */
VE_API ve_status ve_batch_set_options(ve_batch* b, const ve_fit_options* options);
/* Regressors of the jobs: mxreg and vxreg have n_jobs entries, entry i for job i (n_jobs is checked against the
   job count by ve_fit_batch, so jobs may be set in any order); a whole array NULL means no job has that kind,
   one NULL entry that job has none. Each entry is a borrowed panel, like the data panel (same lifetime, same
   strides), with t equal to the data panel's t and n equal to the spec's mxreg or vxreg count K; it has no
   ranges. Job i uses the rows starts[s] .. starts[s] + lengths[s] - 1 of the DATA panel's ranges for its series
   s. A regressor is present if and only if the count of the spec is > 0, and its rows used by the job must be
   finite; ve_fit_batch reports any breach as VE_INVALID_ARGUMENT with "job <i>: ...". Copies the pointers only.
   Mean: m_t = mu + sum_k delta_k x_t,k + xi sigma_t^p (xi only with archm), d_t = r_t - m_t, and the ARMA acts
   on d_t: e_t = d_t - sum phi_i d_(t-i) - sum theta_j e_(t-j) (e_t = d_t for t < m). With arfima, the residuals
   become y_t = sum_{k=0}^{t} pi_k e_(t-k), pi_0 = 1, pi_k = pi_(k-1) (k - 1 - d) / k: the (1 - L)^d of rugarch,
   truncated at the first row of the series, at a cost of O(T^2). y_t replaces e_t everywhere after. The
   pre-sample residuals (var_init sample mean, variance targeting target) are those of the same recursion with
   xi = 0.
   Variance: omega becomes omega + sum_k zeta_k v_t,k at each t, in the scale of the model (sigma^2, log sigma^2
   or sigma^delta); csGARCH adds it to q_t, whose pre-sample stays omega / (1 - rho). Persistence P < 1 is always
   enforced, unlike rugarch, which drops it with vxreg. A sigma^2 <= 0 makes the log-likelihood -inf.
   Variance targeting with vxreg: omega = target - sum_k zeta_k mean(v_k), the means over the rows of the series.
   Backcast (VE_VARINIT_BACKCAST) with mxreg uses the residuals of the OLS of r on [1 (with a mean), X] (as arch);
   the ARMA stays out of the OLS. Without mxreg that is r - mean(r) (r without a mean). */
VE_API ve_status ve_batch_set_regressors(ve_batch* b, const ve_panel* const* mxreg, const ve_panel* const* vxreg,
                                         size_t n_jobs);
VE_API ve_status ve_batch_job_count(const ve_batch* b, size_t* out);
VE_API void ve_batch_free(ve_batch* b);

/* fits_out has ve_batch_job_count elements. On success it receives one new handle per job; on any
   error every element is set to NULL and no handle is created. Missing specs, panel or jobs, a job
   index out of range, a spec that ve_spec_validate rejects (message prefixed "spec <i>: "), or a NaN
   or infinite value in a used series is VE_INVALID_ARGUMENT. A fit that
   does not converge is not an error: see ve_fit_status. The fits own their data: they stay valid after
   the batch, panel, specs and ctx are freed. */
VE_API ve_status ve_fit_batch(ve_context* ctx, const ve_batch* batch, ve_fit** fits_out);

/* VE_FIT_FAILED means "did not converge to a usable optimum": the values of such a fit are those of
   the best finite point the optimizer reached, or NaN (empty series) when it reached none (invalid
   start, constant series). VE_FIT_INSUFFICIENT_DATA always has NaN values and empty series. Check
   the status before using the values below. */
VE_API ve_status ve_fit_status(const ve_fit* f, int32_t* out);
VE_API ve_status ve_fit_nobs(const ve_fit* f, size_t* out);
VE_API ve_status ve_fit_param_count(const ve_fit* f, size_t* out);
/* *out is a NUL-terminated string valid until f is freed; i >= param_count is VE_INVALID_ARGUMENT. */
VE_API ve_status ve_fit_param_name(const ve_fit* f, size_t i, const char** out);
/* Each fills `out` with param_count elements. The values are NaN when the fit has no finite point
   (see ve_fit_status). The t value is estimate / standard error and the p value is
   2 (1 - Phi(|t|)); both are NaN when the standard error is NaN. */
VE_API ve_status ve_fit_estimates(const ve_fit* f, double* out);
VE_API ve_status ve_fit_std_errors(const ve_fit* f, int32_t se_kind, double* out);
VE_API ve_status ve_fit_t_values(const ve_fit* f, int32_t se_kind, double* out);
VE_API ve_status ve_fit_p_values(const ve_fit* f, int32_t se_kind, double* out);
/* NaN with VE_OK when the fit has no finite point (see ve_fit_status); the criteria below behave the
   same. */
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
/* Diagnostics: caller-owned output arrays. A fit without a finite point is invalid.
 * Ljung-Box requires 1 <= lag < nobs and lag >= 3*fitdf-1 (fitdf ignored for squares).
 * ARCH-LM requires 1 <= fitdf < lag < nobs. Both return [statistic, p_value].
 * Undefined statistics (constant autocorrelations or singular regressions) return NaN.
 * See docs/guide/validation.md for formulas and rugarch conventions. */
VE_API ve_status ve_fit_ljung_box(const ve_fit* f, int32_t squared, size_t lag, size_t fitdf, double* out);
VE_API ve_status ve_fit_arch_lm(const ve_fit* f, size_t lag, size_t fitdf, double* out);
/* individual and estimated have ve_fit_param_count entries in parameter order. estimated[i] is 1 for a parameter
 * the fit estimated and 0 for a fixed or derived one (variance-targeted omega, iGARCH beta_q), whose individual
 * statistic is NaN; the count of 1s is what ve_nyblom_critical takes for the joint statistic. An estimated
 * parameter can still get NaN, when its scores are all zero. */
VE_API ve_status ve_fit_nyblom(const ve_fit* f, double* individual, double* joint, int32_t* estimated);
/* Critical values at 10%, 5%, 1%, for 1..20 estimated parameters. */
VE_API ve_status ve_nyblom_critical(size_t parameters, double* out);
/* [statistic, p_value] for sign, negative, positive, joint, in that order. */
VE_API ve_status ve_fit_sign_bias(const ve_fit* f, double* out); /* nobs >= 6 */
/* [statistic, p_value], with 2 <= groups <= nobs. */
VE_API ve_status ve_fit_gof(const ve_fit* f, size_t groups, double* out);

VE_API void ve_fit_free(ve_fit* f);

/* ---- Forecast ----
   The forecast of a fit h = 1..horizon steps after its last observation, as rugarch's ugarchforecast with
   n.roll = 0, at the estimate of the fit. A fit with every parameter fixed forecasts from a specification.
   mxreg and vxreg: the regressors of the horizon, panels of exactly `horizon` rows x the spec's mxreg / vxreg
   columns, without ranges; NULL when the spec has none. VE_INVALID_ARGUMENT for horizon 0, a fit with no finite
   point (see ve_fit_status), a missing or extra panel, or one of another shape. A VE_FIT_FAILED fit that has a
   finite point forecasts from it. The forecast copies what it needs: it stays valid after f is freed. */
typedef struct ve_forecast ve_forecast;
VE_API ve_status ve_forecast_create(const ve_fit* f, size_t horizon, const ve_panel* mxreg,
                                    const ve_panel* vxreg, ve_forecast** out);
VE_API ve_status ve_forecast_horizon(const ve_forecast* fc, size_t* out);
/* Views of `horizon` values, valid until fc is freed. sigma is rugarch's sigmaFor: in eGARCH it is
   exp(E[log sigma2] / 2), in apARCH and fGARCH E[sigma^delta]^(1 / delta), which for h > 1 is not
   sqrt(E[sigma2]); in the other models it is sqrt(E[sigma2]). mean is the conditional mean (seriesFor). */
VE_API ve_status ve_forecast_sigma(const ve_forecast* fc, const double** data, size_t* len);
VE_API ve_status ve_forecast_mean(const ve_forecast* fc, const double** data, size_t* len);
VE_API void ve_forecast_free(ve_forecast* fc);

/* ---- Simulation ----
   Paths of `horizon` steps from a fit (continuing after its last observation, as rugarch's ugarchsim with
   startMethod = "sample") or from a spec with every parameter fixed (from its unconditional state, as rugarch's
   ugarchpath). Each step draws z from the distribution of the spec, e = sigma z and r = the conditional mean + e.
   The same seed, source and library version give the same paths on every machine with the same ISA and for any
   number of threads; a new version may change them (see the CHANGELOG). The first k paths of a simulation are
   those of any simulation of the same seed with more paths, and its steps the first of any longer one. */
typedef struct ve_sim_options ve_sim_options;
typedef struct ve_sim ve_sim;
/* horizon >= 1 steps and paths >= 1, any seed. VE_INVALID_ARGUMENT when paths x horizon values do not fit in
   memory addresses ("too large"). */
VE_API ve_status ve_sim_options_create(size_t horizon, size_t paths, uint64_t seed, ve_sim_options** out);
/* Copies the regressors of the horizon, the same for every path: panels of exactly `horizon` rows without
   ranges, NULL for none. Their columns are checked against the spec by ve_simulate_*: the spec's mxreg / vxreg
   count, NULL when it has none. Replaces what an earlier call set. */
VE_API ve_status ve_sim_options_set_regressors(ve_sim_options* o, const ve_panel* mxreg, const ve_panel* vxreg);
VE_API void ve_sim_options_free(ve_sim_options* o);
/* VE_INVALID_ARGUMENT for a fit with no finite point (see ve_fit_status) or regressors that do not match its spec;
   a VE_FIT_FAILED fit that has a finite point simulates from it. */
VE_API ve_status ve_simulate_fit(ve_context* ctx, const ve_fit* f, const ve_sim_options* o, ve_sim** out);
/* VE_INVALID_ARGUMENT, with the reason in ve_last_error, for a spec with a free parameter, variance targeting,
   iGARCH (no unconditional variance), parameters outside their domain, persistence >= 1, or regressors that do not
   match it. The presample holds the unconditional sigma2 (with the first row of vxreg), the unconditional mean
   and residuals drawn from the paths themselves. */
VE_API ve_status ve_simulate_spec(ve_context* ctx, const ve_spec* s, const ve_sim_options* o, ve_sim** out);
VE_API ve_status ve_sim_paths(const ve_sim* sim, size_t* out);
VE_API ve_status ve_sim_horizon(const ve_sim* sim, size_t* out);
/* paths x horizon values, row-major (row p is path p), valid until sim is freed: the simulated series and its
   conditional standard deviation. The simulation copies what it needs: it stays valid after its context, fit,
   spec and options are freed. */
VE_API ve_status ve_sim_returns(const ve_sim* sim, const double** data, size_t* len);
VE_API ve_status ve_sim_sigma(const ve_sim* sim, const double** data, size_t* len);
VE_API void ve_sim_free(ve_sim* sim);

/* ---- DCC ----
   Batches copy the specs, options and univariate fits. Results and their views remain valid independently.
   The public convention follows the papers. At least two finite fits of the same length >= 100 are required. */
typedef struct ve_dcc_spec ve_dcc_spec;
typedef struct ve_dcc_options ve_dcc_options;
typedef struct ve_dcc_batch ve_dcc_batch;
typedef struct ve_dcc_fit ve_dcc_fit;
#define VE_DCC_MODEL_DCC 0
#define VE_DCC_MODEL_ADCC 1
#define VE_DCC_MODEL_FDCC 2
#define VE_DCC_DIST_MVNORM 0
#define VE_DCC_DIST_MVT 1
#define VE_DCC_DIST_MVLAPLACE 2
#define VE_DCC_CRITERION_AIC 0
#define VE_DCC_CRITERION_BIC 1
#define VE_DCC_CRITERION_HQIC 2
#define VE_DCC_CRITERION_SIC 3
VE_API ve_status ve_dcc_spec_create(int32_t model, ve_dcc_spec** out);
VE_API ve_status ve_dcc_spec_set_order(ve_dcc_spec* s, int32_t p, int32_t q);
VE_API ve_status ve_dcc_spec_set_dist(ve_dcc_spec* s, int32_t dist);
VE_API ve_status ve_dcc_spec_set_groups(ve_dcc_spec* s, const int32_t* groups, size_t count);
VE_API ve_status ve_dcc_spec_set_fixed(ve_dcc_spec* s, const char* name, double value);
VE_API ve_status ve_dcc_spec_set_start(ve_dcc_spec* s, const char* name, double value);
VE_API void ve_dcc_spec_free(ve_dcc_spec* s);
/* Standard errors are enabled by default; regressors make them VE_UNSUPPORTED. */
VE_API ve_status ve_dcc_options_create(ve_dcc_options** out);
VE_API ve_status ve_dcc_options_set_standard_errors(ve_dcc_options* o, int32_t on);
VE_API void ve_dcc_options_free(ve_dcc_options* o);
VE_API ve_status ve_dcc_batch_create(ve_dcc_batch** out);
VE_API ve_status ve_dcc_batch_set_specs(ve_dcc_batch* b, const ve_dcc_spec* const* specs, size_t n);
VE_API ve_status ve_dcc_batch_set_fits(ve_dcc_batch* b, const ve_fit* const* fits, size_t m);
VE_API ve_status ve_dcc_batch_set_options(ve_dcc_batch* b, const ve_dcc_options* o);
VE_API ve_status ve_dcc_batch_spec_count(const ve_dcc_batch* b, size_t* out);
VE_API void ve_dcc_batch_free(ve_dcc_batch* b);
/* out holds spec_count handles in spec order. Nonconvergence is a per-fit status. */
VE_API ve_status ve_dcc_fit_batch(ve_context* ctx, const ve_dcc_batch* b, ve_dcc_fit** out);
VE_API ve_status ve_dcc_fit_status(const ve_dcc_fit* f, int32_t* out);
VE_API ve_status ve_dcc_fit_nobs(const ve_dcc_fit* f, size_t* out);
VE_API ve_status ve_dcc_fit_series(const ve_dcc_fit* f, size_t* out);
VE_API ve_status ve_dcc_fit_param_count(const ve_dcc_fit* f, size_t* out);
VE_API ve_status ve_dcc_fit_param_name(const ve_dcc_fit* f, size_t i, const char** out);
VE_API ve_status ve_dcc_fit_estimates(const ve_dcc_fit* f, double* out);
VE_API ve_status ve_dcc_fit_std_errors(const ve_dcc_fit* f, int32_t se_kind, double* out);
VE_API ve_status ve_dcc_fit_t_values(const ve_dcc_fit* f, int32_t se_kind, double* out);
VE_API ve_status ve_dcc_fit_p_values(const ve_dcc_fit* f, int32_t se_kind, double* out);
VE_API ve_status ve_dcc_fit_loglik(const ve_dcc_fit* f, double* out);
/* scale is VE_IC_PER_OBS or VE_IC_TOTAL; which is VE_DCC_CRITERION_*. */
VE_API ve_status ve_dcc_fit_criterion(const ve_dcc_fit* f, int32_t which, int32_t scale, double* out);
/* Zero-copy T x m x m row-major views, valid until f is freed. H is computed once, on first access. */
VE_API ve_status ve_dcc_fit_correlation(const ve_dcc_fit* f, const double** data, size_t* len);
VE_API ve_status ve_dcc_fit_covariance(const ve_dcc_fit* f, const double** data, size_t* len);
/* Returns a new immutable fit sharing R; fits must match the original univariate estimates and order. */
VE_API ve_status ve_dcc_fit_standard_errors(ve_context* ctx, const ve_dcc_fit* f,
                                          const ve_fit* const* fits, size_t m, ve_dcc_fit** out);
VE_API void ve_dcc_fit_free(ve_dcc_fit* f);

#ifdef __cplusplus
}
#endif

#endif
