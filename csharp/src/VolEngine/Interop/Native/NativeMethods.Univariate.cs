using System.Runtime.InteropServices;

namespace VolEngine.Interop;

internal static unsafe partial class NativeMethods
{
    [LibraryImport(LibraryName)] internal static partial int ve_abi_version();
    [LibraryImport(LibraryName)] internal static partial nint ve_last_error();

    [LibraryImport(LibraryName)] internal static partial int ve_context_create(out ContextHandle context);
    [LibraryImport(LibraryName)] internal static partial int ve_context_set_threads(ContextHandle context, int threads);
    [LibraryImport(LibraryName)] internal static partial int ve_context_threads(ContextHandle context, out int threads);
    [LibraryImport(LibraryName)] internal static partial void ve_context_free(nint context);

    [LibraryImport(LibraryName)] internal static partial int ve_spec_create(int model, out SpecHandle spec);
    [LibraryImport(LibraryName)] internal static partial int ve_spec_set_order(SpecHandle spec, int p, int q);
    [LibraryImport(LibraryName)] internal static partial int ve_spec_set_arma(SpecHandle spec, int m, int n);
    [LibraryImport(LibraryName)] internal static partial int ve_spec_set_dist(SpecHandle spec, int dist);
    [LibraryImport(LibraryName)] internal static partial int ve_spec_set_include_mean(SpecHandle spec, int includeMean);
    [LibraryImport(LibraryName)] internal static partial int ve_spec_set_mxreg(SpecHandle spec, int count);
    [LibraryImport(LibraryName)] internal static partial int ve_spec_set_vxreg(SpecHandle spec, int count);
    [LibraryImport(LibraryName)] internal static partial int ve_spec_set_archm(SpecHandle spec, int power);
    [LibraryImport(LibraryName)] internal static partial int ve_spec_set_arfima(SpecHandle spec, int on);

    [LibraryImport(LibraryName, StringMarshalling = StringMarshalling.Utf8)]
    internal static partial int ve_spec_set_fixed(SpecHandle spec, string name, double value);

    [LibraryImport(LibraryName, StringMarshalling = StringMarshalling.Utf8)]
    internal static partial int ve_spec_set_start(SpecHandle spec, string name, double value);

    [LibraryImport(LibraryName)] internal static partial int ve_spec_set_variance_targeting(SpecHandle spec, int on);
    [LibraryImport(LibraryName)] internal static partial int ve_spec_validate(SpecHandle spec);
    [LibraryImport(LibraryName)] internal static partial void ve_spec_free(nint spec);

    [LibraryImport(LibraryName)] internal static partial int ve_fit_options_create(out FitOptionsHandle options);
    [LibraryImport(LibraryName)] internal static partial int ve_fit_options_set_max_iter(FitOptionsHandle options, int maxIter);
    [LibraryImport(LibraryName)] internal static partial int ve_fit_options_set_tolerance(FitOptionsHandle options, double tol);
    [LibraryImport(LibraryName)] internal static partial int ve_fit_options_set_var_init(FitOptionsHandle options, int varInit);
    [LibraryImport(LibraryName)] internal static partial int ve_fit_options_set_standard_errors(FitOptionsHandle options, int on);
    [LibraryImport(LibraryName)] internal static partial void ve_fit_options_free(nint options);

    [LibraryImport(LibraryName)]
    internal static partial int ve_panel_create(double* data, nuint t, nuint n, out PanelHandle panel);

    [LibraryImport(LibraryName)] internal static partial int ve_panel_set_strides(PanelHandle panel, nint strideT, nint strideN);
    [LibraryImport(LibraryName)] internal static partial int ve_panel_set_ranges(PanelHandle panel, nuint* starts, nuint* lengths);
    [LibraryImport(LibraryName)] internal static partial void ve_panel_free(nint panel);

    [LibraryImport(LibraryName)] internal static partial int ve_batch_create(out BatchHandle batch);
    [LibraryImport(LibraryName)] internal static partial int ve_batch_set_specs(BatchHandle batch, nint* specs, nuint count);
    [LibraryImport(LibraryName)] internal static partial int ve_batch_set_panel(BatchHandle batch, PanelHandle panel);

    [LibraryImport(LibraryName)]
    internal static partial int ve_batch_set_jobs(BatchHandle batch, nuint* jobSpec, nuint* jobSeries, nuint jobs);

    [LibraryImport(LibraryName)] internal static partial int ve_batch_set_options(BatchHandle batch, FitOptionsHandle options);

    [LibraryImport(LibraryName)]
    internal static partial int ve_batch_set_regressors(BatchHandle batch, nint* mxreg, nint* vxreg, nuint jobs);

    [LibraryImport(LibraryName)] internal static partial int ve_batch_job_count(BatchHandle batch, out nuint count);
    [LibraryImport(LibraryName)] internal static partial void ve_batch_free(nint batch);

    [LibraryImport(LibraryName)] internal static partial int ve_fit_batch(ContextHandle context, BatchHandle batch, nint* fits);

    [LibraryImport(LibraryName)]
    internal static partial int ve_fit_batch_with_cancel(ContextHandle context, BatchHandle batch, CancelHandle cancel,
                                                         nint* fits);

    [LibraryImport(LibraryName)] internal static partial int ve_fit_status(FitHandle fit, out int status);
    [LibraryImport(LibraryName)] internal static partial int ve_fit_nobs(FitHandle fit, out nuint nobs);
    [LibraryImport(LibraryName)] internal static partial int ve_fit_param_count(FitHandle fit, out nuint count);
    [LibraryImport(LibraryName)] internal static partial int ve_fit_param_name(FitHandle fit, nuint index, out nint name);
    [LibraryImport(LibraryName)] internal static partial int ve_fit_estimates(FitHandle fit, double* values);
    [LibraryImport(LibraryName)] internal static partial int ve_fit_std_errors(FitHandle fit, int kind, double* values);
    [LibraryImport(LibraryName)] internal static partial int ve_fit_t_values(FitHandle fit, int kind, double* values);
    [LibraryImport(LibraryName)] internal static partial int ve_fit_p_values(FitHandle fit, int kind, double* values);
    [LibraryImport(LibraryName)] internal static partial int ve_fit_loglik(FitHandle fit, out double loglik);

    [LibraryImport(LibraryName)]
    internal static partial int ve_fit_criterion(FitHandle fit, int which, int scale, out double value);

    [LibraryImport(LibraryName)] internal static partial int ve_fit_sigma(FitHandle fit, out double* data, out nuint length);
    [LibraryImport(LibraryName)] internal static partial int ve_fit_residuals(FitHandle fit, out double* data, out nuint length);
    [LibraryImport(LibraryName)] internal static partial int ve_fit_std_residuals(FitHandle fit, out double* data, out nuint length);
    [LibraryImport(LibraryName)] internal static partial int ve_fit_fitted(FitHandle fit, out double* data, out nuint length);

    [LibraryImport(LibraryName)]
    internal static partial int ve_fit_ljung_box(FitHandle fit, int squared, nuint lag, nuint fitdf, double* result);

    [LibraryImport(LibraryName)] internal static partial int ve_fit_arch_lm(FitHandle fit, nuint lag, nuint fitdf, double* result);

    [LibraryImport(LibraryName)]
    internal static partial int ve_fit_nyblom(FitHandle fit, double* individual, double* joint, int* estimated);

    [LibraryImport(LibraryName)] internal static partial int ve_nyblom_critical(nuint parameters, double* values);
    [LibraryImport(LibraryName)] internal static partial int ve_fit_sign_bias(FitHandle fit, double* result);
    [LibraryImport(LibraryName)] internal static partial int ve_fit_gof(FitHandle fit, nuint groups, double* result);
    [LibraryImport(LibraryName)] internal static partial void ve_fit_free(nint fit);
}
