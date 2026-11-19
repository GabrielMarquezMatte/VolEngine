using System.Runtime.InteropServices;

namespace VolEngine.Interop;

internal static unsafe partial class NativeMethods
{
    [LibraryImport(LibraryName)] internal static partial int ve_dcc_spec_create(int model, out DccSpecHandle spec);
    [LibraryImport(LibraryName)] internal static partial int ve_dcc_spec_set_order(DccSpecHandle spec, int p, int q);
    [LibraryImport(LibraryName)] internal static partial int ve_dcc_spec_set_dist(DccSpecHandle spec, int dist);

    [LibraryImport(LibraryName)]
    internal static partial int ve_dcc_spec_set_groups(DccSpecHandle spec, int* groups, nuint count);

    [LibraryImport(LibraryName, StringMarshalling = StringMarshalling.Utf8)]
    internal static partial int ve_dcc_spec_set_fixed(DccSpecHandle spec, string name, double value);

    [LibraryImport(LibraryName, StringMarshalling = StringMarshalling.Utf8)]
    internal static partial int ve_dcc_spec_set_start(DccSpecHandle spec, string name, double value);

    [LibraryImport(LibraryName)] internal static partial void ve_dcc_spec_free(nint spec);
    [LibraryImport(LibraryName)] internal static partial int ve_dcc_spec_set_copula(DccSpecHandle spec, int transformation);
    [LibraryImport(LibraryName)] internal static partial int ve_dcc_spec_set_static_method(DccSpecHandle spec, int method);

    [LibraryImport(LibraryName)] internal static partial int ve_dcc_options_create(out DccOptionsHandle options);
    [LibraryImport(LibraryName)] internal static partial int ve_dcc_options_set_standard_errors(DccOptionsHandle options, int on);
    [LibraryImport(LibraryName)] internal static partial void ve_dcc_options_free(nint options);

    [LibraryImport(LibraryName)] internal static partial int ve_dcc_batch_create(out DccBatchHandle batch);
    [LibraryImport(LibraryName)] internal static partial int ve_dcc_batch_set_specs(DccBatchHandle batch, nint* specs, nuint count);
    [LibraryImport(LibraryName)] internal static partial int ve_dcc_batch_set_fits(DccBatchHandle batch, nint* fits, nuint count);

    [LibraryImport(LibraryName)]
    internal static partial int ve_dcc_batch_set_options(DccBatchHandle batch, DccOptionsHandle options);

    [LibraryImport(LibraryName)] internal static partial int ve_dcc_batch_spec_count(DccBatchHandle batch, out nuint count);
    [LibraryImport(LibraryName)] internal static partial void ve_dcc_batch_free(nint batch);

    [LibraryImport(LibraryName)]
    internal static partial int ve_dcc_fit_batch(ContextHandle context, DccBatchHandle batch, nint* fits);

    [LibraryImport(LibraryName)] internal static partial int ve_dcc_fit_status(DccFitHandle fit, out int status);
    [LibraryImport(LibraryName)] internal static partial int ve_dcc_fit_nobs(DccFitHandle fit, out nuint nobs);
    [LibraryImport(LibraryName)] internal static partial int ve_dcc_fit_series(DccFitHandle fit, out nuint series);
    [LibraryImport(LibraryName)] internal static partial int ve_dcc_fit_param_count(DccFitHandle fit, out nuint count);
    [LibraryImport(LibraryName)] internal static partial int ve_dcc_fit_param_name(DccFitHandle fit, nuint index, out nint name);
    [LibraryImport(LibraryName)] internal static partial int ve_dcc_fit_estimates(DccFitHandle fit, double* values);
    [LibraryImport(LibraryName)] internal static partial int ve_dcc_fit_std_errors(DccFitHandle fit, int kind, double* values);
    [LibraryImport(LibraryName)] internal static partial int ve_dcc_fit_t_values(DccFitHandle fit, int kind, double* values);
    [LibraryImport(LibraryName)] internal static partial int ve_dcc_fit_p_values(DccFitHandle fit, int kind, double* values);
    [LibraryImport(LibraryName)] internal static partial int ve_dcc_fit_loglik(DccFitHandle fit, out double loglik);

    [LibraryImport(LibraryName)]
    internal static partial int ve_dcc_fit_criterion(DccFitHandle fit, int which, int scale, out double value);

    [LibraryImport(LibraryName)]
    internal static partial int ve_dcc_fit_correlation(DccFitHandle fit, out double* data, out nuint length);

    [LibraryImport(LibraryName)]
    internal static partial int ve_dcc_fit_covariance(DccFitHandle fit, out double* data, out nuint length);

    [LibraryImport(LibraryName)]
    internal static partial int ve_dcc_standard_errors(ContextHandle context, DccFitHandle fit, nint* fits, nuint count,
                                                       out DccFitHandle result);

    [LibraryImport(LibraryName)] internal static partial void ve_dcc_fit_free(nint fit);

    [LibraryImport(LibraryName)]
    internal static partial int ve_dcc_future_create(nuint series, nuint horizon, out DccFutureHandle future);

    [LibraryImport(LibraryName)]
    internal static partial int ve_dcc_future_set(DccFutureHandle future, nuint series, PanelHandle mxreg, PanelHandle vxreg);

    [LibraryImport(LibraryName)] internal static partial void ve_dcc_future_free(nint future);

    [LibraryImport(LibraryName)]
    internal static partial int ve_dcc_forecast_create(DccFitHandle fit, nuint horizon, DccFutureHandle future,
                                                       out DccForecastHandle forecast);

    [LibraryImport(LibraryName)] internal static partial int ve_dcc_forecast_horizon(DccForecastHandle forecast, out nuint horizon);
    [LibraryImport(LibraryName)] internal static partial int ve_dcc_forecast_series(DccForecastHandle forecast, out nuint series);

    [LibraryImport(LibraryName)]
    internal static partial int ve_dcc_forecast_correlation(DccForecastHandle forecast, out double* data, out nuint length);

    [LibraryImport(LibraryName)]
    internal static partial int ve_dcc_forecast_covariance(DccForecastHandle forecast, out double* data, out nuint length);

    [LibraryImport(LibraryName)]
    internal static partial int ve_dcc_forecast_sigma(DccForecastHandle forecast, out double* data, out nuint length);

    [LibraryImport(LibraryName)]
    internal static partial int ve_dcc_forecast_mean(DccForecastHandle forecast, out double* data, out nuint length);

    [LibraryImport(LibraryName)] internal static partial void ve_dcc_forecast_free(nint forecast);

    [LibraryImport(LibraryName)] internal static partial int ve_sim_options_set_correlations(SimOptionsHandle options, int on);

    [LibraryImport(LibraryName)]
    internal static partial int ve_simulate_dcc(ContextHandle context, DccFitHandle fit, SimOptionsHandle options,
                                                DccFutureHandle future, out DccSimHandle simulation);

    [LibraryImport(LibraryName)] internal static partial int ve_dcc_sim_paths(DccSimHandle simulation, out nuint paths);
    [LibraryImport(LibraryName)] internal static partial int ve_dcc_sim_horizon(DccSimHandle simulation, out nuint horizon);
    [LibraryImport(LibraryName)] internal static partial int ve_dcc_sim_series(DccSimHandle simulation, out nuint series);

    [LibraryImport(LibraryName)]
    internal static partial int ve_dcc_sim_returns(DccSimHandle simulation, out double* data, out nuint length);

    [LibraryImport(LibraryName)]
    internal static partial int ve_dcc_sim_sigma(DccSimHandle simulation, out double* data, out nuint length);

    [LibraryImport(LibraryName)]
    internal static partial int ve_dcc_sim_correlation(DccSimHandle simulation, out double* data, out nuint length);

    [LibraryImport(LibraryName)]
    internal static partial int ve_dcc_sim_covariance(DccSimHandle simulation, out double* data, out nuint length);

    [LibraryImport(LibraryName)] internal static partial void ve_dcc_sim_free(nint simulation);
}
