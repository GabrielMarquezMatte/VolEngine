using System.Runtime.InteropServices;

namespace VolEngine.Interop;

internal static unsafe partial class NativeMethods
{
    [LibraryImport(LibraryName)]
    internal static partial int ve_forecast_create(FitHandle fit, nuint horizon, PanelHandle mxreg, PanelHandle vxreg,
                                                   out ForecastHandle forecast);

    [LibraryImport(LibraryName)] internal static partial int ve_forecast_horizon(ForecastHandle forecast, out nuint horizon);

    [LibraryImport(LibraryName)]
    internal static partial int ve_forecast_sigma(ForecastHandle forecast, out double* data, out nuint length);

    [LibraryImport(LibraryName)]
    internal static partial int ve_forecast_mean(ForecastHandle forecast, out double* data, out nuint length);

    [LibraryImport(LibraryName)] internal static partial void ve_forecast_free(nint forecast);

    [LibraryImport(LibraryName)]
    internal static partial int ve_sim_options_create(nuint horizon, nuint paths, ulong seed, out SimOptionsHandle options);

    [LibraryImport(LibraryName)]
    internal static partial int ve_sim_options_set_regressors(SimOptionsHandle options, PanelHandle mxreg, PanelHandle vxreg);

    [LibraryImport(LibraryName)]
    internal static partial int ve_sim_options_set_checkpoints(SimOptionsHandle options, nuint* ks, nuint count);

    [LibraryImport(LibraryName)] internal static partial void ve_sim_options_free(nint options);

    [LibraryImport(LibraryName)]
    internal static partial int ve_simulate_fit(ContextHandle context, FitHandle fit, SimOptionsHandle options,
                                                out SimHandle simulation);

    [LibraryImport(LibraryName)]
    internal static partial int ve_simulate_spec(ContextHandle context, SpecHandle spec, SimOptionsHandle options,
                                                 out SimHandle simulation);

    [LibraryImport(LibraryName)]
    internal static partial int ve_simulate_fit_with_cancel(ContextHandle context, FitHandle fit, SimOptionsHandle options,
                                                            CancelHandle cancel, out SimHandle simulation);

    [LibraryImport(LibraryName)]
    internal static partial int ve_simulate_spec_with_cancel(ContextHandle context, SpecHandle spec, SimOptionsHandle options,
                                                             CancelHandle cancel, out SimHandle simulation);

    [LibraryImport(LibraryName)] internal static partial int ve_sim_paths(SimHandle simulation, out nuint paths);
    [LibraryImport(LibraryName)] internal static partial int ve_sim_horizon(SimHandle simulation, out nuint horizon);

    [LibraryImport(LibraryName)]
    internal static partial int ve_sim_returns(SimHandle simulation, out double* data, out nuint length);

    [LibraryImport(LibraryName)]
    internal static partial int ve_sim_sigma(SimHandle simulation, out double* data, out nuint length);

    [LibraryImport(LibraryName)]
    internal static partial int ve_sim_cumulative(SimHandle simulation, out double* data, out nuint length);

    [LibraryImport(LibraryName)]
    internal static partial int ve_sim_checkpoints(SimHandle simulation, out nuint* data, out nuint length);

    [LibraryImport(LibraryName)] internal static partial void ve_sim_free(nint simulation);
}
