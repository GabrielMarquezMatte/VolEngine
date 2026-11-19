using VolEngine.Interop;

namespace VolEngine;

/// <summary>Optimizer settings for a fit.</summary>
/// <remarks>
/// <see cref="StandardErrors"/> = false skips the inference at the estimate (Hessian and robust standard errors,
/// scores, Nyblom): <c>Fit.Coef</c> and <c>Fit.Nyblom</c> then throw <see cref="VolEngineException"/>,
/// while the estimates, the criteria, the series, forecasts, simulations and DCC fits stay the same. Useful for a grid
/// chosen by a criterion; refit the chosen model with standard errors.
/// </remarks>
public sealed record FitOptions
{
    /// <summary>The iteration limit of the optimizer, at least 1; 500 by default.</summary>
    public int MaxIter { get; init; } = 500;

    /// <summary>The gradient tolerance, finite and positive; 1e-6 by default.</summary>
    public double Tolerance { get; init; } = 1e-6;

    /// <summary>How the pre-sample variance is initialized; <see cref="VarInit.Sample"/> by default.</summary>
    public VarInit VarInit { get; init; } = VarInit.Sample;

    /// <summary>Whether the fit computes standard errors; true by default.</summary>
    public bool StandardErrors { get; init; } = true;

    internal FitOptionsHandle CreateHandle()
    {
        Library.EnsureLoaded();
        Calls.Check(NativeMethods.ve_fit_options_create(out var handle));
        try
        {
            Calls.Check(NativeMethods.ve_fit_options_set_max_iter(handle, MaxIter));
            Calls.Check(NativeMethods.ve_fit_options_set_tolerance(handle, Tolerance));
            Calls.Check(NativeMethods.ve_fit_options_set_var_init(handle, (int)VarInit));
            Calls.Check(NativeMethods.ve_fit_options_set_standard_errors(handle, StandardErrors ? 1 : 0));
        }
        catch
        {
            handle.Dispose();
            throw;
        }
        return handle;
    }
}
