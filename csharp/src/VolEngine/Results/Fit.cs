using System.Runtime.InteropServices;
using VolEngine.Interop;

namespace VolEngine;

/// <summary>One fitted model. It owns its native result; dispose it when done.</summary>
/// <remarks>
/// <see cref="Status"/>, <see cref="Nobs"/>, <see cref="Names"/> and <see cref="Loglik"/> are read when the fit is
/// made and stay readable after <see cref="Dispose"/>; every other member then throws
/// <see cref="ObjectDisposedException"/>. A <see cref="FitStatus.Failed"/> fit may carry the values of the best finite
/// point the optimizer reached, or NaN (with empty series) when it reached none. A
/// <see cref="FitStatus.InsufficientData"/> fit is always NaN with empty series. Check <see cref="Status"/> before using
/// the values. Every array returned is a copy. A fit may be read from several threads at once.
/// </remarks>
public sealed unsafe class Fit : IDisposable
{
    private readonly FitHandle _handle;

    internal Fit(FitHandle handle)
    {
        _handle = handle;
        Calls.Check(NativeMethods.ve_fit_status(handle, out var status));
        Calls.Check(NativeMethods.ve_fit_nobs(handle, out var nobs));
        Calls.Check(NativeMethods.ve_fit_param_count(handle, out var count));
        Calls.Check(NativeMethods.ve_fit_loglik(handle, out var loglik));
        Status = (FitStatus)status;
        Nobs = checked((int)nobs);
        Loglik = loglik;
        var names = new string[checked((int)count)];
        for (var i = 0; i < names.Length; i++)
        {
            Calls.Check(NativeMethods.ve_fit_param_name(handle, (nuint)i, out var name));
            names[i] = Marshal.PtrToStringUTF8(name) ?? "";
        }
        Names = Array.AsReadOnly(names);
    }

    private delegate int ViewReader(FitHandle fit, out double* data, out nuint length);

    private delegate int ValueReader(FitHandle fit, double* values);

    /// <summary>How the fit ended. A status added by a newer native library is a value outside the named ones.</summary>
    public FitStatus Status { get; }

    /// <summary>The number of observations fitted.</summary>
    public int Nobs { get; }

    /// <summary>The parameter names, in the order of every per-parameter array.</summary>
    public IReadOnlyList<string> Names { get; }

    /// <summary>The log-likelihood at the estimate; NaN when the fit has no finite point.</summary>
    public double Loglik { get; }

    internal FitHandle Handle => _handle;

    /// <summary>The estimates in <see cref="Names"/> order; also on a fit made without standard errors.</summary>
    /// <returns>A copy of the estimates.</returns>
    public double[] Estimate()
    {
        return Values(NativeMethods.ve_fit_estimates);
    }

    /// <summary>The coefficient table.</summary>
    /// <param name="kind">The covariance behind the standard errors.</param>
    /// <returns>Names, estimates, standard errors, t values and p values.</returns>
    /// <exception cref="VolEngineException">The fit was made with <see cref="FitOptions.StandardErrors"/> = false.</exception>
    public CoefTable Coef(StandardErrorKind kind = StandardErrorKind.Hessian)
    {
        var code = (int)kind;
        return new CoefTable(
            Names,
            Values(NativeMethods.ve_fit_estimates),
            Values((fit, values) => NativeMethods.ve_fit_std_errors(fit, code, values)),
            Values((fit, values) => NativeMethods.ve_fit_t_values(fit, code, values)),
            Values((fit, values) => NativeMethods.ve_fit_p_values(fit, code, values)));
    }

    /// <summary>The Akaike criterion, counting the estimated parameters only.</summary>
    /// <param name="perObs">Divide by the number of observations (rugarch); false gives the total (arch).</param>
    /// <returns>The criterion; NaN when the fit has no finite point.</returns>
    public double Aic(bool perObs = true)
    {
        return Criterion(VolEngine.Criterion.Aic, perObs);
    }

    /// <summary>The Bayesian (Schwarz) criterion, counting the estimated parameters only.</summary>
    /// <param name="perObs">Divide by the number of observations (rugarch); false gives the total (arch).</param>
    /// <returns>The criterion; NaN when the fit has no finite point.</returns>
    public double Bic(bool perObs = true)
    {
        return Criterion(VolEngine.Criterion.Bic, perObs);
    }

    /// <summary>The Hannan-Quinn criterion, counting the estimated parameters only.</summary>
    /// <param name="perObs">Divide by the number of observations (rugarch); false gives the total (arch).</param>
    /// <returns>The criterion; NaN when the fit has no finite point.</returns>
    public double Hqic(bool perObs = true)
    {
        return Criterion(VolEngine.Criterion.Hqic, perObs);
    }

    /// <summary>The Shibata criterion, counting the estimated parameters only.</summary>
    /// <param name="perObs">Divide by the number of observations (rugarch); false gives the total (arch).</param>
    /// <returns>The criterion; NaN when the fit has no finite point.</returns>
    public double Sic(bool perObs = true)
    {
        return Criterion(VolEngine.Criterion.Sic, perObs);
    }

    /// <summary>The conditional standard deviation, one value per observation; empty without a finite point.</summary>
    /// <returns>A copy of the series.</returns>
    public double[] Sigma()
    {
        return View(NativeMethods.ve_fit_sigma);
    }

    /// <summary>The residuals e_t = r_t minus the conditional mean; empty without a finite point.</summary>
    /// <returns>A copy of the series.</returns>
    public double[] Residuals()
    {
        return View(NativeMethods.ve_fit_residuals);
    }

    /// <summary>The standardized residuals e_t / sigma_t; empty without a finite point.</summary>
    /// <returns>A copy of the series.</returns>
    public double[] StdResiduals()
    {
        return View(NativeMethods.ve_fit_std_residuals);
    }

    /// <summary>The conditional mean; empty without a finite point.</summary>
    /// <returns>A copy of the series.</returns>
    public double[] Fitted()
    {
        return View(NativeMethods.ve_fit_fitted);
    }

    /// <summary>Frees the native result. Calling it again does nothing.</summary>
    public void Dispose()
    {
        _handle.Dispose();
    }

    private double Criterion(Criterion which, bool perObs)
    {
        var scale = perObs ? CriterionScale.PerObs : CriterionScale.Total;
        Calls.Check(NativeMethods.ve_fit_criterion(_handle, (int)which, (int)scale, out var value));
        return value;
    }

    private double[] Values(ValueReader reader)
    {
        var values = new double[Names.Count];
        fixed (double* output = values)
        {
            Calls.Check(reader(_handle, output));
        }
        return values;
    }

    // The view points into the native result, so the handle is held until the copy is done: a Dispose on another
    // thread then waits for the release instead of freeing the memory under the copy.
    private double[] View(ViewReader reader)
    {
        var added = false;
        _handle.DangerousAddRef(ref added);
        try
        {
            Calls.Check(reader(_handle, out var data, out var length));
            return new ReadOnlySpan<double>(data, checked((int)length)).ToArray();
        }
        finally
        {
            if (added)
            {
                _handle.DangerousRelease();
            }
        }
    }
}
