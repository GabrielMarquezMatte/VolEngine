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

    private delegate int TestReader(double* result);

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

    /// <summary>The weighted Ljung-Box test of the standardized residuals or of their squares.</summary>
    /// <remarks>
    /// rugarch's show() uses lags 1, max(2, 3d - 1) and max(5, 5d - 1), with d the ARMA order sum (residuals) or
    /// p + q (squares). Requires 1 &lt;= lag &lt; nobs and lag &gt;= 3 fitdf - 1; fitdf is ignored for squares.
    /// </remarks>
    /// <param name="lag">The number of lags.</param>
    /// <param name="squared">Test the squared standardized residuals.</param>
    /// <param name="fitdf">The degrees of freedom the fit used up.</param>
    /// <returns>The statistic and its p value.</returns>
    public DiagnosticResult LjungBox(int lag, bool squared = false, int fitdf = 0)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(lag);
        ArgumentOutOfRangeException.ThrowIfNegative(fitdf);
        var square = squared ? 1 : 0;
        return Test(output => NativeMethods.ve_fit_ljung_box(_handle, square, (nuint)lag, (nuint)fitdf, output));
    }

    /// <summary>The weighted ARCH-LM test.</summary>
    /// <remarks>rugarch's show() uses fitdf = p + q and lags fitdf + 1, fitdf + 3, fitdf + 5. Requires
    /// 1 &lt;= fitdf &lt; lag &lt; nobs.</remarks>
    /// <param name="lag">The number of lags.</param>
    /// <param name="fitdf">The degrees of freedom the fit used up.</param>
    /// <returns>The statistic and its p value.</returns>
    public DiagnosticResult ArchLm(int lag, int fitdf)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(lag);
        ArgumentOutOfRangeException.ThrowIfNegative(fitdf);
        return Test(output => NativeMethods.ve_fit_arch_lm(_handle, (nuint)lag, (nuint)fitdf, output));
    }

    /// <summary>The Engle-Ng sign bias tests with lagged raw residuals, as in rugarch; needs nobs &gt;= 6.</summary>
    /// <returns>The sign, negative, positive and joint tests.</returns>
    public SignBiasResult SignBias()
    {
        var values = new double[8];
        fixed (double* output = values)
        {
            Calls.Check(NativeMethods.ve_fit_sign_bias(_handle, output));
        }
        return new SignBiasResult(new DiagnosticResult(values[0], values[1]), new DiagnosticResult(values[2], values[3]),
                                  new DiagnosticResult(values[4], values[5]), new DiagnosticResult(values[6], values[7]));
    }

    /// <summary>The adjusted Pearson goodness of fit test of the distribution.</summary>
    /// <remarks>rugarch's show() uses 20, 30, 40 and 50 groups. Requires 2 &lt;= groups &lt;= nobs.</remarks>
    /// <param name="groups">The number of groups.</param>
    /// <returns>The statistic and its p value.</returns>
    public DiagnosticResult Gof(int groups = 20)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(groups);
        return Test(output => NativeMethods.ve_fit_gof(_handle, (nuint)groups, output));
    }

    /// <summary>The Nyblom (Hansen) parameter stability test, with critical values at 10%, 5% and 1%.</summary>
    /// <returns>The individual and joint statistics.</returns>
    /// <exception cref="VolEngineException">The fit was made with <see cref="FitOptions.StandardErrors"/> = false.</exception>
    public NyblomResult Nyblom()
    {
        var statistics = new double[Names.Count];
        var estimated = new int[Names.Count];
        double joint;
        fixed (double* individual = statistics)
        fixed (int* flags = estimated)
        {
            Calls.Check(NativeMethods.ve_fit_nyblom(_handle, individual, &joint, flags));
        }
        var values = new Dictionary<string, double>(StringComparer.Ordinal);
        for (var i = 0; i < Names.Count; i++)
        {
            if (estimated[i] == 1)
            {
                values.Add(Names[i], statistics[i]);
            }
        }
        var individualCritical = CriticalValuesFor(1) ?? throw new VolEngineException("no critical values for one parameter");
        return new NyblomResult(values.AsReadOnly(), joint, individualCritical, CriticalValuesFor(values.Count));
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

    private static DiagnosticResult Test(TestReader reader)
    {
        var values = new double[2];
        fixed (double* output = values)
        {
            Calls.Check(reader(output));
        }
        return new DiagnosticResult(values[0], values[1]);
    }

    private static CriticalValues? CriticalValuesFor(int parameters)
    {
        if (parameters < 1 || parameters > 20)
        {
            return null;
        }
        var values = new double[3];
        fixed (double* output = values)
        {
            Calls.Check(NativeMethods.ve_nyblom_critical((nuint)parameters, output));
        }
        return new CriticalValues(values[0], values[1], values[2]);
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
