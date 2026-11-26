using System.Runtime.InteropServices;
using VolEngine.Interop;

namespace VolEngine;

/// <summary>One fitted DCC model. It owns its native result; dispose it when done.</summary>
/// <remarks>
/// <para>
/// <see cref="Status"/>, <see cref="Nobs"/>, <see cref="Series"/>, <see cref="Names"/> and <see cref="Loglik"/> are
/// read when the fit is made and stay readable after <see cref="Dispose"/>; every other member then throws
/// <see cref="ObjectDisposedException"/>. Every array returned is a copy. A fit may be read from several threads at
/// once; H is computed once, on the first <see cref="Covariance"/>.
/// </para>
/// <para>
/// The fit keeps the final state of each series and of its correlation recursion, so it forecasts and simulates
/// after the univariate <see cref="Fit"/> objects are disposed. <see cref="Context.StandardErrors(DccFit, IReadOnlyList{Fit}, CancellationToken)"/> does need them,
/// because it refilters the whole sample.
/// </para>
/// </remarks>
public sealed unsafe class DccFit : IDisposable
{
    private readonly DccFitHandle _handle;

    internal DccFit(DccFitHandle handle)
    {
        _handle = handle;
        Calls.Check(NativeMethods.ve_dcc_fit_status(handle, out var status));
        Calls.Check(NativeMethods.ve_dcc_fit_nobs(handle, out var nobs));
        Calls.Check(NativeMethods.ve_dcc_fit_series(handle, out var series));
        Calls.Check(NativeMethods.ve_dcc_fit_param_count(handle, out var count));
        Calls.Check(NativeMethods.ve_dcc_fit_loglik(handle, out var loglik));
        Status = (FitStatus)status;
        Nobs = checked((int)nobs);
        Series = checked((int)series);
        Loglik = loglik;
        var names = new string[checked((int)count)];
        for (var i = 0; i < names.Length; i++)
        {
            Calls.Check(NativeMethods.ve_dcc_fit_param_name(handle, (nuint)i, out var name));
            names[i] = Marshal.PtrToStringUTF8(name) ?? "";
        }
        Names = Array.AsReadOnly(names);
    }

    /// <summary>How the fit ended. A status added by a newer native library is a value outside the named ones.</summary>
    public FitStatus Status { get; }

    /// <summary>The number of observations T.</summary>
    public int Nobs { get; }

    /// <summary>The number of series m.</summary>
    public int Series { get; }

    /// <summary>Every parameter of the correlation model, in the order of every per-parameter array: dcca*, dccb*,
    /// dccg* (aDCC), fdcca*, fdccb* (FDCC), C1.. (static ML copula) and mshape (MvT).</summary>
    public IReadOnlyList<string> Names { get; }

    /// <summary>The log-likelihood of the second stage at the estimate; NaN when the fit has no finite point.</summary>
    public double Loglik { get; }

    internal DccFitHandle Handle => _handle;

    /// <summary>The estimates in <see cref="Names"/> order; also on a fit made without standard errors.</summary>
    /// <returns>A copy of the estimates.</returns>
    public double[] Estimate()
    {
        return Values(NativeMethods.ve_dcc_fit_estimates);
    }

    /// <summary>The coefficient table; a fixed parameter has NaN standard errors, t and p values.</summary>
    /// <param name="kind">The covariance behind the standard errors.</param>
    /// <returns>Names, estimates, standard errors, t values and p values.</returns>
    /// <exception cref="VolEngineException">The fit was made without standard errors: call
    /// <see cref="Context.StandardErrors(DccFit, IReadOnlyList{Fit}, CancellationToken)"/>.</exception>
    public CoefTable Coef(StandardErrorKind kind = StandardErrorKind.Hessian)
    {
        var code = (int)kind;
        return new CoefTable(
            Names,
            Values(NativeMethods.ve_dcc_fit_estimates),
            Values((fit, values) => NativeMethods.ve_dcc_fit_std_errors(fit, code, values)),
            Values((fit, values) => NativeMethods.ve_dcc_fit_t_values(fit, code, values)),
            Values((fit, values) => NativeMethods.ve_dcc_fit_p_values(fit, code, values)));
    }

    /// <summary>The Akaike criterion of the two stages, as rmgarch counts it.</summary>
    /// <remarks>The count is the parameters each univariate criterion counts, the estimated parameters of this fit
    /// and the m (m - 1) / 2 entries of the correlation intercept Qbar, so it is larger than <see cref="Names"/>.
    /// The same holds for <see cref="Bic"/>, <see cref="Hqic"/> and <see cref="Sic"/>.</remarks>
    /// <param name="perObs">Divide by the number of observations (rmgarch); false gives the total.</param>
    /// <returns>The criterion; NaN when the fit has no finite point.</returns>
    public double Aic(bool perObs = true)
    {
        return Criterion(VolEngine.Criterion.Aic, perObs);
    }

    /// <summary>The Bayesian (Schwarz) criterion of the two stages, as rmgarch counts it.</summary>
    /// <param name="perObs">Divide by the number of observations (rmgarch); false gives the total.</param>
    /// <returns>The criterion; NaN when the fit has no finite point.</returns>
    public double Bic(bool perObs = true)
    {
        return Criterion(VolEngine.Criterion.Bic, perObs);
    }

    /// <summary>The Hannan-Quinn criterion of the two stages, as rmgarch counts it.</summary>
    /// <param name="perObs">Divide by the number of observations (rmgarch); false gives the total.</param>
    /// <returns>The criterion; NaN when the fit has no finite point.</returns>
    public double Hqic(bool perObs = true)
    {
        return Criterion(VolEngine.Criterion.Hqic, perObs);
    }

    /// <summary>The Shibata criterion of the two stages, as rmgarch counts it.</summary>
    /// <param name="perObs">Divide by the number of observations (rmgarch); false gives the total.</param>
    /// <returns>The criterion; NaN when the fit has no finite point.</returns>
    public double Sic(bool perObs = true)
    {
        return Criterion(VolEngine.Criterion.Sic, perObs);
    }

    /// <summary>The conditional correlation R_t of every observation.</summary>
    /// <returns>A copy, T x m x m.</returns>
    public double[][][] Correlation()
    {
        return NativeReads.Matrices(NativeReads.View(_handle, NativeMethods.ve_dcc_fit_correlation), Series);
    }

    /// <summary>The conditional covariance H_t = D_t R_t D_t of every observation, D_t the univariate sigmas.</summary>
    /// <returns>A copy, T x m x m.</returns>
    public double[][][] Covariance()
    {
        return NativeReads.Matrices(NativeReads.View(_handle, NativeMethods.ve_dcc_fit_covariance), Series);
    }

    /// <summary>R, H, sigma and mean for h = 1..horizon after the last observation, as rmgarch's dccforecast
    /// (n.roll = 0).</summary>
    /// <remarks>The fit keeps what it needs: the univariate fits may already be disposed. A
    /// <see cref="FitStatus.Failed"/> fit with a finite point forecasts from that point. See <see cref="DccForecast"/>
    /// for the rules of R.</remarks>
    /// <param name="horizon">The number of steps, at least 1.</param>
    /// <param name="mxreg">For each series, its mean regressors of the horizon (horizon x k) or null for a series
    /// without; null when no series has them.</param>
    /// <param name="vxreg">For each series, its variance regressors of the horizon (horizon x k) or null; null when
    /// no series has them.</param>
    /// <returns>The forecast, a copy that outlives the fit.</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="horizon"/> is below 1.</exception>
    /// <exception cref="ArgumentException">A list without one entry per series, or an entry without horizon
    /// rows.</exception>
    /// <exception cref="VolEngineException">A series' regressors do not match its spec (the message names the series,
    /// from 1), or the fit has no finite point.</exception>
    public DccForecast Forecast(int horizon, IReadOnlyList<double[,]?>? mxreg = null, IReadOnlyList<double[,]?>? vxreg = null)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(horizon, 1);
        using var future = DccFutures.Create(Series, horizon, mxreg, vxreg);
        Calls.Check(NativeMethods.ve_dcc_forecast_create(_handle, (nuint)horizon, future, out var forecast));
        using (forecast)
        {
            return DccForecast.Read(forecast);
        }
    }

    /// <summary>Frees the native result. Calling it again does nothing.</summary>
    public void Dispose()
    {
        _handle.Dispose();
    }

    private double Criterion(Criterion which, bool perObs)
    {
        var scale = perObs ? CriterionScale.PerObs : CriterionScale.Total;
        Calls.Check(NativeMethods.ve_dcc_fit_criterion(_handle, (int)which, (int)scale, out var value));
        return value;
    }

    private double[] Values(ValueReader<DccFitHandle> reader)
    {
        return NativeReads.Values(_handle, Names.Count, reader);
    }
}
