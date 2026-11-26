using VolEngine.Interop;

namespace VolEngine;

/// <summary>The forecast of a DCC fit for h = 1..<see cref="Horizon"/> after its last observation, as rmgarch's
/// dccforecast (n.roll = 0).</summary>
/// <remarks>
/// <see cref="Sigma"/> and <see cref="Mean"/> are those of each series' own <see cref="Fit.Forecast"/>.
/// Correlation[0] is the exact recursion one step after the last observation and, for h &gt;= 2,
/// Correlation[h - 1] = (1 - k^(h-1)) Rbar + k^(h-1) Correlation[0], with k the total persistence sum(a) + sum(b) (the
/// aDCC g does not enter; the FDCC takes k per element; a static copula keeps its R) and Rbar the correlation of Qbar.
/// Covariance = D R D with D = diag(sigma). Every array is a copy; the forecast does not depend on its fit.
/// </remarks>
public sealed class DccForecast
{
    private unsafe DccForecast(DccForecastHandle forecast, int horizon, int series)
    {
        Horizon = horizon;
        Series = series;
        Correlation = NativeReads.Matrices(NativeReads.View(forecast, NativeMethods.ve_dcc_forecast_correlation), series);
        Covariance = NativeReads.Matrices(NativeReads.View(forecast, NativeMethods.ve_dcc_forecast_covariance), series);
        Sigma = NativeReads.View(forecast, NativeMethods.ve_dcc_forecast_sigma).Chunk(series).ToArray();
        Mean = NativeReads.View(forecast, NativeMethods.ve_dcc_forecast_mean).Chunk(series).ToArray();
        CumulativeMean = NativeReads.View(forecast, NativeMethods.ve_dcc_forecast_cumulative_mean).Chunk(series).ToArray();
        CumulativeCovariance = NativeReads.Matrices(
            NativeReads.View(forecast, NativeMethods.ve_dcc_forecast_cumulative_covariance), series);
    }

    /// <summary>The number of steps.</summary>
    public int Horizon { get; }

    /// <summary>The number of series m.</summary>
    public int Series { get; }

    /// <summary>The correlation of each step, Horizon x m x m.</summary>
    public double[][][] Correlation { get; }

    /// <summary>The covariance of each step, Horizon x m x m.</summary>
    public double[][][] Covariance { get; }

    /// <summary>The conditional standard deviation of each step and series, Horizon x m.</summary>
    public double[][] Sigma { get; }

    /// <summary>The conditional mean of each step and series, Horizon x m.</summary>
    public double[][] Mean { get; }

    /// <summary>The mean of the cumulative return R_h = r_(T+1) + ... + r_(T+h) of each series, Horizon x m: the
    /// left-to-right sum of <see cref="Mean"/>.</summary>
    public double[][] CumulativeMean { get; }

    /// <summary>The covariance of R_h, Horizon x m x m, exactly symmetric: sum_k a_i(h-k) a_j(h-k) H_(T+k)[i, j], with
    /// a(n) = psi_0 + ... + psi_n the MA(inf) weights of each series' ARMA/ARFIMA mean, so it is the plain sum of
    /// <see cref="Covariance"/> only without one.</summary>
    public double[][][] CumulativeCovariance { get; }

    internal static unsafe DccForecast Read(DccForecastHandle forecast)
    {
        Calls.Check(NativeMethods.ve_dcc_forecast_horizon(forecast, out var horizon));
        Calls.Check(NativeMethods.ve_dcc_forecast_series(forecast, out var series));
        return new DccForecast(forecast, checked((int)horizon), checked((int)series));
    }
}
