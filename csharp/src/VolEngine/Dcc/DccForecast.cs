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
    private DccForecast(int horizon, int series, double[][][] correlation, double[][][] covariance, double[][] sigma,
                        double[][] mean)
    {
        Horizon = horizon;
        Series = series;
        Correlation = correlation;
        Covariance = covariance;
        Sigma = sigma;
        Mean = mean;
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

    internal static unsafe DccForecast Read(DccForecastHandle forecast)
    {
        Calls.Check(NativeMethods.ve_dcc_forecast_horizon(forecast, out var horizon));
        Calls.Check(NativeMethods.ve_dcc_forecast_series(forecast, out var series));
        var m = checked((int)series);
        return new DccForecast(
            checked((int)horizon),
            m,
            NativeReads.Matrices(NativeReads.View(forecast, NativeMethods.ve_dcc_forecast_correlation), m),
            NativeReads.Matrices(NativeReads.View(forecast, NativeMethods.ve_dcc_forecast_covariance), m),
            NativeReads.View(forecast, NativeMethods.ve_dcc_forecast_sigma).Chunk(m).ToArray(),
            NativeReads.View(forecast, NativeMethods.ve_dcc_forecast_mean).Chunk(m).ToArray());
    }
}
