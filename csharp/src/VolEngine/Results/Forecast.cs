using VolEngine.Interop;

namespace VolEngine;

/// <summary>The forecast of a fit for h = 1..horizon after its last observation, as rugarch's ugarchforecast with
/// n.roll = 0. It is a copy: it does not depend on the fit.</summary>
public sealed class Forecast
{
    private Forecast(int horizon, double[] sigma, double[] mean)
    {
        Horizon = horizon;
        Sigma = sigma;
        Mean = mean;
    }

    /// <summary>The number of steps.</summary>
    public int Horizon { get; }

    /// <summary>rugarch's sigmaFor: in eGARCH exp(E[log sigma^2] / 2), in apARCH and fGARCH
    /// E[sigma^delta]^(1 / delta), which for h &gt; 1 is not sqrt(E[sigma^2]); in the other models sqrt(E[sigma^2]).</summary>
    public double[] Sigma { get; }

    /// <summary>The conditional mean (rugarch's seriesFor).</summary>
    public double[] Mean { get; }

    internal static unsafe Forecast Read(ForecastHandle handle, int horizon)
    {
        double* data;
        nuint length;
        var sigma = Futures.Copy(NativeMethods.ve_forecast_sigma(handle, out data, out length), data, length);
        var mean = Futures.Copy(NativeMethods.ve_forecast_mean(handle, out data, out length), data, length);
        return new Forecast(horizon, sigma, mean);
    }
}
