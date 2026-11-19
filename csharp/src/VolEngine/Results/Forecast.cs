using VolEngine.Interop;

namespace VolEngine;

/// <summary>The forecast of a fit for h = 1..horizon after its last observation, as rugarch's ugarchforecast with
/// n.roll = 0. It is a copy: it does not depend on the fit.</summary>
/// <param name="Horizon">The number of steps.</param>
/// <param name="Sigma">rugarch's sigmaFor: in eGARCH exp(E[log sigma^2] / 2), in apARCH and fGARCH
/// E[sigma^delta]^(1 / delta), which for h &gt; 1 is not sqrt(E[sigma^2]); in the other models sqrt(E[sigma^2]).</param>
/// <param name="Mean">The conditional mean (rugarch's seriesFor).</param>
public sealed record Forecast(int Horizon, double[] Sigma, double[] Mean)
{
    internal static unsafe Forecast Read(ForecastHandle handle, int horizon)
    {
        double* data;
        nuint length;
        var sigma = Futures.Copy(NativeMethods.ve_forecast_sigma(handle, out data, out length), data, length);
        var mean = Futures.Copy(NativeMethods.ve_forecast_mean(handle, out data, out length), data, length);
        return new Forecast(horizon, sigma, mean);
    }
}
