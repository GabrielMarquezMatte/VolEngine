using VolEngine.Interop;

namespace VolEngine;

/// <summary>The forecast of a fit for h = 1..horizon after its last observation, as rugarch's ugarchforecast with
/// n.roll = 0. It is a copy: it does not depend on the fit.</summary>
public sealed class Forecast
{
    private Forecast(int horizon, double[] sigma, double[] mean, double[] cumulativeMean,
                     double[] cumulativeSigma)
    {
        Horizon = horizon;
        Sigma = sigma;
        Mean = mean;
        CumulativeMean = cumulativeMean;
        CumulativeSigma = cumulativeSigma;
    }

    /// <summary>The number of steps.</summary>
    public int Horizon { get; }

    /// <summary>rugarch's sigmaFor: in eGARCH exp(E[log sigma^2] / 2), in apARCH and fGARCH
    /// E[sigma^delta]^(1 / delta), which for h &gt; 1 is not sqrt(E[sigma^2]); in the other models sqrt(E[sigma^2]).</summary>
    public double[] Sigma { get; }

    /// <summary>The conditional mean (rugarch's seriesFor).</summary>
    public double[] Mean { get; }

    /// <summary>The mean of the cumulative return R_h = r_(T+1) + ... + r_(T+h) for h = 1..Horizon: the left-to-right
    /// sum of <see cref="Mean"/>.</summary>
    public double[] CumulativeMean { get; }

    /// <summary>The standard deviation of R_h, sqrt(sum_k a(h-k)^2 sigma_(T+k)^2), with a(n) = psi_0 + ... + psi_n from
    /// the MA(inf) weights of an ARMA/ARFIMA mean, so it is the plain root of the summed variance only without one; with
    /// GARCH-in-mean the randomness of the future sigma in the mean is ignored.</summary>
    public double[] CumulativeSigma { get; }

    internal static unsafe Forecast Read(ForecastHandle handle)
    {
        Calls.Check(NativeMethods.ve_forecast_horizon(handle, out var horizon));
        double* data;
        nuint length;
        var sigma = Futures.Copy(NativeMethods.ve_forecast_sigma(handle, out data, out length), data, length);
        var mean = Futures.Copy(NativeMethods.ve_forecast_mean(handle, out data, out length), data, length);
        var cumulativeMean = Futures.Copy(NativeMethods.ve_forecast_cumulative_mean(handle, out data, out length), data, length);
        var cumulativeSigma = Futures.Copy(NativeMethods.ve_forecast_cumulative_sigma(handle, out data, out length), data, length);
        return new Forecast(checked((int)horizon), sigma, mean, cumulativeMean, cumulativeSigma);
    }
}
