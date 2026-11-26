using VolEngine.Tests.Support;

namespace VolEngine.Tests;

public sealed class ForecastTests
{
    [Fact]
    public void Forecast_matches_rugarch()
    {
        var result = ReferenceData.Result("rugarch_forecast.json", "case", "t5000/sgarch11_norm");
        var n = result.GetProperty("n").GetInt32();
        var horizon = result.GetProperty("horizon").GetInt32();
        var spec = new Spec { Fixed = ReferenceData.Named(result.GetProperty("fixed")) };
        var series = ReferenceData.Series("sgarch11_norm.csv", "t5000");
        using var context = new Context();
        using var fit = context.Fit(spec, series.AsSpan(0, n), new FitOptions { VarInit = VarInit.Sample });
        var forecast = fit.Forecast(horizon);
        Assert.Equal(horizon, forecast.Horizon);
        var sigma = ReferenceData.Doubles(result.GetProperty("sigma"));
        var mean = ReferenceData.Doubles(result.GetProperty("mean"));
        for (var h = 0; h < horizon; h++)
        {
            Assert.True(Math.Abs(forecast.Sigma[h] / sigma[h] - 1) <= 1e-8, $"sigma {h}: {forecast.Sigma[h]} vs {sigma[h]}");
            Assert.Equal(mean[h], forecast.Mean[h], 1e-8);
        }
    }

    [Fact]
    public void Forecast_horizon_must_be_positive()
    {
        using var context = new Context();
        using var fit = context.Fit(new Spec(), ReferenceData.Series("sgarch11_norm.csv", "t1500"));
        Assert.Throws<ArgumentOutOfRangeException>(() => fit.Forecast(0));
    }

    [Fact]
    public void Forecast_regressors_follow_the_fit()
    {
        var series = ReferenceData.Series("sgarch11_norm.csv", "t1500");
        using var context = new Context();
        using var plain = context.Fit(new Spec(), series);
        Assert.Throws<ArgumentException>(() => plain.Forecast(5, mxreg: new double[5, 1]));
        using var withRegressor = context.Fit(new Spec { Mxreg = 1 }, series, mxreg: new double[series.Length, 1]);
        Assert.Throws<ArgumentException>(() => withRegressor.Forecast(5));
        Assert.Throws<ArgumentException>(() => withRegressor.Forecast(5, mxreg: new double[4, 1]));
        Assert.Equal(5, withRegressor.Forecast(5, mxreg: new double[5, 1]).Sigma.Length);
    }

    [Fact]
    public void A_forecast_outlives_its_fit()
    {
        using var context = new Context();
        var fit = context.Fit(new Spec(), ReferenceData.Series("sgarch11_norm.csv", "t1500"));
        var forecast = fit.Forecast(10);
        fit.Dispose();
        Assert.Equal(10, forecast.Sigma.Length);
        Assert.All(forecast.Sigma, value => Assert.True(value > 0));
    }

    private static Fit Fitted(Context context, Spec spec)
    {
        return context.Fit(spec, ReferenceData.Series("sgarch11_norm.csv", "t1500"));
    }

    [Fact]
    public void Cumulative_mean_is_the_left_sum_of_the_mean()
    {
        using var context = new Context();
        using var fit = Fitted(context, new Spec());
        var forecast = fit.Forecast(12);
        Assert.Equal(12, forecast.CumulativeMean.Length);
        var sum = 0.0;
        for (var h = 0; h < 12; h++)
        {
            sum += forecast.Mean[h];
            Assert.Equal(BitConverter.DoubleToInt64Bits(sum), BitConverter.DoubleToInt64Bits(forecast.CumulativeMean[h]));
        }
    }

    [Fact]
    public void Cumulative_sigma_without_arma_is_the_root_of_the_summed_variance()
    {
        using var context = new Context();
        using var fit = Fitted(context, new Spec());
        var forecast = fit.Forecast(12);
        Assert.Equal(12, forecast.CumulativeSigma.Length);
        var variance = 0.0;
        for (var h = 0; h < 12; h++)
        {
            variance += forecast.Sigma[h] * forecast.Sigma[h];
            Assert.True(Math.Abs(forecast.CumulativeSigma[h] / Math.Sqrt(variance) - 1) <= 1e-14, $"h = {h + 1}");
        }
        Assert.Equal(forecast.Sigma[0], forecast.CumulativeSigma[0]);
    }

    [Fact]
    public void Cumulative_sigma_with_a_positive_ar_root_exceeds_the_plain_root_sum()
    {
        var spec = new Spec
        {
            Arma = (1, 0),
            Fixed = new Dictionary<string, double>(StringComparer.Ordinal)
            {
                ["mu"] = 0.0001, ["ar1"] = 0.4, ["omega"] = 2e-6, ["alpha1"] = 0.08, ["beta1"] = 0.9,
            },
        };
        using var context = new Context();
        using var fit = Fitted(context, spec);
        var forecast = fit.Forecast(8);
        Assert.Equal(forecast.Sigma[0], forecast.CumulativeSigma[0]);
        var variance = forecast.Sigma[0] * forecast.Sigma[0];
        for (var h = 1; h < 8; h++)
        {
            variance += forecast.Sigma[h] * forecast.Sigma[h];
            Assert.True(forecast.CumulativeSigma[h] > Math.Sqrt(variance), $"h = {h + 1}");
        }
    }
}
