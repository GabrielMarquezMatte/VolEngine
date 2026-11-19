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
}
