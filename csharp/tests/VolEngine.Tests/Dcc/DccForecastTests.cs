using System.Text.Json;
using VolEngine.Tests.Support;

namespace VolEngine.Tests;

public sealed class DccForecastTests
{
    private static double[][][] Cube(JsonElement values)
    {
        return values.EnumerateArray().Select(Matrix).ToArray();
    }

    private static double[][] Matrix(JsonElement values)
    {
        return values.EnumerateArray().Select(ReferenceData.Doubles).ToArray();
    }

    private static void AssertClose(double[][] expected, double[][] actual, double absolute, double relative)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (var i = 0; i < expected.Length; i++)
        {
            Assert.Equal(expected[i].Length, actual[i].Length);
            for (var j = 0; j < expected[i].Length; j++)
            {
                var tolerance = absolute + (relative * Math.Abs(expected[i][j]));
                Assert.True(Math.Abs(actual[i][j] - expected[i][j]) <= tolerance,
                            $"[{i}][{j}]: {actual[i][j]} is not within {tolerance} of {expected[i][j]}");
            }
        }
    }

    [Fact]
    public void Forecast_matches_rmgarch()
    {
        using var univariate = Univariate.DccForecast();
        var expected = ReferenceData.Result("rmgarch_dcc_forecast.json", "id", "dcc_mvnorm");
        var spec = new DccSpec { Fixed = ReferenceData.Named(expected.GetProperty("params")) };
        using var dcc = univariate.Context.FitDcc(spec, univariate.Fits, standardErrors: false);
        var ahead = expected.GetProperty("ahead").GetInt32();
        var forecast = dcc.Forecast(ahead);
        Assert.Equal((ahead, 3), (forecast.Horizon, forecast.Series));
        var correlation = Cube(expected.GetProperty("R"));
        var covariance = Cube(expected.GetProperty("H"));
        for (var h = 0; h < ahead; h++)
        {
            AssertClose(correlation[h], forecast.Correlation[h], 1e-8, 0);
            AssertClose(covariance[h], forecast.Covariance[h], 1e-8, 0);
        }
        AssertClose(Matrix(expected.GetProperty("sigma")), forecast.Sigma, 0, 1e-8);
    }

    [Fact]
    public void Series_forecasts_are_the_univariate_forecasts()
    {
        using var univariate = Univariate.Dcc3();
        using var dcc = univariate.Context.FitDcc(new DccSpec(), univariate.Fits, standardErrors: false);
        var forecast = dcc.Forecast(5);
        for (var j = 0; j < 3; j++)
        {
            var own = univariate.Fits[j].Forecast(5);
            AssertClose([own.Sigma, own.Mean], [forecast.Sigma.Select(row => row[j]).ToArray(),
                                                 forecast.Mean.Select(row => row[j]).ToArray()], 0, 1e-12);
        }
    }

    [Fact]
    public void Forecast_horizon_must_be_positive()
    {
        using var univariate = Univariate.Dcc3();
        using var dcc = univariate.Context.FitDcc(new DccSpec(), univariate.Fits, standardErrors: false);
        Assert.Throws<ArgumentOutOfRangeException>(() => dcc.Forecast(0));
    }

    [Fact]
    public void Regressor_lists_have_one_entry_per_series()
    {
        using var univariate = Univariate.Dcc3();
        using var dcc = univariate.Context.FitDcc(new DccSpec(), univariate.Fits, standardErrors: false);
        Assert.Throws<ArgumentException>("mxreg", () => dcc.Forecast(4, mxreg: [null, null]));
        Assert.Throws<ArgumentException>("vxreg", () => dcc.Forecast(4, vxreg: [null, null, null, null]));
        Assert.Throws<ArgumentException>("mxreg", () => dcc.Forecast(4, mxreg: [new double[3, 1], null, null]));
    }

    [Fact]
    public void Mixed_regressors_per_series()
    {
        using var univariate = Univariate.MixedRegressors();
        using var dcc = univariate.Context.FitDcc(new DccSpec(), univariate.Fits, standardErrors: false);
        var future = new double[4, 1] { { 0.1 }, { 0.2 }, { -0.1 }, { 0.3 } };
        var forecast = dcc.Forecast(4, mxreg: [future, null, null]);
        var own = univariate.Fits[0].Forecast(4, mxreg: future);
        AssertClose([own.Mean], [forecast.Mean.Select(row => row[0]).ToArray()], 0, 1e-12);
        var error = Assert.Throws<VolEngineException>(() => dcc.Forecast(4));
        Assert.Contains("1", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void A_dcc_fit_outlives_its_univariate_fits()
    {
        var univariate = Univariate.Dcc3();
        using var context = new Context();
        DccFit dcc;
        using (univariate)
        {
            dcc = context.FitDcc(new DccSpec(), univariate.Fits, standardErrors: false);
        }
        using (dcc)
        {
            var simulation = context.Simulate(dcc, new DccSimulationOptions(5, 2, 1));
            Assert.All(simulation.Returns.SelectMany(path => path).SelectMany(step => step),
                       value => Assert.True(double.IsFinite(value)));
            var sigma = dcc.Forecast(5).Sigma;
            Assert.Equal(5, sigma.Length);
            Assert.All(sigma, row => Assert.All(row, value => Assert.True(double.IsFinite(value))));
        }
    }

    [Fact]
    public void Cumulative_mean_is_the_left_sum_of_the_mean()
    {
        using var univariate = Univariate.Dcc3();
        using var dcc = univariate.Context.FitDcc(new DccSpec(), univariate.Fits, standardErrors: false);
        var forecast = dcc.Forecast(6);
        Assert.Equal(6, forecast.CumulativeMean.Length);
        var sum = new double[3];
        for (var h = 0; h < 6; h++)
        {
            for (var j = 0; j < 3; j++)
            {
                sum[j] += forecast.Mean[h][j];
                Assert.Equal(BitConverter.DoubleToInt64Bits(sum[j]),
                             BitConverter.DoubleToInt64Bits(forecast.CumulativeMean[h][j]));
            }
        }
    }

    [Fact]
    public void Cumulative_covariance_is_symmetric()
    {
        using var univariate = Univariate.Dcc3();
        using var dcc = univariate.Context.FitDcc(new DccSpec(), univariate.Fits, standardErrors: false);
        var forecast = dcc.Forecast(6);
        Assert.Equal(6, forecast.CumulativeCovariance.Length);
        for (var h = 0; h < 6; h++)
        {
            var matrix = forecast.CumulativeCovariance[h];
            Assert.Equal(3, matrix.Length);
            for (var i = 0; i < 3; i++)
            {
                for (var j = 0; j < 3; j++)
                {
                    Assert.Equal(BitConverter.DoubleToInt64Bits(matrix[i][j]), BitConverter.DoubleToInt64Bits(matrix[j][i]));
                }
            }
        }
        for (var i = 0; i < 3; i++)
        {
            Assert.Equal(forecast.Covariance[0][i][i], forecast.CumulativeCovariance[0][i][i], 1e-14);
        }
    }

    [Fact]
    public void Cumulative_covariance_is_the_summed_covariance_without_arma()
    {
        using var univariate = Univariate.Dcc3();
        using var dcc = univariate.Context.FitDcc(new DccSpec(), univariate.Fits, standardErrors: false);
        var forecast = dcc.Forecast(6);
        for (var h = 0; h < 6; h++)
        {
            for (var i = 0; i < 3; i++)
            {
                var plain = Enumerable.Range(0, h + 1).Sum(k => forecast.Covariance[k][i][i]);
                Assert.True(Math.Abs(forecast.CumulativeCovariance[h][i][i] / plain - 1) <= 1e-14, $"h = {h + 1}, i = {i}");
            }
        }
    }
}
