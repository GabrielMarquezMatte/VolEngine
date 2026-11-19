using VolEngine.Tests.Support;

namespace VolEngine.Tests;

public sealed class FitTests
{
    private static readonly string[] SgarchNames = ["mu", "omega", "alpha1", "beta1"];

    [Fact]
    public void Sgarch11_norm_matches_rugarch()
    {
        var result = ReferenceData.Result("rugarch.json", "case", "t5000/sgarch11_norm");
        using var context = new Context();
        using var fit = context.Fit(new Spec(), ReferenceData.Series("sgarch11_norm.csv", "t5000"),
                                    new FitOptions { Tolerance = 1e-7 });
        Assert.Equal(FitStatus.Converged, fit.Status);
        Assert.Equal(SgarchNames, fit.Names);
        var loglik = result.GetProperty("loglik").GetDouble();
        Assert.True(fit.Loglik >= loglik - 1e-6 * Math.Abs(loglik), $"loglik {fit.Loglik} below rugarch {loglik}");
        Assert.InRange(Math.Abs(fit.Loglik / loglik - 1), 0, 1e-4);
        AssertParameters(fit.Estimate(), ReferenceData.Named(result.GetProperty("params")));
        AssertRelative(fit.Coef().StdError, ReferenceData.Named(result.GetProperty("se")).Values, 5e-2);
        AssertRelative(fit.Coef(StandardErrorKind.Robust).StdError,
                       ReferenceData.Named(result.GetProperty("qml_robust_se")).Values, 5e-2);
        AssertRelative(fit.Sigma(), ReferenceData.Doubles(result.GetProperty("sigma")), 1e-3);
        Assert.Equal(result.GetProperty("aic").GetDouble(), fit.Aic(), 1e-6);
        Assert.Equal(result.GetProperty("bic").GetDouble(), fit.Bic(), 1e-6);
        Assert.Equal(result.GetProperty("hqic").GetDouble(), fit.Hqic(), 1e-6);
        Assert.Equal(result.GetProperty("sic").GetDouble(), fit.Sic(), 1e-6);
    }

    [Fact]
    public void Criteria_scale_with_the_number_of_observations()
    {
        using var context = new Context();
        using var fit = context.Fit(new Spec(), ReferenceData.Series("sgarch11_norm.csv", "t1500"));
        Assert.Equal(fit.Aic() * fit.Nobs, fit.Aic(perObs: false), 1e-9);
        Assert.Equal(fit.Bic() * fit.Nobs, fit.Bic(perObs: false), 1e-9);
    }

    [Fact]
    public void The_coef_table_lines_up_with_the_names()
    {
        using var context = new Context();
        using var fit = context.Fit(new Spec(), ReferenceData.Series("sgarch11_norm.csv", "t1500"));
        var table = fit.Coef();
        Assert.Equal(fit.Names, table.Names);
        Assert.Equal(fit.Estimate(), table.Estimate);
        for (var i = 0; i < table.Names.Count; i++)
        {
            Assert.Equal(table.Estimate[i] / table.StdError[i], table.TValue[i], 1e-12);
            Assert.InRange(table.PValue[i], 0, 1);
        }
    }

    [Fact]
    public void Standard_errors_off_keeps_estimates_and_refuses_coef()
    {
        var series = ReferenceData.Series("sgarch11_norm.csv", "t1500");
        using var context = new Context();
        using var with = context.Fit(new Spec(), series);
        using var without = context.Fit(new Spec(), series, new FitOptions { StandardErrors = false });
        Assert.Equal(with.Estimate(), without.Estimate());
        var error = Assert.Throws<VolEngineException>(() => without.Coef());
        Assert.Equal(Status.InvalidArgument, error.Status);
    }

    [Fact]
    public void An_insufficient_series_is_a_status_with_empty_series()
    {
        using var context = new Context();
        using var fit = context.Fit(new Spec(), [0.1, -0.2, 0.3]);
        Assert.Equal(FitStatus.InsufficientData, fit.Status);
        Assert.Empty(fit.Sigma());
        Assert.Empty(fit.Residuals());
        Assert.True(double.IsNaN(fit.Loglik));
        Assert.All(fit.Estimate(), value => Assert.True(double.IsNaN(value)));
    }

    [Fact]
    public void Series_views_have_one_value_per_observation()
    {
        var series = ReferenceData.Series("sgarch11_norm.csv", "t1500");
        using var context = new Context();
        using var fit = context.Fit(new Spec(), series);
        Assert.Equal(series.Length, fit.Nobs);
        Assert.Equal(fit.Nobs, fit.Sigma().Length);
        Assert.Equal(fit.Nobs, fit.StdResiduals().Length);
        var mu = fit.Estimate()[0];
        Assert.All(fit.Fitted(), value => Assert.Equal(mu, value, 1e-12));
        Assert.Equal(series[10] - mu, fit.Residuals()[10], 1e-12);
        Assert.Equal(fit.Residuals()[10] / fit.Sigma()[10], fit.StdResiduals()[10], 1e-12);
    }

    [Fact]
    public void A_nan_in_the_series_is_rejected()
    {
        var series = ReferenceData.Series("sgarch11_norm.csv", "t1500");
        series[700] = double.NaN;
        using var context = new Context();
        var error = Assert.Throws<VolEngineException>(() => context.Fit(new Spec(), series));
        Assert.Equal(Status.InvalidArgument, error.Status);
    }

    [Fact]
    public void Regressors_must_match_the_spec()
    {
        var series = ReferenceData.Series("sgarch11_norm.csv", "t1500");
        using var context = new Context();
        Assert.Throws<ArgumentException>(() => context.Fit(new Spec { Mxreg = 1 }, series));
        Assert.Throws<ArgumentException>(() => context.Fit(new Spec(), series, mxreg: new double[series.Length, 1]));
        Assert.Throws<ArgumentException>(() => context.Fit(new Spec { Mxreg = 1 }, series, mxreg: new double[10, 1]));
        Assert.Throws<ArgumentException>(() => context.Fit(new Spec { Vxreg = 2 }, series, vxreg: new double[series.Length, 1]));
    }

    [Fact]
    public void A_mean_regressor_enters_the_names()
    {
        var series = ReferenceData.Series("sgarch11_norm.csv", "t1500");
        var regressor = new double[series.Length, 1];
        for (var t = 0; t < series.Length; t++)
        {
            regressor[t, 0] = Math.Sin(t * 0.1);
        }
        using var context = new Context();
        using var fit = context.Fit(new Spec { Mxreg = 1 }, series, mxreg: regressor);
        Assert.Equal(["mu", "mxreg1", "omega", "alpha1", "beta1"], fit.Names);
        Assert.Equal(FitStatus.Converged, fit.Status);
    }

    [Fact]
    public async Task Disposing_while_reading_never_reads_freed_memory()
    {
        using var context = new Context();
        var fit = context.Fit(new Spec(), ReferenceData.Series("sgarch11_norm.csv", "t5000"));
        var expected = fit.Sigma();
        var reader = Task.Run(() => ReadUntilDisposed(fit, expected), TestContext.Current.CancellationToken);
        await Task.Delay(5, TestContext.Current.CancellationToken);
        fit.Dispose();
        await reader;
        Assert.Equal(SgarchNames, fit.Names);
        Assert.Throws<ObjectDisposedException>(() => fit.Estimate());
    }

    private static void ReadUntilDisposed(Fit fit, double[] expected)
    {
        for (var i = 0; i < 10_000; i++)
        {
            try
            {
                Assert.Equal(expected, fit.Sigma());
            }
            catch (ObjectDisposedException)
            {
                return;
            }
        }
    }

    // assert_params of the Python oracle suite: absolute 1e-5 for |reference| < 1e-2, relative 1e-3 otherwise.
    private static void AssertParameters(double[] estimate, Dictionary<string, double> expected)
    {
        var reference = expected.Values.ToArray();
        Assert.Equal(reference.Length, estimate.Length);
        for (var i = 0; i < reference.Length; i++)
        {
            var allowed = Math.Abs(reference[i]) < 1e-2 ? 1e-5 : 1e-3 * Math.Abs(reference[i]);
            Assert.True(Math.Abs(estimate[i] - reference[i]) <= allowed, $"parameter {i}: {estimate[i]} vs {reference[i]}");
        }
    }

    private static void AssertRelative(double[] actual, IEnumerable<double> expected, double tolerance)
    {
        var reference = expected.ToArray();
        Assert.Equal(reference.Length, actual.Length);
        for (var i = 0; i < reference.Length; i++)
        {
            Assert.True(Math.Abs(actual[i] / reference[i] - 1) <= tolerance, $"value {i}: {actual[i]} vs {reference[i]}");
        }
    }
}
