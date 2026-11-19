using VolEngine.Tests.Support;

namespace VolEngine.Tests;

public sealed class DccFitTests : IDisposable
{
    private readonly Univariate _univariate = Univariate.Dcc3();

    private Context Context => _univariate.Context;

    private Fit[] Fits => _univariate.Fits;

    public void Dispose()
    {
        _univariate.Dispose();
    }

    internal static void AssertNear(double expected, double actual, double relative)
    {
        Assert.True(Math.Abs(actual - expected) <= relative * Math.Abs(expected),
                    $"{actual} is not within {relative} of {expected}");
    }

    [Fact]
    public void Dcc_estimates_near_rmgarch()
    {
        using var dcc = Context.FitDcc(new DccSpec(), Fits, standardErrors: false);
        Assert.Equal(FitStatus.Converged, dcc.Status);
        Assert.Equal(["dcca1", "dccb1"], dcc.Names);
        Assert.Equal((1500, 3), (dcc.Nobs, dcc.Series));
        var expected = ReferenceData.Result("rmgarch_dcc.json", "id", "dcc3_DCC_mvnorm").GetProperty("params");
        var estimate = dcc.Estimate();
        for (var i = 0; i < estimate.Length; i++)
        {
            AssertNear(expected.GetProperty(dcc.Names[i]).GetDouble(), estimate[i], 0.05);
        }
        var correlation = dcc.Correlation();
        Assert.Equal(1500, correlation.Length);
        foreach (var matrix in correlation)
        {
            Assert.Equal(3, matrix.Length);
            for (var i = 0; i < 3; i++)
            {
                Assert.Equal(1, matrix[i][i], 1e-12);
                for (var j = 0; j < 3; j++)
                {
                    Assert.Equal(matrix[i][j], matrix[j][i]);
                }
            }
        }
    }

    [Fact]
    public void Fdcc_names_follow_the_groups()
    {
        var spec = new DccSpec { Model = DccModel.Fdcc, Groups = [1, 1, 2] };
        using var dcc = Context.FitDcc(spec, Fits, standardErrors: false);
        Assert.Equal(["fdcca1", "fdcca2", "fdccb1", "fdccb2"], dcc.Names);
    }

    [Fact]
    public void Fixed_parameters_stay_in_the_table()
    {
        var spec = new DccSpec { Fixed = new Dictionary<string, double>(StringComparer.Ordinal) { ["dcca1"] = 0.05 } };
        using var dcc = Context.FitDcc(spec, Fits);
        var table = dcc.Coef();
        Assert.Equal("dcca1", table.Names[0]);
        Assert.Equal(0.05, table.Estimate[0]);
        Assert.True(double.IsNaN(table.StdError[0]));
        Assert.True(double.IsFinite(table.StdError[1]));
    }

    [Fact]
    public void Standard_errors_can_come_later()
    {
        using var dcc = Context.FitDcc(new DccSpec(), Fits, standardErrors: false);
        Assert.Throws<VolEngineException>(() => dcc.Coef());
        using var withErrors = Context.StandardErrors(dcc, Fits);
        var table = withErrors.Coef(StandardErrorKind.Robust);
        Assert.All(table.StdError, value => Assert.True(double.IsFinite(value) && value > 0));
        Assert.Equal(dcc.Estimate(), withErrors.Estimate());
        Assert.Throws<VolEngineException>(() => dcc.Coef());
    }

    [Fact]
    public void Standard_errors_need_the_same_fits()
    {
        using var dcc = Context.FitDcc(new DccSpec(), Fits, standardErrors: false);
        Assert.Throws<VolEngineException>(() => Context.StandardErrors(dcc, [Fits[1], Fits[0], Fits[2]]));
    }

    [Fact]
    public void A_batch_fits_every_spec()
    {
        using var batch = Context.FitDcc([new DccSpec(), new DccSpec { Model = DccModel.Adcc }], Fits, standardErrors: false);
        Assert.Equal(2, batch.Count);
        Assert.Contains("dccg1", batch[1].Names, StringComparer.Ordinal);
        Assert.DoesNotContain("dccg1", batch[0].Names, StringComparer.Ordinal);
        Assert.Throws<ArgumentOutOfRangeException>(() => batch[2]);
    }

    [Fact]
    public void Bad_univariate_inputs_are_rejected()
    {
        Assert.Throws<ArgumentException>("specs", () => Context.FitDcc(Array.Empty<DccSpec>(), Fits));
        Assert.Throws<ArgumentNullException>("univariate", () => Context.FitDcc(new DccSpec(), [Fits[0], null!]));
        Assert.Throws<VolEngineException>(() => Context.FitDcc(new DccSpec(), [Fits[0]]));
        using var shorter = Context.Fit(new Spec(), ReferenceData.Series("sgarch11_norm.csv", "t1500"));
        using var longer = Context.Fit(new Spec(), ReferenceData.Series("sgarch11_norm.csv", "t3000"));
        var error = Assert.Throws<VolEngineException>(() => Context.FitDcc(new DccSpec(), [shorter, longer]));
        Assert.NotEmpty(error.Message);
    }

    [Fact]
    public void Fits_without_a_finite_point_are_rejected()
    {
        var series = ReferenceData.Columns("dcc3.csv").Select(column => column[..2]).ToArray();
        var tiny = series.Select(column => Context.Fit(new Spec(), column)).ToArray();
        try
        {
            Assert.All(tiny, fit => Assert.Equal(FitStatus.InsufficientData, fit.Status));
            var error = Assert.Throws<VolEngineException>(() => Context.FitDcc(new DccSpec(), tiny));
            Assert.NotEmpty(error.Message);
        }
        finally
        {
            foreach (var fit in tiny)
            {
                fit.Dispose();
            }
        }
    }

    [Fact]
    public void A_disposed_univariate_fit_is_an_object_disposed_error()
    {
        Fits[1].Dispose();
        Assert.Throws<ObjectDisposedException>(() => Context.FitDcc(new DccSpec(), Fits));
    }

    [Fact]
    public async Task Concurrent_covariance_reads_agree()
    {
        using var dcc = Context.FitDcc(new DccSpec(), Fits, standardErrors: false);
        var reads = Enumerable.Range(0, 16).Select(_ => Task.Run(dcc.Covariance, TestContext.Current.CancellationToken));
        var all = await Task.WhenAll(reads);
        Assert.Equal(1500, all[0].Length);
        Assert.All(all, covariance => Assert.Equal(all[0], covariance));
    }

    [Fact]
    public void Dcc_spec_equality_compares_groups_by_content()
    {
        int[] groups = [1, 1, 2];
        var spec = new DccSpec { Model = DccModel.Fdcc, Groups = groups };
        groups[2] = 1;
        var same = new DccSpec { Model = DccModel.Fdcc, Groups = new List<int> { 1, 1, 2 } };
        Assert.Equal(same, spec);
        Assert.Equal(same.GetHashCode(), spec.GetHashCode());
        Assert.NotEqual(new DccSpec { Model = DccModel.Fdcc, Groups = [1, 2, 2] }, spec);
        Assert.NotEqual(new DccSpec { Model = DccModel.Fdcc }, spec);
        Assert.Contains("Groups = [1, 1, 2]", spec.ToString(), StringComparison.Ordinal);
    }
}
