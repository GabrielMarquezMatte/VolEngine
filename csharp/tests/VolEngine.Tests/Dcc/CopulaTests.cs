using VolEngine.Tests.Support;

namespace VolEngine.Tests;

public sealed class CopulaTests : IDisposable
{
    private readonly Univariate _univariate = Univariate.Copula3();

    public void Dispose()
    {
        _univariate.Dispose();
    }

    [Fact]
    public void Copula_dcc_estimates_near_rmgarch()
    {
        var spec = new DccSpec { Copula = Copula.Parametric };
        using var dcc = _univariate.Context.FitDcc(spec, _univariate.Fits, standardErrors: false);
        Assert.Equal(FitStatus.Converged, dcc.Status);
        Assert.Equal(["dcca1", "dccb1"], dcc.Names);
        var expected = ReferenceData.Result("rmgarch_copula.json", "id", "dcc_mvnorm_param").GetProperty("params");
        var estimate = dcc.Estimate();
        for (var i = 0; i < estimate.Length; i++)
        {
            DccFitTests.AssertNear(expected.GetProperty(dcc.Names[i]).GetDouble(), estimate[i], 0.05);
        }
    }

    [Fact]
    public void A_static_copula_has_a_constant_correlation()
    {
        var spec = new DccSpec { Model = DccModel.Static, Copula = Copula.Parametric, StaticMethod = StaticMethod.Kendall };
        using var dcc = _univariate.Context.FitDcc(spec, _univariate.Fits, standardErrors: false);
        var correlation = dcc.Correlation();
        Assert.Equal(correlation[0], correlation[^1]);
        Assert.NotEqual(0, correlation[0][0][1]);
    }

    [Fact]
    public void A_static_copula_with_no_parameter_has_an_empty_table()
    {
        var spec = new DccSpec { Model = DccModel.Static, Copula = Copula.Parametric };
        using var dcc = _univariate.Context.FitDcc(spec, _univariate.Fits);
        Assert.Empty(dcc.Names);
        Assert.Empty(dcc.Estimate());
        var table = dcc.Coef(StandardErrorKind.Robust);
        Assert.Empty(table.Estimate);
        Assert.Empty(table.StdError);
        Assert.Empty(table.TValue);
        Assert.Empty(table.PValue);
    }

    [Fact]
    public void A_static_model_needs_a_copula()
    {
        var spec = new DccSpec { Model = DccModel.Static };
        Assert.Throws<VolEngineException>(() => _univariate.Context.FitDcc(spec, _univariate.Fits));
    }
}
