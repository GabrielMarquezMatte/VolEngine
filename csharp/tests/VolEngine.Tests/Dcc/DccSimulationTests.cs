using VolEngine.Tests.Support;

namespace VolEngine.Tests;

public sealed class DccSimulationTests : IDisposable
{
    private readonly Univariate _univariate = Univariate.Dcc3();
    private readonly DccFit _dcc;

    public DccSimulationTests()
    {
        _dcc = _univariate.Context.FitDcc(new DccSpec(), _univariate.Fits, standardErrors: false);
    }

    private Context Context => _univariate.Context;

    public void Dispose()
    {
        _dcc.Dispose();
        _univariate.Dispose();
    }

    [Fact]
    public void Same_seed_gives_same_paths()
    {
        var first = Context.Simulate(_dcc, new DccSimulationOptions(20, 3, 7));
        var second = Context.Simulate(_dcc, new DccSimulationOptions(20, 3, 7));
        Assert.Equal((3, 20, 3), (first.Paths, first.Horizon, first.Series));
        Assert.Equal(3, first.Returns.Length);
        Assert.All(first.Returns, path => Assert.Equal(20, path.Length));
        Assert.All(first.Returns.SelectMany(path => path), step => Assert.Equal(3, step.Length));
        Assert.Equal(first.Returns, second.Returns);
        Assert.Equal(first.Sigma, second.Sigma);
        Assert.NotEqual(first.Returns[0], first.Returns[1]);
    }

    [Fact]
    public void Thread_count_does_not_change_paths()
    {
        using var one = new Context(1);
        using var four = new Context(4);
        var serial = one.Simulate(_dcc, new DccSimulationOptions(30, 16, 99));
        var parallel = four.Simulate(_dcc, new DccSimulationOptions(30, 16, 99));
        Assert.Equal(serial.Returns, parallel.Returns);
        Assert.Equal(serial.Sigma, parallel.Sigma);
    }

    [Fact]
    public void Correlations_are_opt_in()
    {
        var without = Context.Simulate(_dcc, new DccSimulationOptions(5, 2, 1));
        Assert.Null(without.Correlation);
        Assert.Null(without.Covariance);
        var with = Context.Simulate(_dcc, new DccSimulationOptions(5, 2, 1) { Correlations = true });
        Assert.NotNull(with.Correlation);
        Assert.NotNull(with.Covariance);
        Assert.Equal(2, with.Correlation.Length);
        Assert.All(with.Correlation, path => Assert.Equal(5, path.Length));
        foreach (var matrix in with.Correlation.SelectMany(path => path))
        {
            Assert.Equal(3, matrix.Length);
            for (var i = 0; i < 3; i++)
            {
                Assert.Equal(1, matrix[i][i], 1e-12);
            }
        }
        Assert.Equal(without.Returns, with.Returns);
    }

    [Fact]
    public void The_first_step_matches_the_forecast()
    {
        var simulation = Context.Simulate(_dcc, new DccSimulationOptions(3, 4, 5));
        var forecast = _dcc.Forecast(1);
        foreach (var path in simulation.Sigma)
        {
            for (var j = 0; j < 3; j++)
            {
                DccFitTests.AssertNear(forecast.Sigma[0][j], path[0][j], 1e-12);
            }
        }
    }

    [Fact]
    public void A_copula_dcc_simulates()
    {
        using var copula = Univariate.Copula3();
        using var dcc = copula.Context.FitDcc(new DccSpec { Copula = Copula.Parametric }, copula.Fits, standardErrors: false);
        var simulation = copula.Context.Simulate(dcc, new DccSimulationOptions(10, 3, 2));
        Assert.All(simulation.Returns.SelectMany(path => path).SelectMany(step => step),
                   value => Assert.True(double.IsFinite(value)));
    }

    [Fact]
    public void Mixed_regressors_per_series_simulate()
    {
        using var mixed = Univariate.MixedRegressors();
        using var dcc = mixed.Context.FitDcc(new DccSpec(), mixed.Fits, standardErrors: false);
        var options = new DccSimulationOptions(5, 2, 3) { Mxreg = [new double[5, 1], null, null] };
        var simulation = mixed.Context.Simulate(dcc, options);
        Assert.Equal(2, simulation.Returns.Length);
        Assert.Throws<VolEngineException>(() => mixed.Context.Simulate(dcc, new DccSimulationOptions(5, 2, 3)));
        Assert.Throws<ArgumentException>(() => mixed.Context.Simulate(dcc, options with { Mxreg = [null, null] }));
    }

    [Fact]
    public void Simulation_sizes_must_be_positive()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => Context.Simulate(_dcc, new DccSimulationOptions(0, 1, 1)));
        Assert.Throws<ArgumentOutOfRangeException>(() => Context.Simulate(_dcc, new DccSimulationOptions(1, 0, 1)));
    }

    [Fact]
    public void Oversized_dcc_simulations_are_rejected_before_allocating()
    {
        var options = new DccSimulationOptions(int.MaxValue / 4, 2, 1);
        Assert.Throws<ArgumentOutOfRangeException>(() => Context.Simulate(_dcc, options));
        var correlations = new DccSimulationOptions(int.MaxValue / 16, 2, 1) { Correlations = true };
        Assert.Throws<ArgumentOutOfRangeException>(() => Context.Simulate(_dcc, correlations));
    }
}
