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

    private static void AssertCumulativeMatches(DccSimulation full, DccSimulation cumulative, int[] checkpoints)
    {
        Assert.Equal(checkpoints, cumulative.Checkpoints);
        Assert.Empty(cumulative.Returns);
        Assert.Empty(cumulative.Sigma);
        Assert.NotNull(cumulative.Cumulative);
        Assert.Equal(full.Paths, cumulative.Cumulative.Length);
        for (var path = 0; path < full.Paths; path++)
        {
            Assert.Equal(checkpoints.Length, cumulative.Cumulative[path].Length);
            for (var series = 0; series < full.Series; series++)
            {
                var column = LeftSum.At(full.Returns[path].Select(step => step[series]), checkpoints);
                LeftSum.AssertSameBits(column, cumulative.Cumulative[path].Select(row => row[series]).ToArray());
            }
        }
    }

    [Fact]
    public void Cumulative_matches_the_left_sum_of_the_full_returns()
    {
        int[] checkpoints = [1, 5, 20];
        var options = new DccSimulationOptions(20, 6, 7);
        var full = Context.Simulate(_dcc, options);
        Assert.Null(full.Cumulative);
        Assert.Empty(full.Checkpoints);
        AssertCumulativeMatches(full, Context.Simulate(_dcc, options with { Checkpoints = checkpoints }), checkpoints);
    }

    [Fact]
    public void A_copula_dcc_cumulates()
    {
        int[] checkpoints = [2, 10];
        using var copula = Univariate.Copula3();
        using var dcc = copula.Context.FitDcc(new DccSpec { Copula = Copula.Parametric }, copula.Fits, standardErrors: false);
        var options = new DccSimulationOptions(10, 4, 2);
        var full = copula.Context.Simulate(dcc, options);
        AssertCumulativeMatches(full, copula.Context.Simulate(dcc, options with { Checkpoints = checkpoints }), checkpoints);
    }

    [Fact]
    public void Checkpoints_are_validated()
    {
        int[][] invalid = [[0, 20], [5, 5, 20], [5, 10]];
        foreach (var checkpoints in invalid)
        {
            var options = new DccSimulationOptions(20, 2, 1) { Checkpoints = checkpoints };
            var error = Assert.Throws<VolEngineException>(() => Context.Simulate(_dcc, options));
            Assert.Equal(Status.InvalidArgument, error.Status);
        }
        var withCorrelations = new DccSimulationOptions(20, 2, 1) { Checkpoints = [20], Correlations = true };
        var refused = Assert.Throws<VolEngineException>(() => Context.Simulate(_dcc, withCorrelations));
        Assert.Equal(Status.InvalidArgument, refused.Status);
    }

    [Fact]
    public void Cumulative_size_counts_checkpoints_and_series()
    {
        using var source = new CancellationTokenSource();
        source.Cancel();
        // Paths x Horizon x m holds more than Array.MaxLength values, Paths x K x m does not.
        var paths = Array.MaxLength / 6;
        var small = new DccSimulationOptions(4, paths, 1) { Checkpoints = [4] };
        Assert.Throws<OperationCanceledException>(() => Context.Simulate(_dcc, small, source.Token));
        var large = small with { Checkpoints = [1, 2, 3, 4] };
        Assert.Throws<ArgumentOutOfRangeException>(() => Context.Simulate(_dcc, large, source.Token));
    }
}
