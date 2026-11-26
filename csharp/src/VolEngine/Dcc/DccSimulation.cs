using VolEngine.Interop;

namespace VolEngine;

/// <summary>Paths simulated from a DCC fit, after its last observation.</summary>
/// <remarks>
/// Each step takes R from the exact recursion, draws a shock of covariance R (a Gaussian scale mixture for the MvT and
/// the MvLaplace), and turns it into the standardized shock of each series: itself in a DCC, or through the copula and
/// the margin of the series in a Copula-DCC. Path p reads only the stream (seed, p): the same seed gives the same
/// paths for any number of threads, within one version of the library. Every array is a copy; the simulation does not
/// depend on its fit. With <see cref="DccSimulationOptions.Checkpoints"/> the paths are kept only as their sums at those
/// steps: <see cref="Cumulative"/> is set, and <see cref="Returns"/> and <see cref="Sigma"/> are empty.
/// </remarks>
public sealed class DccSimulation
{
    private DccSimulation(int paths, int horizon, int series, double[][][] returns, double[][][] sigma)
    {
        Paths = paths;
        Horizon = horizon;
        Series = series;
        Returns = returns;
        Sigma = sigma;
    }

    /// <summary>The number of paths.</summary>
    public int Paths { get; }

    /// <summary>The number of steps of each path.</summary>
    public int Horizon { get; }

    /// <summary>The number of series m.</summary>
    public int Series { get; }

    /// <summary>The simulated returns, Paths x Horizon x m; empty with checkpoints.</summary>
    public double[][][] Returns { get; }

    /// <summary>The conditional standard deviation of each step and series, Paths x Horizon x m; empty with
    /// checkpoints.</summary>
    public double[][][] Sigma { get; }

    /// <summary>The sum of the returns of each path and series up to each checkpoint, Paths x K x m, as the
    /// left-to-right sum of the first k steps of the full output with the same seed; null without checkpoints.</summary>
    public double[][][]? Cumulative { get; private init; }

    /// <summary>The checkpoints, in increasing order; empty without them.</summary>
    public IReadOnlyList<int> Checkpoints { get; private init; } = [];

    /// <summary>The correlation of each step, Paths x Horizon x m x m; null unless
    /// <see cref="DccSimulationOptions.Correlations"/> was set.</summary>
    public double[][][][]? Correlation { get; private init; }

    /// <summary>The covariance of each step, Paths x Horizon x m x m; null unless
    /// <see cref="DccSimulationOptions.Correlations"/> was set.</summary>
    public double[][][][]? Covariance { get; private init; }

    internal static DccSimulation Run(ContextHandle context, DccFit source, DccSimulationOptions options,
                                      CancellationToken token)
    {
        ArgumentNullException.ThrowIfNull(options);
        CheckSize(options, source.Series);
        Calls.Check(NativeMethods.ve_sim_options_create((nuint)options.Horizon, (nuint)options.Paths, options.Seed,
                                                        out var handle));
        using (handle)
        {
            Calls.Check(NativeMethods.ve_sim_options_set_correlations(handle, options.Correlations ? 1 : 0));
            Interop.Checkpoints.Apply(handle, options.Checkpoints);
            using var future = DccFutures.Create(source.Series, options.Horizon, options.Mxreg, options.Vxreg);
            DccSimHandle? simulation = null;
            Calls.Check(Calls.WithCancel(cancel =>
                NativeMethods.ve_simulate_dcc_with_cancel(context, source.Handle, handle, future, cancel, out simulation),
                token));
            using (simulation)
            {
                return Read(simulation!, options.Correlations);
            }
        }
    }

    // Every copy must fit in one managed array: Paths x Horizon x m values (Paths x K x m with checkpoints), and x m
    // again for R and H.
    private static void CheckSize(DccSimulationOptions options, int series)
    {
        if (options.Horizon < 1 || options.Paths < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(options), options, "Horizon and Paths must be at least 1");
        }
        var values = Interop.Checkpoints.Values(options.Horizon, options.Paths, series, options.Checkpoints) *
                     (options.Correlations ? series : 1);
        if (values > Array.MaxLength)
        {
            throw new ArgumentOutOfRangeException(nameof(options), options,
                                                  $"the simulation would hold {values} values per array, more than {Array.MaxLength}");
        }
    }

    private static unsafe DccSimulation Read(DccSimHandle simulation, bool correlations)
    {
        Calls.Check(NativeMethods.ve_dcc_sim_paths(simulation, out var pathCount));
        Calls.Check(NativeMethods.ve_dcc_sim_horizon(simulation, out var steps));
        Calls.Check(NativeMethods.ve_dcc_sim_series(simulation, out var seriesCount));
        var horizon = checked((int)steps);
        var m = checked((int)seriesCount);
        var paths = checked((int)pathCount);
        var checkpoints = NativeReads.Sizes(simulation, NativeMethods.ve_dcc_sim_checkpoints);
        if (checkpoints.Length > 0)
        {
            return new DccSimulation(paths, horizon, m, [], [])
            {
                Cumulative = NativeReads.Matrices(simulation, NativeMethods.ve_dcc_sim_cumulative, checkpoints.Length, m),
                Checkpoints = checkpoints,
            };
        }
        var returns = NativeReads.Matrices(simulation, NativeMethods.ve_dcc_sim_returns, horizon, m);
        var sigma = NativeReads.Matrices(simulation, NativeMethods.ve_dcc_sim_sigma, horizon, m);
        return new DccSimulation(paths, horizon, m, returns, sigma)
        {
            Correlation = correlations ? NativeReads.Blocks(simulation, NativeMethods.ve_dcc_sim_correlation, horizon, m, m) : null,
            Covariance = correlations ? NativeReads.Blocks(simulation, NativeMethods.ve_dcc_sim_covariance, horizon, m, m) : null,
        };
    }
}
