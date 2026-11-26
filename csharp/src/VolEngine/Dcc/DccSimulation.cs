using VolEngine.Interop;

namespace VolEngine;

/// <summary>Paths simulated from a DCC fit, after its last observation.</summary>
/// <remarks>
/// Each step takes R from the exact recursion, draws a shock of covariance R (a Gaussian scale mixture for the MvT and
/// the MvLaplace), and turns it into the standardized shock of each series: itself in a DCC, or through the copula and
/// the margin of the series in a Copula-DCC. Path p reads only the stream (seed, p): the same seed gives the same
/// paths for any number of threads, within one version of the library. Every array is a copy; the simulation does not
/// depend on its fit.
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

    /// <summary>The simulated returns, Paths x Horizon x m.</summary>
    public double[][][] Returns { get; }

    /// <summary>The conditional standard deviation of each step and series, Paths x Horizon x m.</summary>
    public double[][][] Sigma { get; }

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

    // Every copy must fit in one managed array: Paths x Horizon x m values, and x m again for R and H.
    private static void CheckSize(DccSimulationOptions options, int series)
    {
        if (options.Horizon < 1 || options.Paths < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(options), options, "Horizon and Paths must be at least 1");
        }
        var values = (long)options.Horizon * options.Paths * series * (options.Correlations ? series : 1);
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
        var returns = NativeReads.View(simulation, NativeMethods.ve_dcc_sim_returns).Chunk(m).Chunk(horizon).ToArray();
        var sigma = NativeReads.View(simulation, NativeMethods.ve_dcc_sim_sigma).Chunk(m).Chunk(horizon).ToArray();
        return new DccSimulation(checked((int)pathCount), horizon, m, returns, sigma)
        {
            Correlation = correlations ? PerPath(NativeReads.View(simulation, NativeMethods.ve_dcc_sim_correlation), horizon, m) : null,
            Covariance = correlations ? PerPath(NativeReads.View(simulation, NativeMethods.ve_dcc_sim_covariance), horizon, m) : null,
        };
    }

    private static double[][][][] PerPath(double[] values, int horizon, int m)
    {
        return NativeReads.Matrices(values, m).Chunk(horizon).ToArray();
    }
}
