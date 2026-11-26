using VolEngine.Interop;

namespace VolEngine;

/// <summary>Simulated paths, path by path. It is a copy: it does not depend on its source or context.</summary>
/// <remarks>With <see cref="SimulationOptions.Checkpoints"/> the paths are kept only as their sums at those steps:
/// <see cref="Cumulative"/> is set, and <see cref="Returns"/> and <see cref="Sigma"/> are empty.</remarks>
public sealed class Simulation
{
    private Simulation(int paths, int horizon, double[][] returns, double[][] sigma)
    {
        Paths = paths;
        Horizon = horizon;
        Returns = returns;
        Sigma = sigma;
    }

    internal delegate int Simulator(SimOptionsHandle options, CancelHandle cancel, out SimHandle simulation);

    /// <summary>The number of paths.</summary>
    public int Paths { get; }

    /// <summary>The number of steps of each path.</summary>
    public int Horizon { get; }

    /// <summary>The simulated returns, Paths x Horizon; empty with checkpoints.</summary>
    public double[][] Returns { get; }

    /// <summary>The conditional standard deviation of each step, Paths x Horizon; empty with checkpoints.</summary>
    public double[][] Sigma { get; }

    /// <summary>The sum of the returns of each path up to each checkpoint, Paths x K, as the left-to-right sum of the
    /// first k steps of the full output with the same seed; null without checkpoints.</summary>
    public double[][]? Cumulative { get; private init; }

    /// <summary>The checkpoints, in increasing order; empty without them.</summary>
    public IReadOnlyList<int> Checkpoints { get; private init; } = [];

    internal static unsafe Simulation Run(SimulationOptions options, int mxreg, int vxreg, Simulator simulate,
                                          CancellationToken token)
    {
        ArgumentNullException.ThrowIfNull(options);
        if (options.Horizon < 1 || options.Paths < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(options), options, "Horizon and Paths must be at least 1");
        }
        if (Interop.Checkpoints.Values(options.Horizon, options.Paths, 1, options.Checkpoints) > Array.MaxLength)
        {
            throw new ArgumentOutOfRangeException(nameof(options), options,
                                                  $"Paths x steps (the horizon, or the checkpoints) must be at most {Array.MaxLength} values");
        }
        Futures.Check(options.Mxreg, nameof(options.Mxreg), mxreg, options.Horizon);
        Futures.Check(options.Vxreg, nameof(options.Vxreg), vxreg, options.Horizon);
        Calls.Check(NativeMethods.ve_sim_options_create((nuint)options.Horizon, (nuint)options.Paths, options.Seed,
                                                        out var handle));
        using (handle)
        {
            fixed (double* mean = options.Mxreg)
            fixed (double* variance = options.Vxreg)
            {
                using var meanPanel = Futures.Panel(mean, options.Mxreg);
                using var variancePanel = Futures.Panel(variance, options.Vxreg);
                Calls.Check(NativeMethods.ve_sim_options_set_regressors(handle, meanPanel, variancePanel));
            }
            Interop.Checkpoints.Apply(handle, options.Checkpoints);
            SimHandle? simulation = null;
            Calls.Check(Calls.WithCancel(cancel => simulate(handle, cancel, out simulation), token));
            using (simulation)
            {
                return Read(simulation!);
            }
        }
    }

    private static unsafe Simulation Read(SimHandle simulation)
    {
        Calls.Check(NativeMethods.ve_sim_paths(simulation, out var pathCount));
        Calls.Check(NativeMethods.ve_sim_horizon(simulation, out var steps));
        var paths = checked((int)pathCount);
        var horizon = checked((int)steps);
        var checkpoints = NativeReads.Sizes(simulation, NativeMethods.ve_sim_checkpoints);
        if (checkpoints.Length > 0)
        {
            return new Simulation(paths, horizon, [], [])
            {
                Cumulative = NativeReads.Rows(simulation, NativeMethods.ve_sim_cumulative, checkpoints.Length),
                Checkpoints = checkpoints,
            };
        }
        var returns = NativeReads.Rows(simulation, NativeMethods.ve_sim_returns, horizon);
        var sigma = NativeReads.Rows(simulation, NativeMethods.ve_sim_sigma, horizon);
        return new Simulation(paths, horizon, returns, sigma);
    }
}
