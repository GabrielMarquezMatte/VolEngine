using VolEngine.Interop;

namespace VolEngine;

/// <summary>Simulated paths, path by path. It is a copy: it does not depend on its source or context.</summary>
public sealed class Simulation
{
    private Simulation(int paths, int horizon, double[][] returns, double[][] sigma)
    {
        Paths = paths;
        Horizon = horizon;
        Returns = returns;
        Sigma = sigma;
    }

    internal delegate int Simulator(SimOptionsHandle options, out SimHandle simulation);

    /// <summary>The number of paths.</summary>
    public int Paths { get; }

    /// <summary>The number of steps of each path.</summary>
    public int Horizon { get; }

    /// <summary>The simulated returns, Paths x Horizon.</summary>
    public double[][] Returns { get; }

    /// <summary>The conditional standard deviation of each step, Paths x Horizon.</summary>
    public double[][] Sigma { get; }

    internal static unsafe Simulation Run(SimulationOptions options, int mxreg, int vxreg, Simulator simulate)
    {
        ArgumentNullException.ThrowIfNull(options);
        if (options.Horizon < 1 || options.Paths < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(options), options, "Horizon and Paths must be at least 1");
        }
        if ((long)options.Horizon * options.Paths > Array.MaxLength)
        {
            throw new ArgumentOutOfRangeException(nameof(options), options,
                                                  $"Horizon x Paths must be at most {Array.MaxLength} values");
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
            Calls.Check(simulate(handle, out var simulation));
            using (simulation)
            {
                return Read(simulation);
            }
        }
    }

    private static unsafe Simulation Read(SimHandle simulation)
    {
        Calls.Check(NativeMethods.ve_sim_paths(simulation, out var pathCount));
        Calls.Check(NativeMethods.ve_sim_horizon(simulation, out var steps));
        var paths = checked((int)pathCount);
        var horizon = checked((int)steps);
        double* data;
        nuint length;
        var returns = Futures.Copy(NativeMethods.ve_sim_returns(simulation, out data, out length), data, length);
        var sigma = Futures.Copy(NativeMethods.ve_sim_sigma(simulation, out data, out length), data, length);
        return new Simulation(paths, horizon, Rows(returns, paths, horizon), Rows(sigma, paths, horizon));
    }

    private static double[][] Rows(double[] values, int paths, int horizon)
    {
        return Enumerable.Range(0, paths).Select(path => values.AsSpan(path * horizon, horizon).ToArray()).ToArray();
    }
}
