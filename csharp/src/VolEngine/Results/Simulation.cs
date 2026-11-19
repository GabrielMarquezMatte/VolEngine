using VolEngine.Interop;

namespace VolEngine;

/// <summary>Simulated paths, path by path. It is a copy: it does not depend on its source or context.</summary>
/// <param name="Paths">The number of paths.</param>
/// <param name="Horizon">The number of steps of each path.</param>
/// <param name="Returns">The simulated returns, Paths x Horizon.</param>
/// <param name="Sigma">The conditional standard deviation of each step, Paths x Horizon.</param>
public sealed record Simulation(int Paths, int Horizon, double[][] Returns, double[][] Sigma)
{
    internal delegate int Simulator(SimOptionsHandle options, out SimHandle simulation);

    internal static unsafe Simulation Run(SimulationOptions options, int mxreg, int vxreg, Simulator simulate)
    {
        ArgumentNullException.ThrowIfNull(options);
        if (options.Horizon < 1 || options.Paths < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(options), options, "Horizon and Paths must be at least 1");
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
                return Read(simulation, options.Paths, options.Horizon);
            }
        }
    }

    private static unsafe Simulation Read(SimHandle simulation, int paths, int horizon)
    {
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
