namespace VolEngine;

/// <summary>The size, seed, horizon regressors and checkpoints of a simulation.</summary>
/// <param name="Horizon">The number of steps of each path, at least 1.</param>
/// <param name="Paths">The number of paths, at least 1.</param>
/// <param name="Seed">The seed; the same seed, source and library version give the same paths for any thread count.</param>
public sealed record SimulationOptions(int Horizon, int Paths, ulong Seed)
{
    /// <summary>The mean regressors of the horizon, Horizon x k, the same for every path; required exactly when the
    /// source has k &gt; 0 of them.</summary>
    public double[,]? Mxreg { get; init; }

    /// <summary>The variance regressors of the horizon, Horizon x k, the same for every path; required exactly when
    /// the source has k &gt; 0 of them.</summary>
    public double[,]? Vxreg { get; init; }

    /// <summary>The steps at which to keep the sum of each path, strictly increasing from 1, the last one equal to
    /// <see cref="Horizon"/>; null or empty keeps every step. With them the simulation holds only
    /// <see cref="Simulation.Cumulative"/>, Paths x K values whatever the horizon, and
    /// <see cref="Simulation.Returns"/> and <see cref="Simulation.Sigma"/> are empty. An invalid list is a
    /// <see cref="VolEngineException"/> with <see cref="Status.InvalidArgument"/>.</summary>
    public IReadOnlyList<int>? Checkpoints { get; init; }
}
