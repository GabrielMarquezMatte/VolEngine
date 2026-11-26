namespace VolEngine;

/// <summary>The size, seed, horizon regressors and checkpoints of a DCC simulation.</summary>
/// <param name="Horizon">The number of steps of each path, at least 1.</param>
/// <param name="Paths">The number of paths, at least 1.</param>
/// <param name="Seed">The seed; the same seed, fit and library version give the same paths for any thread count.</param>
public sealed record DccSimulationOptions(int Horizon, int Paths, ulong Seed)
{
    /// <summary>Whether to keep R and H of every path and step (memory: Paths x Horizon x m^2 x 2 doubles); false by
    /// default, and then <see cref="DccSimulation.Correlation"/> and <see cref="DccSimulation.Covariance"/> are
    /// null.</summary>
    public bool Correlations { get; init; }

    /// <summary>For each series, its mean regressors of the horizon (Horizon x k, the same for every path) or null for
    /// a series without; null when no series has them.</summary>
    public IReadOnlyList<double[,]?>? Mxreg { get; init; }

    /// <summary>For each series, its variance regressors of the horizon (Horizon x k) or null; null when no series has
    /// them.</summary>
    public IReadOnlyList<double[,]?>? Vxreg { get; init; }

    /// <summary>The steps at which to keep the sum of each path and series, strictly increasing from 1, the last one
    /// equal to <see cref="Horizon"/>; null or empty keeps every step. With them the simulation holds only
    /// <see cref="DccSimulation.Cumulative"/>, Paths x K x m values whatever the horizon, and
    /// <see cref="DccSimulation.Returns"/> and <see cref="DccSimulation.Sigma"/> are empty. They do not combine with
    /// <see cref="Correlations"/>. An invalid list is a <see cref="VolEngineException"/> with
    /// <see cref="Status.InvalidArgument"/>.</summary>
    public IReadOnlyList<int>? Checkpoints { get; init; }
}
