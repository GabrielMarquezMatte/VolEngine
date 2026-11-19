namespace VolEngine;

/// <summary>The size, seed and horizon regressors of a DCC simulation.</summary>
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
}
