namespace VolEngine;

/// <summary>The size, seed and horizon regressors of a simulation.</summary>
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
}
