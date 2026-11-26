namespace VolEngine;

/// <summary>A Ledoit-Wolf estimate.</summary>
/// <param name="Covariance">The shrunk covariance, N rows of N, exactly symmetric.</param>
/// <param name="Shrinkage">The shrinkage intensity, in [0, 1].</param>
public sealed record LedoitWolfResult(double[][] Covariance, double Shrinkage);
