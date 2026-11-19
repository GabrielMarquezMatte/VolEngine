namespace VolEngine;

/// <summary>The coefficient table of a fit, one entry per parameter in <see cref="Names"/> order.</summary>
/// <param name="Names">The parameter names.</param>
/// <param name="Estimate">The estimates (fixed and derived parameters included).</param>
/// <param name="StdError">The standard errors; NaN for fixed and derived parameters.</param>
/// <param name="TValue">Estimate divided by standard error; NaN where the standard error is.</param>
/// <param name="PValue">2 (1 - Phi(|t|)); NaN where the standard error is.</param>
public sealed record CoefTable(IReadOnlyList<string> Names, double[] Estimate, double[] StdError, double[] TValue,
                               double[] PValue);
