namespace VolEngine;

/// <summary>The coefficient table of a fit, one entry per parameter in <see cref="Names"/> order.</summary>
public sealed class CoefTable
{
    internal CoefTable(IReadOnlyList<string> names, double[] estimate, double[] stdError, double[] tValue, double[] pValue)
    {
        Names = names;
        Estimate = estimate;
        StdError = stdError;
        TValue = tValue;
        PValue = pValue;
    }

    /// <summary>The parameter names.</summary>
    public IReadOnlyList<string> Names { get; }

    /// <summary>The estimates (fixed and derived parameters included).</summary>
    public double[] Estimate { get; }

    /// <summary>The standard errors; NaN for fixed and derived parameters.</summary>
    public double[] StdError { get; }

    /// <summary>Estimate divided by standard error; NaN where the standard error is.</summary>
    public double[] TValue { get; }

    /// <summary>2 (1 - Phi(|t|)); NaN where the standard error is.</summary>
    public double[] PValue { get; }
}
