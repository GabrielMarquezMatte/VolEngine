using VolEngine.Interop;

namespace VolEngine;

/// <summary>
/// VaR and ES at a confidence in (0, 1) (0.99: the 1% tail), both positive as losses, in units of the return. Inputs
/// out of range throw a <see cref="VolEngineException"/> with <see cref="Status.InvalidArgument"/>.
/// </summary>
public static class Risk
{
    /// <summary>
    /// Over the rows of a T x N array (dates or simulated paths in rows): each row is the portfolio's simple return
    /// sum_j weights[j] s_j (s = exp(r) - 1 for <see cref="ReturnKind.Log"/>); the VaR is minus its
    /// <c>numpy.quantile</c> at 1 - confidence and the ES minus the mean at or below it. One weight per column, not
    /// normalized.
    /// </summary>
    public static unsafe RiskMeasure Empirical(double[,] returns, double[] weights, ReturnKind kind, double confidence)
    {
        ArgumentNullException.ThrowIfNull(returns);
        ArgumentNullException.ThrowIfNull(weights);
        Library.EnsureLoaded();
        if (weights.Length != returns.GetLength(1))
        {
            throw new ArgumentException($"weights has {weights.Length} values for {returns.GetLength(1)} series",
                nameof(weights));
        }
        double var;
        double es;
        fixed (double* data = returns)
        fixed (double* weight = weights)
        {
            Calls.Check(NativeMethods.ve_panel_create(data, (nuint)returns.GetLength(0), (nuint)returns.GetLength(1),
                out var panel));
            using (panel)
            {
                Calls.Check(NativeMethods.ve_risk_empirical(panel, weight, (int)kind, confidence, &var, &es));
            }
        }
        return new RiskMeasure(var, es);
    }

    /// <summary>
    /// The return mean + sigma z, z the standardized density: VaR = -(mean + sigma q), ES = -(mean + sigma E[z | z &lt;= q]),
    /// q its quantile at 1 - confidence. The ES is a closed form for the normal, the Student t and the Johnson SU, a quadrature otherwise.
    /// </summary>
    public static unsafe RiskMeasure Parametric(ShockDistribution distribution, double mean, double sigma, double confidence)
    {
        ArgumentNullException.ThrowIfNull(distribution);
        Library.EnsureLoaded();
        double var;
        double es;
        Calls.Check(NativeMethods.ve_risk_parametric((int)distribution.Distribution, distribution.Skew, distribution.Shape,
            distribution.GhLambda, mean, sigma, confidence, &var, &es));
        return new RiskMeasure(var, es);
    }
}
