namespace VolEngine;

/// <summary>A test statistic and its p value.</summary>
/// <param name="Statistic">The statistic; NaN when it is undefined (constant autocorrelations, singular regression).</param>
/// <param name="PValue">The p value.</param>
public sealed record DiagnosticResult(double Statistic, double PValue);

/// <summary>The Engle-Ng sign bias tests, with lagged raw residuals as in rugarch.</summary>
/// <param name="Sign">The sign bias t test.</param>
/// <param name="Negative">The negative size bias t test.</param>
/// <param name="Positive">The positive size bias t test.</param>
/// <param name="Joint">The joint F-type test.</param>
public sealed record SignBiasResult(DiagnosticResult Sign, DiagnosticResult Negative, DiagnosticResult Positive,
                                    DiagnosticResult Joint);

/// <summary>Critical values at 10%, 5% and 1%.</summary>
/// <param name="TenPercent">The 10% critical value.</param>
/// <param name="FivePercent">The 5% critical value.</param>
/// <param name="OnePercent">The 1% critical value.</param>
public sealed record CriticalValues(double TenPercent, double FivePercent, double OnePercent);

/// <summary>The Nyblom (Hansen) parameter stability test.</summary>
public sealed class NyblomResult
{
    internal NyblomResult(IReadOnlyDictionary<string, double> individual, double joint, CriticalValues individualCritical,
                          CriticalValues? jointCritical)
    {
        Individual = individual;
        Joint = joint;
        IndividualCritical = individualCritical;
        JointCritical = jointCritical;
    }

    /// <summary>The statistic of each estimated parameter, in parameter order; fixed and derived parameters have none.
    /// An estimated parameter whose scores are all zero has NaN.</summary>
    public IReadOnlyDictionary<string, double> Individual { get; }

    /// <summary>The joint statistic over the estimated parameters.</summary>
    public double Joint { get; }

    /// <summary>The critical values for one parameter.</summary>
    public CriticalValues IndividualCritical { get; }

    /// <summary>The critical values for the number of estimated parameters; null outside 1..20.</summary>
    public CriticalValues? JointCritical { get; }
}
