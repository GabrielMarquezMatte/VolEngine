namespace VolEngine;

/// <summary>A VaR and an ES, both positive as losses, in units of the return.</summary>
/// <param name="Var">The value at risk.</param>
/// <param name="Es">The expected shortfall, the mean loss beyond the VaR.</param>
public sealed record RiskMeasure(double Var, double Es);

/// <summary>A standardized density with the fit's parameters; the ones it lacks are not read.</summary>
/// <param name="Distribution">The density.</param>
/// <param name="Skew">The skew (sstd, sged, nig, ghyp, jsu).</param>
/// <param name="Shape">The shape (all but the normal).</param>
/// <param name="GhLambda">The ghyp lambda.</param>
public sealed record ShockDistribution(Distribution Distribution, double Skew = double.NaN, double Shape = double.NaN,
                                       double GhLambda = double.NaN);

/// <summary>What a return panel holds.</summary>
public enum ReturnKind
{
    /// <summary>Log returns; each is repriced as exp(r) - 1 before the weights apply.</summary>
    Log,
    /// <summary>Simple returns.</summary>
    Simple,
}
