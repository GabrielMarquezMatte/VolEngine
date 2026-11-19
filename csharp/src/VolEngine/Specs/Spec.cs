using System.Collections.ObjectModel;
using System.Globalization;
using System.Text;
using VolEngine.Interop;

namespace VolEngine;

/// <summary>
/// A univariate model specification: conditional mean, conditional variance and innovation distribution.
/// </summary>
/// <remarks>
/// <para>
/// <see cref="P"/> is the ARCH order (number of alpha terms, lagged squared residuals) and <see cref="Q"/> the GARCH
/// order (number of beta terms, lagged variances), as in arch's p and q and in rugarch's
/// garchOrder = c(alpha order, beta order) (rugarch calls those orders q and p, the other way round).
/// <see cref="Arma"/> is the ARMA(m, n) order of the conditional mean. Orders are 1..6 for p and q and 0..6 for the
/// ARMA.
/// </para>
/// <para>
/// Parameter names (those of <c>Fit.Names</c>) are, in order: mu, ar1.., ma1.., arfima or archm, mxreg1..,
/// omega, alpha1.., beta1.., gamma1.., eta11.., eta21.., delta or lambda, vxreg1.., skew, shape, ghlambda; a model has
/// only its own blocks, and mu is absent without a mean. Every parameter is estimated, fixed or derived. A fixed
/// parameter (<see cref="Fixed"/>) keeps its value, is not optimized, has NaN standard errors and does not count in the
/// criteria. A derived parameter (the iGARCH beta_q, the EWMA omega and alpha1, and omega under variance targeting) is
/// a function of the others: it has a name and a value, NaN standard errors, and is not counted either.
/// <see cref="Start"/> replaces default starting values; a start for a derived or fixed parameter is ignored, so the
/// names and estimates of a previous fit are a valid warm start. A spec with every parameter fixed does not optimize:
/// the fit evaluates the model at the fixed point and is converged when the log-likelihood is finite.
/// </para>
/// <para>
/// <see cref="VarianceTargeting"/> makes omega derived, computed inside the likelihood from the residuals of the
/// current mean parameters over the whole sample (m2 = mean of e_t^2, P the persistence): omega = m2 (1 - P) for
/// sGARCH and GJR-GARCH; log(m2) (1 - sum beta) for eGARCH; mean(|e_t|^delta) (1 - P) for apARCH and fGARCH. The other
/// standard errors come from the likelihood in which omega is a function of them (as in rugarch).
/// </para>
/// <para>
/// <see cref="Mxreg"/> and <see cref="Vxreg"/> are the numbers of mean and variance regressor columns (0..8), whose
/// data go to the fit; <see cref="Archm"/> is ARCH-in-mean (0 none, 1 adds xi sigma_t, 2 adds xi sigma_t^2).
/// Mean: m_t = mu + sum delta_k x_t,k + xi sigma_t^p and d_t = r_t - m_t, on which the ARMA acts. Variance: omega
/// becomes omega + sum zeta_k v_t,k in the scale of the model. Persistence &lt; 1 is always enforced, and a
/// sigma^2 &lt;= 0 gives a log-likelihood of -inf. <see cref="Arfima"/> is rugarch's ARFIMA mean: the parameter
/// arfima is d in (1e-8, 0.5), applied as (1 - L)^d truncated at the start of the series, at a cost of O(T^2); a spec
/// cannot have both ARFIMA and ARCH-in-mean.
/// </para>
/// <para>
/// Rejected by the native library (<see cref="VolEngineException"/>, the message names the problem) when the spec is
/// used or validated: a name the spec does not have (a misspelled name, or shape under a normal distribution); fixing
/// a derived parameter; variance targeting with iGARCH, csGARCH or EWMA; an EWMA order other than (1, 1); a csGARCH
/// with eta21 fixed and a free beta. Equality compares every property, <see cref="Fixed"/> and <see cref="Start"/> by
/// content.
/// </para>
/// </remarks>
public sealed record Spec
{
    private readonly IReadOnlyDictionary<string, double> _fixed = ReadOnlyDictionary<string, double>.Empty;
    private readonly IReadOnlyDictionary<string, double> _start = ReadOnlyDictionary<string, double>.Empty;

    /// <summary>The variance model; <see cref="Model.SGarch"/> by default.</summary>
    public Model Model { get; init; } = Model.SGarch;

    /// <summary>The ARCH order, 1..6; 1 by default.</summary>
    public int P { get; init; } = 1;

    /// <summary>The GARCH order, 1..6; 1 by default.</summary>
    public int Q { get; init; } = 1;

    /// <summary>The ARMA order of the mean, each 0..6; (0, 0) by default.</summary>
    public (int Ar, int Ma) Arma { get; init; }

    /// <summary>The innovation distribution; <see cref="Distribution.Norm"/> by default.</summary>
    public Distribution Distribution { get; init; } = Distribution.Norm;

    /// <summary>Whether the mean has the constant mu; true by default.</summary>
    public bool IncludeMean { get; init; } = true;

    /// <summary>Parameters held at the given values (a copy of what is assigned).</summary>
    public IReadOnlyDictionary<string, double> Fixed
    {
        get => _fixed;
        init => _fixed = FrozenValues.Frozen(value);
    }

    /// <summary>Starting values that replace the defaults (a copy of what is assigned).</summary>
    public IReadOnlyDictionary<string, double> Start
    {
        get => _start;
        init => _start = FrozenValues.Frozen(value);
    }

    /// <summary>Whether omega is derived by variance targeting; false by default.</summary>
    public bool VarianceTargeting { get; init; }

    /// <summary>The number of mean regressor columns, 0..8.</summary>
    public int Mxreg { get; init; }

    /// <summary>The number of variance regressor columns, 0..8.</summary>
    public int Vxreg { get; init; }

    /// <summary>ARCH-in-mean: 0 none, 1 for xi sigma_t, 2 for xi sigma_t^2.</summary>
    public int Archm { get; init; }

    /// <summary>Whether the mean is ARFIMA (rugarch's arfima = TRUE).</summary>
    public bool Arfima { get; init; }

    /// <summary>Checks the spec in the native library without fitting.</summary>
    /// <exception cref="VolEngineException">The spec cannot be fitted; the message says why.</exception>
    /// <exception cref="ArgumentException">A name in <see cref="Fixed"/> or <see cref="Start"/> contains '\0'.</exception>
    public void Validate()
    {
        using var handle = CreateHandle();
        Calls.Check(NativeMethods.ve_spec_validate(handle));
    }

    /// <summary>Whether both specs have the same properties, with <see cref="Fixed"/> and <see cref="Start"/> by content.</summary>
    /// <param name="other">The other spec.</param>
    /// <returns>True when they are equal.</returns>
    public bool Equals(Spec? other)
    {
        if (other is null)
        {
            return false;
        }
        return (Model, P, Q, Arma, Distribution, IncludeMean, VarianceTargeting, Mxreg, Vxreg, Archm, Arfima)
               == (other.Model, other.P, other.Q, other.Arma, other.Distribution, other.IncludeMean,
                   other.VarianceTargeting, other.Mxreg, other.Vxreg, other.Archm, other.Arfima)
               && FrozenValues.SameValues(_fixed, other._fixed) && FrozenValues.SameValues(_start, other._start);
    }

    /// <summary>A hash consistent with <see cref="Equals(Spec)"/>.</summary>
    /// <returns>The hash.</returns>
    public override int GetHashCode()
    {
        var hash = new HashCode();
        hash.Add((Model, P, Q, Arma, Distribution, IncludeMean, VarianceTargeting, Mxreg, Vxreg, Archm, Arfima));
        FrozenValues.AddValues(ref hash, _fixed);
        FrozenValues.AddValues(ref hash, _start);
        return hash.ToHashCode();
    }

    private bool PrintMembers(StringBuilder builder)
    {
        builder.Append(CultureInfo.InvariantCulture,
                       $"Model = {Model}, P = {P}, Q = {Q}, Arma = {Arma}, Distribution = {Distribution}, ");
        builder.Append(CultureInfo.InvariantCulture, $"IncludeMean = {IncludeMean}, ");
        builder.Append(CultureInfo.InvariantCulture,
                       $"Fixed = {FrozenValues.Print(_fixed)}, Start = {FrozenValues.Print(_start)}, ");
        builder.Append(CultureInfo.InvariantCulture, $"VarianceTargeting = {VarianceTargeting}, Mxreg = {Mxreg}, ");
        builder.Append(CultureInfo.InvariantCulture, $"Vxreg = {Vxreg}, Archm = {Archm}, Arfima = {Arfima}");
        return true;
    }

    internal SpecHandle CreateHandle()
    {
        FrozenValues.RejectNul(_fixed, nameof(Fixed));
        FrozenValues.RejectNul(_start, nameof(Start));
        Library.EnsureLoaded();
        Calls.Check(NativeMethods.ve_spec_create((int)Model, out var handle));
        try
        {
            Apply(handle);
        }
        catch
        {
            handle.Dispose();
            throw;
        }
        return handle;
    }

    private void Apply(SpecHandle handle)
    {
        Calls.Check(NativeMethods.ve_spec_set_order(handle, P, Q));
        Calls.Check(NativeMethods.ve_spec_set_arma(handle, Arma.Ar, Arma.Ma));
        Calls.Check(NativeMethods.ve_spec_set_dist(handle, (int)Distribution));
        Calls.Check(NativeMethods.ve_spec_set_include_mean(handle, IncludeMean ? 1 : 0));
        Calls.Check(NativeMethods.ve_spec_set_mxreg(handle, Mxreg));
        Calls.Check(NativeMethods.ve_spec_set_vxreg(handle, Vxreg));
        Calls.Check(NativeMethods.ve_spec_set_archm(handle, Archm));
        Calls.Check(NativeMethods.ve_spec_set_arfima(handle, Arfima ? 1 : 0));
        Calls.Check(NativeMethods.ve_spec_set_variance_targeting(handle, VarianceTargeting ? 1 : 0));
        foreach (var (name, value) in _fixed)
        {
            Calls.Check(NativeMethods.ve_spec_set_fixed(handle, name, value));
        }
        foreach (var (name, value) in _start)
        {
            Calls.Check(NativeMethods.ve_spec_set_start(handle, name, value));
        }
    }
}
