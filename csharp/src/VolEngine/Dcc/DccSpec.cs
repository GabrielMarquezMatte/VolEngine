using System.Collections.ObjectModel;
using System.Globalization;
using System.Text;
using VolEngine.Interop;

namespace VolEngine;

/// <summary>
/// The second stage of a DCC model: the correlation dynamics fitted on the standardized residuals of the univariate
/// fits, in the paper's convention.
/// </summary>
/// <remarks>
/// <para>
/// <see cref="Order"/> (p, q) has p shock terms (dcca1..dccap, and dccg1..dccgp in the aDCC) and q persistence terms
/// (dccb1..dccbq), each 1..6. The FDCC needs one group label per series in <see cref="Groups"/>, labels from 1 with
/// every label 1..G used, and order (1, 1); its parameters are fdcca1..fdccaG and fdccb1..fdccbG.
/// </para>
/// <para>
/// A <see cref="Fixed"/> parameter keeps its value: it is listed in the coefficient table with NaN standard errors and
/// is not counted by the information criteria. <see cref="Start"/> replaces default starting values.
/// </para>
/// <para>
/// <see cref="Copula"/> = <see cref="VolEngine.Copula.Parametric"/> (each fit's cdf) or
/// <see cref="VolEngine.Copula.Empirical"/> (ranks) makes it a Copula-DCC on the normal or Student scores of the
/// margins (distribution MvNorm or MvT). <see cref="DccModel.Static"/> is a constant-correlation copula and needs a
/// copula; its <see cref="StaticMethod"/> is Kendall (the default) or ML, whose C1.. are the free entries of A in
/// R = cov2cor(A A'), as in rmgarch. The FDCC and the MvLaplace take no copula. The native fit rejects the
/// combinations that do not exist.
/// </para>
/// <para>Equality compares every property, <see cref="Groups"/>, <see cref="Fixed"/> and <see cref="Start"/> by
/// content.</para>
/// </remarks>
public sealed record DccSpec
{
    private readonly ReadOnlyCollection<int>? _groups;
    private readonly IReadOnlyDictionary<string, double> _fixed = ReadOnlyDictionary<string, double>.Empty;
    private readonly IReadOnlyDictionary<string, double> _start = ReadOnlyDictionary<string, double>.Empty;

    /// <summary>The correlation model; <see cref="DccModel.Dcc"/> by default.</summary>
    public DccModel Model { get; init; } = DccModel.Dcc;

    /// <summary>The (p, q) order; (1, 1) by default.</summary>
    public (int P, int Q) Order { get; init; } = (1, 1);

    /// <summary>The multivariate distribution; <see cref="DccDistribution.MvNorm"/> by default.</summary>
    public DccDistribution Distribution { get; init; } = DccDistribution.MvNorm;

    /// <summary>FDCC only: the group of each series, from 1 (a copy of what is assigned); null for the other
    /// models.</summary>
    public IReadOnlyList<int>? Groups
    {
        get => _groups;
        init => _groups = value is null ? null : Array.AsReadOnly(value.ToArray());
    }

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

    /// <summary>The copula transformation of the margins; <see cref="VolEngine.Copula.None"/> (the plain DCC) by
    /// default.</summary>
    public Copula Copula { get; init; } = Copula.None;

    /// <summary>The estimator of a static copula; <see cref="VolEngine.StaticMethod.Default"/> (Kendall) by
    /// default.</summary>
    public StaticMethod StaticMethod { get; init; } = StaticMethod.Default;

    /// <summary>Whether both specs have the same properties, with <see cref="Groups"/>, <see cref="Fixed"/> and
    /// <see cref="Start"/> by content.</summary>
    /// <param name="other">The other spec.</param>
    /// <returns>True when they are equal.</returns>
    public bool Equals(DccSpec? other)
    {
        if (other is null)
        {
            return false;
        }
        var sameGroups = _groups is null ? other._groups is null : other._groups is not null && _groups.SequenceEqual(other._groups);
        return (Model, Order, Distribution, Copula, StaticMethod)
               == (other.Model, other.Order, other.Distribution, other.Copula, other.StaticMethod)
               && sameGroups && FrozenValues.SameValues(_fixed, other._fixed)
               && FrozenValues.SameValues(_start, other._start);
    }

    /// <summary>A hash consistent with <see cref="Equals(DccSpec)"/>.</summary>
    /// <returns>The hash.</returns>
    public override int GetHashCode()
    {
        var hash = new HashCode();
        hash.Add((Model, Order, Distribution, Copula, StaticMethod));
        hash.Add(_groups is null);
        foreach (var group in _groups ?? [])
        {
            hash.Add(group);
        }
        FrozenValues.AddValues(ref hash, _fixed);
        FrozenValues.AddValues(ref hash, _start);
        return hash.ToHashCode();
    }

    internal DccSpecHandle CreateHandle()
    {
        FrozenValues.RejectNul(_fixed, nameof(Fixed));
        FrozenValues.RejectNul(_start, nameof(Start));
        Library.EnsureLoaded();
        Calls.Check(NativeMethods.ve_dcc_spec_create((int)Model, out var handle));
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

    private bool PrintMembers(StringBuilder builder)
    {
        var groups = _groups is null ? "null" : "[" + string.Join(", ", _groups) + "]";
        builder.Append(CultureInfo.InvariantCulture, $"Model = {Model}, Order = {Order}, Distribution = {Distribution}, ");
        builder.Append(CultureInfo.InvariantCulture,
                       $"Groups = {groups}, Fixed = {FrozenValues.Print(_fixed)}, Start = {FrozenValues.Print(_start)}, ");
        builder.Append(CultureInfo.InvariantCulture, $"Copula = {Copula}, StaticMethod = {StaticMethod}");
        return true;
    }

    // The order of the Python binding: order, distribution, groups, fixed, start, then copula and static method only
    // when they are not the defaults.
    private unsafe void Apply(DccSpecHandle handle)
    {
        Calls.Check(NativeMethods.ve_dcc_spec_set_order(handle, Order.P, Order.Q));
        Calls.Check(NativeMethods.ve_dcc_spec_set_dist(handle, (int)Distribution));
        if (_groups is not null)
        {
            var groups = _groups.ToArray();
            fixed (int* labels = groups)
            {
                Calls.Check(NativeMethods.ve_dcc_spec_set_groups(handle, labels, (nuint)groups.Length));
            }
        }
        foreach (var (name, value) in _fixed)
        {
            Calls.Check(NativeMethods.ve_dcc_spec_set_fixed(handle, name, value));
        }
        foreach (var (name, value) in _start)
        {
            Calls.Check(NativeMethods.ve_dcc_spec_set_start(handle, name, value));
        }
        if (Copula != Copula.None)
        {
            Calls.Check(NativeMethods.ve_dcc_spec_set_copula(handle, (int)Copula));
        }
        if (StaticMethod != StaticMethod.Default)
        {
            Calls.Check(NativeMethods.ve_dcc_spec_set_static_method(handle, (int)StaticMethod));
        }
    }
}
