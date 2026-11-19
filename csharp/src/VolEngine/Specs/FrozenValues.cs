using System.Collections.ObjectModel;
using System.Globalization;

namespace VolEngine;

// The parameter-value maps of the specs (Fixed, Start): copied and frozen on assignment, compared and hashed by
// content, printed by name.
internal static class FrozenValues
{
    // The UTF-8 marshalling stops a name at its first NUL, so the native library would see another name.
    internal static void RejectNul(IReadOnlyDictionary<string, double> values, string property)
    {
        foreach (var name in values.Keys)
        {
            if (name.Contains('\0', StringComparison.Ordinal))
            {
                var shown = name.Replace("\0", "\\0", StringComparison.Ordinal);
                throw new ArgumentException($"parameter name {shown} contains a NUL character", property);
            }
        }
    }

    // "{ mu = 0.5, shape = 5 }", by name, in the invariant culture.
    internal static string Print(IReadOnlyDictionary<string, double> values)
    {
        var entries = values.OrderBy(entry => entry.Key, StringComparer.Ordinal)
            .Select(entry => string.Create(CultureInfo.InvariantCulture, $"{entry.Key} = {entry.Value}"));
        return values.Count == 0 ? "{ }" : "{ " + string.Join(", ", entries) + " }";
    }

    internal static ReadOnlyDictionary<string, double> Frozen(IReadOnlyDictionary<string, double> values)
    {
        ArgumentNullException.ThrowIfNull(values);
        return new ReadOnlyDictionary<string, double>(new Dictionary<string, double>(values, StringComparer.Ordinal));
    }

    internal static bool SameValues(IReadOnlyDictionary<string, double> left, IReadOnlyDictionary<string, double> right)
    {
        if (left.Count != right.Count)
        {
            return false;
        }
        foreach (var (name, value) in left)
        {
            if (!right.TryGetValue(name, out var other) || !value.Equals(other))
            {
                return false;
            }
        }
        return true;
    }

    internal static void AddValues(ref HashCode hash, IReadOnlyDictionary<string, double> values)
    {
        foreach (var name in values.Keys.Order(StringComparer.Ordinal))
        {
            hash.Add(name, StringComparer.Ordinal);
            hash.Add(values[name]);
        }
    }
}
