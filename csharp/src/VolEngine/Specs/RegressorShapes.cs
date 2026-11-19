namespace VolEngine;

// The regressor rules of the Python Context.fit: a regressor array is given exactly when some spec has a count > 0
// of its kind, and its shape is the data shape followed by that count, the same for every such spec.
internal static class RegressorShapes
{
    public static int Width(Array? values, string name, IEnumerable<int> counts, params int[] dataShape)
    {
        var widths = counts.Where(count => count > 0).Distinct().ToArray();
        if (widths.Length == 0 && values is not null)
        {
            throw new ArgumentException($"{name} was given but no spec has {name} > 0", name);
        }
        if (widths.Length == 0)
        {
            return 0;
        }
        if (values is null)
        {
            throw new ArgumentException($"a spec has {name} > 0, so {name} is required", name);
        }
        foreach (var width in widths)
        {
            int[] expected = [.. dataShape, width];
            if (!HasShape(values, expected))
            {
                throw new ArgumentException(
                    $"{name} must have shape ({string.Join(", ", expected)}) for a spec with {name} = {width}, " +
                    $"got ({string.Join(", ", Shape(values))})", name);
            }
        }
        return widths[0];
    }

    private static bool HasShape(Array values, int[] expected)
    {
        return values.Rank == expected.Length && Shape(values).SequenceEqual(expected);
    }

    private static IEnumerable<int> Shape(Array values)
    {
        return Enumerable.Range(0, values.Rank).Select(values.GetLength);
    }
}
