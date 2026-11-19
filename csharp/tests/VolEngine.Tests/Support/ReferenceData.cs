using System.Globalization;
using System.Text.Json;

namespace VolEngine.Tests.Support;

// The data and oracle fixtures of tests/reference, the same files the Python and C++ suites compare against.
internal static class ReferenceData
{
    // One column of a CSV in tests/reference/data: the header is skipped and empty cells dropped, as the Python suite
    // reads them (columns of different lengths share one file).
    public static double[] Series(string file, string column)
    {
        var lines = File.ReadAllLines(Path.Combine(Repository.Root, "tests", "reference", "data", file));
        var index = Array.IndexOf(lines[0].Split(','), column);
        if (index < 0)
        {
            throw new InvalidOperationException($"{file} has no column {column}");
        }
        var values = lines.Skip(1)
            .Select(line => line.Split(',')[index])
            .Where(cell => cell.Length > 0)
            .Select(cell => double.Parse(cell, CultureInfo.InvariantCulture))
            .ToArray();
        if (values.Length == 0)
        {
            throw new InvalidOperationException($"{file} column {column} is empty");
        }
        return values;
    }

    // Every column of a CSV in tests/reference/data, in header order.
    public static double[][] Columns(string file)
    {
        var header = File.ReadLines(Path.Combine(Repository.Root, "tests", "reference", "data", file)).First();
        return header.Split(',').Select(column => Series(file, column)).ToArray();
    }

    public static JsonElement Fixture(string file)
    {
        var text = File.ReadAllText(Path.Combine(Repository.Root, "tests", "reference", "fixtures", file));
        using var document = JsonDocument.Parse(text);
        return document.RootElement.Clone();
    }

    // The entry of a fixture's "results" array whose key field names the case.
    public static JsonElement Result(string file, string key, string caseId)
    {
        return Fixture(file).GetProperty("results").EnumerateArray()
            .Single(item => string.Equals(item.GetProperty(key).GetString(), caseId, StringComparison.Ordinal));
    }

    public static double[] Doubles(JsonElement array)
    {
        return array.EnumerateArray().Select(item => item.GetDouble()).ToArray();
    }

    // A JSON object of name to number in its order.
    public static Dictionary<string, double> Named(JsonElement values)
    {
        return values.EnumerateObject().ToDictionary(item => item.Name, item => item.Value.GetDouble(), StringComparer.Ordinal);
    }
}
