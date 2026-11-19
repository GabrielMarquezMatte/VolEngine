using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace VolEngine;

/// <summary>
/// The series of a grid fit, with optional row ranges and regressors. It holds managed arrays only; nothing native.
/// </summary>
/// <remarks>
/// A panel is T rows by N series. Series j may use only rows starts[j] .. starts[j] + lengths[j] - 1, so series of
/// different lengths share one panel; rows outside a series' range are never read, so they may hold anything (NaN
/// included). Regressors are T x N x k: series j uses the k columns of its rows. <see cref="FromColumns"/> keeps the
/// arrays given, without a copy: do not change them while a fit on the panel runs.
/// </remarks>
public sealed class Panel
{
    private Panel(double[,] data, nuint[]? starts, nuint[]? lengths, double[,,]? mxreg, double[,,]? vxreg)
    {
        Data = data;
        Starts = starts;
        Lengths = lengths;
        Mxreg = mxreg;
        Vxreg = vxreg;
    }

    /// <summary>The number of rows T.</summary>
    public int Rows => Data.GetLength(0);

    /// <summary>The number of series N.</summary>
    public int SeriesCount => Data.GetLength(1);

    internal double[,] Data { get; }

    internal nuint[]? Starts { get; }

    internal nuint[]? Lengths { get; }

    internal double[,,]? Mxreg { get; }

    internal double[,,]? Vxreg { get; }

    /// <summary>A panel over a T x N array, one series per column.</summary>
    /// <param name="data">The returns, T x N.</param>
    /// <param name="starts">The first row of each series; given together with <paramref name="lengths"/>.</param>
    /// <param name="lengths">The number of rows of each series; given together with <paramref name="starts"/>.</param>
    /// <param name="mxreg">The mean regressors, T x N x k, for the specs with <see cref="Spec.Mxreg"/> = k.</param>
    /// <param name="vxreg">The variance regressors, T x N x k, for the specs with <see cref="Spec.Vxreg"/> = k.</param>
    /// <returns>The panel, sharing the arrays given.</returns>
    /// <exception cref="ArgumentException">Ranges without one non-negative entry per series, or given alone; a
    /// regressor whose first two dimensions are not T x N.</exception>
    public static Panel FromColumns(double[,] data, int[]? starts = null, int[]? lengths = null, double[,,]? mxreg = null,
                                    double[,,]? vxreg = null)
    {
        ArgumentNullException.ThrowIfNull(data);
        var series = data.GetLength(1);
        if ((starts is null) != (lengths is null))
        {
            throw new ArgumentException("starts and lengths must be given together", nameof(starts));
        }
        var first = Range(starts, series, nameof(starts));
        var size = Range(lengths, series, nameof(lengths));
        CheckRegressor(mxreg, data, nameof(mxreg));
        CheckRegressor(vxreg, data, nameof(vxreg));
        return new Panel(data, first, size, mxreg, vxreg);
    }

    /// <summary>A panel over series of any lengths, copied into one array with a range per series.</summary>
    /// <param name="series">The returns of each series.</param>
    /// <param name="mxreg">The mean regressors of each series, its length x k, the same k for all.</param>
    /// <param name="vxreg">The variance regressors of each series, its length x k, the same k for all.</param>
    /// <returns>The panel; T is the longest series.</returns>
    /// <exception cref="ArgumentException">No series, or regressors of another count, length or width.</exception>
    public static Panel FromSeries(IReadOnlyList<double[]> series, IReadOnlyList<double[,]>? mxreg = null,
                                   IReadOnlyList<double[,]>? vxreg = null)
    {
        ArgumentNullException.ThrowIfNull(series);
        if (series.Count == 0)
        {
            throw new ArgumentException("a panel needs at least one series", nameof(series));
        }
        var rows = series.Max(values => values.Length);
        var data = new double[rows, series.Count];
        FillNaN(data);
        for (var j = 0; j < series.Count; j++)
        {
            for (var t = 0; t < series[j].Length; t++)
            {
                data[t, j] = series[j][t];
            }
        }
        var lengths = series.Select(values => (nuint)values.Length).ToArray();
        var starts = new nuint[series.Count];
        var mean = Stack(mxreg, series, rows, nameof(mxreg));
        var variance = Stack(vxreg, series, rows, nameof(vxreg));
        return new Panel(data, starts, lengths, mean, variance);
    }

    private static nuint[]? Range(int[]? values, int series, string name)
    {
        if (values is null)
        {
            return null;
        }
        if (values.Length != series || values.Any(value => value < 0))
        {
            throw new ArgumentException($"{name} must have one non-negative entry per series ({series})", name);
        }
        return values.Select(value => (nuint)value).ToArray();
    }

    private static void CheckRegressor(double[,,]? values, double[,] data, string name)
    {
        if (values is null)
        {
            return;
        }
        if (values.GetLength(0) != data.GetLength(0) || values.GetLength(1) != data.GetLength(1))
        {
            throw new ArgumentException(
                $"{name} must be {data.GetLength(0)} x {data.GetLength(1)} x k, got " +
                $"{values.GetLength(0)} x {values.GetLength(1)} x {values.GetLength(2)}", name);
        }
    }

    // Regressor j becomes series j of a rows x N x k block, its rows past the series' length left NaN like the data.
    private static double[,,]? Stack(IReadOnlyList<double[,]>? values, IReadOnlyList<double[]> series, int rows, string name)
    {
        if (values is null)
        {
            return null;
        }
        if (values.Count != series.Count)
        {
            throw new ArgumentException($"{name} must have one entry per series ({series.Count})", name);
        }
        var width = values[0].GetLength(1);
        var block = new double[rows, series.Count, width];
        FillNaN(block);
        for (var j = 0; j < series.Count; j++)
        {
            if (values[j].GetLength(0) != series[j].Length || values[j].GetLength(1) != width)
            {
                throw new ArgumentException($"{name}[{j}] must be {series[j].Length} x {width}", name);
            }
            CopySeries(values[j], block, j);
        }
        return block;
    }

    private static void FillNaN(Array values)
    {
        ref var first = ref Unsafe.As<byte, double>(ref MemoryMarshal.GetArrayDataReference(values));
        MemoryMarshal.CreateSpan(ref first, values.Length).Fill(double.NaN);
    }

    private static void CopySeries(double[,] values, double[,,] block, int series)
    {
        for (var t = 0; t < values.GetLength(0); t++)
        {
            for (var k = 0; k < values.GetLength(1); k++)
            {
                block[t, series, k] = values[t, k];
            }
        }
    }
}
