using VolEngine.Interop;

namespace VolEngine;

/// <summary>
/// Covariance estimators over a T x N array of returns (dates in rows, series in columns), each series' mean removed.
/// The matrices are N rows of N and exactly symmetric. Fewer than 2 rows or a value that is not finite throws a
/// <see cref="VolEngineException"/> with <see cref="Status.InvalidArgument"/>.
/// </summary>
public static class Covariances
{
    /// <summary>The sample covariance Z'Z / (T - 1), as <c>numpy.cov(returns, rowvar=False)</c>.</summary>
    public static unsafe double[][] Sample(double[,] returns)
    {
        ArgumentNullException.ThrowIfNull(returns);
        var series = returns.GetLength(1);
        var covariance = new double[checked(series * series)];
        fixed (double* data = returns)
        fixed (double* output = covariance)
        {
            using var panel = Panel(data, returns);
            Calls.Check(NativeMethods.ve_cov_sample(panel, output));
        }
        return Rows(covariance, series);
    }

    /// <summary>
    /// The Ledoit-Wolf (2004) shrinkage of Z'Z / T towards mu * I, mu the mean variance, and its intensity, as
    /// <c>sklearn.covariance.ledoit_wolf(returns, assume_centered=False)</c>.
    /// </summary>
    public static unsafe LedoitWolfResult LedoitWolf(double[,] returns)
    {
        ArgumentNullException.ThrowIfNull(returns);
        var series = returns.GetLength(1);
        var covariance = new double[checked(series * series)];
        double shrinkage;
        fixed (double* data = returns)
        fixed (double* output = covariance)
        {
            using var panel = Panel(data, returns);
            Calls.Check(NativeMethods.ve_cov_ledoit_wolf(panel, output, &shrinkage));
        }
        return new LedoitWolfResult(Rows(covariance, series), shrinkage);
    }

    private static unsafe PanelHandle Panel(double* data, double[,] returns)
    {
        Calls.Check(NativeMethods.ve_panel_create(data, (nuint)returns.GetLength(0), (nuint)returns.GetLength(1),
            out var panel));
        return panel;
    }

    private static double[][] Rows(double[] matrix, int series)
    {
        var rows = new double[series][];
        for (var row = 0; row < series; row++)
        {
            rows[row] = matrix.AsSpan(row * series, series).ToArray();
        }
        return rows;
    }
}
