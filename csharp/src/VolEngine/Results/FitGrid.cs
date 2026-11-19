namespace VolEngine;

/// <summary>The fits of a grid: one per spec and series. Disposing the grid disposes every fit in it.</summary>
public sealed class FitGrid : IDisposable
{
    private readonly Fit[] _fits;

    internal FitGrid(Fit[] fits, int specCount, int seriesCount)
    {
        _fits = fits;
        SpecCount = specCount;
        SeriesCount = seriesCount;
    }

    /// <summary>The number of specs.</summary>
    public int SpecCount { get; }

    /// <summary>The number of series.</summary>
    public int SeriesCount { get; }

    /// <summary>The fit of one spec on one series.</summary>
    /// <param name="spec">The index of the spec in the list given to the fit.</param>
    /// <param name="series">The index of the series in the panel.</param>
    /// <returns>The fit, owned by the grid: do not dispose it; dispose the grid, which frees every fit.</returns>
    /// <exception cref="ArgumentOutOfRangeException">An index is outside the grid.</exception>
    public Fit this[int spec, int series]
    {
        get
        {
            ArgumentOutOfRangeException.ThrowIfNegative(spec);
            ArgumentOutOfRangeException.ThrowIfGreaterThanOrEqual(spec, SpecCount);
            ArgumentOutOfRangeException.ThrowIfNegative(series);
            ArgumentOutOfRangeException.ThrowIfGreaterThanOrEqual(series, SeriesCount);
            return _fits[(spec * SeriesCount) + series];
        }
    }

    /// <summary>Disposes every fit of the grid. Calling it again does nothing.</summary>
    public void Dispose()
    {
        foreach (var fit in _fits)
        {
            fit.Dispose();
        }
    }
}
