using VolEngine.Interop;

namespace VolEngine;

/// <summary>
/// The native thread pool that fits and simulations run on. Results do not depend on it once they exist.
/// </summary>
/// <remarks>
/// Several threads may use one context at the same time for <see cref="Threads"/> and for fitting; the native library
/// promises nothing more, so other calls on a shared context need the caller's own coordination. Disposing waits for
/// nothing: do not dispose a context another thread is still using.
/// </remarks>
public sealed class Context : IDisposable
{
    private readonly ContextHandle _handle;

    /// <summary>Creates a context.</summary>
    /// <param name="threads">
    /// The pool size, 1 to 1024; 0 (the default) means the hardware concurrency, capped at 1024.
    /// </param>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="threads"/> is outside 0..1024.</exception>
    public Context(int threads = 0)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(threads);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(threads, NativeMethods.MaxThreads);
        Library.EnsureLoaded();
        Calls.Check(NativeMethods.ve_context_create(out var handle));
        try
        {
            Calls.Check(NativeMethods.ve_context_set_threads(handle, threads));
        }
        catch
        {
            handle.Dispose();
            throw;
        }
        _handle = handle;
    }

    /// <summary>The size of the thread pool.</summary>
    public int Threads
    {
        get
        {
            Calls.Check(NativeMethods.ve_context_threads(_handle, out var threads));
            return threads;
        }
    }

    internal ContextHandle Handle => _handle;

    /// <summary>Fits one spec on one series.</summary>
    /// <param name="spec">The model.</param>
    /// <param name="series">The returns, finite; fewer observations than the model needs give an
    /// <see cref="FitStatus.InsufficientData"/> fit, not an exception.</param>
    /// <param name="options">The optimizer settings; the defaults of <see cref="FitOptions"/> when null.</param>
    /// <param name="mxreg">The mean regressors, T x <see cref="Spec.Mxreg"/>; required exactly when the spec has them.</param>
    /// <param name="vxreg">The variance regressors, T x <see cref="Spec.Vxreg"/>; required exactly when the spec has them.</param>
    /// <returns>The fit; a fit that does not converge is a status, not an exception.</returns>
    /// <exception cref="ArgumentException">A regressor is missing, unexpected, or of another shape.</exception>
    /// <exception cref="VolEngineException">The spec is rejected, or the series has a NaN or infinite value.</exception>
    public unsafe Fit Fit(Spec spec, ReadOnlySpan<double> series, FitOptions? options = null, double[,]? mxreg = null,
                          double[,]? vxreg = null)
    {
        ArgumentNullException.ThrowIfNull(spec);
        var mxregWidth = RegressorShapes.Width(mxreg, nameof(mxreg), [spec.Mxreg], series.Length);
        var vxregWidth = RegressorShapes.Width(vxreg, nameof(vxreg), [spec.Vxreg], series.Length);
        fixed (double* data = series)
        fixed (double* mean = mxreg)
        fixed (double* variance = vxreg)
        {
            var panel = new PinnedPanel
            {
                Data = data,
                Rows = series.Length,
                Series = 1,
                Mxreg = mean,
                MxregWidth = mxregWidth,
                Vxreg = variance,
                VxregWidth = vxregWidth,
            };
            return Batch.Run(_handle, [spec], options, panel)[0];
        }
    }

    /// <summary>Fits every spec on every series of a panel, in one native batch.</summary>
    /// <param name="specs">The models; at least one.</param>
    /// <param name="data">The series, with their ranges and regressors.</param>
    /// <param name="options">The optimizer settings, shared by every fit; the defaults when null.</param>
    /// <returns>The grid: <c>grid[i, j]</c> is spec i on series j.</returns>
    /// <exception cref="ArgumentException">No specs; regressors missing, unexpected, or not T x N x k for a spec with
    /// a count k > 0 (every such spec must share that k).</exception>
    /// <exception cref="VolEngineException">A spec is rejected, or a used row has a NaN or infinite value.</exception>
    public unsafe FitGrid Fit(IReadOnlyList<Spec> specs, Panel data, FitOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(specs);
        ArgumentNullException.ThrowIfNull(data);
        if (specs.Count == 0)
        {
            throw new ArgumentException("at least one spec is required", nameof(specs));
        }
        if (specs.Any(spec => spec is null))
        {
            throw new ArgumentNullException(nameof(specs), "a spec is null");
        }
        var mxregWidth = RegressorShapes.Width(data.Mxreg, "mxreg", specs.Select(spec => spec.Mxreg), data.Rows, data.SeriesCount);
        var vxregWidth = RegressorShapes.Width(data.Vxreg, "vxreg", specs.Select(spec => spec.Vxreg), data.Rows, data.SeriesCount);
        fixed (double* values = data.Data)
        fixed (double* mean = data.Mxreg)
        fixed (double* variance = data.Vxreg)
        {
            var panel = new PinnedPanel
            {
                Data = values,
                Rows = data.Rows,
                Series = data.SeriesCount,
                Starts = data.Starts,
                Lengths = data.Lengths,
                Mxreg = mean,
                MxregWidth = mxregWidth,
                Vxreg = variance,
                VxregWidth = vxregWidth,
            };
            return new FitGrid(Batch.Run(_handle, specs, options, panel), specs.Count, data.SeriesCount);
        }
    }

    /// <summary>Frees the thread pool. Calling it again does nothing.</summary>
    public void Dispose()
    {
        _handle.Dispose();
    }
}
