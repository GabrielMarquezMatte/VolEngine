using System.Diagnostics.CodeAnalysis;
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

    /// <inheritdoc cref="Fit(Spec, ReadOnlySpan{double}, FitOptions, double[,], double[,], CancellationToken)"/>
    public Fit Fit(Spec spec, ReadOnlySpan<double> series, FitOptions? options, double[,]? mxreg, double[,]? vxreg)
    {
        return Fit(spec, series, options, mxreg, vxreg, CancellationToken.None);
    }

    /// <summary>Fits one spec on one series.</summary>
    /// <param name="spec">The model.</param>
    /// <param name="series">The returns, finite; fewer observations than the model needs give an
    /// <see cref="FitStatus.InsufficientData"/> fit, not an exception.</param>
    /// <param name="options">The optimizer settings; the defaults of <see cref="FitOptions"/> when null.</param>
    /// <param name="mxreg">The mean regressors, T x <see cref="Spec.Mxreg"/>; required exactly when the spec has them.</param>
    /// <param name="vxreg">The variance regressors, T x <see cref="Spec.Vxreg"/>; required exactly when the spec has them.</param>
    /// <param name="cancellationToken">Stops the call: a cancellation before or during it throws soon after, and a call
    /// that finishes first returns its result.</param>
    /// <returns>The fit; a fit that does not converge is a status, not an exception.</returns>
    /// <exception cref="ArgumentException">A regressor is missing, unexpected, or of another shape.</exception>
    /// <exception cref="VolEngineException">The spec is rejected, or the series has a NaN or infinite value.</exception>
    /// <exception cref="OperationCanceledException"><paramref name="cancellationToken"/> was cancelled and the call
    /// stopped with no result.</exception>
    public unsafe Fit Fit(Spec spec, ReadOnlySpan<double> series, FitOptions? options = null, double[,]? mxreg = null,
                          double[,]? vxreg = null, CancellationToken cancellationToken = default)
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
            return Batch.Run(_handle, [spec], options, panel, cancellationToken)[0];
        }
    }

    /// <inheritdoc cref="Simulate(VolEngine.Fit, SimulationOptions, CancellationToken)"/>
    public Simulation Simulate(Fit source, SimulationOptions options)
    {
        return Simulate(source, options, CancellationToken.None);
    }

    /// <summary>Simulates paths that continue after the last observation of a fit.</summary>
    /// <remarks>
    /// As rugarch's ugarchsim with startMethod = "sample". The threads of the context share the paths; the same seed,
    /// source and library version give the same paths for any thread count, and the first k paths (and steps) of a
    /// simulation are those of any larger one with the same seed. A new library version may change them.
    /// </remarks>
    /// <param name="source">The fit; it must have a finite point.</param>
    /// <param name="options">The horizon, number of paths, seed and horizon regressors.</param>
    /// <param name="cancellationToken">Stops the call: a cancellation before or during it throws soon after, and a call
    /// that finishes first returns its result.</param>
    /// <returns>The paths, a copy.</returns>
    /// <exception cref="ArgumentOutOfRangeException">A horizon or path count below 1.</exception>
    /// <exception cref="ArgumentException">A regressor is missing, unexpected, or of another shape.</exception>
    /// <exception cref="VolEngineException">The fit has no finite point, or the sizes do not fit in memory.</exception>
    /// <exception cref="OperationCanceledException"><paramref name="cancellationToken"/> was cancelled and the call
    /// stopped with no result.</exception>
    public Simulation Simulate(Fit source, SimulationOptions options, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(source);
        var mxreg = Futures.Count(source.Names, "mxreg");
        var vxreg = Futures.Count(source.Names, "vxreg");
        return Simulation.Run(options, mxreg, vxreg,
            (SimOptionsHandle handle, CancelHandle cancel, out SimHandle simulation) =>
                NativeMethods.ve_simulate_fit_with_cancel(_handle, source.Handle, handle, cancel, out simulation),
            cancellationToken);
    }

    /// <inheritdoc cref="Simulate(Spec, SimulationOptions, CancellationToken)"/>
    public Simulation Simulate(Spec source, SimulationOptions options)
    {
        return Simulate(source, options, CancellationToken.None);
    }

    /// <summary>Simulates paths of a spec with every parameter fixed, from its unconditional state.</summary>
    /// <remarks>As rugarch's ugarchpath, with the same seed rules as <see cref="Simulate(VolEngine.Fit, SimulationOptions)"/>.
    /// Needs every parameter fixed, no variance targeting, a stationary model, and not iGARCH or EWMA (no
    /// unconditional variance).</remarks>
    /// <param name="source">The spec.</param>
    /// <param name="options">The horizon, number of paths, seed and horizon regressors.</param>
    /// <param name="cancellationToken">Stops the call: a cancellation before or during it throws soon after, and a call
    /// that finishes first returns its result.</param>
    /// <returns>The paths, a copy.</returns>
    /// <exception cref="ArgumentOutOfRangeException">A horizon or path count below 1.</exception>
    /// <exception cref="ArgumentException">A regressor is missing, unexpected, or of another shape.</exception>
    /// <exception cref="VolEngineException">The spec cannot be simulated; the message says why.</exception>
    /// <exception cref="OperationCanceledException"><paramref name="cancellationToken"/> was cancelled and the call
    /// stopped with no result.</exception>
    public Simulation Simulate(Spec source, SimulationOptions options, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(source);
        using var spec = source.CreateHandle();
        return Simulation.Run(options, source.Mxreg, source.Vxreg,
            (SimOptionsHandle handle, CancelHandle cancel, out SimHandle simulation) =>
                NativeMethods.ve_simulate_spec_with_cancel(_handle, spec, handle, cancel, out simulation),
            cancellationToken);
    }

    /// <inheritdoc cref="Fit(IReadOnlyList{Spec}, Panel, FitOptions, CancellationToken)"/>
    public FitGrid Fit(IReadOnlyList<Spec> specs, Panel data, FitOptions? options)
    {
        return Fit(specs, data, options, CancellationToken.None);
    }

    /// <summary>Fits every spec on every series of a panel, in one native batch.</summary>
    /// <param name="specs">The models; at least one.</param>
    /// <param name="data">The series, with their ranges and regressors.</param>
    /// <param name="options">The optimizer settings, shared by every fit; the defaults when null.</param>
    /// <param name="cancellationToken">Stops the call: a cancellation before or during it throws soon after, and a call
    /// that finishes first returns its result.</param>
    /// <returns>The grid: <c>grid[i, j]</c> is spec i on series j.</returns>
    /// <exception cref="ArgumentException">No specs; regressors missing, unexpected, or not T x N x k for a spec with
    /// a count k > 0 (every such spec must share that k).</exception>
    /// <exception cref="VolEngineException">A spec is rejected, or a used row has a NaN or infinite value.</exception>
    /// <exception cref="OperationCanceledException"><paramref name="cancellationToken"/> was cancelled and the call
    /// stopped with no result.</exception>
    [SuppressMessage("ApiDesign", "RS0027", Justification = "The one-series Fit takes other inputs, with its own optional tail.")]
    public unsafe FitGrid Fit(IReadOnlyList<Spec> specs, Panel data, FitOptions? options = null,
                              CancellationToken cancellationToken = default)
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
            return new FitGrid(Batch.Run(_handle, specs, options, panel, cancellationToken), specs.Count, data.SeriesCount);
        }
    }

    /// <inheritdoc cref="FitDcc(DccSpec, IReadOnlyList{VolEngine.Fit}, bool, CancellationToken)"/>
    public DccFit FitDcc(DccSpec spec, IReadOnlyList<Fit> univariate, bool standardErrors)
    {
        return FitDcc(spec, univariate, standardErrors, CancellationToken.None);
    }

    /// <summary>Fits one DCC spec over ordered univariate fits.</summary>
    /// <remarks>
    /// The fits need at least two series with the same T &gt;= 100 and finite points; they are read during the call
    /// only, and the result outlives them. Disable standard errors to select among many specs, then call
    /// <see cref="StandardErrors(DccFit, IReadOnlyList{VolEngine.Fit}, CancellationToken)"/> on the one selected. Fits
    /// with regressors need standardErrors = false. The native library does not promise that DCC fits, standard errors
    /// and DCC simulations run concurrently on one context.
    /// </remarks>
    /// <param name="spec">The correlation model.</param>
    /// <param name="univariate">The fit of each series, in series order.</param>
    /// <param name="standardErrors">Whether to compute the standard errors.</param>
    /// <param name="cancellationToken">Stops the call: a cancellation before or during it throws soon after, and a call
    /// that finishes first returns its result.</param>
    /// <returns>The fit; a fit that does not converge is a status, not an exception.</returns>
    /// <exception cref="ArgumentNullException">A univariate fit is null.</exception>
    /// <exception cref="ObjectDisposedException">A univariate fit was disposed.</exception>
    /// <exception cref="VolEngineException">The spec is rejected, or the fits do not match (one series, different
    /// lengths, no finite point); the message says why.</exception>
    /// <exception cref="OperationCanceledException"><paramref name="cancellationToken"/> was cancelled and the call
    /// stopped with no result.</exception>
    public DccFit FitDcc(DccSpec spec, IReadOnlyList<Fit> univariate, bool standardErrors = true,
                         CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(spec);
        return DccFitBatch.Run(_handle, [spec], univariate, standardErrors, cancellationToken)[0];
    }

    /// <inheritdoc cref="FitDcc(IReadOnlyList{DccSpec}, IReadOnlyList{VolEngine.Fit}, bool, CancellationToken)"/>
    public DccFitBatch FitDcc(IReadOnlyList<DccSpec> specs, IReadOnlyList<Fit> univariate, bool standardErrors)
    {
        return FitDcc(specs, univariate, standardErrors, CancellationToken.None);
    }

    /// <summary>Fits several DCC specs over the same ordered univariate fits, in one native batch.</summary>
    /// <remarks>As <see cref="FitDcc(DccSpec, IReadOnlyList{VolEngine.Fit}, bool)"/>, once per spec.</remarks>
    /// <param name="specs">The correlation models; at least one.</param>
    /// <param name="univariate">The fit of each series, in series order.</param>
    /// <param name="standardErrors">Whether to compute the standard errors.</param>
    /// <param name="cancellationToken">Stops the call: a cancellation before or during it throws soon after, and a call
    /// that finishes first returns its result.</param>
    /// <returns>The fits, in spec order.</returns>
    /// <exception cref="ArgumentException">No specs.</exception>
    /// <exception cref="ArgumentNullException">A spec or a univariate fit is null.</exception>
    /// <exception cref="ObjectDisposedException">A univariate fit was disposed.</exception>
    /// <exception cref="VolEngineException">A spec is rejected, or the fits do not match.</exception>
    /// <exception cref="OperationCanceledException"><paramref name="cancellationToken"/> was cancelled and the call
    /// stopped with no result.</exception>
    public DccFitBatch FitDcc(IReadOnlyList<DccSpec> specs, IReadOnlyList<Fit> univariate, bool standardErrors = true,
                              CancellationToken cancellationToken = default)
    {
        return new DccFitBatch(DccFitBatch.Run(_handle, specs, univariate, standardErrors, cancellationToken));
    }

    /// <inheritdoc cref="StandardErrors(DccFit, IReadOnlyList{VolEngine.Fit}, CancellationToken)"/>
    public DccFit StandardErrors(DccFit fit, IReadOnlyList<Fit> univariate)
    {
        return StandardErrors(fit, univariate, CancellationToken.None);
    }

    /// <summary>A new fit with the standard errors of a fit made without them; the original stays as it was.</summary>
    /// <remarks>It refilters the whole sample, so it needs the univariate fits the DCC fit was made from, in the same
    /// order. Fits with regressors are unsupported.</remarks>
    /// <param name="fit">The DCC fit.</param>
    /// <param name="univariate">The univariate fits it was made from, in the same order.</param>
    /// <param name="cancellationToken">Stops the call: a cancellation before or during it throws soon after, and a call
    /// that finishes first returns its result.</param>
    /// <returns>A new fit sharing R, with standard errors.</returns>
    /// <exception cref="ArgumentNullException">A univariate fit is null.</exception>
    /// <exception cref="ObjectDisposedException">The fit or a univariate fit was disposed.</exception>
    /// <exception cref="VolEngineException">The univariate fits are not the ones of the fit.</exception>
    /// <exception cref="OperationCanceledException"><paramref name="cancellationToken"/> was cancelled and the call
    /// stopped with no result.</exception>
    public unsafe DccFit StandardErrors(DccFit fit, IReadOnlyList<Fit> univariate, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(fit);
        DccFitBatch.CheckFits(univariate);
        DccFitHandle? result = null;
        try
        {
            Calls.Check(Calls.WithCancel(cancel => DccFitBatch.WithFits(univariate, (fits, count) =>
            {
                var status = NativeMethods.ve_dcc_standard_errors_with_cancel(_handle, fit.Handle, fits, count, cancel,
                                                                              out var handle);
                result = handle;
                return status;
            }), cancellationToken));
            return new DccFit(result!);
        }
        catch
        {
            result?.Dispose();
            throw;
        }
    }

    /// <inheritdoc cref="Simulate(DccFit, DccSimulationOptions, CancellationToken)"/>
    public DccSimulation Simulate(DccFit source, DccSimulationOptions options)
    {
        return Simulate(source, options, CancellationToken.None);
    }

    /// <summary>Simulates multivariate paths of a DCC fit after its last observation.</summary>
    /// <remarks>The fit keeps the final state of each series and needs no univariate <see cref="VolEngine.Fit"/>.
    /// Path p reads only the stream (seed, p), with the seed rule of
    /// <see cref="Simulate(VolEngine.Fit, SimulationOptions)"/>. The native library does not promise that DCC fits,
    /// standard errors and DCC simulations run concurrently on one context.</remarks>
    /// <param name="source">The DCC fit; it must have a finite point.</param>
    /// <param name="options">The horizon, number of paths, seed, correlations and per-series horizon regressors.</param>
    /// <param name="cancellationToken">Stops the call: a cancellation before or during it throws soon after, and a call
    /// that finishes first returns its result.</param>
    /// <returns>The paths, a copy.</returns>
    /// <exception cref="ArgumentOutOfRangeException">A horizon or path count below 1, or arrays too large for one
    /// managed array.</exception>
    /// <exception cref="ArgumentException">A regressor list without one entry per series, or an entry without
    /// Horizon rows.</exception>
    /// <exception cref="VolEngineException">A series' regressors do not match its spec (the message names the series,
    /// from 1), or the fit has no finite point.</exception>
    /// <exception cref="OperationCanceledException"><paramref name="cancellationToken"/> was cancelled and the call
    /// stopped with no result.</exception>
    public DccSimulation Simulate(DccFit source, DccSimulationOptions options, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(source);
        return DccSimulation.Run(_handle, source, options, cancellationToken);
    }

    /// <summary>Frees the thread pool. Calling it again does nothing.</summary>
    public void Dispose()
    {
        _handle.Dispose();
    }
}
