using System.Runtime.InteropServices;
using VolEngine.Interop;

namespace VolEngine;

/// <summary>The DCC fits of several specs on the same univariate fits, in spec order. Disposing it frees every
/// fit.</summary>
public sealed class DccFitBatch : IDisposable
{
    private readonly DccFit[] _fits;

    internal DccFitBatch(DccFit[] fits)
    {
        _fits = fits;
    }

    internal unsafe delegate int FitsCall(nint* fits, nuint count);

    /// <summary>The number of fits, one per spec.</summary>
    public int Count => _fits.Length;

    /// <summary>The fit of one spec.</summary>
    /// <param name="index">The index of the spec in the list given to the fit.</param>
    /// <returns>The fit, owned by the batch: do not dispose it; dispose the batch, which frees every fit.</returns>
    /// <exception cref="ArgumentOutOfRangeException">The index is outside the batch.</exception>
    public DccFit this[int index]
    {
        get
        {
            ArgumentOutOfRangeException.ThrowIfNegative(index);
            ArgumentOutOfRangeException.ThrowIfGreaterThanOrEqual(index, _fits.Length);
            return _fits[index];
        }
    }

    /// <summary>Frees every fit. Calling it again does nothing.</summary>
    public void Dispose()
    {
        foreach (var fit in _fits)
        {
            fit.Dispose();
        }
    }

    // One ve_dcc_fit_batch of every spec over the same univariate fits.
    internal static unsafe DccFit[] Run(ContextHandle context, IReadOnlyList<DccSpec> specs, IReadOnlyList<Fit> univariate,
                                        bool standardErrors, CancellationToken token)
    {
        Check(specs, univariate);
        var owned = new List<SafeHandle>();
        try
        {
            Calls.Check(NativeMethods.ve_dcc_batch_create(out var batch));
            owned.Add(batch);
            SetSpecs(owned, batch, specs);
            Calls.Check(NativeMethods.ve_dcc_options_create(out var options));
            owned.Add(options);
            Calls.Check(NativeMethods.ve_dcc_options_set_standard_errors(options, standardErrors ? 1 : 0));
            Calls.Check(NativeMethods.ve_dcc_batch_set_options(batch, options));
            Calls.Check(NativeMethods.ve_dcc_batch_spec_count(batch, out var count));
            var results = new nint[checked((int)count)];
            Calls.Check(Calls.WithCancel(cancel => WithFits(univariate, (fits, fitCount) =>
            {
                var status = NativeMethods.ve_dcc_batch_set_fits(batch, fits, fitCount);
                if (status != 0)
                {
                    return status;
                }
                fixed (nint* output = results)
                {
                    return NativeMethods.ve_dcc_fit_batch_with_cancel(context, batch, cancel, output);
                }
            }), token));
            return Wrap(results);
        }
        finally
        {
            foreach (var handle in owned)
            {
                handle.Dispose();
            }
        }
    }

    // Calls the native function with the handles of the univariate fits, each held until it returns: a fit disposed
    // before is an ObjectDisposedException, and one disposed on another thread meanwhile waits for the release.
    internal static unsafe int WithFits(IReadOnlyList<Fit> univariate, FitsCall call)
    {
        var handles = univariate.Select(fit => fit.Handle).ToArray();
        var added = new bool[handles.Length];
        var pointers = new nint[handles.Length];
        try
        {
            for (var i = 0; i < handles.Length; i++)
            {
                handles[i].DangerousAddRef(ref added[i]);
                pointers[i] = handles[i].DangerousGetHandle();
            }
            fixed (nint* fits = pointers)
            {
                return call(fits, (nuint)pointers.Length);
            }
        }
        finally
        {
            for (var i = 0; i < handles.Length; i++)
            {
                if (added[i])
                {
                    handles[i].DangerousRelease();
                }
            }
        }
    }

    internal static void CheckFits(IReadOnlyList<Fit> univariate)
    {
        ArgumentNullException.ThrowIfNull(univariate);
        if (univariate.Any(fit => fit is null))
        {
            throw new ArgumentNullException(nameof(univariate), "a univariate fit is null");
        }
    }

    private static void Check(IReadOnlyList<DccSpec> specs, IReadOnlyList<Fit> univariate)
    {
        ArgumentNullException.ThrowIfNull(specs);
        if (specs.Count == 0)
        {
            throw new ArgumentException("at least one spec is required", nameof(specs));
        }
        if (specs.Any(spec => spec is null))
        {
            throw new ArgumentNullException(nameof(specs), "a spec is null");
        }
        CheckFits(univariate);
    }

    private static unsafe void SetSpecs(List<SafeHandle> owned, DccBatchHandle batch, IReadOnlyList<DccSpec> specs)
    {
        var pointers = new nint[specs.Count];
        for (var i = 0; i < specs.Count; i++)
        {
            var spec = specs[i].CreateHandle();
            owned.Add(spec);
            pointers[i] = spec.DangerousGetHandle();
        }
        fixed (nint* specPointers = pointers)
        {
            Calls.Check(NativeMethods.ve_dcc_batch_set_specs(batch, specPointers, (nuint)pointers.Length));
        }
    }

    private static DccFit[] Wrap(nint[] pointers)
    {
        var handles = pointers.Select(pointer => new DccFitHandle(pointer)).ToArray();
        try
        {
            return handles.Select(handle => new DccFit(handle)).ToArray();
        }
        catch
        {
            foreach (var handle in handles)
            {
                handle.Dispose();
            }
            throw;
        }
    }
}
