using System.Runtime.InteropServices;

namespace VolEngine.Interop;

// Fixed managed arrays as the native batch reads them: data T x N and regressors T x N x k, all in C order. Width 0
// means no spec has that kind; a null pointer with a width is an empty block (T = 0), which the native panel accepts.
internal readonly unsafe struct PinnedPanel
{
    public required double* Data { get; init; }
    public required int Rows { get; init; }
    public required int Series { get; init; }
    public nuint[]? Starts { get; init; }
    public nuint[]? Lengths { get; init; }
    public double* Mxreg { get; init; }
    public int MxregWidth { get; init; }
    public double* Vxreg { get; init; }
    public int VxregWidth { get; init; }
}

// One ve_fit_batch over every spec and series: job i * N + j fits spec i on series j.
internal static unsafe class Batch
{
    public static Fit[] Run(ContextHandle context, IReadOnlyList<Spec> specs, FitOptions? options, in PinnedPanel panel,
                            CancellationToken token)
    {
        var owned = new List<SafeHandle>();
        try
        {
            using var batch = CreateBatch(owned, specs, options, panel);
            SetJobs(batch, specs.Count, panel.Series);
            SetRegressors(owned, batch, specs, panel);
            return Execute(context, batch, token);
        }
        finally
        {
            foreach (var handle in owned)
            {
                handle.Dispose();
            }
        }
    }

    private static BatchHandle CreateBatch(List<SafeHandle> owned, IReadOnlyList<Spec> specs, FitOptions? options, in PinnedPanel panel)
    {
        Calls.Check(NativeMethods.ve_batch_create(out var batch));
        owned.Add(batch);
        var pointers = new nint[specs.Count];
        for (var i = 0; i < specs.Count; i++)
        {
            var spec = specs[i].CreateHandle();
            owned.Add(spec);
            pointers[i] = spec.DangerousGetHandle();
        }
        fixed (nint* specPointers = pointers)
        {
            Calls.Check(NativeMethods.ve_batch_set_specs(batch, specPointers, (nuint)pointers.Length));
        }
        var data = CreatePanel(owned, panel.Data, panel.Rows, panel.Series);
        SetRanges(data, panel.Starts, panel.Lengths);
        Calls.Check(NativeMethods.ve_batch_set_panel(batch, data));
        if (options is not null)
        {
            var optionsHandle = options.CreateHandle();
            owned.Add(optionsHandle);
            Calls.Check(NativeMethods.ve_batch_set_options(batch, optionsHandle));
        }
        owned.Remove(batch);
        return batch;
    }

    private static PanelHandle CreatePanel(List<SafeHandle> owned, double* data, int rows, int columns)
    {
        Calls.Check(NativeMethods.ve_panel_create(data, (nuint)rows, (nuint)columns, out var panel));
        owned.Add(panel);
        return panel;
    }

    private static void SetRanges(PanelHandle panel, nuint[]? starts, nuint[]? lengths)
    {
        if (starts is null || lengths is null)
        {
            return;
        }
        fixed (nuint* first = starts)
        fixed (nuint* size = lengths)
        {
            Calls.Check(NativeMethods.ve_panel_set_ranges(panel, first, size));
        }
    }

    private static void SetJobs(BatchHandle batch, int specCount, int seriesCount)
    {
        var jobs = checked(specCount * seriesCount);
        var jobSpec = new nuint[jobs];
        var jobSeries = new nuint[jobs];
        for (var job = 0; job < jobs; job++)
        {
            jobSpec[job] = (nuint)(job / seriesCount);
            jobSeries[job] = (nuint)(job % seriesCount);
        }
        fixed (nuint* specs = jobSpec)
        fixed (nuint* series = jobSeries)
        {
            Calls.Check(NativeMethods.ve_batch_set_jobs(batch, specs, series, (nuint)jobs));
        }
    }

    private static void SetRegressors(List<SafeHandle> owned, BatchHandle batch, IReadOnlyList<Spec> specs, in PinnedPanel panel)
    {
        var mean = JobPanels(owned, panel, panel.Mxreg, panel.MxregWidth, specs.Select(spec => spec.Mxreg));
        var variance = JobPanels(owned, panel, panel.Vxreg, panel.VxregWidth, specs.Select(spec => spec.Vxreg));
        if (mean is null && variance is null)
        {
            return;
        }
        fixed (nint* meanPointers = mean)
        fixed (nint* variancePointers = variance)
        {
            var jobs = (nuint)(specs.Count * panel.Series);
            Calls.Check(NativeMethods.ve_batch_set_regressors(batch, meanPointers, variancePointers, jobs));
        }
    }

    // The regressor panel of every job in job order: series j of the T x N x k block for a spec with a count, NULL for
    // a spec without one; null when no spec has this kind.
    private static nint[]? JobPanels(List<SafeHandle> owned, in PinnedPanel panel, double* values, int width, IEnumerable<int> counts)
    {
        if (width == 0)
        {
            return null;
        }
        var series = new nint[panel.Series];
        for (var j = 0; j < panel.Series; j++)
        {
            var handle = CreatePanel(owned, values + (j * width), panel.Rows, width);
            Calls.Check(NativeMethods.ve_panel_set_strides(handle, panel.Series * width, 1));
            series[j] = handle.DangerousGetHandle();
        }
        var jobs = new List<nint>();
        foreach (var count in counts)
        {
            jobs.AddRange(count > 0 ? series : new nint[panel.Series]);
        }
        return [.. jobs];
    }

    private static Fit[] Execute(ContextHandle context, BatchHandle batch, CancellationToken token)
    {
        Calls.Check(NativeMethods.ve_batch_job_count(batch, out var jobs));
        var pointers = new nint[checked((int)jobs)];
        Calls.Check(Calls.WithCancel(cancel =>
        {
            fixed (nint* fits = pointers)
            {
                return NativeMethods.ve_fit_batch_with_cancel(context, batch, cancel, fits);
            }
        }, token));
        var handles = pointers.Select(pointer => new FitHandle(pointer)).ToArray();
        try
        {
            return handles.Select(handle => new Fit(handle)).ToArray();
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
