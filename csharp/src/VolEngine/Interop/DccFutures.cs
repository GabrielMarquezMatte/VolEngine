namespace VolEngine.Interop;

// The regressors of the horizon of each series of a DCC fit (forecast and simulation): one entry per series,
// horizon x k or null for a series without. The native library checks each width against the series' spec.
internal static unsafe class DccFutures
{
    // An invalid handle (NULL) when neither list is given: no series has regressors.
    public static DccFutureHandle Create(int series, int horizon, IReadOnlyList<double[,]?>? mxreg,
                                         IReadOnlyList<double[,]?>? vxreg)
    {
        if (mxreg is null && vxreg is null)
        {
            return new DccFutureHandle();
        }
        Check(mxreg, nameof(mxreg), series, horizon);
        Check(vxreg, nameof(vxreg), series, horizon);
        Calls.Check(NativeMethods.ve_dcc_future_create((nuint)series, (nuint)horizon, out var future));
        try
        {
            for (var j = 0; j < series; j++)
            {
                Set(future, j, mxreg?[j], vxreg?[j]);
            }
        }
        catch
        {
            future.Dispose();
            throw;
        }
        return future;
    }

    private static void Check(IReadOnlyList<double[,]?>? entries, string name, int series, int horizon)
    {
        if (entries is null)
        {
            return;
        }
        if (entries.Count != series)
        {
            throw new ArgumentException($"{name} must have one entry per series ({series}), got {entries.Count}", name);
        }
        for (var j = 0; j < series; j++)
        {
            if (entries[j] is { } values && values.GetLength(0) != horizon)
            {
                throw new ArgumentException($"{name}[{j}] must have {horizon} rows, got {values.GetLength(0)}", name);
            }
        }
    }

    // The future copies the panels, so each array is pinned only for its call.
    private static void Set(DccFutureHandle future, int series, double[,]? mxreg, double[,]? vxreg)
    {
        fixed (double* mean = mxreg)
        fixed (double* variance = vxreg)
        {
            using var meanPanel = Futures.Panel(mean, mxreg);
            using var variancePanel = Futures.Panel(variance, vxreg);
            Calls.Check(NativeMethods.ve_dcc_future_set(future, (nuint)series, meanPanel, variancePanel));
        }
    }
}
