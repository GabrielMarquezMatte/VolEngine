using Microsoft.Win32.SafeHandles;

namespace VolEngine.Interop;

// One SafeHandle per opaque type of the C interface. The parameterless constructor is the one the LibraryImport
// generator uses for out parameters; the pointer constructor wraps handles that arrive in arrays (ve_fit_batch).

internal sealed class ContextHandle : SafeHandleZeroOrMinusOneIsInvalid
{
    public ContextHandle()
        : base(true)
    {
    }

    internal ContextHandle(nint pointer)
        : this()
    {
        SetHandle(pointer);
    }

    protected override bool ReleaseHandle()
    {
        NativeMethods.ve_context_free(handle);
        return true;
    }
}

internal sealed class SpecHandle : SafeHandleZeroOrMinusOneIsInvalid
{
    public SpecHandle()
        : base(true)
    {
    }

    internal SpecHandle(nint pointer)
        : this()
    {
        SetHandle(pointer);
    }

    protected override bool ReleaseHandle()
    {
        NativeMethods.ve_spec_free(handle);
        return true;
    }
}

internal sealed class FitOptionsHandle : SafeHandleZeroOrMinusOneIsInvalid
{
    public FitOptionsHandle()
        : base(true)
    {
    }

    internal FitOptionsHandle(nint pointer)
        : this()
    {
        SetHandle(pointer);
    }

    protected override bool ReleaseHandle()
    {
        NativeMethods.ve_fit_options_free(handle);
        return true;
    }
}

internal sealed class PanelHandle : SafeHandleZeroOrMinusOneIsInvalid
{
    public PanelHandle()
        : base(true)
    {
    }

    internal PanelHandle(nint pointer)
        : this()
    {
        SetHandle(pointer);
    }

    protected override bool ReleaseHandle()
    {
        NativeMethods.ve_panel_free(handle);
        return true;
    }
}

internal sealed class BatchHandle : SafeHandleZeroOrMinusOneIsInvalid
{
    public BatchHandle()
        : base(true)
    {
    }

    internal BatchHandle(nint pointer)
        : this()
    {
        SetHandle(pointer);
    }

    protected override bool ReleaseHandle()
    {
        NativeMethods.ve_batch_free(handle);
        return true;
    }
}

internal sealed class FitHandle : SafeHandleZeroOrMinusOneIsInvalid
{
    public FitHandle()
        : base(true)
    {
    }

    internal FitHandle(nint pointer)
        : this()
    {
        SetHandle(pointer);
    }

    protected override bool ReleaseHandle()
    {
        NativeMethods.ve_fit_free(handle);
        return true;
    }
}

internal sealed class ForecastHandle : SafeHandleZeroOrMinusOneIsInvalid
{
    public ForecastHandle()
        : base(true)
    {
    }

    internal ForecastHandle(nint pointer)
        : this()
    {
        SetHandle(pointer);
    }

    protected override bool ReleaseHandle()
    {
        NativeMethods.ve_forecast_free(handle);
        return true;
    }
}

internal sealed class SimOptionsHandle : SafeHandleZeroOrMinusOneIsInvalid
{
    public SimOptionsHandle()
        : base(true)
    {
    }

    internal SimOptionsHandle(nint pointer)
        : this()
    {
        SetHandle(pointer);
    }

    protected override bool ReleaseHandle()
    {
        NativeMethods.ve_sim_options_free(handle);
        return true;
    }
}

internal sealed class SimHandle : SafeHandleZeroOrMinusOneIsInvalid
{
    public SimHandle()
        : base(true)
    {
    }

    internal SimHandle(nint pointer)
        : this()
    {
        SetHandle(pointer);
    }

    protected override bool ReleaseHandle()
    {
        NativeMethods.ve_sim_free(handle);
        return true;
    }
}

internal sealed class DccSpecHandle : SafeHandleZeroOrMinusOneIsInvalid
{
    public DccSpecHandle()
        : base(true)
    {
    }

    internal DccSpecHandle(nint pointer)
        : this()
    {
        SetHandle(pointer);
    }

    protected override bool ReleaseHandle()
    {
        NativeMethods.ve_dcc_spec_free(handle);
        return true;
    }
}

internal sealed class DccOptionsHandle : SafeHandleZeroOrMinusOneIsInvalid
{
    public DccOptionsHandle()
        : base(true)
    {
    }

    internal DccOptionsHandle(nint pointer)
        : this()
    {
        SetHandle(pointer);
    }

    protected override bool ReleaseHandle()
    {
        NativeMethods.ve_dcc_options_free(handle);
        return true;
    }
}

internal sealed class DccBatchHandle : SafeHandleZeroOrMinusOneIsInvalid
{
    public DccBatchHandle()
        : base(true)
    {
    }

    internal DccBatchHandle(nint pointer)
        : this()
    {
        SetHandle(pointer);
    }

    protected override bool ReleaseHandle()
    {
        NativeMethods.ve_dcc_batch_free(handle);
        return true;
    }
}

internal sealed class DccFitHandle : SafeHandleZeroOrMinusOneIsInvalid
{
    public DccFitHandle()
        : base(true)
    {
    }

    internal DccFitHandle(nint pointer)
        : this()
    {
        SetHandle(pointer);
    }

    protected override bool ReleaseHandle()
    {
        NativeMethods.ve_dcc_fit_free(handle);
        return true;
    }
}

internal sealed class DccFutureHandle : SafeHandleZeroOrMinusOneIsInvalid
{
    public DccFutureHandle()
        : base(true)
    {
    }

    internal DccFutureHandle(nint pointer)
        : this()
    {
        SetHandle(pointer);
    }

    protected override bool ReleaseHandle()
    {
        NativeMethods.ve_dcc_future_free(handle);
        return true;
    }
}

internal sealed class DccForecastHandle : SafeHandleZeroOrMinusOneIsInvalid
{
    public DccForecastHandle()
        : base(true)
    {
    }

    internal DccForecastHandle(nint pointer)
        : this()
    {
        SetHandle(pointer);
    }

    protected override bool ReleaseHandle()
    {
        NativeMethods.ve_dcc_forecast_free(handle);
        return true;
    }
}

internal sealed class DccSimHandle : SafeHandleZeroOrMinusOneIsInvalid
{
    public DccSimHandle()
        : base(true)
    {
    }

    internal DccSimHandle(nint pointer)
        : this()
    {
        SetHandle(pointer);
    }

    protected override bool ReleaseHandle()
    {
        NativeMethods.ve_dcc_sim_free(handle);
        return true;
    }
}

// The default instance is the NULL handle of a call that cannot be cancelled.
internal sealed class CancelHandle : SafeHandleZeroOrMinusOneIsInvalid
{
    public CancelHandle()
        : base(true)
    {
    }

    protected override bool ReleaseHandle()
    {
        NativeMethods.ve_cancel_free(handle);
        return true;
    }
}
