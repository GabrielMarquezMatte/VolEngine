using System.Runtime.InteropServices;

namespace VolEngine.Interop;

internal static class Calls
{
    // The message of ve_last_error belongs to the calling thread and lives only until its next VolEngine call, so it
    // is read here, before anything else can call into the library.
    public static void Check(int status)
    {
        if (status == 0)
        {
            return;
        }
        throw new VolEngineException((Status)status, LastError());
    }

    public static string LastError()
    {
        return Marshal.PtrToStringUTF8(NativeMethods.ve_last_error()) ?? "";
    }

    // Runs a native call that takes a ve_cancel, requested when the token is cancelled. A token that can never be
    // cancelled passes NULL.
    public static int WithCancel(Func<CancelHandle, int> call, CancellationToken token)
    {
        token.ThrowIfCancellationRequested();
        if (!token.CanBeCanceled)
        {
            using var none = new CancelHandle();
            return call(none);
        }
        Check(NativeMethods.ve_cancel_create(out var created));
        using var cancel = created;
        // Declared after the handle so it is disposed first: Dispose waits for a running callback, so no request
        // reaches the handle once it is freed.
        using var registration = token.Register(Request, cancel);
        var status = call(cancel);
        if (status == (int)Status.Cancelled)
        {
            throw new OperationCanceledException(token);
        }
        return status;
    }

    private static void Request(object? cancel)
    {
        _ = NativeMethods.ve_cancel_request((CancelHandle)cancel!);
    }
}
