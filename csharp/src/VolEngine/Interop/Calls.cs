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
}
