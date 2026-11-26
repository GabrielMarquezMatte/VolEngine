using System.Runtime.InteropServices;

namespace VolEngine.Interop;

// The C interface, one declaration per function of include/volengine/volengine.abi, under its C name. The types follow
// the table of the C#-a plan; ContractTests checks every declaration and constant against the contract.
internal static unsafe partial class NativeMethods
{
    // Without a lib prefix on Windows and with only the extension appended elsewhere, this one name finds
    // libvolengine.dll, libvolengine.so and libvolengine.dylib.
    internal const string LibraryName = "libvolengine";

    internal const int AbiVersion = 1;
    internal const int MaxThreads = 1024;
    internal const int MaxOrder = 6;
    internal const int MaxRegressors = 8;

    [LibraryImport(LibraryName)] internal static partial int ve_cancel_create(out CancelHandle cancel);
    [LibraryImport(LibraryName)] internal static partial int ve_cancel_request(CancelHandle cancel);
    [LibraryImport(LibraryName)] internal static partial void ve_cancel_free(nint cancel);
}
