using System.Runtime.InteropServices;
using VolEngine.Interop;

namespace VolEngine.Tests.Support;

// Declarations the contract check must reject. They are only reflected, never called.
internal static unsafe partial class WrongDeclarations
{
    // const size_t* is an input: only nuint* fits, never out nuint.
    [DefaultDllImportSearchPaths(DllImportSearchPath.SafeDirectories)]
    [LibraryImport("libvolengine")]
    internal static partial int ve_panel_set_ranges(PanelHandle panel, out nuint starts, nuint* lengths);

    // int32_t* is an output: out int or int*, never ref int.
    [DefaultDllImportSearchPaths(DllImportSearchPath.SafeDirectories)]
    [LibraryImport("libvolengine")]
    internal static partial int ve_context_threads(ContextHandle context, ref int threads);

    // const ve_spec* const* is an array of handles going in: only nint*.
    [DefaultDllImportSearchPaths(DllImportSearchPath.SafeDirectories)]
    [LibraryImport("libvolengine")]
    internal static partial int ve_batch_set_specs(BatchHandle batch, out SpecHandle specs, nuint count);

    // Strings cross the boundary as UTF-8.
    [DefaultDllImportSearchPaths(DllImportSearchPath.SafeDirectories)]
    [LibraryImport("libvolengine", StringMarshalling = StringMarshalling.Utf16)]
    internal static partial int ve_spec_set_fixed(SpecHandle spec, string name, double value);
}
