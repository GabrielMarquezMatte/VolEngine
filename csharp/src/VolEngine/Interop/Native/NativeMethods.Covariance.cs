using System.Runtime.InteropServices;

namespace VolEngine.Interop;

internal static unsafe partial class NativeMethods
{
    [LibraryImport(LibraryName)] internal static partial int ve_cov_sample(PanelHandle panel, double* output);
    [LibraryImport(LibraryName)] internal static partial int ve_cov_ledoit_wolf(PanelHandle panel, double* output, double* shrinkage);
}
