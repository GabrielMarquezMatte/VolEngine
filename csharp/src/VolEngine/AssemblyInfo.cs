using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

// Every native signature is blittable (handles, integers, doubles, pointers, UTF-8 strings), so the runtime never
// needs to marshal and the library stays trimming- and AOT-safe.
[assembly: DisableRuntimeMarshalling]

// The NuGet package puts the library where the host's own probing finds it, and VOLENGINE_LIB loads it by full
// path; nothing needs the current or application directory of the process.
[assembly: DefaultDllImportSearchPaths(DllImportSearchPath.SafeDirectories)]
[assembly: InternalsVisibleTo("VolEngine.Tests")]
[assembly: InternalsVisibleTo("VolEngine.Benchmarks")]
