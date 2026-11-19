using VolEngine.Interop;

namespace VolEngine;

/// <summary>The native library this process loaded.</summary>
public static class NativeAbi
{
    /// <summary>The ABI version of the loaded library (<c>ve_abi_version</c>); loading checks it is the one this
    /// package was built for.</summary>
    /// <exception cref="DllNotFoundException">The native library cannot be found.</exception>
    /// <exception cref="VolEngineException">The library has another ABI version.</exception>
    public static int Version
    {
        get
        {
            Library.EnsureLoaded();
            return NativeMethods.ve_abi_version();
        }
    }
}
