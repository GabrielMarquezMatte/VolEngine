using VolEngine.Interop;

namespace VolEngine;

/// <summary>
/// The native thread pool that fits and simulations run on. Results do not depend on it once they exist.
/// </summary>
/// <remarks>
/// Several threads may use one context at the same time for <see cref="Threads"/> and for fitting; the native library
/// promises nothing more, so other calls on a shared context need the caller's own coordination. Disposing waits for
/// nothing: do not dispose a context another thread is still using.
/// </remarks>
public sealed class Context : IDisposable
{
    private readonly ContextHandle _handle;

    /// <summary>Creates a context.</summary>
    /// <param name="threads">
    /// The pool size, 1 to 1024; 0 (the default) means the hardware concurrency, capped at 1024.
    /// </param>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="threads"/> is outside 0..1024.</exception>
    public Context(int threads = 0)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(threads);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(threads, NativeMethods.MaxThreads);
        Library.EnsureLoaded();
        Calls.Check(NativeMethods.ve_context_create(out var handle));
        try
        {
            Calls.Check(NativeMethods.ve_context_set_threads(handle, threads));
        }
        catch
        {
            handle.Dispose();
            throw;
        }
        _handle = handle;
    }

    /// <summary>The size of the thread pool.</summary>
    public int Threads
    {
        get
        {
            Calls.Check(NativeMethods.ve_context_threads(_handle, out var threads));
            return threads;
        }
    }

    internal ContextHandle Handle => _handle;

    /// <summary>Frees the thread pool. Calling it again does nothing.</summary>
    public void Dispose()
    {
        _handle.Dispose();
    }
}
