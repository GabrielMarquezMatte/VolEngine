using System.Runtime.InteropServices;

namespace VolEngine.Interop;

internal unsafe delegate int ViewReader<in THandle>(THandle owner, out double* data, out nuint length);

internal unsafe delegate int ValueReader<in THandle>(THandle owner, double* values);

// Copies out of a native result: zero-copy views (sigma, R, H) and per-parameter values.
internal static unsafe class NativeReads
{
    // The view points into the native result, so the handle is held until the copy is done: a Dispose on another
    // thread then waits for the release instead of freeing the memory under the copy.
    public static double[] View<THandle>(THandle owner, ViewReader<THandle> read)
        where THandle : SafeHandle
    {
        var added = false;
        owner.DangerousAddRef(ref added);
        try
        {
            Calls.Check(read(owner, out var data, out var length));
            return new ReadOnlySpan<double>(data, checked((int)length)).ToArray();
        }
        finally
        {
            if (added)
            {
                owner.DangerousRelease();
            }
        }
    }

    public static double[] Values<THandle>(THandle owner, int count, ValueReader<THandle> read)
        where THandle : SafeHandle
    {
        var values = new double[count];
        fixed (double* output = values)
        {
            Calls.Check(read(owner, output));
        }
        return values;
    }

    // A flat row-major block of count x m x m matrices as count jagged matrices.
    public static double[][][] Matrices(double[] values, int m)
    {
        return values.Chunk(m).Chunk(m).ToArray();
    }
}
