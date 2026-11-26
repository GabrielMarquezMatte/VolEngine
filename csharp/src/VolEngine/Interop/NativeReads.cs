using System.Runtime.InteropServices;

namespace VolEngine.Interop;

internal unsafe delegate int ViewReader<in THandle>(THandle owner, out double* data, out nuint length);

internal unsafe delegate int SizeViewReader<in THandle>(THandle owner, out nuint* data, out nuint length);

internal unsafe delegate int ValueReader<in THandle>(THandle owner, double* values);

// Copies out of a native result: zero-copy views (sigma, R, H) and per-parameter values.
internal static unsafe class NativeReads
{
    // The view points into the native result, so the handle is held until the copy is done: a Dispose on another
    // thread then waits for the release instead of freeing the memory under the copy.
    public static double[] View<THandle>(THandle owner, ViewReader<THandle> read)
        where THandle : SafeHandle
    {
        return Held(owner, () =>
        {
            Calls.Check(read(owner, out var data, out var length));
            return new ReadOnlySpan<double>(data, checked((int)length)).ToArray();
        });
    }

    // A view of size_t values (the checkpoints) as ints; the engine holds them at most as large as an int horizon.
    public static int[] Sizes<THandle>(THandle owner, SizeViewReader<THandle> read)
        where THandle : SafeHandle
    {
        return Held(owner, () =>
        {
            Calls.Check(read(owner, out var data, out var length));
            return new ReadOnlySpan<nuint>(data, checked((int)length)).ToArray().Select(size => checked((int)size)).ToArray();
        });
    }

    private static T Held<THandle, T>(THandle owner, Func<T> copy)
        where THandle : SafeHandle
    {
        var added = false;
        owner.DangerousAddRef(ref added);
        try
        {
            return copy();
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

    // A row-major view as count rows of width values (count = length / width), each row copied straight from the
    // native memory while the handle is held: no flat copy in between, which for the R and H of a simulation was as
    // large as the result.
    public static double[][] Rows<THandle>(THandle owner, ViewReader<THandle> read, int width)
        where THandle : SafeHandle
    {
        return Held(owner, () =>
        {
            Calls.Check(read(owner, out var data, out var length));
            return RowsAt(data, Count(length, width), width);
        });
    }

    // count x rows x width, as Rows.
    public static double[][][] Matrices<THandle>(THandle owner, ViewReader<THandle> read, int rows, int width)
        where THandle : SafeHandle
    {
        return Held(owner, () =>
        {
            Calls.Check(read(owner, out var data, out var length));
            var size = (long)rows * width;
            var matrices = new double[Count(length, size)][][];
            for (var i = 0; i < matrices.Length; i++)
            {
                matrices[i] = RowsAt(data + i * size, rows, width);
            }
            return matrices;
        });
    }

    // count x blocks x rows x width, as Rows.
    public static double[][][][] Blocks<THandle>(THandle owner, ViewReader<THandle> read, int blocks, int rows,
                                                 int width)
        where THandle : SafeHandle
    {
        return Held(owner, () =>
        {
            Calls.Check(read(owner, out var data, out var length));
            var matrix = (long)rows * width;
            var size = blocks * matrix;
            var result = new double[Count(length, size)][][][];
            for (var i = 0; i < result.Length; i++)
            {
                var block = new double[blocks][][];
                for (var j = 0; j < blocks; j++)
                {
                    block[j] = RowsAt(data + i * size + j * matrix, rows, width);
                }
                result[i] = block;
            }
            return result;
        });
    }

    private static int Count(nuint length, long size)
    {
        return size == 0 ? 0 : checked((int)((long)length / size));
    }

    private static double[][] RowsAt(double* data, int count, int width)
    {
        var rows = new double[count][];
        for (var i = 0; i < count; i++)
        {
            rows[i] = new ReadOnlySpan<double>(data + (long)i * width, width).ToArray();
        }
        return rows;
    }
}
