namespace VolEngine.Interop;

// The regressors of a forecast or simulation horizon: horizon x k panels, the same rule as future_panel in Python.
internal static unsafe class Futures
{
    public static void Check(double[,]? values, string name, int count, int horizon)
    {
        if (count == 0 && values is not null)
        {
            throw new ArgumentException($"{name} was given but the spec has no {name}", name);
        }
        if (count == 0)
        {
            return;
        }
        if (values is null)
        {
            throw new ArgumentException($"the spec has {name} = {count}, so {name} is required", name);
        }
        if (values.GetLength(0) != horizon || values.GetLength(1) != count)
        {
            throw new ArgumentException(
                $"{name} must have shape ({horizon}, {count}), got ({values.GetLength(0)}, {values.GetLength(1)})", name);
        }
    }

    // The native panel over fixed data, or an invalid handle, which the native call receives as NULL ("none").
    public static PanelHandle Panel(double* data, double[,]? values)
    {
        if (values is null)
        {
            return new PanelHandle();
        }
        var rows = (nuint)values.GetLength(0);
        var columns = (nuint)values.GetLength(1);
        Calls.Check(NativeMethods.ve_panel_create(data, rows, columns, out var panel));
        return panel;
    }

    public static int Count(IReadOnlyList<string> names, string prefix)
    {
        return names.Count(name => name.StartsWith(prefix, StringComparison.Ordinal));
    }

    public static double[] Copy(int status, double* data, nuint length)
    {
        Calls.Check(status);
        return new ReadOnlySpan<double>(data, checked((int)length)).ToArray();
    }
}
