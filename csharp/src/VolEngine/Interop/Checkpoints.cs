namespace VolEngine.Interop;

// The checkpoints of a simulation: the steps at which it keeps the running sum of each path. The native library
// checks the list (strictly increasing, from 1, the last one the horizon).
internal static unsafe class Checkpoints
{
    // The number of values a simulation copies per array: Paths x K x series when it has checkpoints, Paths x
    // Horizon x series otherwise.
    public static long Values(int horizon, int paths, int series, IReadOnlyList<int>? checkpoints)
    {
        var steps = checkpoints is { Count: > 0 } ? checkpoints.Count : horizon;
        return (long)steps * paths * series;
    }

    // An empty list is the full output, as a null one.
    public static void Apply(SimOptionsHandle options, IReadOnlyList<int>? checkpoints)
    {
        if (checkpoints is null or { Count: 0 })
        {
            return;
        }
        var sizes = checkpoints.Select(step => unchecked((nuint)step)).ToArray();
        fixed (nuint* first = sizes)
        {
            Calls.Check(NativeMethods.ve_sim_options_set_checkpoints(options, first, (nuint)sizes.Length));
        }
    }
}
