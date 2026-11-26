namespace VolEngine.Tests.Support;

// The reference of the cumulative simulation: the left-to-right sum of the first k steps, for each checkpoint k, with
// no compensation and no pairwise grouping, as the engine adds them.
internal static class LeftSum
{
    public static double[] At(IEnumerable<double> steps, IReadOnlyList<int> checkpoints)
    {
        var sums = new double[checkpoints.Count];
        var sum = 0.0;
        var step = 0;
        var next = 0;
        foreach (var value in steps)
        {
            sum += value;
            step++;
            if (next < checkpoints.Count && checkpoints[next] == step)
            {
                sums[next++] = sum;
            }
        }
        return sums;
    }

    public static void AssertSameBits(double[] expected, double[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (var i = 0; i < expected.Length; i++)
        {
            Assert.Equal(BitConverter.DoubleToInt64Bits(expected[i]), BitConverter.DoubleToInt64Bits(actual[i]));
        }
    }
}
