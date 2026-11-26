using System.Text.Json;
using VolEngine.Tests.Support;

namespace VolEngine.Tests;

public sealed class CovarianceTests
{
    private const double Tolerance = 1e-12;

    private static JsonElement Case(string id)
    {
        return ReferenceData.Result("sklearn_cov.json", "id", id);
    }

    private static double[,] Returns(JsonElement item)
    {
        var rows = item.GetProperty("returns").EnumerateArray().Select(ReferenceData.Doubles).ToArray();
        var panel = new double[rows.Length, rows[0].Length];
        for (var row = 0; row < rows.Length; row++)
        {
            for (var column = 0; column < rows[0].Length; column++)
            {
                panel[row, column] = rows[row][column];
            }
        }
        return panel;
    }

    // Every entry within Tolerance times the largest |entry| of the expected matrix.
    private static void AssertClose(JsonElement expected, double[][] actual)
    {
        var reference = expected.EnumerateArray().Select(ReferenceData.Doubles).ToArray();
        Assert.Equal(reference.Length, actual.Length);
        var scale = reference.SelectMany(row => row).Max(Math.Abs);
        for (var row = 0; row < reference.Length; row++)
        {
            Assert.Equal(reference.Length, actual[row].Length);
            for (var column = 0; column < reference.Length; column++)
            {
                Assert.True(Math.Abs(actual[row][column] - reference[row][column]) <= Tolerance * scale, $"{row},{column}");
            }
        }
    }

    [Theory]
    [InlineData("copula3")]
    [InlineData("wide")]
    public void Sample_matches_numpy(string id)
    {
        var item = Case(id);
        AssertClose(item.GetProperty("sample"), Covariances.Sample(Returns(item)));
    }

    [Theory]
    [InlineData("copula3")]
    [InlineData("wide")]
    public void LedoitWolf_matches_sklearn(string id)
    {
        var item = Case(id);
        var result = Covariances.LedoitWolf(Returns(item));
        AssertClose(item.GetProperty("ledoit_wolf"), result.Covariance);
        var expected = item.GetProperty("shrinkage").GetDouble();
        Assert.True(Math.Abs(result.Shrinkage - expected) <= Tolerance * expected);
    }

    [Fact]
    public void Nan_throws()
    {
        var returns = Returns(Case("wide"));
        returns[59, 3] = double.NaN;
        var error = Assert.Throws<VolEngineException>(() => Covariances.LedoitWolf(returns));
        Assert.Equal(Status.InvalidArgument, error.Status);
    }

    [Fact]
    public void Null_throws()
    {
        Assert.Throws<ArgumentNullException>(() => Covariances.Sample(null!));
        Assert.Throws<ArgumentNullException>(() => Covariances.LedoitWolf(null!));
    }

    // 65536 series: 65536 * 65536 wraps to 0 in int, which must not reach the native call as a too small buffer.
    [Fact]
    public void Too_many_series_throws_overflow()
    {
        var returns = new double[2, 65536];
        Assert.Throws<OverflowException>(() => Covariances.Sample(returns));
        Assert.Throws<OverflowException>(() => Covariances.LedoitWolf(returns));
    }

    [Fact]
    public void Result_is_symmetric()
    {
        var returns = Returns(Case("wider"));
        var sample = Covariances.Sample(returns);
        var shrunk = Covariances.LedoitWolf(returns).Covariance;
        for (var row = 0; row < sample.Length; row++)
        {
            for (var column = 0; column < sample.Length; column++)
            {
                Assert.Equal(sample[row][column], sample[column][row]);
                Assert.Equal(shrunk[row][column], shrunk[column][row]);
            }
        }
    }
}
