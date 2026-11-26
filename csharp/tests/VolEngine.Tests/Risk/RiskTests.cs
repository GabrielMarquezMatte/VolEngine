using System.Text.Json;
using VolEngine.Tests.Support;

namespace VolEngine.Tests;

public sealed class RiskTests
{
    private const double Tolerance = 1e-12;

    private static readonly Dictionary<string, Distribution> Names = new(StringComparer.Ordinal)
    {
        ["norm"] = Distribution.Norm,
        ["std"] = Distribution.Std,
        ["ged"] = Distribution.Ged,
        ["sstd"] = Distribution.Sstd,
        ["sged"] = Distribution.Sged,
        ["nig"] = Distribution.Nig,
        ["ghyp"] = Distribution.Ghyp,
        ["jsu"] = Distribution.Jsu,
    };

    private static double OrNaN(JsonElement item, string name)
    {
        var value = item.GetProperty(name);
        if (value.ValueKind == JsonValueKind.Null)
        {
            return double.NaN;
        }
        return value.GetDouble();
    }

    public static TheoryData<int> Cases()
    {
        var data = new TheoryData<int>();
        var results = ReferenceData.Fixture("rugarch_risk.json").GetProperty("results");
        for (var index = 0; index < results.GetArrayLength(); index++)
        {
            if (results[index].GetProperty("confidence").GetDouble() == 0.99)
            {
                data.Add(index);
            }
        }
        return data;
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public void Parametric_matches_rugarch(int index)
    {
        var item = ReferenceData.Fixture("rugarch_risk.json").GetProperty("results")[index];
        var distribution = new ShockDistribution(Names[item.GetProperty("dist").GetString()!], OrNaN(item, "skew"),
            OrNaN(item, "shape"), OrNaN(item, "ghlambda"));
        var measure = Risk.Parametric(distribution, 0.0, 1.0, 0.99);
        var quantile = item.GetProperty("quantile").GetDouble();
        var tail = item.GetProperty("tail_mean").GetDouble();
        Assert.True(Math.Abs(measure.Var + quantile) <= Tolerance * Math.Abs(quantile), $"VaR {measure.Var}");
        Assert.True(Math.Abs(measure.Es + tail) <= Tolerance * Math.Abs(tail), $"ES {measure.Es}");
    }

    [Fact]
    public void Empirical_hand_values()
    {
        double[,] returns = { { -0.05 }, { 0.01 }, { 0.02 }, { -0.02 }, { 0.03 } };
        var measure = Risk.Empirical(returns, [1.0], ReturnKind.Simple, 0.8);
        Assert.Equal(0.026, measure.Var, 1e-15);
        Assert.Equal(0.05, measure.Es, 1e-15);
    }

    [Fact]
    public void Wrong_weight_count_throws()
    {
        var returns = new double[10, 2];
        Assert.Throws<ArgumentException>(() => Risk.Empirical(returns, [1.0], ReturnKind.Log, 0.99));
    }

    [Fact]
    public void Invalid_confidence_throws()
    {
        var returns = new double[10, 1];
        var error = Assert.Throws<VolEngineException>(() => Risk.Empirical(returns, [1.0], ReturnKind.Log, 1.0));
        Assert.Equal(Status.InvalidArgument, error.Status);
        Assert.Throws<VolEngineException>(() => Risk.Parametric(new ShockDistribution(Distribution.Norm), 0.0, 1.0, 0.0));
    }

    [Fact]
    public void Missing_shape_throws()
    {
        var error = Assert.Throws<VolEngineException>(() =>
            Risk.Parametric(new ShockDistribution(Distribution.Std), 0.0, 0.01, 0.99));
        Assert.Equal(Status.InvalidArgument, error.Status);
    }

    [Fact]
    public void Null_throws()
    {
        Assert.Throws<ArgumentNullException>(() => Risk.Empirical(null!, [1.0], ReturnKind.Log, 0.99));
        Assert.Throws<ArgumentNullException>(() => Risk.Empirical(new double[2, 1], null!, ReturnKind.Log, 0.99));
        Assert.Throws<ArgumentNullException>(() => Risk.Parametric(null!, 0.0, 1.0, 0.99));
    }
}
