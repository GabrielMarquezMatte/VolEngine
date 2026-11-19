using System.Reflection;

namespace VolEngine.Tests;

// Result types the library builds and the caller only reads: they must be able to gain members in a 1.x release, so
// no public constructor or Deconstruct whose signature would change, and no record equality that compares arrays by
// reference while looking like value equality.
public sealed class ResultShapeTests
{
    [Theory]
    [InlineData(typeof(Forecast))]
    [InlineData(typeof(Simulation))]
    [InlineData(typeof(CoefTable))]
    [InlineData(typeof(NyblomResult))]
    public void Result_types_can_grow_without_breaking_callers(Type type)
    {
        Assert.Empty(type.GetConstructors(BindingFlags.Public | BindingFlags.Instance));
        Assert.Null(type.GetMethod("Deconstruct"));
        Assert.Null(type.GetMethod("<Clone>$"));
    }
}
