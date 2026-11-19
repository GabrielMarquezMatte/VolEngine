using System.Collections.Concurrent;
using System.Globalization;

namespace VolEngine.Tests;

public sealed class SpecTests
{
    [Fact]
    public void Defaults_match_python()
    {
        var spec = new Spec();
        Assert.Equal((Model.SGarch, 1, 1, (0, 0), Distribution.Norm, true),
                     (spec.Model, spec.P, spec.Q, spec.Arma, spec.Distribution, spec.IncludeMean));
        Assert.Empty(spec.Fixed);
        Assert.Empty(spec.Start);
        var options = new FitOptions();
        Assert.Equal((500, 1e-6, VarInit.Sample, true),
                     (options.MaxIter, options.Tolerance, options.VarInit, options.StandardErrors));
    }

    [Fact]
    public void Equality_compares_fixed_and_start_by_content()
    {
        var a = new Spec { Fixed = new Dictionary<string, double>(StringComparer.Ordinal) { ["shape"] = 5 }, Distribution = Distribution.Std };
        var b = new Spec { Fixed = new Dictionary<string, double>(StringComparer.Ordinal) { ["shape"] = 5 }, Distribution = Distribution.Std };
        Assert.Equal(a, b);
        Assert.Equal(a.GetHashCode(), b.GetHashCode());
        Assert.NotEqual(a, b with { Fixed = new Dictionary<string, double>(StringComparer.Ordinal) { ["shape"] = 6 } });
        Assert.NotEqual(a, b with { Start = new Dictionary<string, double>(StringComparer.Ordinal) { ["shape"] = 5 } });
    }

    [Fact]
    public void A_valid_spec_validates()
    {
        new Spec { Model = Model.Ewma }.Validate();
    }

    [Fact]
    public void An_unknown_fixed_name_is_rejected_with_its_name()
    {
        var spec = new Spec { Fixed = new Dictionary<string, double>(StringComparer.Ordinal) { ["shpe"] = 5 } };
        var error = Assert.Throws<VolEngineException>(spec.Validate);
        Assert.Equal(Status.InvalidArgument, error.Status);
        Assert.Contains("shpe", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void A_non_ascii_name_round_trips_in_the_message()
    {
        var spec = new Spec { Fixed = new Dictionary<string, double>(StringComparer.Ordinal) { ["ômega"] = 0.1 } };
        var error = Assert.Throws<VolEngineException>(spec.Validate);
        Assert.Contains("ômega", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void A_name_with_nul_is_an_argument_error()
    {
        var spec = new Spec { Start = new Dictionary<string, double>(StringComparer.Ordinal) { ["a\0b"] = 1 } };
        Assert.Throws<ArgumentException>(spec.Validate);
    }

    [Theory]
    [InlineData(0, 1)]
    [InlineData(1, 7)]
    public void Orders_outside_1_to_6_are_rejected(int p, int q)
    {
        var error = Assert.Throws<VolEngineException>(new Spec { P = p, Q = q }.Validate);
        Assert.Equal(Status.InvalidArgument, error.Status);
    }

    [Fact]
    public void Ewma_rejects_order_2_1()
    {
        Assert.Throws<VolEngineException>(new Spec { Model = Model.Ewma, P = 2 }.Validate);
    }

    [Fact]
    public void Invalid_fit_options_are_rejected()
    {
        Assert.Throws<VolEngineException>(() => new FitOptions { MaxIter = 0 }.CreateHandle());
        Assert.Throws<VolEngineException>(() => new FitOptions { Tolerance = -1 }.CreateHandle());
    }

    [Fact]
    public void Errors_on_many_threads_keep_their_own_message()
    {
        var messages = new ConcurrentBag<string>();
        Parallel.For(0, 64, i =>
        {
            var name = "bad" + i.ToString(CultureInfo.InvariantCulture);
            if (i % 2 == 0)
            {
                new Spec().Validate();
                return;
            }
            var error = Assert.Throws<VolEngineException>(
                new Spec { Fixed = new Dictionary<string, double>(StringComparer.Ordinal) { [name] = 1 } }.Validate);
            Assert.Contains(name, error.Message, StringComparison.Ordinal);
            messages.Add(error.Message);
        });
        Assert.Equal(32, messages.Count);
    }
}
