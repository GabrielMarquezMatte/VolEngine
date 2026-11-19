using VolEngine.Tests.Support;

namespace VolEngine.Tests;

public sealed class SimulationTests
{
    private static readonly Dictionary<string, double> Stationary = new(StringComparer.Ordinal)
    {
        ["mu"] = 0,
        ["omega"] = 0.01,
        ["alpha1"] = 0.05,
        ["beta1"] = 0.9,
    };

    private static Fit SeriesFit(Context context)
    {
        return context.Fit(new Spec(), ReferenceData.Series("sgarch11_norm.csv", "t1500"));
    }

    [Fact]
    public void Same_seed_gives_same_paths()
    {
        using var context = new Context();
        using var fit = SeriesFit(context);
        var first = context.Simulate(fit, new SimulationOptions(20, 4, 7));
        var second = context.Simulate(fit, new SimulationOptions(20, 4, 7));
        Assert.Equal((4, 20), (first.Paths, first.Horizon));
        Assert.Equal(first.Returns, second.Returns);
        Assert.Equal(first.Sigma, second.Sigma);
        Assert.NotEqual(first.Returns[0], first.Returns[1]);
    }

    [Fact]
    public void Paths_and_steps_are_prefixes()
    {
        using var context = new Context();
        using var fit = SeriesFit(context);
        var small = context.Simulate(fit, new SimulationOptions(10, 3, 7));
        var large = context.Simulate(fit, new SimulationOptions(20, 5, 7));
        for (var path = 0; path < 3; path++)
        {
            Assert.Equal(small.Returns[path], large.Returns[path][..10]);
            Assert.Equal(small.Sigma[path], large.Sigma[path][..10]);
        }
    }

    [Fact]
    public void Thread_count_does_not_change_paths()
    {
        using var one = new Context(1);
        using var four = new Context(4);
        using var fit = SeriesFit(one);
        var serial = one.Simulate(fit, new SimulationOptions(30, 16, 99));
        var parallel = four.Simulate(fit, new SimulationOptions(30, 16, 99));
        Assert.Equal(serial.Returns, parallel.Returns);
        Assert.Equal(serial.Sigma, parallel.Sigma);
    }

    [Fact]
    public void A_fixed_spec_simulates_from_its_unconditional_state()
    {
        using var context = new Context();
        var simulation = context.Simulate(new Spec { Fixed = Stationary }, new SimulationOptions(50, 2, 3));
        Assert.Equal(2, simulation.Returns.Length);
        Assert.All(simulation.Returns, path => Assert.Equal(50, path.Length));
        Assert.All(simulation.Returns.SelectMany(path => path), value => Assert.True(double.IsFinite(value)));
        Assert.All(simulation.Sigma.SelectMany(path => path), value => Assert.True(value > 0));
    }

    [Fact]
    public void A_spec_with_a_free_parameter_cannot_simulate()
    {
        using var context = new Context();
        var error = Assert.Throws<VolEngineException>(() => context.Simulate(new Spec(), new SimulationOptions(5, 1, 1)));
        Assert.Equal(Status.InvalidArgument, error.Status);
    }

    [Fact]
    public void Ewma_spec_cannot_simulate()
    {
        var fixedValues = new Dictionary<string, double>(StringComparer.Ordinal) { ["mu"] = 0, ["lambda"] = 0.94 };
        using var context = new Context();
        var spec = new Spec { Model = Model.Ewma, Fixed = fixedValues };
        Assert.Throws<VolEngineException>(() => context.Simulate(spec, new SimulationOptions(5, 1, 1)));
    }

    [Fact]
    public void Simulation_sizes_must_be_positive()
    {
        using var context = new Context();
        using var fit = SeriesFit(context);
        Assert.Throws<ArgumentOutOfRangeException>(() => context.Simulate(fit, new SimulationOptions(0, 1, 1)));
        Assert.Throws<ArgumentOutOfRangeException>(() => context.Simulate(fit, new SimulationOptions(1, 0, 1)));
    }

    [Fact]
    public void Regressors_of_the_horizon_follow_the_source()
    {
        using var context = new Context();
        using var fit = SeriesFit(context);
        Assert.Throws<ArgumentException>(() => context.Simulate(fit, new SimulationOptions(5, 1, 1) { Mxreg = new double[5, 1] }));
        var spec = new Spec { Fixed = Stationary, Mxreg = 1 } with
        {
            Fixed = new Dictionary<string, double>(Stationary, StringComparer.Ordinal) { ["mxreg1"] = 0.5 },
        };
        Assert.Throws<ArgumentException>(() => context.Simulate(spec, new SimulationOptions(5, 1, 1)));
        var simulation = context.Simulate(spec, new SimulationOptions(5, 1, 1) { Mxreg = new double[5, 1] });
        Assert.Equal(5, simulation.Returns[0].Length);
    }

    [Fact]
    public void Oversized_simulations_are_rejected_before_allocating()
    {
        using var context = new Context();
        using var fit = SeriesFit(context);
        Assert.Throws<ArgumentOutOfRangeException>(() => context.Simulate(fit, new SimulationOptions(int.MaxValue, 2, 1)));
    }
}
