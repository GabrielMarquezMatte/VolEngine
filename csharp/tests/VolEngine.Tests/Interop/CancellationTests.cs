using System.Security.Cryptography;
using VolEngine.Tests.Support;

namespace VolEngine.Tests;

public sealed class CancellationTests
{
    // Grids many times the work that fits in SignalDelay, so a cancel at SignalDelay lands inside the call.
    private const int LongGrid = 500;
    private const int LongDccGrid = 100;
    private static readonly TimeSpan SignalDelay = TimeSpan.FromMilliseconds(50);

    private static readonly Dictionary<string, double> Stationary = new(StringComparer.Ordinal)
    {
        ["mu"] = 0,
        ["omega"] = 0.01,
        ["alpha1"] = 0.05,
        ["beta1"] = 0.9,
    };

    private static readonly Spec StudentT = new()
    {
        Distribution = Distribution.Std,
        Fixed = new Dictionary<string, double>(Stationary, StringComparer.Ordinal) { ["shape"] = 6 },
    };

    // 400 million steps, many times the work that fits in SignalDelay even on a fast runner (10 million finished before
    // it on one); the single checkpoint keeps only one sum per path, so the memory stays small.
    private static readonly SimulationOptions LongSimulation =
        new(Horizon: 2000, Paths: 200_000, Seed: 1) { Checkpoints = [2000] };

    private static double[] Series()
    {
        return ReferenceData.Series("sgarch11_norm.csv", "t1500");
    }

    // The univariate fits of 30 independent Student-t GARCH series of 4000 days: a DCC fit with standard errors, or the
    // standard errors alone, take seconds on them.
    private static Fit[] ManySeries(Context context)
    {
        var series = context.Simulate(StudentT, new SimulationOptions(Horizon: 4000, Paths: 30, Seed: 3)).Returns;
        return series.Select(column => context.Fit(new Spec(), column)).ToArray();
    }

    // The token is cancelled SignalDelay into the call, which the work is sized never to finish before.
    private static void AssertCancelledMidRun(Func<CancellationToken, object> call)
    {
        using var source = new CancellationTokenSource(SignalDelay);
        var error = Assert.Throws<OperationCanceledException>(() => call(source.Token));
        Assert.Equal(source.Token, error.CancellationToken);
    }

    private static void DisposeAll(IEnumerable<Fit> fits)
    {
        foreach (var fit in fits)
        {
            fit.Dispose();
        }
    }

    [Fact]
    public void Canceled_token_throws_OperationCanceledException()
    {
        using var univariate = Univariate.Dcc3();
        var context = univariate.Context;
        var fits = univariate.Fits;
        using var dcc = context.FitDcc(new DccSpec(), fits, standardErrors: false);
        var series = Series();
        var options = new SimulationOptions(5, 3, 1);
        using var source = new CancellationTokenSource();
        source.Cancel();
        var token = source.Token;
        Func<object>[] calls =
        [
            () => context.Fit(new Spec(), series, cancellationToken: token),
            () => context.Fit([new Spec()], Panel.FromSeries([series]), cancellationToken: token),
            () => context.FitDcc(new DccSpec(), fits, cancellationToken: token),
            () => context.FitDcc([new DccSpec(), new DccSpec()], fits, cancellationToken: token),
            () => context.StandardErrors(dcc, fits, token),
            () => context.Simulate(fits[0], options, token),
            () => context.Simulate(new Spec { Fixed = Stationary }, options, token),
            () => context.Simulate(dcc, new DccSimulationOptions(5, 3, 1), token),
        ];
        foreach (var call in calls)
        {
            var error = Assert.Throws<OperationCanceledException>(call);
            Assert.Equal(token, error.CancellationToken);
        }
    }

    [Fact]
    public void Cancel_during_a_long_fit_stops_it()
    {
        using var context = new Context(2);
        var series = ReferenceData.Series("sgarch11_norm.csv", "t5000");
        var specs = Enumerable.Range(0, LongGrid)
            .Select(_ => new Spec { Model = Model.GjrGarch, Distribution = Distribution.Std })
            .ToArray();
        var panel = Panel.FromSeries([series]);
        using var source = new CancellationTokenSource(SignalDelay);
        var error = Assert.Throws<OperationCanceledException>(() => context.Fit(specs, panel, cancellationToken: source.Token));
        Assert.Equal(source.Token, error.CancellationToken);
        using var fit = context.Fit(new Spec(), series);
        Assert.Equal(FitStatus.Converged, fit.Status);
    }

    [Fact]
    public void Cancel_during_a_long_dcc_batch_stops_it()
    {
        using var univariate = Univariate.Dcc3();
        using var context = new Context(1);
        var specs = Enumerable.Range(0, LongDccGrid).Select(_ => new DccSpec { Distribution = DccDistribution.MvT }).ToArray();
        using var source = new CancellationTokenSource(SignalDelay);
        var error = Assert.Throws<OperationCanceledException>(
            () => context.FitDcc(specs, univariate.Fits, cancellationToken: source.Token));
        Assert.Equal(source.Token, error.CancellationToken);
        using var dcc = context.FitDcc(new DccSpec(), univariate.Fits, standardErrors: false);
        Assert.Equal(FitStatus.Converged, dcc.Status);
    }

    [Fact]
    public void Cancel_during_a_dcc_fit_with_standard_errors_stops_it()
    {
        using var context = new Context();
        var fits = ManySeries(context);
        try
        {
            DccSpec[] specs = [new DccSpec(), new DccSpec { Model = DccModel.Adcc }, new DccSpec { Distribution = DccDistribution.MvT }];
            AssertCancelledMidRun(token => context.FitDcc(specs, fits, standardErrors: true, token));
        }
        finally
        {
            DisposeAll(fits);
        }
    }

    [Fact]
    public void Cancel_during_the_dcc_standard_errors_stops_them()
    {
        using var context = new Context();
        var fits = ManySeries(context);
        try
        {
            using var dcc = context.FitDcc(new DccSpec { Model = DccModel.Adcc, Distribution = DccDistribution.MvT }, fits,
                                           standardErrors: false);
            AssertCancelledMidRun(token => context.StandardErrors(dcc, fits, token));
        }
        finally
        {
            DisposeAll(fits);
        }
    }

    [Fact]
    public void Cancel_during_a_simulation_stops_it()
    {
        using var context = new Context();
        var series = context.Simulate(StudentT, new SimulationOptions(Horizon: 4000, Paths: 1, Seed: 5)).Returns[0];
        using var fit = context.Fit(new Spec { Distribution = Distribution.Std }, series);
        AssertCancelledMidRun(token => context.Simulate(fit, LongSimulation, token));
        AssertCancelledMidRun(token => context.Simulate(StudentT, LongSimulation, token));
    }

    // 20000 paths of 100 steps over 3 series: 6 million draws of a multivariate t.
    [Fact]
    public void Cancel_during_a_dcc_simulation_stops_it()
    {
        using var univariate = Univariate.Dcc3();
        var context = univariate.Context;
        using var dcc = context.FitDcc(new DccSpec { Distribution = DccDistribution.MvT }, univariate.Fits, standardErrors: false);
        AssertCancelledMidRun(token => context.Simulate(dcc, new DccSimulationOptions(Horizon: 100, Paths: 20_000, Seed: 1), token));
    }

    [Fact]
    public void Cancelable_token_not_canceled_gives_the_same_fit()
    {
        using var context = new Context();
        var series = Series();
        var options = new SimulationOptions(20, 4, 7);
        using var source = new CancellationTokenSource();
        using var plain = context.Fit(new Spec(), series, cancellationToken: CancellationToken.None);
        using var cancelable = context.Fit(new Spec(), series, cancellationToken: source.Token);
        Assert.Equal(plain.Estimate(), cancelable.Estimate());
        Assert.Equal(plain.Loglik, cancelable.Loglik);
        Assert.Equal(context.Simulate(plain, options, CancellationToken.None).Returns,
                     context.Simulate(plain, options, source.Token).Returns);
    }

    // A cancel that lands just before, during or just after the native call must give a fit or an
    // OperationCanceledException, never a crash from the callback reaching a freed handle.
    [Fact]
    public void Racing_cancellations_never_crash()
    {
        using var context = new Context(2);
        var series = Series();
        var fitted = 0;
        var canceled = 0;
        for (var i = 0; i < 200; i++)
        {
            using var source = new CancellationTokenSource();
            source.CancelAfter(TimeSpan.FromMilliseconds(RandomNumberGenerator.GetInt32(3)));
            try
            {
                using var fit = context.Fit(new Spec(), series, cancellationToken: source.Token);
                Assert.True(Enum.IsDefined(fit.Status));
                fitted++;
            }
            catch (OperationCanceledException error)
            {
                Assert.Equal(source.Token, error.CancellationToken);
                canceled++;
            }
        }
        Assert.Equal(200, fitted + canceled);
    }
}
