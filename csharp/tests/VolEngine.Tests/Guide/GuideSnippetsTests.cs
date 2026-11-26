using VolEngine.Tests.Support;

namespace VolEngine.Tests;

// The code blocks of docs/guide/csharp.md, copied verbatim between the BEGIN and END lines, so the guide compiles and
// runs. A change to the guide's code goes here too.
public sealed class GuideSnippetsTests
{
    [Fact]
    public void Fit_forecast_simulate()
    {
        // BEGIN guide: Fit, forecast, simulate
        using var context = new Context();

        // Data to fit: a GARCH(1,1) with Student-t shocks, simulated by the library itself.
        var truth = new Spec
        {
            Distribution = Distribution.Std,
            Fixed = new Dictionary<string, double>(StringComparer.Ordinal)
            {
                ["mu"] = 0.0003, ["omega"] = 2e-6, ["alpha1"] = 0.08, ["beta1"] = 0.9, ["shape"] = 8,
            },
        };
        double[] returns = context.Simulate(truth, new SimulationOptions(Horizon: 3000, Paths: 1, Seed: 1)).Returns[0];

        using var fit = context.Fit(new Spec { Model = Model.GjrGarch, Distribution = Distribution.Std }, returns);
        if (fit.Status != FitStatus.Converged)
        {
            throw new InvalidOperationException("the fit did not converge");
        }
        CoefTable table = fit.Coef(StandardErrorKind.Robust);
        Forecast forecast = fit.Forecast(10);
        Console.WriteLine($"loglik {fit.Loglik:F3}, BIC {fit.Bic():F4}, sigma in 10 steps {forecast.Sigma[^1]:F5}");
        Console.WriteLine($"{table.Names[1]} = {table.Estimate[1]:F6} ({table.StdError[1]:F6})");
        // END guide
        Assert.Equal(10, forecast.Sigma.Length);
    }

    [Fact]
    public void Many_series_and_specs()
    {
        using var context = new Context();
        var columns = ReferenceData.Columns("dcc3.csv");
        var panel = new double[columns[0].Length, columns.Length];
        for (var t = 0; t < columns[0].Length; t++)
        {
            for (var j = 0; j < columns.Length; j++)
            {
                panel[t, j] = columns[j][t];
            }
        }
        // BEGIN guide: Many series and specs
        // Three specs on every column of a T x N array, in one native batch on the context's threads.
        Spec[] specs =
        [
            new Spec(),
            new Spec { Model = Model.GjrGarch },
            new Spec { Model = Model.EGarch, Distribution = Distribution.Std },
        ];
        using FitGrid grid = context.Fit(specs, Panel.FromColumns(panel), new FitOptions { StandardErrors = false });
        for (var j = 0; j < grid.SeriesCount; j++)
        {
            var best = Enumerable.Range(0, grid.SpecCount).MinBy(i => grid[i, j].Bic());
            Console.WriteLine($"series {j}: spec {best}, BIC {grid[best, j].Bic():F4}");
        }
        // END guide
        Assert.Equal((3, 3), (grid.SpecCount, grid.SeriesCount));
    }

    [Fact]
    public void Diagnostics()
    {
        using var context = new Context();
        using var fit = context.Fit(new Spec(), ReferenceData.Series("sgarch11_norm.csv", "t1500"));
        // BEGIN guide: Diagnostics
        DiagnosticResult ljungBox = fit.LjungBox(lag: 5, squared: true);
        DiagnosticResult archLm = fit.ArchLm(lag: 5, fitdf: 2);
        SignBiasResult signBias = fit.SignBias();
        NyblomResult nyblom = fit.Nyblom();
        Console.WriteLine($"Ljung-Box p {ljungBox.PValue:F3}, ARCH-LM p {archLm.PValue:F3}, sign bias p {signBias.Joint.PValue:F3}");
        Console.WriteLine($"Nyblom joint {nyblom.Joint:F3}, 5% critical value {nyblom.JointCritical?.FivePercent:F3}");
        // END guide
        Assert.InRange(ljungBox.PValue, 0, 1);
    }

    [Fact]
    public void Dcc()
    {
        using var univariate = Univariate.Dcc3();
        var context = univariate.Context;
        var fits = univariate.Fits;
        // BEGIN guide: DCC
        // `fits` holds one univariate Fit per series, all of one length (at least 100).
        using DccFitBatch candidates = context.FitDcc(
            [new DccSpec(), new DccSpec { Model = DccModel.Adcc }, new DccSpec { Distribution = DccDistribution.MvT }],
            fits, standardErrors: false);
        DccFit chosen = Enumerable.Range(0, candidates.Count).Select(i => candidates[i]).MinBy(candidate => candidate.Bic())!;
        using DccFit withErrors = context.StandardErrors(chosen, fits);
        CoefTable dccTable = withErrors.Coef();
        double[][][] correlation = withErrors.Correlation();      // T x m x m
        DccForecast dccForecast = withErrors.Forecast(5);        // R, H, sigma and mean for 5 steps
        DccSimulation paths = context.Simulate(withErrors, new DccSimulationOptions(Horizon: 20, Paths: 100, Seed: 7));
        Console.WriteLine($"{dccTable.Names[0]} = {dccTable.Estimate[0]:F4}, R12 in 5 steps {dccForecast.Correlation[^1][0][1]:F3}");
        // END guide
        Assert.Equal(1500, correlation.Length);
        Assert.Equal(100, paths.Paths);
    }

    [Fact]
    public void Cancellation()
    {
        using var context = new Context();
        var data = Panel.FromSeries(ReferenceData.Columns("dcc3.csv"));
        Spec[] specs = [new Spec(), new Spec { Model = Model.GjrGarch }];
        // BEGIN guide: Cancellation
        // `specs` and `data` as in "Many series and specs": stop the grid if it takes more than two seconds.
        using var deadline = new CancellationTokenSource(TimeSpan.FromSeconds(2));
        try
        {
            using FitGrid grid = context.Fit(specs, data, cancellationToken: deadline.Token);
            Console.WriteLine($"{grid.SpecCount} specs on {grid.SeriesCount} series");
        }
        catch (OperationCanceledException error) when (deadline.IsCancellationRequested)
        {
            throw new TimeoutException("the grid took more than two seconds", error);
        }
        // END guide
        Assert.False(deadline.IsCancellationRequested);
    }
}
