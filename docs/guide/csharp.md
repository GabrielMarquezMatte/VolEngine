# C# quick guide

The `VolEngine` package binds the C API for .NET 10. Results are immutable classes that own their native memory
(dispose them with `using`), specs and options are records compared by value, every array returned is a managed copy,
and errors are exceptions: `VolEngineException` (with the native `Status` and message) for what the library rejects,
`ArgumentException` and its subclasses for what the binding checks first, `ObjectDisposedException` for a result used
after `Dispose`. The public API is stable from 1.1.0 and follows semantic versioning.

## Install

```sh
dotnet add package VolEngine
```

The package carries the native library for Windows x64, macOS arm64 and x64, and Linux glibc and musl on x64 and
arm64; .NET finds it in `runtimes/<rid>/native/` with no setting. To use a library you built yourself, set the
environment variable `VOLENGINE_LIB` to its path before the first call: the binding loads that file instead, and
checks that its ABI version (`NativeAbi.Version`) is the one it was built for.

## Fit, forecast, simulate

```csharp
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
```

A fit that does not converge is a status, not an exception. The information criteria are per observation by default,
as in rugarch (`Bic(perObs: false)` gives arch's total). `Spec` has the orders (`P`, `Q`, `Arma`), the mean
(`IncludeMean`, `Archm`, `Arfima`), `Fixed` and `Start` values, `VarianceTargeting`, and the regressor counts `Mxreg`
and `Vxreg`; the regressors themselves are T x k arrays given to `Fit`, and their horizon values to `Forecast` and
`SimulationOptions`. `Context.Simulate` also takes a spec with every parameter fixed. The forecast follows `rugarch`, so it
is not the mean of the simulated paths for a gjrGARCH with a skewed distribution (P(z < 0) stands in for
E[1{z < 0} z²]) and for a csGARCH from h = 2 (the closed form leaves out the last shock); see
[validation](validation.md).

## Many series and specs

```csharp
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
```

`grid[i, j]` is spec i on series j; the grid owns its fits, so dispose the grid, not them. `Panel.FromSeries` takes
series of different lengths (one range per series), `Panel.FromColumns` takes T x N x k regressors and `Panel.FromSeries` one length x k array per series.

## Diagnostics

```csharp
DiagnosticResult ljungBox = fit.LjungBox(lag: 5, squared: true);
DiagnosticResult archLm = fit.ArchLm(lag: 5, fitdf: 2);
SignBiasResult signBias = fit.SignBias();
NyblomResult nyblom = fit.Nyblom();
Console.WriteLine($"Ljung-Box p {ljungBox.PValue:F3}, ARCH-LM p {archLm.PValue:F3}, sign bias p {signBias.Joint.PValue:F3}");
Console.WriteLine($"Nyblom joint {nyblom.Joint:F3}, 5% critical value {nyblom.JointCritical?.FivePercent:F3}");
```

`Gof(groups)` is the adjusted Pearson test. The defaults rugarch's `show()` uses are in each method's documentation.

## DCC

```csharp
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
```

Fitting without standard errors and adding them to the chosen fit is the cheap way to select among many specs. The
FDCC takes `Groups`, one label per series from 1; `Copula` makes a Copula-DCC, and `DccModel.Static` with a copula a
constant correlation. A `DccFit` keeps what it needs to forecast and simulate, so the univariate fits may be disposed
after `FitDcc`; `StandardErrors` needs them. `DccSimulationOptions { Correlations = true }` keeps R and H of every path
and step, and `Mxreg`/`Vxreg` take one entry per series (null for a series without regressors).

The `MvLaplace` density is singular where every standardized residual of a date is exactly 0 (a common holiday with
`mu` fixed at 0): the log-likelihood is then not finite and the fit ends `Failed`, with no message besides the status.
Drop such dates before the univariate fits.

## Cancellation

`Fit`, `FitDcc`, `StandardErrors` and `Simulate` take an optional `CancellationToken` as their last argument. A token
cancelled before or during the call makes it throw `OperationCanceledException` soon after (within one optimizer
iteration, one block of the standard errors or one simulated path) and no result is made; a call that finishes first
returns its result. Without a token, or with one that can never be cancelled, nothing changes. On a `Context` that
other calls keep busy, a cancelled call waits for its turn in the context's queue before it stops: give each caller
that needs a deadline its own `Context`.

```csharp
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
```

One token may go to several calls at once, on any threads, and cancelling it stops all of them.

## Threads and Dispose

A `Context` is a native thread pool; `new Context(threads)` sizes it (0 means the hardware concurrency). Results do
not depend on the thread count, and the same seed gives the same paths for any number of threads. Several threads may
fit on one context at the same time; the native library promises nothing more for one context, so any other call on
a shared context (simulations of either kind, DCC fits and DCC standard errors) needs the caller's coordination. A result may be read from
several threads at once.

`Status`, `Nobs`, `Names` and `Loglik` (and `Series` of a `DccFit`) stay readable after `Dispose`; every other member
then throws `ObjectDisposedException`. Forecasts and simulations are managed copies and need no `Dispose`.

## Native AOT

The library is trimming and AOT compatible: `dotnet publish -p:PublishAot=true -r <rid>` gives an executable that
loads the package's native library from beside it, with no warning.
