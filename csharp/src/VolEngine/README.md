# VolEngine

GARCH-family and DCC volatility models for .NET, on the native C++23 engine of
[VolEngine](https://github.com/GabrielMarquezMatte/VolEngine). The models follow rugarch and rmgarch: sGARCH,
eGARCH, gjrGARCH, apARCH, iGARCH, csGARCH, the fGARCH family and EWMA, ARMA/ARFIMA means with regressors, eight
innovation distributions, and DCC, aDCC, FDCC and Copula-DCC correlations. Fits run in parallel over many specs and
series in one call; results carry standard errors, information criteria, diagnostics, forecasts and simulations.

The package targets .NET 10 and ships the native library for linux-x64, linux-arm64, linux-musl-x64,
linux-musl-arm64, osx-arm64, osx-x64 and win-x64. It is Native AOT compatible.

## Install

```sh
dotnet add package VolEngine
```

## Example

```csharp
using VolEngine;

double[] returns = LoadReturns();   // your series, finite
using var context = new Context();
using var fit = context.Fit(new Spec { Model = Model.GjrGarch, Distribution = Distribution.Std }, returns);
if (fit.Status != FitStatus.Converged)
{
    return;
}
var table = fit.Coef();
for (var i = 0; i < table.Names.Count; i++)
{
    Console.WriteLine($"{table.Names[i]}: {table.Estimate[i]:F4} ({table.StdError[i]:F4})");
}
Console.WriteLine($"BIC {fit.Bic():F4}; sigma in 10 steps {fit.Forecast(10).Sigma[^1]:F4}");
```

Fits own native memory: dispose them (`using`). The
[C# guide](https://github.com/GabrielMarquezMatte/VolEngine/blob/master/docs/guide/csharp.md) covers grids over a
panel, regressors, diagnostics, simulation, DCC models and threading.
