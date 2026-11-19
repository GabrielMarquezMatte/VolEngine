using System.Globalization;
using VolEngine;

// Simulates a fully fixed sGARCH(1,1), fits it back and exits 0 only when the fit converged: the package found its
// native library by itself.
var parameters = new Dictionary<string, double>(StringComparer.Ordinal)
{
    ["mu"] = 0,
    ["omega"] = 0.02,
    ["alpha1"] = 0.08,
    ["beta1"] = 0.9,
};
using var context = new Context();
var simulation = context.Simulate(new Spec { Fixed = parameters }, new SimulationOptions(2000, 1, 42));
using var fit = context.Fit(new Spec(), simulation.Returns[0]);
Console.Out.WriteLine(string.Create(CultureInfo.InvariantCulture, $"VolEngine ABI {NativeAbi.Version}: {fit.Status}"));
return fit.Status == FitStatus.Converged ? 0 : 1;
