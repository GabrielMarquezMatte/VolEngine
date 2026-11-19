using System.Text.Json;

namespace VolEngine.Tests.Support;

// The univariate fits of a DCC fixture: each series fitted from the oracle's own estimates with a tolerance that
// stops the optimizer there, so the second stage starts from the margins the oracle used.
internal sealed class Univariate : IDisposable
{
    private Univariate(Context context, Fit[] fits)
    {
        Context = context;
        Fits = fits;
    }

    public Context Context { get; }

    public Fit[] Fits { get; }

    public static Univariate Dcc3()
    {
        var parameters = ReferenceData.Fixture("rmgarch_dcc.json").GetProperty("panels").GetProperty("dcc3")
            .GetProperty("univariate");
        return Make("dcc3.csv", parameters, Distribution.Norm);
    }

    public static Univariate Copula3()
    {
        return Make("copula3.csv", ReferenceData.Fixture("rmgarch_copula.json").GetProperty("univariate"), Distribution.Std);
    }

    public void Dispose()
    {
        foreach (var fit in Fits)
        {
            fit.Dispose();
        }
        Context.Dispose();
    }

    private static Univariate Make(string file, JsonElement parameters, Distribution distribution)
    {
        var columns = ReferenceData.Columns(file);
        var starts = parameters.EnumerateArray().Select(ReferenceData.Named).ToArray();
        var context = new Context();
        var options = new FitOptions { Tolerance = 1 };
        var fits = columns.Select((column, j) =>
            context.Fit(new Spec { Distribution = distribution, Start = starts[j] }, column, options)).ToArray();
        return new Univariate(context, fits);
    }
}
