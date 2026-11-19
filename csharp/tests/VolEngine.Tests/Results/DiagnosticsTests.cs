using System.Text.Json;
using VolEngine.Tests.Support;

namespace VolEngine.Tests;

// Against rugarch 1.5.6 (tests/reference/fixtures/rugarch_diagnostics.json) at rugarch's own coefficients, with the
// tolerances of the Python oracle suite.
public sealed class DiagnosticsTests
{
    private const string CaseId = "t5000/sgarch11_norm";

    private static JsonElement Expected()
    {
        return ReferenceData.Result("rugarch_diagnostics.json", "id", CaseId);
    }

    private static Fit FixedFit(Context context)
    {
        var spec = new Spec { Fixed = ReferenceData.Named(Expected().GetProperty("params")) };
        var fit = context.Fit(spec, ReferenceData.Series("sgarch11_norm.csv", "t5000"), new FitOptions { Tolerance = 1e-7 });
        Assert.Equal(FitStatus.Converged, fit.Status);
        return fit;
    }

    private static void AssertTest(DiagnosticResult actual, JsonElement expected, double relative)
    {
        var statistic = expected[0].GetDouble();
        Assert.True(Math.Abs(actual.Statistic - statistic) <= relative * Math.Abs(statistic),
                    $"statistic {actual.Statistic} vs {statistic}");
        Assert.Equal(expected[1].GetDouble(), actual.PValue, 1e-7);
    }

    [Fact]
    public void Ljung_box_matches_rugarch()
    {
        using var context = new Context();
        using var fit = FixedFit(context);
        var expected = Expected();
        DiagnosticResult[] plain = [fit.LjungBox(1, fitdf: 0), fit.LjungBox(2, fitdf: 0), fit.LjungBox(5, fitdf: 0)];
        DiagnosticResult[] squared = [fit.LjungBox(1, squared: true), fit.LjungBox(5, squared: true), fit.LjungBox(9, squared: true)];
        for (var i = 0; i < 3; i++)
        {
            AssertTest(plain[i], expected.GetProperty("ljung_box")[i], 1e-7);
            AssertTest(squared[i], expected.GetProperty("ljung_box_squared")[i], 1e-7);
        }
    }

    [Fact]
    public void Arch_lm_matches_rugarch()
    {
        using var context = new Context();
        using var fit = FixedFit(context);
        int[] lags = [3, 5, 7];
        for (var i = 0; i < lags.Length; i++)
        {
            AssertTest(fit.ArchLm(lags[i], fitdf: 2), Expected().GetProperty("arch_lm")[i], 1e-7);
        }
    }

    [Fact]
    public void Sign_bias_matches_rugarch()
    {
        using var context = new Context();
        using var fit = FixedFit(context);
        var result = fit.SignBias();
        DiagnosticResult[] tests = [result.Sign, result.Negative, result.Positive, result.Joint];
        for (var i = 0; i < tests.Length; i++)
        {
            AssertTest(tests[i], Expected().GetProperty("sign_bias")[i], 1e-7);
        }
    }

    [Fact]
    public void Gof_matches_rugarch()
    {
        using var context = new Context();
        using var fit = FixedFit(context);
        int[] groups = [20, 30, 40, 50];
        for (var i = 0; i < groups.Length; i++)
        {
            var expected = Expected().GetProperty("gof")[i];
            var actual = fit.Gof(groups[i]);
            Assert.Equal(expected[0].GetDouble(), actual.Statistic, 1e-12);
            Assert.Equal(expected[1].GetDouble(), actual.PValue, 1e-7);
        }
    }

    [Fact]
    public void Nyblom_matches_rugarch()
    {
        var expected = Expected();
        var spec = new Spec { Start = ReferenceData.Named(expected.GetProperty("params")) };
        using var context = new Context();
        using var fit = context.Fit(spec, ReferenceData.Series("sgarch11_norm.csv", "t5000"), new FitOptions { Tolerance = 1.0 });
        var result = fit.Nyblom();
        Assert.Equal(["mu", "omega", "alpha1", "beta1"], result.Individual.Keys, StringComparer.Ordinal);
        var joint = expected.GetProperty("nyblom").GetProperty("joint").GetDouble();
        Assert.True(Math.Abs(result.Joint / joint - 1) <= 1e-3, $"joint {result.Joint} vs {joint}");
        Assert.NotNull(result.JointCritical);
        Assert.True(result.IndividualCritical.FivePercent > 0);
        Assert.True(result.IndividualCritical.TenPercent < result.IndividualCritical.FivePercent);
        Assert.True(result.IndividualCritical.FivePercent < result.IndividualCritical.OnePercent);
    }

    [Fact]
    public void Nyblom_skips_fixed_parameters()
    {
        var spec = new Spec { Fixed = new Dictionary<string, double>(StringComparer.Ordinal) { ["mu"] = 0.04 } };
        using var context = new Context();
        using var fit = context.Fit(spec, ReferenceData.Series("sgarch11_norm.csv", "t1500"));
        Assert.Equal(["omega", "alpha1", "beta1"], fit.Nyblom().Individual.Keys, StringComparer.Ordinal);
    }

    [Fact]
    public void Negative_arguments_are_argument_errors()
    {
        using var context = new Context();
        using var fit = context.Fit(new Spec(), ReferenceData.Series("sgarch11_norm.csv", "t1500"));
        Assert.Throws<ArgumentOutOfRangeException>(() => fit.LjungBox(-1));
        Assert.Throws<ArgumentOutOfRangeException>(() => fit.LjungBox(5, fitdf: -1));
        Assert.Throws<ArgumentOutOfRangeException>(() => fit.ArchLm(-3, fitdf: 2));
        Assert.Throws<ArgumentOutOfRangeException>(() => fit.Gof(-20));
    }

    [Fact]
    public void Arguments_the_native_library_rejects_are_volengine_errors()
    {
        using var context = new Context();
        using var fit = context.Fit(new Spec(), ReferenceData.Series("sgarch11_norm.csv", "t1500"));
        Assert.Equal(Status.InvalidArgument, Assert.Throws<VolEngineException>(() => fit.LjungBox(0)).Status);
        Assert.Equal(Status.InvalidArgument, Assert.Throws<VolEngineException>(() => fit.Gof(1)).Status);
    }
}
