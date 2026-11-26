using BenchmarkDotNet.Attributes;

namespace VolEngine.Benchmarks;

// What the binding costs to read large results into managed arrays: a DCC simulation with R and H, a univariate
// simulation and the R path of a DCC fit, with the memory they allocate. 10 series of 2000 rows keep the DCC
// simulation near 100 MB of managed arrays on a small machine.
[InProcess]
[MemoryDiagnoser]
public class ReadOut
{
    private const int Rows = 2000;
    private const int Series = 10;

    private Context _context = null!;
    private Fit[] _univariate = [];
    private DccFit _dcc = null!;

    [GlobalSetup]
    public void Setup()
    {
        _context = new Context();
        var spec = new Spec
        {
            Fixed = new Dictionary<string, double>(StringComparer.Ordinal)
            {
                ["mu"] = 0, ["omega"] = 0.02, ["alpha1"] = 0.08, ["beta1"] = 0.9,
            },
        };
        var paths = _context.Simulate(spec, new SimulationOptions(Rows, Series, 7)).Returns;
        _univariate = paths.Select(path => _context.Fit(new Spec(), path, new FitOptions { StandardErrors = false }))
            .ToArray();
        _dcc = _context.FitDcc(new DccSpec(), _univariate, standardErrors: false);
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        _dcc.Dispose();
        foreach (var fit in _univariate)
        {
            fit.Dispose();
        }
        _context.Dispose();
    }

    [Benchmark]
    public int SimulateDccWithCorrelations()
    {
        var simulation = _context.Simulate(_dcc, new DccSimulationOptions(60, 1000, 3) { Correlations = true });
        return simulation.Correlation!.Length;
    }

    [Benchmark]
    public int SimulateFit()
    {
        return _context.Simulate(_univariate[0], new SimulationOptions(60, 10000, 3)).Returns.Length;
    }

    [Benchmark]
    public int DccCorrelation()
    {
        return _dcc.Correlation().Length;
    }
}
