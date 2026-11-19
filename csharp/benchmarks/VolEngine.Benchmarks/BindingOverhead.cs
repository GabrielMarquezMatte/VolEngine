using System.Globalization;
using BenchmarkDotNet.Attributes;
using VolEngine.Interop;

namespace VolEngine.Benchmarks;

// What the C# binding adds over the same native calls written by hand: one univariate fit, and a batch of 12 specs
// over 21 series. Each iteration builds its specs, panel, batch and options on both sides, and checks that the sum of
// the log-likelihoods is bit for bit the one of the other side. In process: both sides run in the same process anyway,
// and the out-of-process generator scans the repository and fails on build/ (access denied to the pytest loader's
// folders there).
[InProcess]
public class BindingOverhead
{
    private static readonly Model[] Models = [Model.SGarch, Model.GjrGarch, Model.EGarch];
    private static readonly Distribution[] Distributions = [Distribution.Norm, Distribution.Std, Distribution.Ged, Distribution.Sstd];

    private Context _context = null!;
    private double[] _series = [];
    private double[,] _panel = new double[0, 0];
    private Spec[] _specs = [];
    private double _oneLoglik;
    private double _batchLoglik;

    [GlobalSetup]
    public void Setup()
    {
        _context = new Context();
        _series = ReadColumn("sgarch11_norm.csv", "t3000");
        _panel = SimulatedPanel(_context);
        _specs = Models.SelectMany(model => Distributions.Select(dist => new Spec { Model = model, Distribution = dist })).ToArray();
        _oneLoglik = DirectOne();
        _batchLoglik = DirectBatch();
        Require(BindingOne(), _oneLoglik, "one fit");
        Require(BindingBatch(), _batchLoglik, "batch");
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        _context.Dispose();
    }

    [Benchmark]
    public double FitOne()
    {
        return Require(BindingOne(), _oneLoglik, "one fit");
    }

    [Benchmark]
    public double FitOneDirect()
    {
        return Require(DirectOne(), _oneLoglik, "one fit");
    }

    [Benchmark]
    public double Batch()
    {
        return Require(BindingBatch(), _batchLoglik, "batch");
    }

    [Benchmark]
    public double BatchDirect()
    {
        return Require(DirectBatch(), _batchLoglik, "batch");
    }

    private double BindingOne()
    {
        using var fit = _context.Fit(new Spec(), _series);
        return fit.Loglik;
    }

    private double BindingBatch()
    {
        using var grid = _context.Fit(_specs, Panel.FromColumns(_panel), new FitOptions { StandardErrors = false });
        var sum = 0.0;
        for (var i = 0; i < grid.SpecCount; i++)
        {
            for (var j = 0; j < grid.SeriesCount; j++)
            {
                sum += grid[i, j].Loglik;
            }
        }
        return sum;
    }

    private unsafe double DirectOne()
    {
        fixed (double* data = _series)
        {
            return DirectFit([(Model.SGarch, Distribution.Norm)], data, _series.Length, 1, true);
        }
    }

    private unsafe double DirectBatch()
    {
        var specs = Models.SelectMany(model => Distributions.Select(dist => (model, dist))).ToArray();
        fixed (double* data = _panel)
        {
            return DirectFit(specs, data, _panel.GetLength(0), _panel.GetLength(1), false);
        }
    }

    // The C sequence a caller without the binding writes: specs, panel, jobs, options, batch, then the fits.
    private unsafe double DirectFit((Model Model, Distribution Dist)[] specs, double* data, int rows, int columns, bool standardErrors)
    {
        var specHandles = new SpecHandle[specs.Length];
        var specPointerValues = new nint[specs.Length];
        for (var i = 0; i < specs.Length; i++)
        {
            Check(NativeMethods.ve_spec_create((int)specs[i].Model, out specHandles[i]));
            Check(NativeMethods.ve_spec_set_dist(specHandles[i], (int)specs[i].Dist));
            specPointerValues[i] = specHandles[i].DangerousGetHandle();
        }
        Check(NativeMethods.ve_panel_create(data, (nuint)rows, (nuint)columns, out var panel));
        Check(NativeMethods.ve_fit_options_create(out var options));
        Check(NativeMethods.ve_fit_options_set_standard_errors(options, standardErrors ? 1 : 0));
        Check(NativeMethods.ve_batch_create(out var batch));
        var jobs = specs.Length * columns;
        var jobSpec = new nuint[jobs];
        var jobSeries = new nuint[jobs];
        for (var job = 0; job < jobs; job++)
        {
            jobSpec[job] = (nuint)(job / columns);
            jobSeries[job] = (nuint)(job % columns);
        }
        var fits = new nint[jobs];
        fixed (nint* specPointers = specPointerValues)
        fixed (nuint* specIndex = jobSpec)
        fixed (nuint* seriesIndex = jobSeries)
        fixed (nint* fitPointers = fits)
        {
            Check(NativeMethods.ve_batch_set_specs(batch, specPointers, (nuint)specs.Length));
            Check(NativeMethods.ve_batch_set_panel(batch, panel));
            Check(NativeMethods.ve_batch_set_jobs(batch, specIndex, seriesIndex, (nuint)jobs));
            Check(NativeMethods.ve_batch_set_options(batch, options));
            Check(NativeMethods.ve_fit_batch(_context.Handle, batch, fitPointers));
        }
        var sum = SumAndFree(fits);
        batch.Dispose();
        options.Dispose();
        panel.Dispose();
        foreach (var spec in specHandles)
        {
            spec.Dispose();
        }
        return sum;
    }

    private static double SumAndFree(nint[] fits)
    {
        var sum = 0.0;
        foreach (var pointer in fits)
        {
            using var fit = new FitHandle(pointer);
            Check(NativeMethods.ve_fit_loglik(fit, out var loglik));
            sum += loglik;
        }
        return sum;
    }

    private static double[,] SimulatedPanel(Context context)
    {
        const int rows = 5000;
        const int series = 21;
        var spec = new Spec
        {
            Fixed = new Dictionary<string, double>(StringComparer.Ordinal)
            {
                ["mu"] = 0, ["omega"] = 0.02, ["alpha1"] = 0.08, ["beta1"] = 0.9,
            },
        };
        var paths = context.Simulate(spec, new SimulationOptions(rows, series, 42)).Returns;
        var panel = new double[rows, series];
        for (var t = 0; t < rows; t++)
        {
            for (var j = 0; j < series; j++)
            {
                panel[t, j] = paths[j][t];
            }
        }
        return panel;
    }

    private static double[] ReadColumn(string file, string column)
    {
        var lines = File.ReadAllLines(Path.Combine(RepositoryRoot(), "tests", "reference", "data", file));
        var index = Array.IndexOf(lines[0].Split(','), column);
        var values = lines.Skip(1).Select(line => line.Split(',')[index]).Where(cell => cell.Length > 0)
            .Select(cell => double.Parse(cell, CultureInfo.InvariantCulture)).ToArray();
        if (values.Length == 0)
        {
            throw new InvalidOperationException($"{file} column {column} is empty");
        }
        return values;
    }

    // BenchmarkDotNet runs from a generated folder below the benchmark project, inside the repository.
    private static string RepositoryRoot()
    {
        for (var directory = new DirectoryInfo(AppContext.BaseDirectory); directory is not null; directory = directory.Parent)
        {
            if (File.Exists(Path.Combine(directory.FullName, "include", "volengine", "volengine.abi")))
            {
                return directory.FullName;
            }
        }
        throw new InvalidOperationException($"no repository above {AppContext.BaseDirectory}");
    }

    private static double Require(double actual, double expected, string what)
    {
        if (BitConverter.DoubleToInt64Bits(actual) != BitConverter.DoubleToInt64Bits(expected))
        {
            throw new InvalidOperationException($"{what}: binding and direct calls differ ({actual} vs {expected})");
        }
        return actual;
    }

    private static void Check(int status)
    {
        Calls.Check(status);
    }
}
