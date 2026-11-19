using VolEngine.Tests.Support;

namespace VolEngine.Tests;

public sealed class GridTests
{
    private static double[] Column(string name)
    {
        return ReferenceData.Series("sgarch11_norm.csv", name);
    }

    [Fact]
    public void A_grid_equals_the_single_fits()
    {
        double[][] columns = [Column("t5000"), Column("t3000"), Column("t1500")];
        Spec[] specs = [new Spec(), new Spec { Distribution = Distribution.Std }];
        using var context = new Context();
        using var grid = context.Fit(specs, Panel.FromSeries(columns));
        Assert.Equal((2, 3), (grid.SpecCount, grid.SeriesCount));
        for (var i = 0; i < specs.Length; i++)
        {
            for (var j = 0; j < columns.Length; j++)
            {
                using var single = context.Fit(specs[i], columns[j]);
                Assert.Equal(single.Estimate(), grid[i, j].Estimate());
                Assert.Equal(single.Nobs, grid[i, j].Nobs);
            }
        }
    }

    [Fact]
    public void Ranges_select_rows()
    {
        var series = Column("t5000");
        var data = new double[series.Length, 1];
        for (var t = 0; t < series.Length; t++)
        {
            data[t, 0] = series[t];
        }
        using var context = new Context();
        using var grid = context.Fit([new Spec()], Panel.FromColumns(data, starts: [1000], lengths: [2000]));
        using var single = context.Fit(new Spec(), series.AsSpan(1000, 2000));
        Assert.Equal(2000, grid[0, 0].Nobs);
        Assert.Equal(single.Estimate(), grid[0, 0].Estimate());
    }

    [Fact]
    public void Ranges_must_come_together_with_one_entry_per_series()
    {
        var data = new double[10, 2];
        Assert.Throws<ArgumentException>(() => Panel.FromColumns(data, starts: [0, 0]));
        Assert.Throws<ArgumentException>(() => Panel.FromColumns(data, starts: [0], lengths: [5]));
        Assert.Throws<ArgumentException>(() => Panel.FromColumns(data, starts: [0, -1], lengths: [5, 5]));
        Assert.Throws<ArgumentException>(() => Panel.FromColumns(data, mxreg: new double[9, 2, 1]));
    }

    [Fact]
    public void Nan_in_a_used_row_is_rejected()
    {
        var series = Column("t1500");
        series[700] = double.NaN;
        using var context = new Context();
        var error = Assert.Throws<VolEngineException>(() => context.Fit([new Spec()], Panel.FromSeries([series])));
        Assert.Equal(Status.InvalidArgument, error.Status);
    }

    [Fact]
    public void Padding_nan_outside_ranges_is_accepted()
    {
        var panel = Panel.FromSeries([Column("t5000"), Column("t1500")]);
        Assert.Equal((5000, 2), (panel.Rows, panel.SeriesCount));
        using var context = new Context();
        using var grid = context.Fit([new Spec()], panel);
        Assert.Equal(FitStatus.Converged, grid[0, 0].Status);
        Assert.Equal(FitStatus.Converged, grid[0, 1].Status);
        Assert.Equal(1500, grid[0, 1].Nobs);
    }

    [Fact]
    public void Grid_regressors_equal_single_fits()
    {
        var x1 = ReferenceData.Series("regressors_2e.csv", "x1");
        double[][] columns = [Column("t1500"), Column("t3000")];
        var regressors = columns.Select(series => Matrix(x1, series.Length)).ToArray();
        Spec[] specs = [new Spec { Mxreg = 1 }, new Spec()];
        using var context = new Context();
        using var grid = context.Fit(specs, Panel.FromSeries(columns, mxreg: regressors));
        for (var j = 0; j < columns.Length; j++)
        {
            using var withRegressor = context.Fit(specs[0], columns[j], mxreg: regressors[j]);
            using var without = context.Fit(specs[1], columns[j]);
            Assert.Equal(withRegressor.Estimate(), grid[0, j].Estimate());
            Assert.Equal(without.Estimate(), grid[1, j].Estimate());
        }
    }

    [Fact]
    public void Regressor_shapes_are_checked()
    {
        double[][] columns = [Column("t1500")];
        var two = new[] { new double[1500, 2] };
        using var context = new Context();
        Assert.Throws<ArgumentException>(() => context.Fit([new Spec { Mxreg = 1 }], Panel.FromSeries(columns, mxreg: two)));
        Assert.Throws<ArgumentException>(() => context.Fit([new Spec { Mxreg = 1 }], Panel.FromSeries(columns)));
        Assert.Throws<ArgumentException>(() => context.Fit([new Spec()], Panel.FromSeries(columns, mxreg: two)));
        Assert.Throws<ArgumentException>(() => Panel.FromSeries(columns, vxreg: [new double[10, 1]]));
        Assert.Throws<ArgumentException>(() => Panel.FromSeries([Column("t1500"), Column("t3000")],
                                                                 mxreg: [new double[1500, 1], new double[3000, 2]]));
    }

    [Fact]
    public void Specs_must_be_given()
    {
        var panel = Panel.FromSeries([Column("t1500")]);
        using var context = new Context();
        Assert.Throws<ArgumentException>(() => context.Fit([], panel));
        Assert.Throws<ArgumentNullException>(() => context.Fit([new Spec(), null!], panel));
        Assert.Throws<ArgumentException>(() => Panel.FromSeries([]));
    }

    [Fact]
    public void Indices_outside_the_grid_are_rejected()
    {
        using var context = new Context();
        using var grid = context.Fit([new Spec()], Panel.FromSeries([Column("t1500")]));
        Assert.Throws<ArgumentOutOfRangeException>(() => grid[1, 0]);
        Assert.Throws<ArgumentOutOfRangeException>(() => grid[0, -1]);
    }

    [Fact]
    public void Disposing_the_grid_disposes_its_fits()
    {
        using var context = new Context();
        var grid = context.Fit([new Spec()], Panel.FromSeries([Column("t1500")]));
        var fit = grid[0, 0];
        grid.Dispose();
        Assert.Throws<ObjectDisposedException>(() => fit.Estimate());
    }

    [Fact]
    public async Task Concurrent_fits_on_one_context_match_sequential()
    {
        Spec[] specs = [new Spec(), new Spec { Distribution = Distribution.Std }];
        var panel = Panel.FromSeries([Column("t3000"), Column("t1500")]);
        using var context = new Context();
        var expected = Estimates(context, specs, panel);
        var runs = Enumerable.Range(0, 16)
            .Select(_ => Task.Run(() => Estimates(context, specs, panel), TestContext.Current.CancellationToken));
        foreach (var actual in await Task.WhenAll(runs))
        {
            Assert.Equal(expected, actual);
        }
    }

    private static double[][] Estimates(Context context, Spec[] specs, Panel panel)
    {
        using var grid = context.Fit(specs, panel);
        return Enumerable.Range(0, specs.Length * panel.SeriesCount)
            .Select(job => grid[job / panel.SeriesCount, job % panel.SeriesCount].Estimate())
            .ToArray();
    }

    private static double[,] Matrix(double[] values, int rows)
    {
        var matrix = new double[rows, 1];
        for (var t = 0; t < rows; t++)
        {
            matrix[t, 0] = values[t];
        }
        return matrix;
    }

    [Fact]
    public void An_empty_regressor_is_passed_as_given()
    {
        using var context = new Context();
        using var fit = context.Fit(new Spec { Mxreg = 1 }, [], mxreg: new double[0, 1]);
        Assert.Equal(FitStatus.InsufficientData, fit.Status);
    }

    [Fact]
    public void Null_series_are_argument_null_errors()
    {
        var series = Column("t1500");
        Assert.Throws<ArgumentNullException>("series", () => Panel.FromSeries([series, null!]));
        Assert.Throws<ArgumentNullException>("mxreg", () => Panel.FromSeries([series], mxreg: [null!]));
    }

    [Fact]
    public void A_panel_without_series_is_an_argument_error()
    {
        Assert.Throws<ArgumentException>("data", () => Panel.FromColumns(new double[10, 0]));
    }
}
