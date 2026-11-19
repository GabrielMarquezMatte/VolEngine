namespace VolEngine.Tests;

public sealed class LibraryTests
{
    [Fact]
    public void A_native_failure_carries_status_and_message()
    {
        Library.EnsureLoaded();
        Calls.Check(NativeMethods.ve_spec_create(0, out var spec));
        using (spec)
        {
            var error = Assert.Throws<VolEngineException>(() => Calls.Check(NativeMethods.ve_spec_set_order(spec, 0, 1)));
            Assert.Equal(Status.InvalidArgument, error.Status);
            Assert.NotEmpty(error.Message);
        }
    }

    [Fact]
    public void A_success_leaves_no_message()
    {
        Library.EnsureLoaded();
        Calls.Check(NativeMethods.ve_spec_create(0, out var spec));
        spec.Dispose();
        Assert.Equal("", Calls.LastError());
    }

    [Fact]
    public void Missing_override_falls_back_to_default_probing()
    {
        Assert.Null(Library.ResolvePath(null));
        Assert.Null(Library.ResolvePath(""));
    }

    [Fact]
    public void An_override_that_is_not_a_file_is_reported()
    {
        var error = Assert.Throws<DllNotFoundException>(() => Library.ResolvePath("C:/nowhere/libvolengine.dll"));
        Assert.Contains("VOLENGINE_LIB", error.Message, StringComparison.Ordinal);
        Assert.Contains("C:/nowhere/libvolengine.dll", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void An_override_that_is_a_file_is_its_full_path()
    {
        var path = Environment.GetEnvironmentVariable("VOLENGINE_LIB");
        Assert.False(string.IsNullOrEmpty(path), "the tests run against VOLENGINE_LIB");
        Assert.Equal(Path.GetFullPath(path), Library.ResolvePath(path));
    }

    [Fact]
    public void Native_abi_is_one()
    {
        Assert.Equal(1, NativeAbi.Version);
    }
}
