namespace VolEngine.Tests.Support;

// The repository checkout the tests run from: the contract and the reference data live there, not in the build output.
internal static class Repository
{
    public static string Root { get; } = FindRoot();

    public static string[] ReadContract()
    {
        return File.ReadAllLines(Path.Combine(Root, "include", "volengine", "volengine.abi"));
    }

    private static string FindRoot()
    {
        for (var directory = new DirectoryInfo(AppContext.BaseDirectory); directory is not null; directory = directory.Parent)
        {
            if (File.Exists(Path.Combine(directory.FullName, "include", "volengine", "volengine.abi")))
            {
                return directory.FullName;
            }
        }
        throw new InvalidOperationException($"no include/volengine/volengine.abi above {AppContext.BaseDirectory}");
    }
}
