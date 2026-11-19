using System.Text.RegularExpressions;
using VolEngine.Tests.Support;

namespace VolEngine.Tests;

// What cpp/scripts/check_abi_coverage.py checks for the C++ wrapper: every function of the frozen contract is called
// by the library itself, outside the declarations.
public sealed partial class CoverageTests
{
    [GeneratedRegex(@"^function .+? (?<name>ve_\w+)\(", RegexOptions.None, 1000)]
    private static partial Regex FunctionName();

    [Fact]
    public void Every_contract_function_is_called_by_the_library()
    {
        var source = Path.Combine(Repository.Root, "csharp", "src", "VolEngine");
        var native = Path.Combine(source, "Interop", "Native");
        var code = string.Concat(Directory.EnumerateFiles(source, "*.cs", SearchOption.AllDirectories)
            .Where(file => !file.StartsWith(native, StringComparison.Ordinal))
            .Where(file => !file.Contains($"{Path.DirectorySeparatorChar}obj{Path.DirectorySeparatorChar}", StringComparison.Ordinal))
            .Select(File.ReadAllText));
        var missing = Repository.ReadContract()
            .Select(line => FunctionName().Match(line))
            .Where(match => match.Success)
            .Select(match => match.Groups["name"].Value)
            .Where(name => !Regex.IsMatch(code, $@"NativeMethods\.{name}[(,)]", RegexOptions.None, TimeSpan.FromSeconds(1)))
            .ToList();
        Assert.Empty(missing);
    }
}
