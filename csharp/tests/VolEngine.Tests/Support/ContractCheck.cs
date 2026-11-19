using System.Globalization;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Text.RegularExpressions;

namespace VolEngine.Tests.Support;

// Compares the hand-written declarations of Native and the C# enums with the lines of include/volengine/volengine.abi.
// The C to C# type table here is the one the plan fixes; a declaration may use any C# type the table accepts.
internal static partial class ContractCheck
{
    private static readonly Dictionary<string, Type> Scalars = new(StringComparer.Ordinal)
    {
        ["ve_status"] = typeof(int),
        ["int32_t"] = typeof(int),
        ["size_t"] = typeof(nuint),
        ["ptrdiff_t"] = typeof(nint),
        ["uint64_t"] = typeof(ulong),
        ["double"] = typeof(double),
    };

    private static readonly string[] StatusNames =
        ["VE_OK", "VE_INVALID_ARGUMENT", "VE_INVALID_HANDLE", "VE_UNSUPPORTED", "VE_OUT_OF_MEMORY", "VE_INTERNAL_ERROR"];

    private static readonly Dictionary<string, string> Constants = new(StringComparer.Ordinal)
    {
        ["VE_ABI_VERSION"] = "AbiVersion",
        ["VE_MAX_THREADS"] = "MaxThreads",
        ["VE_MAX_ORDER"] = "MaxOrder",
        ["VE_MAX_REGRESSORS"] = "MaxRegressors",
    };

    private static readonly (string Prefix, Type Enum)[] Prefixes =
    [
        ("VE_DCC_MODEL_", typeof(DccModel)),
        ("VE_DCC_DIST_", typeof(DccDistribution)),
        ("VE_DCC_COPULA_", typeof(Copula)),
        ("VE_DCC_STATIC_", typeof(StaticMethod)),
        ("VE_MODEL_", typeof(Model)),
        ("VE_DIST_", typeof(Distribution)),
        ("VE_VARINIT_", typeof(VarInit)),
        ("VE_FIT_", typeof(FitStatus)),
        ("VE_SE_", typeof(StandardErrorKind)),
        ("VE_CRITERION_", typeof(Criterion)),
        ("VE_IC_", typeof(CriterionScale)),
    ];

    [GeneratedRegex(@"^function (?<ret>.+?) (?<name>ve_\w+)\((?<params>.*)\)$", RegexOptions.None, 1000)]
    private static partial Regex FunctionLine();

    [GeneratedRegex(@"^define (?<name>VE_\w+) (?<value>-?\d+)$", RegexOptions.None, 1000)]
    private static partial Regex DefineLine();

    [GeneratedRegex(@"\bconst\b", RegexOptions.None, 1000)]
    private static partial Regex ConstWord();

    public static List<string> FunctionProblems(IEnumerable<string> lines, Type native)
    {
        var declared = native.GetMethods(BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic)
            .Where(method => method.GetCustomAttribute<LibraryImportAttribute>() is not null)
            .ToDictionary(method => method.Name, StringComparer.Ordinal);
        var problems = new List<string>();
        var seen = new HashSet<string>(StringComparer.Ordinal);
        foreach (var match in lines.Select(line => FunctionLine().Match(line)).Where(match => match.Success))
        {
            var name = match.Groups["name"].Value;
            seen.Add(name);
            if (!declared.TryGetValue(name, out var method))
            {
                problems.Add($"{name} is in the contract but not declared");
                continue;
            }
            problems.AddRange(SignatureProblems(method, match, native.Assembly));
        }
        problems.AddRange(declared.Keys.Where(name => !seen.Contains(name))
            .Select(name => $"{name} is declared but not in the contract"));
        return problems;
    }

    public static List<string> DefineProblems(IEnumerable<string> lines)
    {
        var problems = new List<string>();
        var matched = new HashSet<(Type, string)>();
        foreach (var match in lines.Select(line => DefineLine().Match(line)).Where(match => match.Success))
        {
            var value = int.Parse(match.Groups["value"].Value, CultureInfo.InvariantCulture);
            var problem = DefineProblem(match.Groups["name"].Value, value, matched);
            if (problem is not null)
            {
                problems.Add(problem);
            }
        }
        foreach (var type in Prefixes.Select(entry => entry.Enum).Append(typeof(Status)))
        {
            problems.AddRange(Enum.GetNames(type).Where(member => !matched.Contains((type, member)))
                .Select(member => $"{type.Name}.{member} has no define"));
        }
        return problems;
    }

    private static string? DefineProblem(string name, int value, HashSet<(Type, string)> matched)
    {
        if (Constants.TryGetValue(name, out var field))
        {
            var constant = typeof(NativeMethods).GetField(field, BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic);
            return Equals(constant?.GetRawConstantValue(), value) ? null : $"{name} = {value} but NativeMethods.{field} differs";
        }
        if (StatusNames.Contains(name, StringComparer.Ordinal))
        {
            return MemberProblem(name, name["VE_".Length..], value, typeof(Status), matched);
        }
        foreach (var (prefix, type) in Prefixes.Where(entry => name.StartsWith(entry.Prefix, StringComparison.Ordinal)))
        {
            return MemberProblem(name, name[prefix.Length..], value, type, matched);
        }
        return $"{name} has no C# destination";
    }

    private static string? MemberProblem(string name, string suffix, int value, Type type, HashSet<(Type, string)> matched)
    {
        var wanted = suffix.Replace("_", "", StringComparison.Ordinal);
        var member = Enum.GetNames(type)
            .FirstOrDefault(candidate => string.Equals(candidate, wanted, StringComparison.OrdinalIgnoreCase));
        if (member is null)
        {
            return $"{name} has no member in {type.Name}";
        }
        matched.Add((type, member));
        var actual = Convert.ToInt32(Enum.Parse(type, member), CultureInfo.InvariantCulture);
        return actual == value ? null : $"{name} = {value} but {type.Name}.{member} = {actual}";
    }

    private static IEnumerable<string> SignatureProblems(MethodInfo method, Match match, Assembly assembly)
    {
        var name = method.Name;
        var free = name.EndsWith("_free", StringComparison.Ordinal);
        var returned = match.Groups["ret"].Value;
        if (!Accepted(returned, true, free, assembly).Contains(method.ReturnType))
        {
            yield return $"{name} returns {method.ReturnType} for C {returned}";
        }
        var parameters = match.Groups["params"].Value;
        string[] cTypes = string.Equals(parameters, "void", StringComparison.Ordinal) ? [] : parameters.Split(", ");
        var actual = method.GetParameters();
        if (actual.Length != cTypes.Length)
        {
            yield return $"{name} has {actual.Length} parameters, the contract {cTypes.Length}";
            yield break;
        }
        for (var i = 0; i < cTypes.Length; i++)
        {
            if (!Accepted(cTypes[i], false, free, assembly).Contains(actual[i].ParameterType))
            {
                yield return $"{name} parameter {i} is {actual[i].ParameterType} for C {cTypes[i]}";
            }
        }
    }

    // The C# types a declaration may use for one C type of the contract.
    private static Type[] Accepted(string cType, bool isReturn, bool isFree, Assembly assembly)
    {
        var bare = ConstWord().Replace(cType, "").Replace(" ", "", StringComparison.Ordinal);
        var stars = bare.Count(character => character == '*');
        var baseName = bare.TrimEnd('*');
        if (string.Equals(baseName, "void", StringComparison.Ordinal) && stars == 0)
        {
            return [typeof(void)];
        }
        if (string.Equals(baseName, "char", StringComparison.Ordinal))
        {
            return Characters(stars, isReturn);
        }
        if (Scalars.TryGetValue(baseName, out var scalar))
        {
            return Pointers(scalar, stars);
        }
        var handle = assembly.GetType("VolEngine.Interop." + HandleName(baseName));
        if (handle is null)
        {
            return [];
        }
        if (stars == 1)
        {
            return isFree ? [typeof(nint)] : [handle];
        }
        return stars == 2 ? [handle.MakeByRefType(), typeof(nint).MakePointerType()] : [];
    }

    private static Type[] Characters(int stars, bool isReturn)
    {
        if (stars == 1)
        {
            return isReturn ? [typeof(nint), typeof(byte).MakePointerType()] : [typeof(string)];
        }
        return stars == 2 ? [typeof(nint).MakeByRefType(), typeof(byte).MakePointerType().MakeByRefType()] : [];
    }

    private static Type[] Pointers(Type scalar, int stars)
    {
        if (stars == 0)
        {
            return [scalar];
        }
        if (stars == 1)
        {
            return [scalar.MakeByRefType(), scalar.MakePointerType()];
        }
        return stars == 2 && scalar == typeof(double) ? [typeof(double).MakePointerType().MakeByRefType()] : [];
    }

    private static string HandleName(string cName)
        => string.Concat(cName["ve_".Length..].Split('_').Select(part => char.ToUpperInvariant(part[0]) + part[1..]))
           + "Handle";
}
