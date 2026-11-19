# VolEngine C#-a: núcleo univariado do binding .NET (implementação)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Objetivo:** um projeto .NET 10 ajusta, lê, diagnostica, prevê e simula modelos univariados do VolEngine por uma API
C# idiomática, sobre as 130 funções do contrato declaradas à mão e conferidas por teste contra `volengine.abi`.

**Arquitetura:**
- `csharp/src/VolEngine/`: pacote `VolEngine` (`net10.0`). `Interop/` tem a classe interna `Native` (um
  `[LibraryImport("libvolengine")]` por função do contrato, **inclusive as do DCC**, que o C#-b usa), os `SafeHandle`
  de cada tipo opaco, a carga da biblioteca e a conversão de status em exceção. `Specs/` e `Results/` têm a API
  pública.
- A API pública espelha o Python (`python/src/volengine/api.py`, `_spec.py`, `_fit.py`): `Context`, `Spec`,
  `FitOptions`, `Panel`, `Fit`, `FitGrid`, `Forecast`, `Simulation`. Só `Context`, `Fit` e `FitGrid` seguram handle
  nativo; `Forecast`, `Simulation`, a tabela de coeficientes e os diagnósticos são records com arrays copiados (o
  handle nativo do forecast e da simulação é liberado antes de devolver).
- Testes em xUnit contra a biblioteca local (`VOLENGINE_LIB`), com os dados e fixtures de `tests/reference/`.

**Stack:** .NET SDK 10 (C# 14), `LibraryImport` + `DisableRuntimeMarshalling`, xUnit v3, BenchmarkDotNet,
Microsoft.CodeAnalysis.PublicApiAnalyzers, GitHub Actions (`actions/setup-dotnet`).

**Spec:** `docs/superpowers/specs/2026-11-19-volengine-csharp-design.md` (aprovado, dc18a10). O C#-b (DCC, cobertura
das 130 funções, AOT, NuGet, guia) é outro plano.

## Restrições globais

- Só `net10.0`. `Nullable` habilitado, `TreatWarningsAsErrors`, `GenerateDocumentationFile` (todo membro público tem
  doc XML; as regras copiam os docstrings do Python), `AllowUnsafeBlocks`, `IsAotCompatible`,
  `[assembly: DisableRuntimeMarshalling]`.
- Interface C intocada: nenhuma linha nova em `include/volengine/volengine.h`. Se algo não fecha sem mudar o header,
  parar e perguntar ao autor.
- API pública estável desde 1.1.0 (decisão do autor): todo membro público entra em `PublicAPI.Unshipped.txt`
  (PublicApiAnalyzers, RS0016/RS0017 como erro). O C#-b move tudo para `Shipped` antes da tag.
- Erros: status ≠ `VE_OK` vira `VolEngineException(Status, mensagem de ve_last_error lida logo após a chamada, na
  mesma thread)`. Argumento que o C# pode checar sozinho (horizonte < 1, contagem negativa, forma de array, nome com
  `'\0'`) é `ArgumentException`/`ArgumentOutOfRangeException`/`ArgumentNullException` antes do nativo. Objeto liberado é
  `ObjectDisposedException`.
- Arrays de saída: cópias gerenciadas, jagged (`double[]`, `double[][]`); nunca view sobre memória nativa.
- Padrões iguais aos do Python: `P = Q = 1`, ARMA (0, 0), `Distribution.Norm`, `IncludeMean = true`,
  `FitOptions { MaxIter = 500, Tolerance = 1e-6, VarInit = VarInit.Sample, StandardErrors = true }`.
- Threads: a doc XML promete só o que o header promete (`ve_context_*` exceto free, e `ve_fit_batch`, concorrentes num
  mesmo `Context`; leituras de resultados imutáveis concorrentes). Sem lock no C#.
- STYLEGUIDE nas regras independentes de linguagem: nunca `else`, retorno cedo, `if` sempre com chaves e corpo na linha
  seguinte, ≤ 60 linhas por método, ≤ 5 parâmetros, aninhamento ≤ 3, ≤ 8 arquivos `.cs` por pasta, sem
  `Utils`/`Helpers`, comentário diz o porquê e não guarda achados. Sem checador automático para C# neste plano (minha
  escolha; a revisão confere). Inglês no código; `docs/superpowers/` em português.
- Biblioteca nativa local (Windows, PowerShell, com `C:\msys64\ucrt64\bin` primeiro no PATH, porque o `ld` chamado pelo
  Bash grava `--output-def` vazio e a libstdc++ do Git mingw64 não tem os símbolos):
  `$env:PATH = "C:\msys64\ucrt64\bin;$env:PATH"; python python/scripts/build_native.py` gera
  `python/src/volengine/_lib/libvolengine.dll`.
- Testes: `$env:VOLENGINE_LIB = "$PWD\python\src\volengine\_lib\libvolengine.dll"; dotnet test csharp/VolEngine.slnx -c Release`
  (no Bash: `VOLENGINE_LIB=$PWD/python/src/volengine/_lib/libvolengine.dll PATH=/c/msys64/ucrt64/bin:$PATH dotnet test csharp/VolEngine.slnx -c Release`).
- Versões de pacote NuGet de teste/benchmark/analisador: a última estável no momento da Task 1, fixada exatamente em
  `csharp/Directory.Packages.props` (gestão central). O pacote `VolEngine` não tem dependência de runtime.
- Commits em inglês no estilo do repositório (`feat(csharp): ...`), sem `Co-Authored-By`. Push só com o ok do autor.

## Interface pública (decisões que os testes usam)

Namespace único `VolEngine` para tudo o que é público; `VolEngine.Interop` para o interno.

```csharp
public enum Status { Ok, InvalidArgument, InvalidHandle, Unsupported, OutOfMemory, InternalError }
public enum Model { SGarch, EGarch, GjrGarch, Aparch, IGarch, CsGarch, FGarchGarch, FGarchTGarch, FGarchAvGarch,
                    FGarchNGarch, FGarchNaGarch, FGarchAparch, FGarchGjrGarch, FGarchAllGarch, Ewma }
public enum Distribution { Norm, Std, Ged, Sstd, Sged, Nig, Ghyp, Jsu }
public enum VarInit { Sample, Backcast }
public enum FitStatus { Converged, MaxIter, Failed, InsufficientData }
public enum StandardErrorKind { Hessian, Robust }
public enum DccModel { Dcc, Adcc, Fdcc, Static }          // usados pelo C#-b; aqui só pelo teste do contrato
public enum DccDistribution { MvNorm, MvT, MvLaplace }
public enum Copula { None, Parametric, Empirical }
public enum StaticMethod { Default, Kendall, Ml }
internal enum Criterion { Aic, Bic, Hqic, Sic }
internal enum CriterionScale { PerObs, Total }
// Valores = os VE_* na ordem do header (todos começam em 0 e são consecutivos).

public sealed class VolEngineException : Exception { public Status Status { get; } }

public sealed class Context : IDisposable
{
    public Context(int threads = 0);                       // 0..1024; fora: ArgumentOutOfRangeException
    public int Threads { get; }
    public Fit Fit(Spec spec, ReadOnlySpan<double> series, FitOptions? options = null,
                   double[,]? mxreg = null, double[,]? vxreg = null);        // regressores T x k
    public FitGrid Fit(IReadOnlyList<Spec> specs, Panel data, FitOptions? options = null);
    public Simulation Simulate(Fit source, SimulationOptions options);
    public Simulation Simulate(Spec source, SimulationOptions options);
}

public sealed record Spec
{
    public Model Model { get; init; } = Model.SGarch;
    public int P { get; init; } = 1;
    public int Q { get; init; } = 1;
    public (int Ar, int Ma) Arma { get; init; } = (0, 0);
    public Distribution Distribution { get; init; } = Distribution.Norm;
    public bool IncludeMean { get; init; } = true;
    public IReadOnlyDictionary<string, double> Fixed { get; init; }   // vazio por padrão
    public IReadOnlyDictionary<string, double> Start { get; init; }   // vazio por padrão
    public bool VarianceTargeting { get; init; }
    public int Mxreg { get; init; }
    public int Vxreg { get; init; }
    public int Archm { get; init; }
    public bool Arfima { get; init; }
    public void Validate();                                // VolEngineException com o motivo
    // Equals/GetHashCode por valor, inclusive o conteúdo de Fixed e Start.
}

public sealed record FitOptions
{
    public int MaxIter { get; init; } = 500;
    public double Tolerance { get; init; } = 1e-6;
    public VarInit VarInit { get; init; } = VarInit.Sample;
    public bool StandardErrors { get; init; } = true;
}

public sealed class Panel                                  // só dados gerenciados, sem handle
{
    public static Panel FromColumns(double[,] data, int[]? starts = null, int[]? lengths = null,
                                    double[,,]? mxreg = null, double[,,]? vxreg = null);  // T x N, T x N x k
    public static Panel FromSeries(IReadOnlyList<double[]> series, IReadOnlyList<double[,]>? mxreg = null,
                                   IReadOnlyList<double[,]>? vxreg = null);              // len_j x k por série
    public int Rows { get; }
    public int SeriesCount { get; }
}

public sealed class FitGrid : IDisposable                  // Dispose libera todos os fits
{
    public int SpecCount { get; }
    public int SeriesCount { get; }
    public Fit this[int spec, int series] { get; }
}

public sealed class Fit : IDisposable
{
    public FitStatus Status { get; }                       // lidos no construtor; valem depois do Dispose
    public int Nobs { get; }
    public IReadOnlyList<string> Names { get; }
    public double Loglik { get; }
    public double[] Estimate();
    public CoefTable Coef(StandardErrorKind kind = StandardErrorKind.Hessian);
    public double Aic(bool perObs = true);  public double Bic(bool perObs = true);
    public double Hqic(bool perObs = true); public double Sic(bool perObs = true);
    public double[] Sigma(); public double[] Residuals(); public double[] StdResiduals(); public double[] Fitted();
    public DiagnosticResult LjungBox(int lag, bool squared = false, int fitdf = 0);
    public DiagnosticResult ArchLm(int lag, int fitdf);
    public SignBiasResult SignBias();
    public DiagnosticResult Gof(int groups = 20);
    public NyblomResult Nyblom();
    public Forecast Forecast(int horizon, double[,]? mxreg = null, double[,]? vxreg = null);  // horizon x k
}

public sealed record CoefTable(IReadOnlyList<string> Names, double[] Estimate, double[] StdError,
                               double[] TValue, double[] PValue);
public sealed record DiagnosticResult(double Statistic, double PValue);
public sealed record SignBiasResult(DiagnosticResult Sign, DiagnosticResult Negative,
                                    DiagnosticResult Positive, DiagnosticResult Joint);
public sealed record CriticalValues(double TenPercent, double FivePercent, double OnePercent);
public sealed record NyblomResult(IReadOnlyDictionary<string, double> Individual, double Joint,
                                  CriticalValues IndividualCritical, CriticalValues? JointCritical);
public sealed record Forecast(int Horizon, double[] Sigma, double[] Mean);
public sealed record SimulationOptions(int Horizon, int Paths, ulong Seed)
{
    public double[,]? Mxreg { get; init; }                 // horizon x k, iguais para todos os caminhos
    public double[,]? Vxreg { get; init; }
}
public sealed record Simulation(int Paths, int Horizon, double[][] Returns, double[][] Sigma);  // paths x horizon
```

## Review Focus

1. **Dispose concorrente com leitura** (serviço desligando): uma thread lê `fit.Sigma()` em laço enquanto outra chama
   `Dispose()`. Esperado: cada leitura termina com dados completos ou lança `ObjectDisposedException`; nunca lê memória
   liberada. Teste na Task 4.
2. **Mensagem de erro de outra thread**: muitas threads validam specs, metade inválida. Esperado: cada
   `VolEngineException` traz a mensagem da própria chamada, nunca vazia nem a de outra thread. Teste na Task 2.
3. **Séries de tamanhos diferentes** (`Panel.FromSeries`): esperado que cada fit da grade seja igual (bit a bit) ao
   fit da série sozinha. Teste na Task 5.
4. **Nome de parâmetro não ASCII** em `Fixed` (`"ômega"`): esperado `VolEngineException` cuja mensagem contém
   `"ômega"` (ida e volta UTF-8). Teste na Task 3.
5. **NaN dentro e fora do range**: NaN numa linha usada é `VolEngineException(InvalidArgument)`; NaN no enchimento de
   `FromSeries` (fora dos ranges) é aceito. Teste na Task 5.

---

### Task 1: Projeto, declarações nativas e teste do contrato

**Files:**
- Create: `csharp/VolEngine.slnx`, `csharp/Directory.Build.props`, `csharp/Directory.Packages.props`,
  `csharp/.gitignore` (`bin/`, `obj/`, `BenchmarkDotNet.Artifacts/`)
- Create: `csharp/src/VolEngine/VolEngine.csproj`, `csharp/src/VolEngine/PublicAPI.Shipped.txt` (vazio, com
  `#nullable enable`), `csharp/src/VolEngine/PublicAPI.Unshipped.txt`, `csharp/src/VolEngine/AssemblyInfo.cs`
  (`DisableRuntimeMarshalling`, `InternalsVisibleTo("VolEngine.Tests")`, `InternalsVisibleTo("VolEngine.Benchmarks")`)
- Create: `csharp/src/VolEngine/Enums.cs` (os enums públicos e internos da seção de interface)
- Create: `csharp/src/VolEngine/Interop/Native.cs` (constantes e partial), `Interop/Native.Univariate.cs` (context,
  spec, fit_options, panel, batch, fit, nyblom_critical), `Interop/Native.Results.cs` (forecast, sim_options,
  simulate, sim), `Interop/Native.Dcc.cs` (todas as `ve_dcc_*` e `ve_sim_options_set_correlations`,
  `ve_simulate_dcc`), `Interop/Handles.cs` (os 16 `SafeHandle`)
- Create: `csharp/tests/VolEngine.Tests/VolEngine.Tests.csproj`, `tests/VolEngine.Tests/ContractTests.cs`,
  `tests/VolEngine.Tests/Support/Repository.cs`, `tests/VolEngine.Tests/Support/ContractCheck.cs`

**Interfaces:**
- Produces: `internal static unsafe partial class Native` em `VolEngine.Interop`, um método por função do contrato com
  o **nome C exato** (`ve_fit_batch`) e a assinatura pela tabela abaixo; constantes `internal const int AbiVersion = 1,
  MaxThreads = 1024, MaxOrder = 6, MaxRegressors = 8`.
- Produces: `internal sealed class XHandle : SafeHandleZeroOrMinusOneIsInvalid` para cada tipo opaco, nome =
  PascalCase do tipo C sem `ve_` + `Handle` (`ve_fit_options` → `FitOptionsHandle`, `ve_dcc_sim` → `DccSimHandle`),
  com construtor público sem parâmetros (o gerador do `LibraryImport` precisa), `internal XHandle(nint pointer)` (para
  os handles que chegam em arrays, como os de `ve_fit_batch`) e `ReleaseHandle` chamando o `ve_x_free`.
- Produces (testes): `Repository.Root` (pasta que contém `include/volengine/volengine.abi`, achada subindo de
  `AppContext.BaseDirectory`), `Repository.ReadContract() -> string[]` (linhas do `.abi`),
  `ContractCheck.FunctionProblems(IEnumerable<string> lines, Type native) -> List<string>`,
  `ContractCheck.DefineProblems(IEnumerable<string> lines) -> List<string>`.

Tabela de tipos (C do contrato → C# aceito na declaração; o `ContractCheck` a implementa):

| C | C# |
| --- | --- |
| `ve_status`, `int32_t` | `int` |
| `size_t` / `ptrdiff_t` / `uint64_t` / `double` / `void` (retorno) | `nuint` / `nint` / `ulong` / `double` / `void` |
| `const char*` parâmetro | `string` (`StringMarshalling = StringMarshalling.Utf8` no atributo) |
| `const char*` retorno; `const char**` | `nint` ou `byte*`; `out nint` ou `out byte*` |
| `ve_x*`, `const ve_x*` | `XHandle`, exceto o parâmetro dos `ve_*_free`, que é `nint` (o `ReleaseHandle` não pode passar o próprio handle) |
| `ve_x**` | `out XHandle` (funções `*_create`) ou `nint*` (arrays de saída: `ve_fit_batch`, `ve_dcc_fit_batch`) |
| `const ve_x* const*` | `nint*` |
| `int32_t*`, `size_t*`, `double*` (e `const`) | `out T` ou `T*` (escalar de saída ou array) |
| `const double**` | `out double*` |

`define`: `VE_ABI_VERSION`, `VE_MAX_THREADS`, `VE_MAX_ORDER`, `VE_MAX_REGRESSORS` → constantes de `Native` (nomes
acima); `VE_OK`, `VE_INVALID_ARGUMENT`, `VE_INVALID_HANDLE`, `VE_UNSUPPORTED`, `VE_OUT_OF_MEMORY`,
`VE_INTERNAL_ERROR` → `Status`; prefixos `VE_MODEL_` → `Model`, `VE_DIST_` → `Distribution`, `VE_VARINIT_` →
`VarInit`, `VE_FIT_` → `FitStatus`, `VE_SE_` → `StandardErrorKind`, `VE_CRITERION_` → `Criterion`, `VE_IC_` →
`CriterionScale`, `VE_DCC_MODEL_` → `DccModel`, `VE_DCC_DIST_` → `DccDistribution`, `VE_DCC_COPULA_` → `Copula`,
`VE_DCC_STATIC_` → `StaticMethod`. O membro casa quando o sufixo sem `_` e em minúsculas é igual ao nome do membro em
minúsculas (`FGARCH_AVGARCH` ↔ `FGarchAvGarch`) e o valor é igual. Um `define` sem destino é problema; um membro de
enum sem `define` também.

- [ ] **Step 1: Recompilar a biblioteca nativa** (comando das restrições globais). Esperado: `libvolengine.dll` nova
  em `python/src/volengine/_lib/` (a atual pode ser anterior ao EWMA, 6e25248).

- [ ] **Step 2: Escrever os testes do contrato** em `ContractTests.cs`:

```csharp
[Fact]
public void Every_contract_function_has_a_matching_declaration()
    => Assert.Empty(ContractCheck.FunctionProblems(Repository.ReadContract(), typeof(Native)));

[Fact]
public void Every_contract_define_matches_a_member()
    => Assert.Empty(ContractCheck.DefineProblems(Repository.ReadContract()));

[Fact]
public void A_missing_declaration_is_reported()
{
    var problems = ContractCheck.FunctionProblems(["function ve_status ve_not_declared(int32_t)"], typeof(Native));
    Assert.Contains(problems, problem => problem.Contains("ve_not_declared"));
}

[Fact]
public void A_wrong_parameter_type_is_reported()
{
    var line = "function ve_status ve_context_threads(const ve_context*, double*)";
    Assert.Contains(ContractCheck.FunctionProblems([line], typeof(Native)), p => p.Contains("ve_context_threads"));
}

[Fact]
public void A_declaration_outside_the_contract_is_reported()
{
    var problems = ContractCheck.FunctionProblems(["function int32_t ve_abi_version(void)"], typeof(Native));
    Assert.Contains(problems, problem => problem.Contains("ve_fit_batch"));
}

[Fact]
public void An_unmapped_define_is_reported()
    => Assert.Contains(ContractCheck.DefineProblems(["define VE_NEW_THING 3"]), p => p.Contains("VE_NEW_THING"));
```

  `FunctionProblems` também confere o retorno e o número de parâmetros, e reporta todo método com
  `[LibraryImport]` de `Native` cujo nome não está nas linhas dadas (por isso o quinto teste acusa `ve_fit_batch`).

- [ ] **Step 3: Rodar e ver falhar.** Run: o comando de testes. Esperado: falha de compilação (`Native` não existe).

- [ ] **Step 4: Criar projetos, enums, handles e as 130 declarações.** `Directory.Build.props` com as propriedades
  das restrições globais; o `VolEngine.csproj` referencia `Microsoft.CodeAnalysis.PublicApiAnalyzers`
  (`PrivateAssets=all`); `RS0016`/`RS0017` como erro. Declarações em ordem do header, separadas pelos quatro arquivos
  listados. Preencher `PublicAPI.Unshipped.txt` com o que for público (`dotnet format analyzers csharp/VolEngine.slnx
  --diagnostics RS0016` aplica a correção). Cada enum público e cada membro com doc XML (texto curto vindo do header).

- [ ] **Step 5: Rodar e ver passar.** Esperado: 6 testes PASS, build sem warning.

- [ ] **Step 6: Commit**

```bash
git add csharp
git commit -m "feat(csharp): project, native declarations and the contract test"
```

### Task 2: Carga da biblioteca, erros e `Context`

**Files:**
- Create: `csharp/src/VolEngine/Interop/Library.cs`, `csharp/src/VolEngine/Interop/Calls.cs`,
  `csharp/src/VolEngine/VolEngineException.cs`, `csharp/src/VolEngine/Context.cs`
- Test: `csharp/tests/VolEngine.Tests/ContextTests.cs`, `csharp/tests/VolEngine.Tests/LibraryTests.cs`

**Interfaces:**
- Consumes: `Native`, `ContextHandle`, `Status` (Task 1).
- Produces: `internal static class Library { static void EnsureLoaded(); static string? ResolvePath(string?
  environmentValue); }`. `EnsureLoaded` roda uma vez (thread-safe, `Lazy`/lock): registra
  `NativeLibrary.SetDllImportResolver(typeof(Native).Assembly, ...)`, que, para o nome `"libvolengine"`, carrega o
  caminho de `ResolvePath(Environment.GetEnvironmentVariable("VOLENGINE_LIB"))` quando não nulo e devolve
  `IntPtr.Zero` (resolução padrão do NuGet) quando nulo; depois confere `ve_abi_version() == Native.AbiVersion`, senão
  `VolEngineException(Status.Unsupported, "<caminho ou libvolengine> has ABI version <n>, this package needs 1")`.
  `ResolvePath(null or "")` → `null`; caminho que não é arquivo → `DllNotFoundException` com
  `"VOLENGINE_LIB points to <valor>, which is not a file"`.
- Produces: `internal static class Calls { static void Check(int status); static string LastError(); }` — `Check`
  lança `new VolEngineException((Status)status, LastError())` quando `status != 0`; `LastError` lê
  `ve_last_error()` com `Marshal.PtrToStringUTF8` (`""` para nulo).
- Produces: `Context` como na seção de interface. O construtor chama `Library.EnsureLoaded()`. `Threads` lê
  `ve_context_threads`.

- [ ] **Step 1: Escrever os testes**

```csharp
[Fact] public void Default_threads_is_the_hardware_concurrency_capped_at_1024()
{ using var context = new Context(); Assert.Equal(Math.Min(Environment.ProcessorCount, 1024), context.Threads); }

[Fact] public void Threads_can_be_set() { using var context = new Context(2); Assert.Equal(2, context.Threads); }

[Theory, InlineData(-1), InlineData(1025)]
public void Threads_outside_0_to_1024_are_rejected(int threads)
    => Assert.Throws<ArgumentOutOfRangeException>(() => new Context(threads));

[Fact] public void A_disposed_context_throws()
{ var context = new Context(); context.Dispose(); context.Dispose();
  Assert.Throws<ObjectDisposedException>(() => context.Threads); }

[Fact] public void A_native_failure_carries_status_and_message()
{
    Library.EnsureLoaded();
    Calls.Check(Native.ve_spec_create(0, out var spec));
    using (spec)
    {
        var error = Assert.Throws<VolEngineException>(() => Calls.Check(Native.ve_spec_set_order(spec, 0, 1)));
        Assert.Equal(Status.InvalidArgument, error.Status);
        Assert.NotEmpty(error.Message);
    }
}

[Fact] public void Missing_override_falls_back_to_default_probing() => Assert.Null(Library.ResolvePath(null));

[Fact] public void An_override_that_is_not_a_file_is_reported()
{
    var error = Assert.Throws<DllNotFoundException>(() => Library.ResolvePath("C:/nowhere/libvolengine.dll"));
    Assert.Contains("VOLENGINE_LIB", error.Message);
    Assert.Contains("C:/nowhere/libvolengine.dll", error.Message);
}
```

  O teste da Review Focus 2 fica na Task 3 (precisa de `Spec.Validate`).

- [ ] **Step 2: Rodar e ver falhar** (compilação: `Context`, `Library`, `Calls` não existem).
- [ ] **Step 3: Implementar** os quatro arquivos. `Environment.ProcessorCount` é o `hardware_concurrency` no teste;
  se divergir numa máquina com afinidade restrita, trocar a asserção para `Threads >= 1` e anotar o porquê.
- [ ] **Step 4: Rodar e ver passar.**
- [ ] **Step 5: Commit** — `feat(csharp): library loading, native errors and Context`

### Task 3: `Spec` e `FitOptions`

**Files:**
- Create: `csharp/src/VolEngine/Specs/Spec.cs`, `csharp/src/VolEngine/Specs/FitOptions.cs`
- Test: `csharp/tests/VolEngine.Tests/SpecTests.cs`

**Interfaces:**
- Consumes: `Native`, `Calls`, `Library` (Tasks 1-2).
- Produces: `Spec` e `FitOptions` como na seção de interface, mais `internal SpecHandle Spec.CreateHandle()` e
  `internal FitOptionsHandle FitOptions.CreateHandle()`. `CreateHandle` chama `Library.EnsureLoaded()`, cria o handle e
  aplica, nesta ordem, `set_order`, `set_arma`, `set_dist`, `set_include_mean`, `set_mxreg`, `set_vxreg`,
  `set_archm`, `set_arfima`, `set_variance_targeting`, `set_fixed` de cada entrada, `set_start` de cada entrada; libera
  o handle se um setter falha. `Validate()` = `CreateHandle()` + `ve_spec_validate` + liberar. Nome com `'\0'` em
  `Fixed`/`Start` → `ArgumentException` antes do nativo (o marshalling UTF-8 truncaria). `FitOptions.CreateHandle`
  aplica `max_iter`, `tolerance`, `var_init`, `standard_errors`.

- [ ] **Step 1: Escrever os testes**

```csharp
[Fact] public void Defaults_match_python()
{
    var spec = new Spec();
    Assert.Equal((Model.SGarch, 1, 1, (0, 0), Distribution.Norm, true),
                 (spec.Model, spec.P, spec.Q, spec.Arma, spec.Distribution, spec.IncludeMean));
    Assert.Empty(spec.Fixed); Assert.Empty(spec.Start);
    Assert.Equal((500, 1e-6, VarInit.Sample, true),
                 (new FitOptions().MaxIter, new FitOptions().Tolerance, new FitOptions().VarInit, new FitOptions().StandardErrors));
}

[Fact] public void Equality_compares_fixed_and_start_by_content()
{
    var a = new Spec { Fixed = new Dictionary<string, double> { ["shape"] = 5 }, Distribution = Distribution.Std };
    var b = new Spec { Fixed = new Dictionary<string, double> { ["shape"] = 5 }, Distribution = Distribution.Std };
    Assert.Equal(a, b); Assert.Equal(a.GetHashCode(), b.GetHashCode());
    Assert.NotEqual(a, b with { Fixed = new Dictionary<string, double> { ["shape"] = 6 } });
}

[Fact] public void A_valid_spec_validates() => new Spec { Model = Model.Ewma }.Validate();

[Fact] public void An_unknown_fixed_name_is_rejected_with_its_name()
{
    var error = Assert.Throws<VolEngineException>(
        () => new Spec { Fixed = new Dictionary<string, double> { ["shpe"] = 5 } }.Validate());
    Assert.Equal(Status.InvalidArgument, error.Status); Assert.Contains("shpe", error.Message);
}

[Fact] public void A_non_ascii_name_round_trips_in_the_message()
{
    var error = Assert.Throws<VolEngineException>(
        () => new Spec { Fixed = new Dictionary<string, double> { ["ômega"] = 0.1 } }.Validate());
    Assert.Contains("ômega", error.Message);
}

[Fact] public void A_name_with_nul_is_an_argument_error()
    => Assert.Throws<ArgumentException>(
        () => new Spec { Start = new Dictionary<string, double> { ["a\0b"] = 1 } }.Validate());

[Theory, InlineData(0, 1), InlineData(1, 7)]
public void Orders_outside_1_to_6_are_rejected(int p, int q)
    => Assert.Equal(Status.InvalidArgument,
        Assert.Throws<VolEngineException>(() => new Spec { P = p, Q = q }.Validate()).Status);

[Fact] public void Ewma_rejects_order_2_1()
    => Assert.Throws<VolEngineException>(() => new Spec { Model = Model.Ewma, P = 2 }.Validate());

[Fact] public void Errors_on_many_threads_keep_their_own_message()
{
    var messages = new System.Collections.Concurrent.ConcurrentBag<string>();
    Parallel.For(0, 64, i =>
    {
        if (i % 2 == 0) { new Spec().Validate(); return; }
        try { new Spec { Fixed = new Dictionary<string, double> { ["bad" + i] = 1 } }.Validate(); }
        catch (VolEngineException error) { Assert.Contains("bad" + i, error.Message); messages.Add(error.Message); }
    });
    Assert.Equal(32, messages.Count);
}
```

- [ ] **Step 2: Rodar e ver falhar.**
- [ ] **Step 3: Implementar.** `Fixed`/`Start` padrão = dicionário vazio imutável compartilhado; o `Equals` compara
  os dois por conteúdo (mesmas chaves, mesmos valores) e o `GetHashCode` combina as entradas em ordem de chave.
  A doc XML de `Spec` copia as regras do docstring de `Spec` em `python/src/volengine/_spec.py` (ordem p/q, papéis
  dos parâmetros, EWMA, distribuições, variance targeting, regressores, ARFIMA, recusas), trocando as strings por
  membros de enum.
- [ ] **Step 4: Rodar e ver passar.**
- [ ] **Step 5: Commit** — `feat(csharp): Spec and FitOptions`

### Task 4: Ajuste de uma série e leitura do `Fit`

**Files:**
- Create: `csharp/src/VolEngine/Results/Fit.cs`, `csharp/src/VolEngine/Results/CoefTable.cs`,
  `csharp/src/VolEngine/Interop/Batch.cs` (montagem e execução de um lote a partir de handles e ponteiros)
- Modify: `csharp/src/VolEngine/Context.cs` (sobrecarga de uma série)
- Test: `csharp/tests/VolEngine.Tests/FitTests.cs`, `csharp/tests/VolEngine.Tests/Support/ReferenceData.cs`

**Interfaces:**
- Consumes: `Spec.CreateHandle`, `FitOptions.CreateHandle`, `Calls.Check` (Tasks 2-3).
- Produces: `internal static unsafe class Batch { static Fit[] Run(ContextHandle context, IReadOnlyList<Spec> specs,
  FitOptions? options, in PinnedPanel panel); }` e
  `internal readonly unsafe record struct PinnedPanel(double* Data, int Rows, int Series, nuint[]? Starts,
  nuint[]? Lengths, double* Mxreg, int MxregWidth, double* Vxreg, int VxregWidth)`: dados T × N em ordem C;
  regressores T × N × k em ordem C (ponteiro nulo e largura 0 quando ausentes). Uma série com regressores `double[,]`
  T × k é o caso N = 1 do mesmo layout. `Run` cria panel, ranges, batch, jobs em ordem spec-major (job `i * N + j` =
  spec `i`, série `j`, como `set_jobs` do Python), um panel de regressor por série (`data = X + j * k`, `t = T`,
  `n = k`, strides `(N * k, 1)`), entrada `NULL` nos jobs de specs com contagem 0, chama `ve_fit_batch` e embrulha cada
  `nint` num `FitHandle`. Handles temporários liberados em `finally`; quem chama mantém os arrays fixados (`fixed`)
  durante o `Run`.
- Produces: `Fit` como na seção de interface (`internal Fit(FitHandle handle)` lê status, nobs, nomes e loglik no
  construtor). Toda leitura de view (`Sigma`, `Residuals`, `StdResiduals`, `Fitted`) faz `DangerousAddRef` antes da
  chamada e `DangerousRelease` depois da cópia (o ponteiro só vale enquanto o handle vive; isso cobre a Review Focus 1).
  `ve_fit_param_name` com `Marshal.PtrToStringUTF8`. `Aic/Bic/Hqic/Sic` usam `Criterion` e `CriterionScale`.
- Produces (testes): `ReferenceData.Series(string file, string column) -> double[]` (CSV de `tests/reference/data`,
  cabeçalho pulado, células vazias descartadas, como `read_columns` do Python) e `ReferenceData.Fixture(string file)
  -> JsonElement` (raiz de `tests/reference/fixtures/<file>`).
- Context: `Fit(Spec, ReadOnlySpan<double>, FitOptions?, double[,]? mxreg, double[,]? vxreg)` — painel T × 1;
  regressores com `GetLength(0) == T` e `GetLength(1) == spec.Mxreg` (ou `Vxreg`), senão `ArgumentException` com as
  regras de `regressor_array` do Python: dado sem a spec ter → "was given but no spec has", spec com e dado ausente →
  "is required".

- [ ] **Step 1: Escrever os testes**

```csharp
[Fact] public void Sgarch11_norm_matches_rugarch()
{
    var result = ReferenceData.Fixture("rugarch.json").GetProperty("results").EnumerateArray()
        .Single(item => item.GetProperty("case").GetString() == "t5000/sgarch11_norm");
    using var context = new Context();
    using var fit = context.Fit(new Spec(), ReferenceData.Series("sgarch11_norm.csv", "t5000"),
                                new FitOptions { Tolerance = 1e-7 });
    Assert.Equal(FitStatus.Converged, fit.Status);
    Assert.Equal(["mu", "omega", "alpha1", "beta1"], fit.Names);
    var loglik = result.GetProperty("loglik").GetDouble();
    Assert.True(fit.Loglik >= loglik - 1e-6 * Math.Abs(loglik));
    Assert.InRange(Math.Abs(fit.Loglik / loglik - 1), 0, 1e-4);
    // estimativas: |ref| < 1e-2 -> erro absoluto <= 1e-5, senão relativo <= 1e-3 (assert_params do Python)
    // SE hessiana vs "se" e robusta vs "qml_robust_se": relativo <= 5e-2
    // sigma: mesmo tamanho e max |s / ref - 1| <= 1e-3
    // Aic/Bic/Hqic/Sic(perObs: true) vs "aic"/"bic"/"hqic"/"sic": absoluto <= 1e-6
}

[Fact] public void Standard_errors_off_keeps_estimates_and_refuses_coef()
{
    var series = ReferenceData.Series("sgarch11_norm.csv", "t1500");
    using var context = new Context();
    using var with = context.Fit(new Spec(), series);
    using var without = context.Fit(new Spec(), series, new FitOptions { StandardErrors = false });
    Assert.Equal(with.Estimate(), without.Estimate());
    Assert.Equal(Status.InvalidArgument, Assert.Throws<VolEngineException>(() => without.Coef()).Status);
}

[Fact] public void An_insufficient_series_is_a_status_with_empty_series()
{
    using var context = new Context();
    using var fit = context.Fit(new Spec(), new double[] { 0.1, -0.2, 0.3 });
    Assert.Equal(FitStatus.InsufficientData, fit.Status);
    Assert.Empty(fit.Sigma()); Assert.True(double.IsNaN(fit.Loglik));
}

[Fact] public void Regressors_must_match_the_spec()
{
    var series = ReferenceData.Series("sgarch11_norm.csv", "t1500");
    using var context = new Context();
    Assert.Throws<ArgumentException>(() => context.Fit(new Spec { Mxreg = 1 }, series));
    Assert.Throws<ArgumentException>(() => context.Fit(new Spec(), series, mxreg: new double[series.Length, 1]));
    Assert.Throws<ArgumentException>(() => context.Fit(new Spec { Mxreg = 1 }, series, mxreg: new double[10, 1]));
}

[Fact] public void Disposing_while_reading_never_reads_freed_memory()
{
    using var context = new Context();
    var fit = context.Fit(new Spec(), ReferenceData.Series("sgarch11_norm.csv", "t5000"));
    var expected = fit.Sigma();
    var reader = Task.Run(() =>
    {
        for (var i = 0; i < 10_000; i++)
        {
            try { Assert.Equal(expected, fit.Sigma()); }
            catch (ObjectDisposedException) { return; }
        }
    });
    Thread.Sleep(5);
    fit.Dispose();
    reader.Wait();
    Assert.Equal(["mu", "omega", "alpha1", "beta1"], fit.Names);
    Assert.Throws<ObjectDisposedException>(() => fit.Estimate());
}
```

  Os comentários do primeiro teste viram asserções com os valores do fixture, nas tolerâncias escritas (as de
  `python/tests/oracles/test_oracles.py`).

- [ ] **Step 2: Rodar e ver falhar.**
- [ ] **Step 3: Implementar** `Batch`, `Fit`, `CoefTable` e a sobrecarga de `Context.Fit`. Doc XML de `Fit` copia o
  docstring da classe Python (status, ponto finito, `insufficient_data`) e a de `Coef` a de `FitOptions.StandardErrors`.
- [ ] **Step 4: Rodar e ver passar.**
- [ ] **Step 5: Commit** — `feat(csharp): fit one series and read the Fit`

### Task 5: Painel, grade, ranges, regressores e concorrência

**Files:**
- Create: `csharp/src/VolEngine/Specs/Panel.cs`, `csharp/src/VolEngine/Results/FitGrid.cs`
- Modify: `csharp/src/VolEngine/Context.cs` (sobrecarga de grade)
- Test: `csharp/tests/VolEngine.Tests/GridTests.cs`

**Interfaces:**
- Consumes: `Batch.Run` (Task 4).
- Produces: `Panel` e `FitGrid` como na seção de interface; `Context.Fit(IReadOnlyList<Spec>, Panel, FitOptions?)`.
  `Panel` guarda `double[,]` (T × N), `starts`/`lengths` opcionais e `double[,,]?` (T × N × k) por tipo de regressor,
  sem cópia em `FromColumns`. `FromColumns`: `starts` e `lengths` juntos ou nenhum, cada um com N entradas ≥ 0, senão
  `ArgumentException`; regressor com `GetLength(0) != T` ou `GetLength(1) != N` → `ArgumentException`.
  `FromSeries`: lista não vazia; T = maior tamanho, linhas além do tamanho de cada série preenchidas com `NaN`,
  `starts = 0`, `lengths = tamanho`; regressores por série `len_j × k` com o mesmo `k` em todas (senão
  `ArgumentException`), copiados para `double[,,]` com o mesmo enchimento. `Context.Fit` de grade: lista de specs não
  vazia e sem nulos; regressores conferidos contra as specs como no Python (`k` de toda spec com contagem > 0 igual ao
  `k` do array; dado sem spec que use; spec que use sem dado). O regressor da série `j` é passado com base
  `&x[0, j, 0]` e stride_t `N * k`, sem cópia.
- `FitGrid[i, j]` = fit da spec `i` na série `j`; índice fora → `ArgumentOutOfRangeException`.

- [ ] **Step 1: Escrever os testes**

```csharp
[Fact] public void A_grid_equals_the_single_fits()
{
    var columns = new[] { "t5000", "t3000", "t1500" }.Select(c => ReferenceData.Series("sgarch11_norm.csv", c)).ToArray();
    Spec[] specs = [new Spec(), new Spec { Distribution = Distribution.Std }];
    using var context = new Context();
    using var grid = context.Fit(specs, Panel.FromSeries(columns));
    Assert.Equal((2, 3), (grid.SpecCount, grid.SeriesCount));
    for (var i = 0; i < 2; i++)
        for (var j = 0; j < 3; j++)
        {
            using var single = context.Fit(specs[i], columns[j]);
            Assert.Equal(single.Estimate(), grid[i, j].Estimate());
            Assert.Equal(single.Nobs, grid[i, j].Nobs);
        }
}

[Fact] public void Ranges_select_rows()
    // t5000 como double[5000, 1], FromColumns(data, starts: [1000], lengths: [2000]): Estimate() e Nobs (2000) iguais
    // aos de context.Fit(new Spec(), t5000[1000..3000]).
[Fact] public void Nan_in_a_used_row_is_rejected()
    // t1500 com o valor 700 trocado por NaN, FromSeries([serie]): VolEngineException com Status.InvalidArgument.
[Fact] public void Padding_nan_outside_ranges_is_accepted()
    // FromSeries([t5000, t1500]): Rows == 5000, as duas Converged, grid[0, 1].Nobs == 1500.
[Fact] public void Grid_regressors_equal_single_fits()
    // Spec { Mxreg = 1 } e Spec(), dados t1500 e t3000 (FromSeries) com a coluna "x1" de regressors_2e.csv cortada
    // ao tamanho de cada série; fit [0, j] igual ao fit de uma série com mxreg; fit [1, j] igual ao sem regressor.
[Fact] public void Regressor_shapes_are_checked()     // k diferentes entre specs com contagem > 0 -> ArgumentException
[Fact] public void Disposing_the_grid_disposes_its_fits() // grid.Dispose(); grid[0,0].Estimate() -> ObjectDisposedException
[Fact] public void Concurrent_fits_on_one_context_match_sequential()
    // 16 Tasks fazem o mesmo Context.Fit(specs, panel) ao mesmo tempo; todos os Estimate() iguais aos de um fit sequencial.
```

  Antes de escrever `Grid_regressors_equal_single_fits`, abrir `tests/reference/data/regressors_2e.csv` e usar o nome
  real da primeira coluna (o teste diz `"x1"` como marcador).

- [ ] **Step 2: Rodar e ver falhar.**
- [ ] **Step 3: Implementar.**
- [ ] **Step 4: Rodar e ver passar.**
- [ ] **Step 5: Commit** — `feat(csharp): panels, fit grids, ranges and regressors`

### Task 6: Diagnósticos

**Files:**
- Create: `csharp/src/VolEngine/Results/Diagnostics.cs` (`DiagnosticResult`, `SignBiasResult`, `CriticalValues`,
  `NyblomResult`)
- Modify: `csharp/src/VolEngine/Results/Fit.cs` (`LjungBox`, `ArchLm`, `SignBias`, `Gof`, `Nyblom`)
- Test: `csharp/tests/VolEngine.Tests/DiagnosticsTests.cs`

**Interfaces:**
- Consumes: `Fit` (Task 4).
- Produces: os records da seção de interface. `lag`, `fitdf`, `groups` negativos → `ArgumentOutOfRangeException`; o
  resto o nativo checa. `Nyblom()`: `Individual` só com os parâmetros de `estimated == 1`, em ordem de `Names`;
  `IndividualCritical = ve_nyblom_critical(1)`; `JointCritical = ve_nyblom_critical(contagem)` quando a contagem está
  em 1..20, senão `null` (como `critical_values` do Python).

- [ ] **Step 1: Escrever os testes** contra `rugarch_diagnostics.json`, id `"t5000/sgarch11_norm"`:

```csharp
// fit: new Spec { Fixed = params do fixture }, série t5000, FitOptions { Tolerance = 1e-7 }; Status Converged.
[Fact] public void Ljung_box_matches_rugarch()
    // LjungBox(1, fitdf: 0), (2, 0), (5, 0) vs "ljung_box"; LjungBox(1|5|9, squared: true) vs "ljung_box_squared";
    // estatística relativa <= 1e-7, p-valor absoluto <= 1e-7.
[Fact] public void Arch_lm_matches_rugarch()      // ArchLm(3|5|7, fitdf: 2) vs "arch_lm", mesmas tolerâncias
[Fact] public void Sign_bias_matches_rugarch()    // Sign, Negative, Positive, Joint vs as 4 linhas de "sign_bias"
[Fact] public void Gof_matches_rugarch()          // Gof(20|30|40|50) vs "gof": estatística absoluta <= 1e-12, p <= 1e-7
[Fact] public void Nyblom_matches_rugarch()
    // new Spec { Start = params }, FitOptions { Tolerance = 1.0 }: chaves de Individual = {mu, omega, alpha1, beta1},
    // Joint relativo <= 1e-3 vs nyblom.joint, JointCritical não nulo (4 parâmetros), IndividualCritical.FivePercent > 0.
[Fact] public void Negative_lag_is_an_argument_error() // LjungBox(-1) -> ArgumentOutOfRangeException
```

- [ ] **Step 2: Rodar e ver falhar.**
- [ ] **Step 3: Implementar.** Doc XML copia os docstrings de `ljung_box`, `arch_lm`, `nyblom`, `sign_bias`, `gof`
  de `_fit.py`.
- [ ] **Step 4: Rodar e ver passar.**
- [ ] **Step 5: Commit** — `feat(csharp): residual diagnostics and Nyblom`

### Task 7: Forecast e simulação

**Files:**
- Create: `csharp/src/VolEngine/Results/Forecast.cs`, `csharp/src/VolEngine/Results/Simulation.cs`,
  `csharp/src/VolEngine/Specs/SimulationOptions.cs`, `csharp/src/VolEngine/Interop/Futures.cs` (panel de regressores
  do horizonte a partir de `double[,]`, compartilhado pelo forecast e pela simulação)
- Modify: `csharp/src/VolEngine/Results/Fit.cs` (`Forecast`), `csharp/src/VolEngine/Context.cs` (`Simulate`)
- Test: `csharp/tests/VolEngine.Tests/ForecastTests.cs`, `csharp/tests/VolEngine.Tests/SimulationTests.cs`

**Interfaces:**
- Consumes: `Fit`, `Spec.CreateHandle`, `Batch` (Tasks 3-5).
- Produces: `Forecast`, `SimulationOptions`, `Simulation` como na seção de interface. `Fit.Forecast`: `horizon < 1` →
  `ArgumentOutOfRangeException`; contagem de regressores do fit = nomes que começam com `"mxreg"`/`"vxreg"` (como
  `_future` do Python); regressor deve ser `horizon × k`, ausente quando `k = 0` e presente quando `k > 0`
  (`ArgumentException` com as mensagens de `future_panel` do Python). Lê `sigma`/`mean`, copia e libera o
  `ForecastHandle` antes de devolver. `Simulate`: `Horizon`/`Paths < 1` → `ArgumentOutOfRangeException`; contagem
  de regressores vem do `Spec` (`Mxreg`/`Vxreg`) ou dos nomes do `Fit`; cria `ve_sim_options`, aplica regressores,
  chama `ve_simulate_fit`/`ve_simulate_spec`, copia `returns` e `sigma` (row-major, linha = caminho) para
  `double[paths][horizon]` e libera os handles.

- [ ] **Step 1: Escrever os testes**

```csharp
[Fact] public void Forecast_matches_rugarch()
    // rugarch_forecast.json, case "t5000/sgarch11_norm": Spec { Fixed = "fixed" }, primeiros "n" (4975) valores de t5000,
    // FitOptions { VarInit = VarInit.Sample }; Forecast("horizon" = 25): Sigma relativo <= 1e-8, Mean absoluto <= 1e-8.
[Fact] public void Forecast_horizon_must_be_positive()   // Forecast(0) -> ArgumentOutOfRangeException
[Fact] public void Forecast_regressors_follow_the_fit()  // fit sem mxreg + Forecast(5, mxreg: ...) -> ArgumentException
[Fact] public void Same_seed_gives_same_paths()          // dois Simulate(fit, new(20, 4, 7)) iguais elemento a elemento
[Fact] public void Paths_and_steps_are_prefixes()
    // Simulate(fit, new(10, 3, 7)) == primeiros 3 caminhos e primeiros 10 passos de Simulate(fit, new(20, 5, 7))
[Fact] public void Thread_count_does_not_change_paths()  // Context(1) e Context(4): mesmos Returns e Sigma
[Fact] public void A_fixed_spec_simulates_from_its_unconditional_state()
    // Spec { Fixed = { mu 0, omega 0.01, alpha1 0.05, beta1 0.9 } }: Returns 2 x 50, todos finitos, Sigma > 0
[Fact] public void A_spec_with_a_free_parameter_cannot_simulate() // VolEngineException, Status.InvalidArgument
[Fact] public void Ewma_spec_cannot_simulate()                     // Spec { Model = Ewma, Fixed = { lambda 0.94, mu 0 } }
[Fact] public void Simulation_sizes_must_be_positive()             // new(0, 1, 1) e new(1, 0, 1) -> ArgumentOutOfRangeException
```

- [ ] **Step 2: Rodar e ver falhar.**
- [ ] **Step 3: Implementar.** Doc XML: `Fit.Forecast` e `Forecast.Sigma` copiam o docstring de `forecast` e o
  significado de sigma em eGARCH/apARCH/fGARCH (header, seção Forecast); `Context.Simulate` copia o de `simulate`
  (regra da seed, prefixos, versão).
- [ ] **Step 4: Rodar e ver passar.**
- [ ] **Step 5: Commit** — `feat(csharp): forecast and simulation`

### Task 8: CI do binding

**Files:**
- Create: `.github/workflows/csharp.yml`

**Interfaces:**
- Consumes: `python/scripts/build_native.py` (como o `cpp.yml`), a solução `csharp/VolEngine.slnx`.

Conteúdo: gatilhos `push` em `master` e `pull_request` com `paths` `csharp/**`, `include/**`, `src/**`,
`CMakeLists.txt`, `python/scripts/build_native.py`, `tests/reference/**`, `.github/workflows/csharp.yml`;
`concurrency` e `permissions: contents: read` como o `cpp.yml`. Três jobs, cada um: checkout, compilar o nativo
exatamente como o job correspondente do `cpp.yml` (linux GCC 14, macOS com `MACOSX_DEPLOYMENT_TARGET: '11.0'`,
Windows MSYS2 UCRT64 + `uv run --no-project --python 3.14 python python/scripts/build_native.py`),
`actions/setup-dotnet@v4` com `dotnet-version: 10.0.x`, depois
`dotnet test csharp/VolEngine.slnx -c Release` com `VOLENGINE_LIB` apontando para
`python/src/volengine/_lib/libvolengine.{so,dylib,dll}`.

- [ ] **Step 1: Escrever o workflow.**
- [ ] **Step 2: Conferir a sintaxe**: `python -c "import yaml,sys; yaml.safe_load(open('.github/workflows/csharp.yml'))"`
  (instalar `pyyaml` num venv temporário se faltar). Esperado: sem erro. Rodar localmente o mesmo `dotnet test`
  (restrições globais). A execução real fica para o push, que precisa do ok do autor; registrar isso no relatório.
- [ ] **Step 3: Commit** — `ci(csharp): build and test the binding on three systems`

### Task 9: Benchmark do binding contra chamadas nativas diretas

**Files:**
- Create: `csharp/benchmarks/VolEngine.Benchmarks/VolEngine.Benchmarks.csproj`, `.../Program.cs`,
  `.../BindingOverhead.cs`
- Modify: `csharp/VolEngine.slnx`, `docs/performance/benchmarks.md` (seção nova depois de "Plan 6b")

**Interfaces:**
- Consumes: API pública e `Native` (InternalsVisibleTo, Task 1).

Desenho (espelha o 6b, `cpp/benchmarks/wrapper_overhead.cpp`): dois pares de benchmarks BenchmarkDotNet, cada
iteração montando specs, painel, lote e opções dos dois lados, para medir só o que o binding acrescenta.
- `FitOne` / `FitOneDirect`: sGARCH(1,1) norm na coluna `t3000` de `sgarch11_norm.csv`, SE ligados.
- `Batch` / `BatchDirect`: os 12 specs sGARCH, gjrGARCH e eGARCH (1,1) × norm, std, ged, sstd sobre 21 séries
  T = 5000 simuladas uma vez no `[GlobalSetup]` por `Context.Simulate` de um sGARCH(1,1) norm todo fixo
  (`mu 0, omega 0.02, alpha1 0.08, beta1 0.9`, seed 42), todas as threads, SE desligados.
- O lado direto usa só `Native` e `Calls.Check`, sem `Batch.Run`: a sequência de chamadas C escrita à mão, como o
  lado C do 6b. Cada iteração confere que os dois lados dão a mesma soma de log-verossimilhanças, bit a bit, e lança
  se não.

- [ ] **Step 1: Escrever o projeto e os benchmarks.**
- [ ] **Step 2: Rodar uma vez curto** para ver que funciona: `dotnet run -c Release --project
  csharp/benchmarks/VolEngine.Benchmarks -- --filter * --job dry` com `VOLENGINE_LIB`. Esperado: 4 benchmarks sem erro.
- [ ] **Step 3: Rodar para valer** (`--filter *`, job padrão do BenchmarkDotNet) com a máquina parada, e escrever em
  `docs/performance/benchmarks.md` a seção "C#-a: the C# binding against direct native calls" no formato da seção do
  6b: o que cada benchmark faz, máquina, commit, a tabela de medianas e desvios, e uma frase que diga se o binding fica
  dentro do ruído. Se o autor preferir rodar ele mesmo, deixar a tabela marcada como pendente.
- [ ] **Step 4: Commit** — `bench(csharp): binding overhead against direct native calls`
