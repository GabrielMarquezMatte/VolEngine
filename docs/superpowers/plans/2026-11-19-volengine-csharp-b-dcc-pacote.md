# VolEngine C#-b: DCC, pacote NuGet e documentação do binding .NET (implementação)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Objetivo:** fechar o binding C# para a tag 1.1.0: DCC/aDCC/FDCC/Copula-DCC completos (ajuste, SE, forecast,
simulação), as 130 funções chamadas pela biblioteca, um pacote `VolEngine` no nuget.org com as 7 bibliotecas nativas,
testado por um consumidor limpo e por Native AOT, a API pública congelada e o guia.

**Arquitetura:**
- Pasta nova `csharp/src/VolEngine/Dcc/` (spec, fit, lote, forecast, simulação, opções), sobre as declarações
  `NativeMethods.Dcc.cs` que o C#-a já escreveu e o teste do contrato já confere. A leitura de views e de valores por
  parâmetro, hoje privada em `Fit`, passa a ser compartilhada por `Fit` e `DccFit` (`Interop/NativeReads.cs`).
- Pacote: o `VolEngine.csproj` empacota `runtimes/<rid>/native/libvolengine.<ext>` a partir de uma pasta com os
  arquivos `volengine-native-<rid>.<ext>` que o `wheels.yml` já produz; um consumidor (`csharp/tests/VolEngine.Consumer`)
  restaura o `.nupkg` de uma pasta local, roda sem `VOLENGINE_LIB` e é publicado com `PublishAot`.
- `release.yml` ganha empacotamento, teste do pacote nos 3 SOs e publicação por Trusted Publishing.

**Stack:** .NET 10, xUnit v3 (MTP), BenchmarkDotNet, GitHub Actions (`actions/setup-dotnet`, `NuGet/login@v1`).

**Spec:** `docs/superpowers/specs/2026-11-19-volengine-csharp-design.md`. Plano anterior (mesmas restrições):
`docs/superpowers/plans/2026-11-19-volengine-csharp-a-nucleo.md`.

## Restrições globais

- Todas as do C#-a valem (seção "Restrições globais" daquele plano): `net10.0`; estilo do ExcelReader
  (`AnalysisMode=All`, Meziantou, PublicApiAnalyzers como erro, corpo em bloco, chaves em todo `if`) mais os limites do
  repositório (nunca `else`, ≤ 60 linhas por método, ≤ 5 parâmetros, aninhamento ≤ 3, ≤ 8 arquivos `.cs` por pasta);
  erros como `VolEngineException`/`Argument*Exception`/`ObjectDisposedException`; arrays de saída copiados e jagged;
  cada projeto referencia só os pacotes que usa; interface C intocada; commits sem `Co-Authored-By`; push só com ok.
- Tipos de resultado (lidos, nunca construídos pelo usuário) são `sealed class` com construtor interno e propriedades
  só de leitura (decisão da revisão do C#-a, `ResultShapeTests`); specs e opções são `sealed record` com `init`.
- Threads: o header não promete concorrência para `ve_dcc_fit_batch`, `ve_dcc_standard_errors` e `ve_simulate_dcc` num
  mesmo `ve_context`; a doc XML diz isso, sem lock no C#.
- Pacote (decisões do autor): ID `VolEngine`, versão = tag sem o `v` (`v1.1.0` → `1.1.0`, `v1.1.0-rc.1` → `1.1.0-rc.1`),
  publicação por Trusted Publishing com `NuGet/login@v1`; o username do nuget.org vem da variável de repositório
  `NUGET_USER` (o autor a cria; nada de username no repositório). A publicação no NuGet roda depois do GitHub Release,
  como a do PyPI, e respeita `dry_run`.
- RIDs e nomes: `linux-x64`, `linux-arm64`, `linux-musl-x64`, `linux-musl-arm64`, `osx-arm64`, `osx-x64`, `win-x64`;
  artefato `native-<rid>` com `volengine-native-<rid>.{so,dylib,dll}`; dentro do pacote `libvolengine.{so,dylib,dll}`.
- Testes: `dotnet test --solution csharp/VolEngine.slnx -c Release` da raiz, com `VOLENGINE_LIB` (ver C#-a).

## Interface pública nova (decisões que os testes usam)

```csharp
public sealed record DccSpec
{
    public DccModel Model { get; init; } = DccModel.Dcc;
    public (int P, int Q) Order { get; init; } = (1, 1);
    public DccDistribution Distribution { get; init; } = DccDistribution.MvNorm;
    public IReadOnlyList<int>? Groups { get; init; }                  // FDCC: um rótulo por série, de 1; copiado
    public IReadOnlyDictionary<string, double> Fixed { get; init; }   // vazio por padrão; copiado
    public IReadOnlyDictionary<string, double> Start { get; init; }
    public Copula Copula { get; init; } = Copula.None;
    public StaticMethod StaticMethod { get; init; } = StaticMethod.Default;
    // Equals/GetHashCode por valor, Groups/Fixed/Start por conteúdo; ToString mostra as entradas.
}

public sealed record DccSimulationOptions(int Horizon, int Paths, ulong Seed)
{
    public bool Correlations { get; init; }                           // guarda R e H por caminho e passo
    public IReadOnlyList<double[,]?>? Mxreg { get; init; }            // uma entrada por série, horizon x k ou null
    public IReadOnlyList<double[,]?>? Vxreg { get; init; }
}

// Context (novos membros)
public DccFit FitDcc(DccSpec spec, IReadOnlyList<Fit> univariate, bool standardErrors = true);
public DccFitBatch FitDcc(IReadOnlyList<DccSpec> specs, IReadOnlyList<Fit> univariate, bool standardErrors = true);
public DccFit StandardErrors(DccFit fit, IReadOnlyList<Fit> univariate);   // novo DccFit com SE; o original fica
public DccSimulation Simulate(DccFit source, DccSimulationOptions options);

public sealed class DccFitBatch : IDisposable { public int Count { get; } public DccFit this[int index] { get; } }

public sealed class DccFit : IDisposable
{
    public FitStatus Status { get; } public int Nobs { get; } public int Series { get; }
    public IReadOnlyList<string> Names { get; } public double Loglik { get; }    // lidos no construtor
    public double[] Estimate(); public CoefTable Coef(StandardErrorKind kind = StandardErrorKind.Hessian);
    public double Aic(bool perObs = true); public double Bic(bool perObs = true);
    public double Hqic(bool perObs = true); public double Sic(bool perObs = true);
    public double[][][] Correlation();                                 // T x m x m
    public double[][][] Covariance();                                  // T x m x m (calculada no 1º acesso, nativo)
    public DccForecast Forecast(int horizon, IReadOnlyList<double[,]?>? mxreg = null, IReadOnlyList<double[,]?>? vxreg = null);
}

public sealed class DccForecast   // Horizon, Series, Correlation/Covariance (h x m x m), Sigma/Mean (h x m)
public sealed class DccSimulation // Paths, Horizon, Series, Returns/Sigma (p x h x m), Correlation?/Covariance? (p x h x m x m, null sem Correlations)

public static class NativeAbi { public static int Version { get; } }   // ve_abi_version da biblioteca carregada
```

## Review Focus

1. **Fits univariados liberados antes do DCC**: `FitDcc` com um `Fit` já liberado lança `ObjectDisposedException`
   (nunca lê handle morto); um `DccFit` feito antes continua prevendo e simulando depois que os univariados são
   liberados. Teste na Task 3.
2. **Fits univariados incompatíveis** (tamanhos diferentes, um `Failed` sem ponto finito, uma só série): esperado
   `VolEngineException` com a mensagem nativa, não crash nem `IndexOutOfRange`. Teste na Task 3.
3. **Leitura concorrente de `DccFit.Covariance()`** (H calculada no primeiro acesso, no nativo): 16 threads lendo ao
   mesmo tempo recebem arrays iguais. Teste na Task 3.
4. **Consumidor sem `VOLENGINE_LIB`**: o pacote acha `libvolengine` em `runtimes/<rid>/native/` sozinho, inclusive
   publicado com AOT. Teste na Task 7 (consumidor) e no CI.
5. **Regressores mistos por série** no forecast e na simulação do DCC (uma série com mxreg, outra com `null`):
   funciona; contagem de entradas diferente do número de séries é `ArgumentException`. Testes nas Tasks 4 e 5.

---

### Task 1: Teste do contrato mais estrito

**Files:**
- Modify: `csharp/tests/VolEngine.Tests/Support/ContractCheck.cs`, `csharp/tests/VolEngine.Tests/Interop/ContractTests.cs`
- Modify (se o teste acusar): `csharp/src/VolEngine/Interop/Native/NativeMethods*.cs`

**Interfaces:**
- Consumes: `ContractCheck.FunctionProblems(IEnumerable<string>, Type)` (C#-a).
- Produces: a mesma assinatura, com a tabela mais estrita: `const T*` (entrada) só `T*`; `T*` sem `const` aceita
  `out T` ou `T*`; `out` exige `ParameterInfo.IsOut` (não `ref`/`in`); `ve_x**` só `out XHandle` nas funções
  `*_create` e no último parâmetro de `ve_forecast_create`, `ve_simulate_*`, `ve_dcc_forecast_create`,
  `ve_dcc_standard_errors`, e só `nint*` em `ve_fit_batch`/`ve_dcc_fit_batch`; `const ve_x* const*` só `nint*`; toda
  declaração com parâmetro `string` tem `LibraryImportAttribute.StringMarshalling == StringMarshalling.Utf8`.

- [ ] **Step 1: Testes novos** em `ContractTests.cs`, cada um com uma linha de contrato e o tipo `NativeMethods`:

```csharp
// "function ve_status ve_panel_set_ranges(ve_panel*, const size_t*, const size_t*)" é aceita (declaração real);
// uma classe de teste interna com [LibraryImport] e assinaturas erradas, passada como 'native', é rejeitada em:
//   const size_t* declarado como out nuint;  int32_t* declarado como ref int;  const ve_spec* const* como out SpecHandle;
//   string sem StringMarshalling.Utf8.
[Fact] public void A_const_input_declared_as_out_is_reported()
[Fact] public void A_ref_output_is_reported()
[Fact] public void A_handle_array_declared_as_out_is_reported()
[Fact] public void A_string_without_utf8_is_reported()
```

  A classe de teste `WrongDeclarations` vive em `Support/` (`internal static partial class` com `[LibraryImport]`
  apontando para `"libvolengine"`; nunca é chamada, só refletida).

- [ ] **Step 2: Rodar e ver falhar** (os quatro testes novos: a checagem atual aceita tudo isso).
- [ ] **Step 3: Implementar** a tabela estrita em `ContractCheck` (o nome da função entra em `Accepted`).
- [ ] **Step 4: Rodar a suíte inteira.** Esperado: os testes novos passam e
  `Every_contract_function_has_a_matching_declaration` continua passando; se acusar uma declaração real, corrigir a
  declaração (e só ela).
- [ ] **Step 5: Commit** — `test(csharp): the contract check tells out, ref and const apart`

### Task 2: Pendências do C#-a antes do congelamento

Os menores adiados da revisão do C#-a que mudam a API pública ou o comportamento visível; depois da 1.1.0 custariam
uma quebra.

**Files:**
- Modify: `csharp/src/VolEngine/Interop/Library.cs`, `Interop/Batch.cs`, `Specs/Panel.cs`, `Specs/Spec.cs`,
  `Results/Simulation.cs`, `Results/FitGrid.cs`, `csharp/Directory.Build.props`, `csharp/.editorconfig`,
  `tests/VolEngine.Tests/Specs/SpecTests.cs`, `tests/VolEngine.Tests/Support/{ContractCheck,Repository}.cs`,
  `csharp/benchmarks/VolEngine.Benchmarks/BindingOverhead.cs`
- Create: `csharp/src/VolEngine/NativeAbi.cs`
- Test: `tests/VolEngine.Tests/Specs/GridTests.cs`, `tests/VolEngine.Tests/Results/SimulationTests.cs`,
  `tests/VolEngine.Tests/Interop/LibraryTests.cs`

**Interfaces:**
- Produces: `NativeAbi.Version` (chama `Library.EnsureLoaded()` e `ve_abi_version`).

- [ ] **Step 1: Testes**

```csharp
[Fact] public void An_empty_regressor_is_passed_as_given()
    // context.Fit(new Spec { Mxreg = 1 }, [], mxreg: new double[0, 1]) -> Status InsufficientData (não "mxreg is missing")
[Fact] public void Null_series_are_argument_null_errors()
    // Panel.FromSeries([t1500, null!]) e FromSeries([t1500], mxreg: [null!]) -> ArgumentNullException
[Fact] public void A_panel_without_series_is_an_argument_error()   // FromColumns(new double[10, 0]) -> ArgumentException
[Fact] public void Oversized_simulations_are_rejected_before_allocating()
    // Simulate(fit, new SimulationOptions(int.MaxValue, 2, 1)) -> ArgumentOutOfRangeException (Horizon * Paths > Array.MaxLength)
[Fact] public void Spec_text_shows_fixed_and_start()   // new Spec { Fixed = { ["shape"] = 5 } }.ToString() contém "shape = 5"
[Fact] public void Native_abi_is_one()                // Assert.Equal(1, NativeAbi.Version)
```

  E no teste de mensagens entre threads (`SpecTests`), nomes delimitados: `"bad" + i + "x"`.

- [ ] **Step 2: Rodar e ver falhar.**
- [ ] **Step 3: Implementar.**
  - `Batch.JobPanels` decide só por `width == 0` (dado nulo é legal com T·k = 0).
  - `Panel.FromSeries` rejeita elementos nulos com `ArgumentNullException(nameof(series))` / `nameof(mxreg)`.
  - `FromColumns` rejeita N = 0 como `FromSeries` rejeita lista vazia.
  - `Simulation.Run` checa `(long)Horizon * Paths > Array.MaxLength` antes de criar as opções nativas.
  - `Spec.PrintMembers` escreve `Fixed = { shape = 5 }` (entradas em ordem de nome, invariant culture).
  - `Library.Load` registra o resolver só quando `VOLENGINE_LIB` está definido.
  - Doc XML do indexador de `FitGrid`: o fit pertence à grade; não o libere, libere a grade.
  - `EnforceCodeStyleInBuild=true` em `Directory.Build.props`.
  - `IDE1006` sai do `NoWarn` global e vai para `[Interop/Native/*.cs]` no `.editorconfig`.
  - Os dois helpers com corpo de expressão (`ContractCheck.HandleName`, `Repository.ReadContract`) passam a ter corpo
    em bloco, junto com o que mais o build passar a acusar.
  - O comentário de `[InProcess]` em `BindingOverhead.cs` passa a dar o motivo real (o gerador fora do processo falha
    ao varrer `build/`).
- [ ] **Step 4: Rodar a suíte inteira.** Esperado: tudo passa, build sem warning.
- [ ] **Step 5: Commit** — `fix(csharp): the C#-a review minors that would change the 1.1.0 API`

### Task 3: `DccSpec`, ajuste DCC, SE e lote

**Files:**
- Create: `csharp/src/VolEngine/Dcc/DccSpec.cs`, `Dcc/DccFit.cs`, `Dcc/DccFitBatch.cs`,
  `csharp/src/VolEngine/Interop/NativeReads.cs`, `csharp/src/VolEngine/Specs/FrozenValues.cs`
- Modify: `csharp/src/VolEngine/Context.cs`, `Results/Fit.cs` (passa a usar `NativeReads`), `Specs/Spec.cs` (passa a
  usar `FrozenValues`)
- Test: `csharp/tests/VolEngine.Tests/Dcc/DccFitTests.cs`, `Dcc/CopulaTests.cs`, `Support/ReferenceData.cs`
  (helper `Columns(file)`)

**Interfaces:**
- Consumes: `Fit.Handle` (interno), `Calls.Check`, `CoefTable` (construtor interno), `Criterion`/`CriterionScale`.
- Produces: `DccSpec`, `DccFit`, `DccFitBatch`, `Context.FitDcc` (duas), `Context.StandardErrors` como na seção de
  interface; `internal DccSpecHandle DccSpec.CreateHandle()` (setters na ordem do Python: order, dist, groups, fixed,
  start, copula se ≠ None, static method se ≠ Default); `internal static unsafe class NativeReads` com
  `double[] View<THandle>(THandle owner, ViewReader<THandle> read) where THandle : SafeHandle` (AddRef durante a cópia,
  como `Fit.View`) e `double[] Values<THandle>(THandle owner, int count, ValueReader<THandle> read)`;
  `internal static class FrozenValues` com `Frozen`, `SameValues`, `AddValues`, `RejectNul`, `Print` (o que `Spec`
  tem hoje, mais a impressão da Task 2), usado por `Spec` e `DccSpec`.
- `FitDcc`: lista de specs não vazia e sem nulos, lista de fits sem nulos (`ArgumentException`/`ArgumentNullException`);
  os handles dos fits vão como `nint*` com `DangerousAddRef` em cada um durante a chamada (um fit liberado →
  `ObjectDisposedException`); opções com `ve_dcc_options_set_standard_errors`; `ve_dcc_fit_batch` → um `DccFit` por
  spec. `StandardErrors` usa `ve_dcc_standard_errors` com o handle do contexto e devolve um `DccFit` novo.
- Dados (decisão minha, como o Python): `dcc3.csv` (colunas A, B, C) com univariados `new Spec { Start = params }`,
  `FitOptions { Tolerance = 1 }` dos `panels.dcc3.univariate` de `rmgarch_dcc.json` (estimativas iguais a 1e-12);
  `copula3.csv` com `Spec { Distribution = Std, Start = params }` de `rmgarch_copula.json` `univariate`.

- [ ] **Step 1: Testes**

```csharp
[Fact] public void Dcc_estimates_near_rmgarch()
    // caso "dcc3_DCC_mvnorm": FitDcc(new DccSpec(), fits, standardErrors: false): Converged, Names [dcca1, dccb1],
    // cada estimativa a 5% relativo de "params"; Correlation() 1500 x 3 x 3 com diagonal 1 e simétrica.
[Fact] public void Fdcc_names_follow_the_groups()       // Groups [1, 1, 2] (o "groups" do fixture): fdcca1, fdcca2, fdccb1, fdccb2
[Fact] public void Fixed_parameters_stay_in_the_table()  // Fixed dcca1 = 0.05: valor 0.05, SE NaN
[Fact] public void Standard_errors_can_come_later()
    // sem SE: Coef() -> VolEngineException; context.StandardErrors(fit, fits).Coef(Robust): SE finitos e > 0;
    // o fit original continua sem SE.
[Fact] public void Standard_errors_need_the_same_fits()   // StandardErrors(fit, outros fits) -> VolEngineException
[Fact] public void A_batch_fits_every_spec()             // [Dcc, Adcc]: Count 2, [1].Names contém dccg1; índice 2 -> ArgumentOutOfRange
[Fact] public void Bad_univariate_inputs_are_rejected()
    // [] specs -> ArgumentException; fits com null -> ArgumentNullException; uma série só -> VolEngineException;
    // séries de tamanhos diferentes (t1500 e t3000 de sgarch11_norm.csv) -> VolEngineException.
[Fact] public void A_disposed_univariate_fit_is_an_object_disposed_error()
[Fact] public void A_dcc_fit_outlives_its_univariate_fits()
    // libera os três Fit; dcc.Forecast(5).Sigma tem 5 x 3 finitos (Forecast vem da Task 4: este teste entra lá)
[Fact] public async Task Concurrent_covariance_reads_agree()   // 16 Tasks chamam Covariance(); todas iguais à primeira
[Fact] public void Dcc_spec_equality_compares_groups_by_content()
// CopulaTests:
[Fact] public void Copula_dcc_estimates_near_rmgarch()   // "dcc_mvnorm_param": Copula Parametric, 5% relativo
[Fact] public void A_static_copula_has_a_constant_correlation() // Static + Parametric + Kendall: R[0] == R[T-1]
[Fact] public void A_static_model_needs_a_copula()       // new DccSpec { Model = Static } -> VolEngineException
```

  `A_dcc_fit_outlives_its_univariate_fits` fica escrito na Task 4 (precisa de `Forecast`).

- [ ] **Step 2: Rodar e ver falhar.**
- [ ] **Step 3: Implementar.** Doc XML copia os docstrings de `DccSpec`, `fit_dcc`, `DccFit` e `standard_errors` de
  `python/src/volengine/_dcc.py` e o comentário do header sobre o critério de duas etapas (`ve_dcc_fit_criterion`).
- [ ] **Step 4: Rodar a suíte inteira.**
- [ ] **Step 5: Commit** — `feat(csharp): DCC, aDCC, FDCC and Copula-DCC fits with standard errors`

### Task 4: Forecast do DCC

**Files:**
- Create: `csharp/src/VolEngine/Dcc/DccForecast.cs`, `csharp/src/VolEngine/Interop/DccFutures.cs`
- Modify: `Dcc/DccFit.cs` (`Forecast`)
- Test: `csharp/tests/VolEngine.Tests/Dcc/DccForecastTests.cs`

**Interfaces:**
- Consumes: `DccFit`, `Futures.Check` (C#-a), `NativeReads`.
- Produces: `DccForecast`; `internal static unsafe class DccFutures { static DccFutureHandle Create(int series, int
  horizon, IReadOnlyList<double[,]?>? mxreg, IReadOnlyList<double[,]?>? vxreg); }`: os dois nulos → handle inválido
  (NULL nativo); lista com contagem ≠ `series` → `ArgumentException`; entrada não nula com `GetLength(0) != horizon` →
  `ArgumentException`; a largura o nativo confere contra a spec de cada série (erro nomeia a série de 1). Fixa cada
  array só durante o `ve_dcc_future_set` (o nativo copia). Também usada pela Task 5.

- [ ] **Step 1: Testes**

```csharp
[Fact] public void Forecast_matches_rmgarch()
    // rmgarch_dcc_forecast.json, caso "dcc_mvnorm": univariados de dcc3 (fixture "univariate", Start + Tolerance 1),
    // new DccSpec { Fixed = params }, FitDcc(..., standardErrors: false).Forecast("ahead" = 10):
    // Correlation e Covariance a 1e-8 absoluto de "R" e "H", Sigma a 1e-8 relativo de "sigma" (como cpp/tests/dcc_future_test.cpp).
[Fact] public void Series_forecasts_are_the_univariate_forecasts() // Sigma[h][j] == fits[j].Forecast(5).Sigma[h] (1e-12 relativo)
[Fact] public void Forecast_horizon_must_be_positive()            // 0 -> ArgumentOutOfRangeException
[Fact] public void Regressor_lists_have_one_entry_per_series()   // 2 entradas para 3 séries -> ArgumentException
[Fact] public void Mixed_regressors_per_series()
    // univariados t1500 de sgarch11_norm.csv: série 0 com Spec { Mxreg = 1 } (coluna x1 de regressors_2e.csv), séries 1 e 2 sem;
    // FitDcc(new DccSpec(), fits, standardErrors: false) (com regressores o SE do DCC é VE_UNSUPPORTED);
    // Forecast(4, mxreg: [new double[4, 1], null, null]) funciona; sem a entrada da série 0 -> VolEngineException
[Fact] public void A_dcc_fit_outlives_its_univariate_fits()   // o da Task 3
```

- [ ] **Step 2: Rodar e ver falhar.**
- [ ] **Step 3: Implementar.** Doc XML copia o docstring de `DccForecast` (regra de R_h) de `_forecast.py`.
- [ ] **Step 4: Rodar a suíte inteira.**
- [ ] **Step 5: Commit** — `feat(csharp): DCC forecast`

### Task 5: Simulação do DCC e as 130 funções chamadas

**Files:**
- Create: `csharp/src/VolEngine/Dcc/DccSimulation.cs`, `Dcc/DccSimulationOptions.cs`
- Modify: `csharp/src/VolEngine/Context.cs` (`Simulate(DccFit, DccSimulationOptions)`)
- Test: `csharp/tests/VolEngine.Tests/Dcc/DccSimulationTests.cs`, `csharp/tests/VolEngine.Tests/Interop/CoverageTests.cs`

**Interfaces:**
- Consumes: `DccFutures.Create` (Task 4), `NativeReads`.
- Produces: `DccSimulationOptions`, `DccSimulation`, `Context.Simulate(DccFit, DccSimulationOptions)`. Tamanhos < 1
  e `Paths * Horizon * Series` (×`Series` de novo com `Correlations`) acima de `Array.MaxLength` →
  `ArgumentOutOfRangeException` antes do nativo. `Correlation`/`Covariance` só são lidos com `Correlations = true`
  (senão `null`).

- [ ] **Step 1: Testes**

```csharp
[Fact] public void Same_seed_gives_same_paths()        // Returns e Sigma 3 caminhos x 20 passos x 3 séries iguais
[Fact] public void Thread_count_does_not_change_paths() // Context(1) e Context(4)
[Fact] public void Correlations_are_opt_in()           // sem: Correlation e Covariance null; com: 2 x 5 x 3 x 3, diagonal 1
[Fact] public void The_first_step_matches_the_forecast() // Sigma[p][0][j] == dcc.Forecast(1).Sigma[0][j] para todo p (1e-12 relativo)
[Fact] public void A_copula_dcc_simulates()            // dcc_mvnorm_param de copula3: Returns finitos
[Fact] public void Mixed_regressors_per_series_simulate() // como no forecast, com Mxreg = [new double[5, 1], null, null]
[Fact] public void Simulation_sizes_must_be_positive()   // (0, 1, 1) e (1, 0, 1) -> ArgumentOutOfRangeException
// CoverageTests:
[Fact] public void Every_contract_function_is_called_by_the_library()
    // lê as linhas "function" de volengine.abi e todos os .cs de csharp/src/VolEngine fora de Interop/Native;
    // cada nome ve_* precisa aparecer como "NativeMethods.<nome>(" ou "NativeMethods.<nome>," (ou "NativeMethods.<nome>)" em
    // grupos de método); a lista de faltantes é vazia. Equivale ao cpp/scripts/check_abi_coverage.py.
```

- [ ] **Step 2: Rodar e ver falhar.**
- [ ] **Step 3: Implementar.** Doc XML copia o docstring de `DccSimulation` e o comentário de `simulate` sobre
  `correlations` (memória: paths x horizon x N² x 2 doubles).
- [ ] **Step 4: Rodar a suíte inteira.** Esperado: tudo passa, inclusive a cobertura das 130 funções.
- [ ] **Step 5: Commit** — `feat(csharp): DCC simulation; every contract function is called`

### Task 6: Benchmark do DCC pelo binding

**Files:**
- Modify: `csharp/benchmarks/VolEngine.Benchmarks/BindingOverhead.cs`, `docs/performance/benchmarks.md` (seção do C#-a)

- [ ] **Step 1: Acrescentar `FitDcc`/`FitDccDirect`**: DCC(1,1) mvnorm sobre as três primeiras séries do painel
  simulado do `Batch*` (fits sGARCH(1,1) norm sem SE feitos uma vez no `[GlobalSetup]`), SE desligados; o lado direto
  monta spec, opções e lote DCC com `NativeMethods`; soma de loglik bit a bit igual, senão lança.
- [ ] **Step 2: Rodar** `dotnet run -c Release --project csharp/benchmarks/VolEngine.Benchmarks -- --filter '*FitDcc*'`
  com `VOLENGINE_LIB`. Esperado: dois benchmarks sem erro.
- [ ] **Step 3: Acrescentar a linha** "DCC(1,1) over 3 series" à tabela da seção "C#-a: the C# binding against direct
  native calls" (renomear a seção para "C#: ...") com máquina e commit, e a frase sobre o ruído.
- [ ] **Step 4: Commit** — `bench(csharp): DCC through the binding against direct calls`

### Task 7: Pacote NuGet, consumidor limpo e Native AOT

**Files:**
- Modify: `csharp/src/VolEngine/VolEngine.csproj`, `.github/workflows/csharp.yml`
- Create: `csharp/src/VolEngine/README.md` (readme do pacote: o que é, instalação, um exemplo de 15 linhas, link para o
  guia), `csharp/src/VolEngine/build/NativeAssets.targets` (ou o mesmo conteúdo no csproj),
  `csharp/tests/VolEngine.Consumer/VolEngine.Consumer.csproj`, `csharp/tests/VolEngine.Consumer/Program.cs`,
  `csharp/scripts/check_package.sh`

**Interfaces:**
- Produces (MSBuild): `dotnet pack csharp/src/VolEngine -c Release -p:Version=<v> -p:VolEngineNativeAssets=<pasta>
  [-p:VolEngineNativeRids=<lista ;>] -o <saída>`. `VolEngineNativeRids` padrão = os 7 RIDs; para cada um, o arquivo
  `<pasta>/volengine-native-<rid>.<ext>` (`ext` = `dll` para `win-*`, `dylib` para `osx-*`, `so` para o resto) vira
  `runtimes/<rid>/native/libvolengine.<ext>`; um arquivo faltando é erro de pack com o caminho. Sem
  `VolEngineNativeAssets`, o pack falha dizendo que falta a pasta (nunca um pacote sem nativos). Metadados como os do
  ExcelReader: `Authors` Gabriel Matte, `PackageLicenseExpression` MIT, `PackageReadmeFile` README.md,
  `RepositoryUrl`/`PackageProjectUrl` https://github.com/GabrielMarquezMatte/VolEngine, `PackageTags`
  garch;dcc;volatility;econometrics;finance, `IncludeSymbols` + `snupkg`, `EnablePackageValidation` true (sem baseline
  na 1.1.0; a baseline entra quando houver um pacote anterior, ver Task 8).
- Produces: `csharp/scripts/check_package.sh <nupkg-dir> <version>`: restaura o consumidor só com
  `-p:RestoreAdditionalProjectSources=<nupkg-dir> -p:VolEngineVersion=<version>`, roda `dotnet run` **sem**
  `VOLENGINE_LIB`, depois `dotnet publish -p:PublishAot=true -r <rid do host>` e roda o executável publicado; falha se
  qualquer passo falhar. O consumidor (fora do `.slnx`): `PackageReference Include="VolEngine"
  Version="$(VolEngineVersion)"`, `TreatWarningsAsErrors` (warning de trimming/AOT é erro); `Program.cs` simula um
  sGARCH(1,1) todo fixo (`Context.Simulate`), ajusta, imprime `NativeAbi.Version` e o status, e sai com 0 só se
  `Converged`.

- [ ] **Step 1: Escrever o consumidor e o script**, e rodar o script contra um pacote ainda inexistente. Esperado:
  falha no restore (`VolEngine` não encontrado) — é o vermelho desta tarefa.
- [ ] **Step 2: Empacotar localmente** com a DLL local copiada como `volengine-native-win-x64.dll` numa pasta de
  rascunho: `dotnet pack ... -p:Version=1.1.0-local -p:VolEngineNativeAssets=<pasta> -p:VolEngineNativeRids=win-x64`.
  Esperado: `.nupkg` com `runtimes/win-x64/native/libvolengine.dll` (`unzip -l`), e o mesmo pack sem
  `VolEngineNativeRids` falha nomeando `volengine-native-linux-x64.so`.
- [ ] **Step 3: Rodar `check_package.sh`.** Esperado: consumidor normal e AOT imprimem ABI 1 e `Converged`, exit 0.
  (Windows: o AOT precisa das ferramentas C++ do Visual Studio; se faltarem na máquina, o passo AOT local fica
  registrado como não rodado e o CI o cobre.)
- [ ] **Step 4: CI** — no `csharp.yml`, depois dos testes, cada job copia a biblioteca de `python/src/volengine/_lib/`
  para `natives/volengine-native-<rid>.<ext>` (`linux-x64`, `osx-arm64`, `win-x64`), empacota com
  `VolEngineNativeRids=<rid>` e roda `check_package.sh`. Conferir a sintaxe com PyYAML.
- [ ] **Step 5: Commit** — `build(csharp): NuGet package with the native libraries, tested by a clean and an AOT consumer`

### Task 8: Release no NuGet, documentação e API congelada

**Files:**
- Modify: `.github/workflows/release.yml`, `csharp/src/VolEngine/PublicAPI.Shipped.txt`, `PublicAPI.Unshipped.txt`,
  `README.md`, `ARCHITECTURE.md`, `CHANGELOG.md`, `CONTRIBUTING.md`
- Create: `docs/guide/csharp.md`

**Interfaces:**
- Consumes: o pack da Task 7, `check_package.sh`.

- [ ] **Step 1: `release.yml`.** Três jobs novos:
  - `nuget` (needs `wheels`, ubuntu): baixa `native-*` (merge), deriva a versão da tag (`${GITHUB_REF_NAME#v}`; num
    run manual, a do `VE_VERSION` do `CMakeLists.txt` com `-dev`), empacota com os 7 RIDs e sobe o artefato
    `nuget-package` (`.nupkg` + `.snupkg`).
  - `nuget-check` (needs `nuget`, matriz ubuntu-24.04 / macos-latest / windows-latest): baixa o artefato e roda
    `check_package.sh`.
  - `publish-nuget` (needs `[publish-github, nuget-check]`, `permissions: id-token: write`): `NuGet/login@v1` com
    `user: ${{ vars.NUGET_USER }}`, depois `dotnet nuget push *.nupkg --api-key ${{ steps.login.outputs.NUGET_API_KEY }}
    --source https://api.nuget.org/v3/index.json --skip-duplicate`, só com `github.event_name == 'push' ||
    !inputs.dry_run`.
  
  Conferir a sintaxe com PyYAML. Não rodar (precisa de tag ou de dispatch, com o ok do autor).
- [ ] **Step 2: Congelar a API.** Mover todo o conteúdo de `PublicAPI.Unshipped.txt` para `PublicAPI.Shipped.txt`
  (o Unshipped fica só com `#nullable enable`); a suíte e o build continuam verdes.
- [ ] **Step 3: Documentação.**
  - `docs/guide/csharp.md` no formato de `docs/guide/cpp.md`: instalação (`dotnet add package VolEngine`), plataformas,
    `VOLENGINE_LIB`, ajuste, grade com `Panel`, forecast, simulação, diagnósticos, DCC (ajuste, SE depois, forecast,
    simulação), regras de threads e de `Dispose`, erros, AOT.
  - Seção "C#" no README ao lado das de Python e C++.
  - Camada nova no diagrama do `ARCHITECTURE.md`: `C# (csharp/src/VolEngine, LibraryImport)` ao lado do Python, mais
    uma linha na tabela de responsabilidades.
  - Entrada "Added" no `[Unreleased]` do CHANGELOG: o binding, o pacote e os 7 RIDs.
  - Seção "C# binding tests" no CONTRIBUTING: build, testes, pack local, `check_package.sh`, e o passo de release
    "depois de publicar a 1.1.0, pôr `PackageValidationBaselineVersion` = 1.1.0 no csproj".
  
  Cada trecho de código do guia compila: copiá-los para um teste `GuideSnippetsTests` que roda os exemplos com os dados
  de `tests/reference/data`.
- [ ] **Step 4: Rodar a suíte inteira e o `check_package.sh` local da Task 7.**
- [ ] **Step 5: Commit** — `docs(csharp): guide, README and changelog; the 1.1.0 API is shipped`

## Fora deste plano

- A tag `v1.1.0` (e um `rc` antes, se o autor quiser): depois deste plano, com o ok do autor, que antes confere o ID
  `VolEngine` no nuget.org, registra a policy do Trusted Publishing para `release.yml` e cria a variável `NUGET_USER`.
- O SV (spec 2026-11-17), que vai para a 1.2.0.
