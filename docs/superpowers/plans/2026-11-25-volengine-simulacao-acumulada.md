# Simulação acumulada — plano de implementação

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** uma lista de prazos (checkpoints) no `ve_sim_options` faz `ve_simulate_fit`, `ve_simulate_spec` e
`ve_simulate_dcc` devolverem só a soma da série simulada em cada prazo, por caminho (e por série no DCC), e os
bindings C++, C# e Python expõem isso.

**Architecture:** o núcleo ganha `checkpoints` nos pedidos e `checkpoints`/`cumulative` nos resultados. Cada caminho
soma o retorno de cada passo numa soma corrente, na ordem dos passos, e grava nos prazos. A C ABI ganha um setter e
quatro views; as views completas dão `VE_INVALID_ARGUMENT` no modo acumulado.

**Tech Stack:** C++23 (GCC/MinGW, gtest), C ABI, wrapper C++ header-only, C# (.NET 10, LibraryImport,
PublicApiAnalyzers, xUnit), Python (ctypes, numpy, pytest, mypy).

**Spec:** [docs/superpowers/specs/2026-11-25-volengine-simulacao-acumulada-design.md](../specs/2026-11-25-volengine-simulacao-acumulada-design.md)

## Global Constraints

- ABI 1 aditiva: `VE_ABI_VERSION` continua `1`; `volengine.abi` só ganha linhas
  (`python scripts/abi_contract.py write`, depois `check`).
- Nomes exatos da C ABI: `ve_sim_options_set_checkpoints(ve_sim_options* o, const size_t* ks, size_t n)`,
  `ve_sim_cumulative(const ve_sim*, const double** data, size_t* len)`,
  `ve_sim_checkpoints(const ve_sim*, const size_t** data, size_t* len)`,
  `ve_dcc_sim_cumulative(const ve_dcc_sim*, const double**, size_t*)`,
  `ve_dcc_sim_checkpoints(const ve_dcc_sim*, const size_t**, size_t*)`. Comentários com "Since 1.3.0.".
- Prazos válidos: estritamente crescentes, o primeiro ≥ 1 e o último igual ao horizonte das opções. `n = 0` limpa
  (`ks` pode ser `NULL`); `NULL` com `n > 0` é `VE_INVALID_ARGUMENT`. Um erro deixa a opção como estava.
- O valor no prazo k é, bit a bit, a soma da esquerda para a direita dos k primeiros passos da saída completa com a
  mesma semente. Sem Kahan, sem soma por pares.
- Sem checkpoints, toda saída fica igual à de hoje, bit a bit.
- Checkpoints com `correlations` ligado: `VE_INVALID_ARGUMENT` no `ve_simulate_dcc`.
- No modo acumulado, `returns` e `sigma` do núcleo ficam vazios e não são alocados.
- Limites do STYLEGUIDE: funções ≤ 60 linhas, ≤ 5 parâmetros (Python também), ≤ 8 módulos por pasta, sem `else`.
  O teste `style` da raiz roda em todo `ctest`. Não criar módulos novos em `python/src/volengine` nem em
  `python/tests`, que já estão em 8.
- `-Werror` (raiz e `cpp/tests`) e `TreatWarningsAsErrors` (C#).
- Testes C++: `cmake --build --preset gcc-release` e `ctest --preset gcc-release -j 8` pelo PowerShell. Wrapper:
  `bash cpp/scripts/ci_check.sh` com `/c/msys64/ucrt64/bin` primeiro no PATH e
  `VOLENGINE_NATIVE_LIB=$PWD/build/gcc-release/libvolengine.dll`. C#: `VOLENGINE_LIB=... dotnet test --solution
  csharp/VolEngine.slnx -c Release`. Python: `VOLENGINE_LIB=... uv run --project python pytest python/tests -q` e
  `uv run --project python mypy python/src`.
- `abi_coverage` (do `ci_check` e de `python/tests/packaging`) falha entre a Tarefa 3 e a Tarefa 4, porque o wrapper
  ainda não chama as funções novas; volta a passar na Tarefa 4.
- Branch `develop`, commits no estilo do histórico, sem trailer `Co-Authored-By`.

## Review Focus

1. **Prazo igual a 1 e prazo único igual ao horizonte**: K = 1 com k = horizonte dá caminhos × 1 e o valor é a soma de
   todos os passos; `{1}` com horizonte 1 dá o próprio retorno do passo. Teste na Tarefa 1.
2. **Regressores no horizonte**: o modo acumulado com `mxreg` soma os mesmos retornos da saída completa com os mesmos
   regressores. Teste na Tarefa 3.
3. **Setter chamado antes de outro setter e duas vezes**: chamar `ve_sim_options_set_checkpoints` duas vezes
   substitui a lista; depois de um erro a lista anterior continua valendo. Teste na Tarefa 3.
4. **Cópula DCC (margens empíricas ou paramétricas)**: o acumulado também bate bit a bit numa Copula-DCC, não só no
   DCC simples. Teste na Tarefa 2.
5. **Tamanho no C#**: Paths × K acima de `Array.MaxLength` é rejeitado antes da chamada nativa, e Paths × Horizon
   grande com K pequeno é aceito no modo acumulado. Teste na Tarefa 5.

---

### Task 1: Núcleo univariado

**Files:**
- Modify: `src/sim/simulate.hpp` (`Request`, `Simulation`), `src/sim/simulate.cpp` (`simulate_path`, `simulate`)
- Test: `tests/unit/sim/simulate_test.cpp`

**Interfaces:**
- Produces:
  - `sim::Request::checkpoints : std::vector<std::size_t>` (vazio = saída completa). Quem chama já validou.
  - `sim::Simulation::checkpoints : std::vector<std::size_t>` e `sim::Simulation::cumulative : std::vector<double>`
    (paths × K, linha p = caminho p). No modo acumulado `returns` e `sigma` ficam vazios.

- [ ] **Step 1: Write the failing tests** em `simulate_test.cpp`, sobre o `Start` que o arquivo já usa:
  - `SimCumulative.EqualsTheLeftSumOfTheFullPath`: horizonte 20, 50 caminhos, `checkpoints = {1, 5, 20}`; para cada
    caminho p e prazo j, `std::bit_cast<std::uint64_t>(cumulative[p * 3 + j])` é igual ao da soma
    `double s = 0; for (h < k_j) s += full.returns[p * 20 + h];` da simulação completa com a mesma semente. Também:
    `returns.empty()`, `sigma.empty()`, `checkpoints == {1, 5, 20}`.
  - `SimCumulative.SameForAnyThreadCount`: o `cumulative` com 1 e com 4 threads é igual bit a bit.
  - `SimCumulative.SingleCheckpointIsTheWholeSum` (Review Focus 1): `{20}` dá paths × 1 igual à soma dos 20 passos;
    horizonte 1 com `{1}` dá o próprio `returns[p]`.
  - `SimCumulative.NoCheckpointsChangesNothing`: sem checkpoints, `returns` e `sigma` são iguais bit a bit aos de
    antes da mudança (compare com uma segunda chamada idêntica e confira `cumulative.empty()`).
- [ ] **Step 2: Run to verify they fail.** `cmake --build --preset gcc-release`. Expected: erro de compilação
  (`checkpoints` não é membro de `Request`).
- [ ] **Step 3: Implement.** Em `simulate`, aloque `returns`/`sigma` só sem checkpoints e `cumulative` (paths × K) só
  com eles. Em `simulate_path`, uma soma corrente `double` somada com `draw.r` logo após cada passo; quando
  `h + 1 == checkpoints[j]`, grava e avança j.
- [ ] **Step 4: Run the tests.** `ctest --preset gcc-release -j 8 -R "SimCumulative|Sim"` e depois a suíte inteira.
  Expected: PASS.
- [ ] **Step 5: Commit** `feat(sim): cumulative sums at checkpoints in the univariate simulation`

### Task 2: Núcleo DCC

**Files:**
- Modify: `src/dcc/future/simulate.hpp` (`DccRequest`, `DccSimulation`), `src/dcc/future/simulate.cpp`
  (`PathRun::step`, `simulate_dcc`)
- Test: `tests/unit/dcc/future/simulate_test.cpp`

**Interfaces:**
- Produces:
  - `DccRequest::checkpoints : std::vector<std::size_t>`.
  - `DccSimulation::checkpoints : std::vector<std::size_t>` e `DccSimulation::cumulative : std::vector<double>`
    (paths × K × m, `cumulative[(p * K + j) * m + i]`). No modo acumulado `returns`, `sigma`, `r` e `h` ficam vazios.
    Quem chama garante que `correlations` está desligado.

- [ ] **Step 1: Write the failing tests** em `tests/unit/dcc/future/simulate_test.cpp`, com o ajuste DCC que o
  arquivo já monta:
  - `DccSimCumulative.EqualsTheLeftSumOfTheFullPath`: horizonte 10, 30 caminhos, `{2, 10}`; cada
    `cumulative[(p * 2 + j) * m + i]` é igual bit a bit à soma da esquerda para a direita de
    `full.returns[(p * 10 + h) * m + i]` para h < k_j; `returns`, `sigma`, `r` e `h` vazios.
  - `DccSimCumulative.CopulaMatchesToo` (Review Focus 4): o mesmo para uma Copula-DCC com margens paramétricas.
  - `DccSimCumulative.SameForAnyThreadCount`: 1 e 4 threads, iguais bit a bit.
- [ ] **Step 2: Run to verify they fail.** Expected: erro de compilação.
- [ ] **Step 3: Implement.** O `PathRun` ganha uma soma corrente por série (`std::vector<double>` de m), somada com
  `draw.r` no laço das séries de `step`; no prazo, grava as m somas. `simulate_dcc` aloca só o que o modo usa.
- [ ] **Step 4: Run the tests.** `ctest --preset gcc-release -j 8 -R DccSim` e a suíte inteira. Expected: PASS.
- [ ] **Step 5: Commit** `feat(dcc): cumulative sums at checkpoints in the DCC simulation`

### Task 3: C ABI

**Files:**
- Modify: `include/volengine/volengine.h` (as cinco declarações, com o texto do spec, seção "C ABI", e as regras
  nos comentários de `ve_sim_returns`/`ve_sim_sigma`/`ve_dcc_sim_returns`/`ve_dcc_sim_sigma`)
- Modify: `include/volengine/volengine.abi` (regenerado)
- Modify: `src/capi/handles.hpp` (`ve_sim_options::checkpoints : std::vector<std::size_t>`)
- Modify: `src/capi/results/sim_api.cpp` (setter, `ve_sim_cumulative`, `ve_sim_checkpoints`, recusa das views
  completas, repasse ao `Request`)
- Modify: `src/capi/dcc/dcc_sim_api.cpp` (`ve_dcc_sim_cumulative`, `ve_dcc_sim_checkpoints`, recusa com
  correlações, tamanho com paths × K × m, repasse ao `DccRequest`)
- Modify: `CHANGELOG.md` (`[Unreleased]`, Added: uma linha da C ABI)
- Test: `tests/unit/capi/results/sim_test.cpp`, `tests/unit/capi/dcc_test.cpp`

**Interfaces:**
- Consumes: Tarefas 1 e 2.
- Produces: as cinco funções da Global Constraints. Mensagens de erro: `"checkpoints must be strictly increasing,
  from 1, and end at the horizon"`, `"ks must not be null"`, `"the simulation has checkpoints: read
  ve_sim_cumulative"` (e a forma `ve_dcc_sim_cumulative` no DCC), `"checkpoints and correlations cannot be on
  together"`.

- [ ] **Step 1: Write the failing tests.**
  - `sim_test.cpp`:
    - `CapiSimCumulative.MatchesTheFullOutput`: fit e spec, `{1, 5, 20}`, horizonte 20; cumulative igual bit a bit
      à soma da saída completa; `ve_sim_checkpoints` devolve `{1, 5, 20}` e len 3; `ve_sim_returns` e
      `ve_sim_sigma` dão `VE_INVALID_ARGUMENT`.
    - `CapiSimCumulative.WithRegressors` (Review Focus 2): um fit com `mxreg`, opções com o painel do horizonte; o
      mesmo critério.
    - `CapiSimCumulative.InvalidListsLeaveTheOption` (Review Focus 3): `{5, 20}` ok; depois `{0, 20}`, `{5, 5, 20}`,
      `{5, 10}` (último ≠ horizonte), `NULL` com n 2 dão `VE_INVALID_ARGUMENT`, e a simulação seguinte ainda tem
      `{5, 20}`; `n = 0` limpa e a simulação volta à saída completa (`ve_sim_cumulative` dá `VE_INVALID_ARGUMENT`,
      `ve_sim_checkpoints` dá len 0).
  - `dcc_test.cpp`:
    - `CapiDccSimCumulative.MatchesTheFullOutput`: o mesmo critério para `ve_simulate_dcc`, mais
      `ve_dcc_sim_checkpoints`, e `ve_dcc_sim_returns`/`ve_dcc_sim_sigma` dão `VE_INVALID_ARGUMENT`.
    - `CapiDccSimCumulative.RefusesCorrelations`: checkpoints com `ve_sim_options_set_correlations(o, 1)` dá
      `VE_INVALID_ARGUMENT` e `out == NULL`.
- [ ] **Step 2: Run to verify they fail.** Expected: erro de compilação (`ve_sim_options_set_checkpoints` não
  declarado).
- [ ] **Step 3: Implement** o que está em Files e Interfaces. A regra dos prazos é conferida no setter e de novo
  nas três simulações, antes de montar o pedido (um único ajudante de validação em `sim_api.cpp`, usado pelos dois
  arquivos). `python scripts/abi_contract.py write`.
- [ ] **Step 4: Verify.** `python scripts/abi_contract.py check` (só linhas novas no `git diff` do `.abi`),
  `cmake --build --preset gcc-release`, `ctest --preset gcc-release -j 8`. Expected: tudo PASS.
- [ ] **Step 5: Commit** `feat(capi): checkpoints for cumulative simulations`

### Task 4: Wrapper C++

**Files:**
- Modify: `cpp/include/volengine/simulation.hpp` (`SimOptions::checkpoints : std::vector<std::size_t>`, aplicado em
  `make_handle` com `ve_sim_options_set_checkpoints`; `Simulation::cumulative()` e `Simulation::checkpoints()`)
- Modify: `cpp/include/volengine/dcc.hpp` (`DccSimulation::cumulative()` e `DccSimulation::checkpoints()`)
- Modify: `docs/guide/cpp.md` (seção "Cumulative simulation", com VaR em 1, 5 e 10 dias dos mesmos caminhos, e a frase
  da escala: o motor é invariante à escala, os dados podem entrar em fração)
- Test: `cpp/tests/future_test.cpp`, `cpp/tests/dcc_future_test.cpp`

**Interfaces:**
- Consumes: Tarefa 3.
- Produces: `Result<std::span<const double>> Simulation::cumulative() const`,
  `Result<std::span<const std::size_t>> Simulation::checkpoints() const`, os mesmos em `DccSimulation`, no padrão
  das views que já existem.

- [ ] **Step 1: Write the failing tests:** `Simulation.CumulativeMatchesTheFullOutput` (fit, `{1, 5, 20}`) e
  `DccSimulation.CumulativeMatchesTheFullOutput`, bit a bit contra a soma de `returns()` da simulação completa;
  `returns()` no modo acumulado é erro com `Status::invalid_argument`.
- [ ] **Step 2: Run to verify they fail.** `bash cpp/scripts/ci_check.sh`. Expected: erro de compilação.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run.** `bash cpp/scripts/ci_check.sh`. Expected: PASS, inclusive `abi_coverage`. Depois
  `uv run --project python pytest python/tests/packaging -q`: PASS.
- [ ] **Step 5: Commit** `feat(cpp): checkpoints and cumulative views in the wrapper`

### Task 5: C#

**Files:**
- Modify: `csharp/src/VolEngine/Interop/Native/NativeMethods.Results.cs` e `NativeMethods.Dcc.cs` (as cinco
  funções)
- Modify: `csharp/src/VolEngine/Specs/SimulationOptions.cs` e o arquivo de `DccSimulationOptions`
  (`IReadOnlyList<int>? Checkpoints { get; init; }`)
- Modify: `csharp/src/VolEngine/Results/Simulation.cs` e `csharp/src/VolEngine/Dcc/DccSimulation.cs`
- Modify: `csharp/src/VolEngine/PublicAPI.Unshipped.txt`
- Modify: `docs/guide/csharp.md` (a mesma seção e a frase da escala) e
  `csharp/tests/VolEngine.Tests/Guide/GuideSnippetsTests.cs`
- Test: `csharp/tests/VolEngine.Tests/Results/SimulationTests.cs`, `csharp/tests/VolEngine.Tests/Dcc/DccSimulationTests.cs`

**Interfaces:**
- Consumes: Tarefa 3.
- Produces: `double[][]? Simulation.Cumulative` (Paths × K), `double[][][]? DccSimulation.Cumulative`
  (Paths × K × m), nulos sem checkpoints; `IReadOnlyList<int> Checkpoints` nos dois (vazio sem eles). Com
  checkpoints, `Returns` e `Sigma` são arrays de comprimento 0.

- [ ] **Step 1: Write the failing tests:**
  - `Cumulative_matches_the_left_sum_of_the_full_returns` (univariado e DCC): bit a bit
    (`BitConverter.DoubleToInt64Bits`) contra a soma da esquerda para a direita de `Returns` da simulação completa
    com a mesma semente; `Returns.Length == 0` no modo acumulado.
  - `Checkpoints_are_validated`: `[0, 20]`, `[5, 5, 20]`, `[5, 10]` com horizonte 20 lançam `VolEngineException`
    com status `InvalidArgument`.
  - `Cumulative_size_counts_checkpoints` (Review Focus 5), com um token já cancelado, que para a chamada logo depois
    da validação gerenciada: Paths × K acima de `Array.MaxLength` lança `ArgumentOutOfRangeException`; Paths ×
    Horizon acima de `Array.MaxLength` com `Checkpoints = [Horizon]` passa da validação e lança
    `OperationCanceledException`.
- [ ] **Step 2: Run to verify they fail.** O comando de teste do C#. Expected: erro de compilação.
- [ ] **Step 3: Implement**, com as linhas novas no `PublicAPI.Unshipped.txt`.
- [ ] **Step 4: Run.** Expected: PASS, sem avisos, inclusive `ContractTests` e `CoverageTests`.
- [ ] **Step 5: Commit** `feat(csharp): checkpoints and cumulative simulations`

### Task 6: Python

**Files:**
- Modify: `python/src/volengine/_native.py` (as cinco funções em `PROTOTYPES`)
- Modify: `python/src/volengine/api.py` (`checkpoints: Sequence[int] | None` em `FutureKeywords` e
  `DccSimulateKeywords`; repasse ao `ve_sim_options_set_checkpoints`)
- Modify: `python/src/volengine/_simulation.py` (`Simulation.cumulative`, `Simulation.checkpoints`,
  `DccSimulation.cumulative`, `DccSimulation.checkpoints`)
- Modify: `docs/guide/python.md` (a mesma seção e a frase da escala; o `tests/docs` roda o snippet)
- Modify: `CHANGELOG.md` (as linhas de C++, C# e Python da simulação acumulada)
- Test: `python/tests/test_simulate.py`, `python/tests/dcc/test_dcc_simulate.py`

**Interfaces:**
- Consumes: Tarefa 3.
- Produces: `Simulation.cumulative -> NDArray[np.float64]` (paths, K), `DccSimulation.cumulative` (paths, K, m),
  `checkpoints -> tuple[int, ...]` (vazia sem eles). Com checkpoints, `returns` e `sigma` levantam
  `VolEngineError`; sem eles, `cumulative` levanta `VolEngineError`.

- [ ] **Step 1: Write the failing tests:** `test_cumulative_matches_the_left_sum` (fit, spec e DccFit), com
  `np.array_equal` sobre a soma feita por um laço Python da esquerda para a direita (não `np.cumsum`, que pode
  agrupar de outro jeito) de `returns` da simulação completa; `test_checkpoints_are_validated` com as três listas
  inválidas levantando `VolEngineError`; `test_cumulative_mode_has_no_full_returns`.
- [ ] **Step 2: Run to verify they fail.** O comando de pytest. Expected: FAIL (`unexpected keyword argument
  'checkpoints'`).
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run.** pytest inteiro, mypy e `ctest --preset gcc-release -R "^style$"`. Expected: PASS.
- [ ] **Step 5: Commit** `feat(python): checkpoints= and the cumulative simulation`
