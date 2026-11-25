# Cancelamento cooperativo — plano de implementação

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `ve_fit_batch`, `ve_dcc_fit_batch`, `ve_dcc_standard_errors` e as três simulações aceitam um handle de
cancelamento e devolvem `VE_CANCELLED` pouco depois de ele ser sinalizado. As funções atuais ficam obsoletas, e os
bindings C++, C# e Python expõem o cancelamento.

**Architecture:** um `ve::CancelFlag` (um `std::atomic<bool>`) desce até os pontos de conferência: a iteração do
L-BFGS, o início de cada job, spec, bloco de erros-padrão ou caminho. Ele lança `ve::Cancelled`. `run_in_blocks` já
repassa a exceção depois que todos os blocos param, e `capi::guard` a converte em `VE_CANCELLED`. Na C ABI entram seis
variantes `_with_cancel`, e as funções antigas delegam para elas com `NULL`.

**Tech Stack:** C++23 (GCC/MinGW, gtest), C ABI, wrapper C++ header-only, C# (.NET, LibraryImport, PublicApiAnalyzers,
xUnit), Python (ctypes, pytest).

**Spec:** [docs/superpowers/specs/2026-11-25-volengine-cancelamento-design.md](../specs/2026-11-25-volengine-cancelamento-design.md)

## Global Constraints

- ABI 1 aditiva: `VE_ABI_VERSION` continua `1`, e `python scripts/abi_contract.py check --base <último tag v1.x>` passa,
  sem nenhuma linha removida ou alterada.
- `#define VE_CANCELLED 6`. `ve_last_error()` devolve `"cancelled"` depois de uma chamada cancelada.
- Nomes exatos: `ve_cancel`, `ve_cancel_create`, `ve_cancel_request`, `ve_cancel_free`, `ve_fit_batch_with_cancel`,
  `ve_dcc_fit_batch_with_cancel`, `ve_dcc_standard_errors_with_cancel`, `ve_simulate_fit_with_cancel`,
  `ve_simulate_spec_with_cancel`, `ve_simulate_dcc_with_cancel`, com as assinaturas da seção "C ABI" do spec. O
  `const ve_cancel*` vem logo antes do primeiro parâmetro de saída.
- `c == NULL` faz exatamente o que a função antiga faz, bit a bit.
- Os argumentos são validados antes do primeiro ponto de conferência.
- Numa chamada cancelada, `out` fica com `NULL`: cada um dos `n` handles nos lotes.
- Nenhum resultado guarda o ponteiro do sinal depois que a chamada volta.
- O projeto compila com `-Werror` (raiz e `cpp/tests`) e com `TreatWarningsAsErrors` (C#).
- Os testes C++ rodam no preset `gcc-release`, com `cmake --build --preset gcc-release` e `ctest --preset gcc-release
  -j 8` a partir do PowerShell (o executável chamado pelo Git Bash quebra). O wrapper é testado com
  `cpp/scripts/ci_check.sh`, conforme o CONTRIBUTING.
- Trabalhar no branch atual (`develop`), sem worktree. Commits no estilo do histórico (`feat:`, `docs:`...).
- Comentários e documentação seguem o `STYLEGUIDE.md`.

## Review Focus

1. **Um handle em duas chamadas simultâneas** (o DataPanel pode reusar o mesmo sinal): um único `request` cancela as
   duas. Teste na Tarefa 2.
2. **O handle sinalizado é reusado na próxima chamada**: ela devolve `VE_CANCELLED` logo de cara, porque o sinal não
   volta atrás. Teste na Tarefa 2.
3. **O contexto depois de um cancelamento**: o pool não fica preso, `ve_context_set_threads` volta e um ajuste normal
   no mesmo contexto dá `VE_OK`. Teste na Tarefa 2.
4. **Nada para otimizar** (cópula estática Kendall com mvnorm, spec univariada com tudo fixo): o L-BFGS não roda, mas o
   ponto de conferência do início do job ou spec ainda cancela. Teste na Tarefa 3.
5. **Corrida entre o token do C# e o retorno nativo**: tokens cancelados perto do fim da chamada nunca derrubam o
   processo (o callback não pode rodar depois de o handle ser liberado). O resultado é um valor ou uma
   `OperationCanceledException`, nunca outra coisa. Teste na Tarefa 6.

---

### Task 1: Sinal interno e ponto de conferência no L-BFGS

**Files:**
- Create: `src/core/cancel.hpp`
- Modify: `src/optim/lbfgs.hpp` (`LbfgsOptions`, laço de `Minimizer::run`)
- Modify: `src/core/spec.hpp` (`FitConfig`)
- Modify: `src/fit/fit.cpp` (`minimize`), `src/dcc/fit.cpp` (`minimize`), `src/dcc/copula/static_fit.cpp` (onde
  monta `LbfgsOptions`)
- Test: `tests/unit/optim/lbfgs_test.cpp`

**Interfaces:**
- Produces:
  ```cpp
  namespace ve {
  class CancelFlag {               // request() e requested() são thread-safe; requested() usa memory_order_relaxed
  public:
      void request() noexcept;
      bool requested() const noexcept;
  };
  struct Cancelled final : std::exception { const char* what() const noexcept override; };  // "cancelled"
  void throw_if_cancelled(const CancelFlag* flag);  // nullptr: nunca lança
  }
  // optim::LbfgsOptions::cancel  : const ve::CancelFlag* = nullptr
  // ve::FitConfig::cancel        : const ve::CancelFlag* = nullptr
  ```
- As três funções `minimize` copiam `config.cancel` para `options.cancel`.

- [ ] **Step 1: Write the failing tests** em `lbfgs_test.cpp`, sobre a quadrática que o arquivo já usa:
  - `Lbfgs.RequestedFlagThrowsCancelled`: com `flag.request()` e `options.cancel = &flag`, `EXPECT_THROW(lbfgs_minimize(...),
    ve::Cancelled)`.
  - `Lbfgs.UnrequestedFlagChangesNothing`: com um sinal não pedido, `stop`, `iterations`, `evaluations`, `f` e os bits
    de `x` são iguais aos da execução sem sinal.
  - `Lbfgs.RequestDuringRunStops`: um objetivo que chama `flag.request()` na 3ª avaliação. Espera-se `ve::Cancelled`,
    com menos avaliações que a execução completa.
  - `Lbfgs.ConvergedStartReturnsNormally`: começando no mínimo com o sinal pedido, `stop == Stop::Converged` e nenhuma
    exceção, porque a conferência vem depois dos testes de convergência.
- [ ] **Step 2: Run them to verify they fail.** Run: `cmake --build --preset gcc-release`. Expected: falha de
  compilação (`cancel` não é membro de `LbfgsOptions`).
- [ ] **Step 3: Implement.** Crie `cancel.hpp`. Em `Minimizer::run`, chame `throw_if_cancelled(options_.cancel)` a cada
  volta do laço, logo antes de `iterate()`. Acrescente o campo a `FitConfig` e repasse-o nas três `minimize`.
- [ ] **Step 4: Run the tests to verify they pass.** Run: `ctest --preset gcc-release -j 8 -R Lbfgs`. Expected: PASS.
  Depois rode a suíte inteira: `ctest --preset gcc-release -j 8`, tudo PASS.
- [ ] **Step 5: Commit** `feat(core): cancel flag checked by the L-BFGS loop`

### Task 2: C ABI — handle, status e as variantes univariadas

**Files:**
- Modify: `include/volengine/volengine.h`: `VE_CANCELLED`, `ve_cancel` com as três funções e
  `ve_fit_batch_with_cancel`, `ve_simulate_fit_with_cancel` e `ve_simulate_spec_with_cancel`. O comentário do handle
  traz a semântica do spec (thread-safe, idempotente, não volta atrás, vale para várias chamadas, liberar durante o uso
  é indefinido, `VE_OK` possível quando o sinal chega tarde, `out` em `NULL`).
- Modify: `include/volengine/volengine.abi`, regenerado por `python scripts/abi_contract.py write`.
- Create: `src/capi/cancel_api.cpp`, listado em `CMakeLists.txt` junto dos outros `src/capi`.
- Modify: `src/capi/handles.hpp` (`struct ve_cancel { ve::CancelFlag flag; };`)
- Modify: `src/capi/error.hpp`: `guard` captura `ve::Cancelled` antes de `std::exception` e devolve
  `set_error(VE_CANCELLED, "cancelled")`.
- Modify: `src/capi/batch_api.cpp`: `ve_fit_batch` passa a delegar para a variante com `nullptr`.
- Modify: `src/capi/results/sim_api.cpp`: o mesmo para `ve_simulate_fit` e `ve_simulate_spec`.
- Modify: `src/fit/batch.hpp`, `src/fit/batch.cpp`, `src/fit/fit.cpp`
- Modify: `src/sim/simulate.hpp`, `src/sim/simulate.cpp`
- Test: Create `tests/unit/capi/cancel_test.cpp` e registre-o na lista de `volengine_tests` no `CMakeLists.txt`.

**Interfaces:**
- Consumes: `ve::CancelFlag`, `ve::Cancelled`, `throw_if_cancelled` e `FitConfig::cancel`, da Tarefa 1.
- Produces:
  - `std::vector<FitResult> run_batch(Context&, const Batch&, const CancelFlag* cancel = nullptr)`. Usa uma cópia de
    `batch.config` com `cancel` e confere o sinal no início de cada job.
  - `fit_one` zera `result.config.cancel` antes de voltar.
  - `fill_inference` confere o sinal antes dos erros-padrão.
  - `sim::Request::cancel : const CancelFlag* = nullptr`, conferido a cada caminho dentro do bloco.
  - `ve_cancel_create(NULL)` devolve `VE_INVALID_ARGUMENT`, `ve_cancel_request(NULL)` devolve `VE_INVALID_HANDLE`, e
    `ve_cancel_free(NULL)` não faz nada.

- [ ] **Step 1: Write the failing tests** em `cancel_test.cpp`, usando os ajudantes de `tests/unit/capi_scope.hpp` e
  `simulated_garch.hpp`:
  - `CancelHandle.Lifecycle`: create dá `VE_OK`, dois `request` seguidos dão `VE_OK`, e os casos `NULL` acima.
  - `FitBatchWithCancel.RequestedBeforeCall`: devolve `VE_CANCELLED`, todos os `out[i] == nullptr` e
    `ve_last_error() == std::string("cancelled")`.
  - `FitBatchWithCancel.NullEqualsUnrequested`: com `NULL` e com um handle não sinalizado, as estimativas e o `loglik`
    de cada ajuste são iguais bit a bit.
  - `FitBatchWithCancel.InvalidArgumentWinsOverCancel`: um lote sem jobs, com o handle sinalizado, dá
    `VE_INVALID_ARGUMENT`.
  - `FitBatchWithCancel.RequestFromAnotherThreadStops`: um lote calibrado para levar ≥ 1 s no `gcc-release` (sGARCH,
    norm, muitos jobs de uma série simulada). Um `std::jthread` sinaliza depois de 50 ms. Espera-se `VE_CANCELLED` e
    menos de 500 ms do início à volta.
  - `FitBatchWithCancel.OneHandleCancelsTwoConcurrentCalls` (Review Focus 1): duas threads, o mesmo handle, os dois
    lotes longos. Um único `request` faz as duas devolverem `VE_CANCELLED`.
  - `FitBatchWithCancel.RequestedHandleStaysRequested` (Review Focus 2): depois de uma chamada cancelada, a próxima com
    o mesmo handle devolve `VE_CANCELLED` mesmo com um lote pequeno.
  - `FitBatchWithCancel.ContextUsableAfterCancel` (Review Focus 3): depois de um cancelamento no meio,
    `ve_context_set_threads(ctx, 2)` dá `VE_OK`, e `ve_fit_batch_with_cancel(..., NULL, ...)` no mesmo contexto também.
  - `SimulateWithCancel.RequestedBeforeCall`: para `ve_simulate_fit_with_cancel` e `ve_simulate_spec_with_cancel`,
    `VE_CANCELLED` e `out == nullptr`.
  - `SimulateWithCancel.NullEqualsUnrequested`: retornos e sigma iguais bit a bit.
  - `SimulateWithCancel.RequestFromAnotherThreadStops`: muitos caminhos e horizonte curto, com até ~200 MB de saída e
    ≥ 1 s de trabalho. Sinal depois de 50 ms; espera-se `VE_CANCELLED` em menos de 500 ms.
- [ ] **Step 2: Run them to verify they fail.** Run: `cmake --build --preset gcc-release`. Expected: falha de
  compilação (`ve_cancel_create` não declarado).
- [ ] **Step 3: Implement** a declaração, a definição, o `guard` e o encanamento listado em Interfaces. A conferência
  vem depois de `check_fit_batch_arguments` e das validações de simulação que já existem.
- [ ] **Step 4: Regenerate the contract and run the tests.** Run: `python scripts/abi_contract.py write`, depois
  `python scripts/abi_contract.py check`, `cmake --build --preset gcc-release` e `ctest --preset gcc-release -j 8`.
  Expected: o contrato só ganha linhas (confira no `git diff include/volengine/volengine.abi`) e tudo PASS.
- [ ] **Step 5: Commit** `feat(capi): ve_cancel and the cancellable univariate fit and simulations`

### Task 3: C ABI — variantes DCC

**Files:**
- Modify: `include/volengine/volengine.h`, `include/volengine/volengine.abi` (regenerado)
- Modify: o arquivo que define `ve_dcc_fit_batch` (em `src/capi/dcc/`), `src/capi/dcc/dcc_standard_errors_api.cpp` e
  `src/capi/dcc/dcc_sim_api.cpp`. As três funções atuais delegam para as variantes com `nullptr`.
- Modify: `src/dcc/fit.hpp`, `src/dcc/fit.cpp`
- Modify: `src/dcc/standard_errors.hpp`, `src/dcc/standard_errors.cpp`
- Modify: `src/dcc/future/simulate.hpp`, `src/dcc/future/simulate.cpp`
- Test: `tests/unit/capi/cancel_test.cpp`

**Interfaces:**
- Consumes: Tarefas 1 e 2 (o handle, o `guard` e `FitConfig::cancel`).
- Produces:
  - `fit_dcc_batch(Context&, std::span<const FitResult* const>, std::span<const DccSpec>, const DccOptions&, const
    CancelFlag* cancel = nullptr)`. Usa uma cópia das opções com `config.cancel` (vale para DCC, cópula e cópula
    estática) e confere o sinal no início de cada spec, em `fit_in_batch`.
  - A sobrecarga de `dcc_standard_errors` com pool ganha `const CancelFlag* cancel = nullptr` e confere o sinal no
    início de cada bloco do lambda `run` (todas as etapas paralelas passam por ele). `add_standard_errors` repassa o
    sinal.
  - `DccRequest::cancel : const CancelFlag* = nullptr`, conferido a cada caminho.
  - Se algum resultado DCC guardar um `FitConfig` ou um `DccOptions`, o `cancel` dele é zerado antes da volta.

- [ ] **Step 1: Write the failing tests** em `cancel_test.cpp`, com o painel de `synthetic_panel.hpp` ou os dados
  `dcc3` que `dcc_test.cpp` já lê:
  - `DccFitBatchWithCancel.RequestedBeforeCall`: `VE_CANCELLED`, cada `out[i] == nullptr` e a mensagem `"cancelled"`.
  - `DccFitBatchWithCancel.NullEqualsUnrequested`: estimativas, `loglik` e erros-padrão iguais bit a bit.
  - `DccFitBatchWithCancel.NothingToEstimateStillCancels` (Review Focus 4): cópula estática Kendall com mvnorm e o
    handle sinalizado dão `VE_CANCELLED`.
  - `DccFitBatchWithCancel.RequestFromAnotherThreadStops`: vários specs (dcc, adcc, mvt) com erros-padrão ligados,
    calibrados para ≥ 1 s. Sinal depois de 50 ms; `VE_CANCELLED` em menos de 500 ms.
  - `DccStandardErrorsWithCancel.RequestedBeforeCall`: `VE_CANCELLED` e `out == nullptr`.
  - `DccStandardErrorsWithCancel.InvalidArgumentWinsOverCancel`: `m` diferente do número de séries dá
    `VE_INVALID_ARGUMENT`.
  - `SimulateDccWithCancel.RequestedBeforeCall`, `SimulateDccWithCancel.NullEqualsUnrequested` (retornos, sigma e,
    com correlações ligadas, R e H) e `SimulateDccWithCancel.RequestFromAnotherThreadStops`, com os mesmos limites de
    tempo e até ~200 MB de saída.
- [ ] **Step 2: Run them to verify they fail.** Run: `cmake --build --preset gcc-release`. Expected: falha de
  compilação (`ve_dcc_fit_batch_with_cancel` não declarado).
- [ ] **Step 3: Implement** o que está listado em Interfaces. A conferência vem depois de `check_dcc_batch` e das
  validações atuais de erros-padrão e simulação.
- [ ] **Step 4: Regenerate the contract and run the tests.** Run: `python scripts/abi_contract.py write`, `python
  scripts/abi_contract.py check`, `cmake --build --preset gcc-release` e `ctest --preset gcc-release -j 8`. Expected:
  tudo PASS.
- [ ] **Step 5: Commit** `feat(capi): cancellable DCC fit, standard errors and simulation`

### Task 4: Funções antigas obsoletas

**Files:**
- Modify: `include/volengine/volengine.h`: a macro `VE_DEPRECATED`, logo depois das definições de `VE_API`, e a
  anotação das seis funções antigas.
  ```c
  #if defined(VE_NO_DEPRECATED_WARNINGS)
  #define VE_DEPRECATED(message)
  #elif defined(_MSC_VER)
  #define VE_DEPRECATED(message) __declspec(deprecated(message))
  #else
  #define VE_DEPRECATED(message) __attribute__((deprecated(message)))
  #endif
  ```
  A macro vem **antes** de `VE_API`: `VE_DEPRECATED("use ve_fit_batch_with_cancel with a NULL ve_cancel; removed in ABI
  2") VE_API ve_status ve_fit_batch(...);`. Cada uma das seis recebe uma mensagem nesse formato, com o nome da
  variante.
- Modify: todos os chamadores C/C++ das seis funções passam para `_with_cancel(..., nullptr, ...)`:
  - `tests/unit/capi/batch_test.cpp`, `tests/unit/capi/dcc_test.cpp`, `tests/unit/capi/results/sim_test.cpp`,
    `tests/unit/capi_scope.hpp`
  - `tests/benchmarks/fit_benchmarks.cpp`
  - `cpp/benchmarks/wrapper_overhead.cpp`, `cpp/include/volengine/context.hpp`

  Para achar todos: `grep -rnE 've_(fit_batch|dcc_fit_batch|dcc_standard_errors|simulate_fit|simulate_spec|simulate_dcc)\('
  --include=*.cpp --include=*.hpp`.
- Modify: `cpp/scripts/check_abi_coverage.py`. As seis entram em `UNUSED_ALLOWED`, com um comentário: obsoletas, o
  wrapper chama as variantes `_with_cancel`.
- Create: `tests/unit/capi/deprecated_test.cpp`, com `#define VE_NO_DEPRECATED_WARNINGS` antes de qualquer `#include`.
  Registre-o no `CMakeLists.txt`.
- Modify: `CHANGELOG.md`, seção `## [Unreleased]` com `### Added` (o handle, `VE_CANCELLED` e as seis variantes) e
  `### Deprecated` (as seis funções, com remoção na ABI 2).

**Interfaces:**
- Consumes: as seis variantes das Tarefas 2 e 3.
- Produces: a macro `VE_DEPRECATED(message)` e `VE_NO_DEPRECATED_WARNINGS`.

- [ ] **Step 1: Write the test** `Deprecated.EachDelegatesToItsVariant` em `deprecated_test.cpp`: cada função antiga dá
  o mesmo resultado, bit a bit, que a variante com `NULL` (estimativas e `loglik` nos ajustes, retornos nas
  simulações, erros-padrão no `ve_dcc_standard_errors`).
- [ ] **Step 2: Annotate the header** (sem migrar os chamadores ainda) e rode `cmake --build --preset gcc-release`.
  Expected: FAIL com `-Werror=deprecated-declarations` nos chamadores antigos. Isso mostra que a anotação pega.
- [ ] **Step 3: Migrate every caller** listado em Files e atualize `UNUSED_ALLOWED`.
- [ ] **Step 4: Verify.**
  - `python scripts/abi_contract.py check`: sem diferenças, porque a macro antes de `VE_API` não muda nenhuma linha.
  - `cmake --build --preset gcc-release` e `ctest --preset gcc-release -j 8`: PASS sem avisos. O teste `c_header_check`
    garante que o header ainda compila em C.
  - `python cpp/scripts/check_abi_coverage.py`: PASS.
  - `bash cpp/scripts/ci_check.sh` (PowerShell, com `VOLENGINE_NATIVE_LIB` apontando para a biblioteca do
    `gcc-release`): PASS.
  - `uv run --project python pytest python/tests/packaging -q`: PASS.
- [ ] **Step 5: Commit** `feat(capi): deprecate the six calls without a cancel handle`

### Task 5: Wrapper C++ com `std::stop_token`

**Files:**
- Modify: `cpp/include/volengine/error.hpp`: `Status::cancelled = VE_CANCELLED`.
- Modify: `cpp/include/volengine/context.hpp`: todo método de `Context` que chama uma das seis variantes ganha
  `std::stop_token stop = {}` como último parâmetro. São `fit` (as três sobrecargas), `simulate` (as três),
  `fit_dcc` (as duas) e `standard_errors`.
- Test: Create `cpp/tests/cancel_test.cpp` e registre-o em `cpp/tests/CMakeLists.txt`.
- Modify: `docs/guide/cpp.md`, com uma seção "Cancelamento" (um `std::stop_source` e um `std::jthread` que pede a
  parada).

**Interfaces:**
- Consumes: as variantes `_with_cancel` (Tarefas 2 e 3).
- Produces: um ajudante em `detail`, que vive o tempo da chamada. Com `stop.stop_possible()`, ele cria o `ve_cancel` e
  um `std::stop_callback` que chama `ve_cancel_request`, e o callback é destruído antes de `ve_cancel_free`. Sem isso,
  ele passa `nullptr`. Um token já parado chega ao nativo como handle sinalizado, porque o `stop_callback` roda na hora.

- [ ] **Step 1: Write the failing tests** em `cancel_test.cpp`, com os dados de `fixture_data.hpp`:
  - `Cancel.StoppedTokenGivesCancelled`: com `std::stop_source s; s.request_stop();`, `fit`, `fit_dcc`,
    `standard_errors` e cada `simulate` dão `!result`, com `result.error().status == ve::Status::cancelled`.
  - `Cancel.DefaultTokenUnchanged`: `fit(spec, series)` e `fit(spec, series, {}, std::stop_token{})` dão estimativas
    iguais bit a bit.
  - `Cancel.StopFromAnotherThread`: um `fit` de grade longa (≥ 1 s), com `request_stop()` vindo de um `std::jthread`
    depois de 50 ms. Espera-se `Status::cancelled` em menos de 500 ms.
- [ ] **Step 2: Run them to verify they fail.** Run: `bash cpp/scripts/ci_check.sh`. Expected: falha de compilação (não
  há sobrecarga com `std::stop_token`).
- [ ] **Step 3: Implement** o ajudante e os parâmetros.
- [ ] **Step 4: Run the tests.** Run: `bash cpp/scripts/ci_check.sh`. Expected: PASS, inclusive `abi_coverage` e o
  consumidor do pacote.
- [ ] **Step 5: Commit** `feat(cpp): std::stop_token cancels the wrapper's long calls`

### Task 6: Binding C# com `CancellationToken`

**Files:**
- Modify: `csharp/src/VolEngine/Enums.cs` (`Status.Cancelled = 6`)
- Modify: `csharp/src/VolEngine/Interop/Native/NativeMethods.cs` (handle), `NativeMethods.Univariate.cs`,
  `NativeMethods.Results.cs` e `NativeMethods.Dcc.cs`, com as nove funções novas.
- Modify: `csharp/src/VolEngine/Interop/Handles.cs`: `CancelHandle`, um SafeHandle que libera por `ve_cancel_free`,
  cuja instância padrão vai como `NULL` (o mesmo padrão de `PanelHandle`).
- Modify: `csharp/src/VolEngine/Interop/Calls.cs`, com o ajudante de cancelamento.
- Modify: `csharp/src/VolEngine/Context.cs`, `Interop/Batch.cs`, `Dcc/DccFitBatch.cs`, `Dcc/DccSimulation.cs` e
  `Results/Simulation.cs`, onde o token entra até a chamada nativa.
- Modify: `csharp/src/VolEngine/PublicAPI.Unshipped.txt`, com as linhas novas. As sobrecargas antigas, que perdem os
  valores padrão, entram como `*REMOVED*` da linha antiga mais a linha nova.
- Modify: `csharp/tests/VolEngine.Tests/Interop/CoverageTests.cs`, com um conjunto das seis funções obsoletas excluído
  da exigência de chamada, e um comentário dizendo por quê.
- Modify: `docs/guide/csharp.md` (seção "Cancelamento") e `csharp/tests/VolEngine.Tests/Guide/GuideSnippetsTests.cs`.
- Test: Create `csharp/tests/VolEngine.Tests/Interop/CancellationTests.cs`

**Interfaces:**
- Consumes: as variantes `_with_cancel`.
- Produces:
  - O ajudante interno `Calls.WithCancel(CancellationToken token, Func<CancelHandle, int> call) -> int`. Ele chama
    `token.ThrowIfCancellationRequested()` e passa a instância padrão quando `!token.CanBeCanceled`. Caso contrário,
    cria o handle e faz `using var registration = token.Register(...)`, declarado depois do handle para ser desfeito
    antes. Quando o nativo devolve `VE_CANCELLED`, lança `new OperationCanceledException(token)`.
  - As sobrecargas públicas novas, cada uma com os parâmetros atuais e `CancellationToken cancellationToken = default` no
    fim:
    `Fit(Spec, ReadOnlySpan<double>, FitOptions? = null, double[,]? = null, double[,]? = null, CancellationToken = default)`,
    `Fit(IReadOnlyList<Spec>, Panel, FitOptions? = null, CancellationToken = default)`,
    `FitDcc(DccSpec, IReadOnlyList<Fit>, bool = true, CancellationToken = default)`,
    `FitDcc(IReadOnlyList<DccSpec>, IReadOnlyList<Fit>, bool = true, CancellationToken = default)`,
    `StandardErrors(DccFit, IReadOnlyList<Fit>, CancellationToken = default)` e
    `Simulate(Fit|Spec|DccFit, ..., CancellationToken = default)`.
  - As sobrecargas atuais perdem todos os valores padrão e delegam com `CancellationToken.None`.

- [ ] **Step 1: Write the failing tests** em `CancellationTests.cs`:
  - `Canceled_token_throws_OperationCanceledException`: para `Fit`, `FitDcc`, `StandardErrors` e `Simulate` (os três), um
    token já cancelado lança `OperationCanceledException` com `CancellationToken == token`.
  - `Cancel_during_a_long_fit_stops_it`: `CancelAfter(50 ms)` numa grade de ≥ 1 s lança `OperationCanceledException`
    em menos de 500 ms. Em seguida, `Fit` no mesmo `Context` funciona.
  - `Cancelable_token_not_canceled_gives_the_same_fit`: as estimativas com `new CancellationTokenSource().Token` são
    iguais às de `CancellationToken.None`.
  - `Racing_cancellations_never_crash` (Review Focus 5): 200 ajustes pequenos com `CancelAfter` aleatório entre 0 e
    2 ms. Cada um devolve um `Fit` ou lança `OperationCanceledException`; nenhum lança outra coisa.
  - Verifique se o `ContractTests` já confere `Status` contra os `define` do contrato. Se não conferir, acrescente
    `Status.Cancelled == 6`.
- [ ] **Step 2: Run them to verify they fail.** Run: `VOLENGINE_LIB=$PWD/build/gcc-release/libvolengine.dll dotnet test
  --solution csharp/VolEngine.slnx -c Release`. Expected: falha de compilação (não há sobrecarga com `CancellationToken`).
- [ ] **Step 3: Implement** o que está listado em Files e Interfaces, e a seção do guia com o snippet espelhado em
  `GuideSnippetsTests`.
- [ ] **Step 4: Run the tests.** O mesmo comando. Expected: PASS, inclusive `ContractTests`, `CoverageTests` e o build
  sem avisos (RS0016, RS0026, RS0027).
- [ ] **Step 5: Commit** `feat(csharp): CancellationToken overloads for fits and simulations`

### Task 7: Binding Python

**Files:**
- Modify: `python/src/volengine/_native.py`:
  - `VE_CANCELLED = 6`;
  - as nove funções em `PROTOTYPES`, com as variantes no lugar das antigas nos chamadores;
  - `class Cancelled(VolEngineError)`;
  - `check()` lança `Cancelled` quando o status é `VE_CANCELLED`.
- Create: `python/src/volengine/_cancel.py`, com `class Cancel`: o handle por `create_handle(lib.ve_cancel_create,
  lib.ve_cancel_free)` e `cancel() -> None`, que chama `ve_cancel_request`.
- Modify: `python/src/volengine/api.py`: `cancel: Cancel | None = None`, só de palavra-chave, em `Context.fit` (no
  `FitKeywords`), `fit_dcc`, `standard_errors` e `simulate`, com `None` virando um ponteiro nulo.
- Modify: `python/src/volengine/_dcc.py`: `dcc_standard_errors`, o lote DCC e `DccFit.standard_errors` repassam o
  `cancel`.
- Modify: `python/src/volengine/__init__.py`, exportando `Cancel` e `Cancelled` (no `__all__` também).
- Modify: `docs/guide/python.md`, com a seção "Cancelamento" usando `threading.Timer(prazo, c.cancel)`. O
  `tests/docs/test_docs.py` roda o snippet.
- Modify: `CHANGELOG.md`, nas linhas de Python e C# da seção `[Unreleased]`, se as Tarefas 5 e 6 não as acrescentaram.
- Test: Create `python/tests/test_cancel.py`

**Interfaces:**
- Consumes: as variantes `_with_cancel`.
- Produces: `ve.Cancel` (com `.cancel()`), `ve.Cancelled` (subclasse de `ve.VolEngineError`, com `status == 6`) e o
  argumento `cancel=` nos métodos acima.

- [ ] **Step 1: Write the failing tests** em `test_cancel.py`:
  - `test_cancelled_before_the_call_raises`, parametrizado em `fit`, `fit_dcc`, `standard_errors`,
    `simulate(fit)`, `simulate(spec)` e `simulate(dcc_fit)`: com `c = ve.Cancel(); c.cancel()`, a chamada com
    `cancel=c` lança `ve.Cancelled`.
  - `test_cancelled_is_a_volengine_error`: `issubclass(ve.Cancelled, ve.VolEngineError)` e `error.status == 6`.
  - `test_timer_cancels_a_long_fit`: `threading.Timer(0.05, c.cancel)` numa grade de ≥ 1 s lança `ve.Cancelled` em
    menos de 0,5 s, o que mostra que o GIL está solto durante a chamada.
  - `test_unrequested_cancel_gives_the_same_fit`: as estimativas com `cancel=ve.Cancel()` são iguais às sem `cancel`.
- [ ] **Step 2: Run them to verify they fail.** Run: `VOLENGINE_LIB=$PWD/build/gcc-release/libvolengine.dll uv run
  --project python pytest python/tests/test_cancel.py -q`. Expected: FAIL (`module 'volengine' has no attribute
  'Cancel'`).
- [ ] **Step 3: Implement** o que está listado em Files.
- [ ] **Step 4: Run the tests.** `uv run --project python pytest python/tests -q` e `uv run --project python mypy
  python/src`. Expected: PASS, sem erros do mypy.
- [ ] **Step 5: Commit** `feat(python): Cancel handle and the cancel= keyword`
