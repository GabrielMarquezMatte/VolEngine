# VolEngine plano 6b: wrapper C++ (implementação)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Objetivo:** um projeto C++ consome o VolEngine por `FetchContent` ou `find_package` e usa uma API C++23 em RAII,
sem tocar na interface C. O wrapper é o segundo consumidor da ABI: o que ele achar de estranho vai para o 6c.

**Arquitetura:**
- Header-only em `cpp/include/volengine/`, namespace `ve`, sobre `<volengine/volengine.h>`. Cada handle C vira um tipo
  move-only; cada chamada que devolve `ve_status` vira `std::expected<T, ve::Error>`.
- `cpp/CMakeLists.txt` define `volengine::volengine` (INTERFACE) e `volengine::native` (IMPORTED, a biblioteca baixada
  da release da tag ou a indicada por `VOLENGINE_NATIVE_LIB`).
- Testes em GoogleTest contra a biblioteca do build. Um script confere que cada `VE_API` é chamada por algum header.

**Stack:** C++23, CMake 3.28, GoogleTest 1.17, Google Benchmark 1.9.1, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-11-15-volengine-06-release-design.md` §5.2 (e §1 a §4). Modelo de referência:
`C:/Users/x/Documents/Programs/Csharp/ExcelReader/cpp` (`CMakeLists.txt`, `cmake/FetchNativeLib.cmake`,
`cmake/CopyNativeLibrary.cmake`, `cmake/excelreaderConfig.cmake.in`, `tests/package`).

## Restrições globais

- STYLEGUIDE do repositório vale para `cpp/` (o `check_style.py` varre a pasta): nunca `else`, função com até 60
  linhas, até 5 parâmetros, aninhamento até 3, no máximo 8 módulos por pasta, sem `utils`/`helpers`. Inglês no código e
  nos READMEs; os documentos de `docs/superpowers/` ficam em português.
- Interface C intocada: nenhuma assinatura, enum ou função nova em `include/volengine/volengine.h`. Se o wrapper não
  fecha sem uma mudança, parar e perguntar ao autor. Todo atrito com a ABI vai para
  `docs/superpowers/notes/2026-11-15-abi-findings-6b.md` (insumo do 6c): função, o que atrapalhou, o que o wrapper fez.
- Erros: `ve::Result<T> = std::expected<T, ve::Error>`. `Error` copia `ve_last_error()` na hora (o ponteiro vale só até a
  próxima chamada na thread). Nenhuma exceção sai do wrapper.
- Visões (`std::span<const double>`) são zero-copy e valem enquanto o objeto dono viver; cada tipo diz isso no
  comentário do header. `Forecast`, `Simulation` e as variantes DCC não dependem do `Fit` de origem (a C copia).
- Nomes de enumeradores: o sufixo minúsculo da macro C (`VE_MODEL_GJRGARCH` vira `Model::gjrgarch`). Exceções:
  `VE_DIST_STD` vira `Dist::student`, `VE_DCC_MODEL_STATIC` vira `DccModel::static_` (`static` é palavra reservada), e `VE_FIT_*` vira `FitStatus::{converged, max_iter, failed, insufficient_data}`.
- Padrões iguais aos do Python (`python/src/volengine/_spec.py`): `p = q = 1`, ARMA(0,0), `Dist::norm`, com média,
  `FitOptions{500, 1e-6, VarInit::sample, standard_errors = true}`.
- Assets e nomes: `volengine-native-<os>-<arch>.<ext>` do 6a (`os` em `win`, `osx`, `linux`, `linux-musl`; `arch` em
  `x64`, `arm64`). No Windows há também `.dll.a` (MinGW) e `.lib` (MSVC) com o mesmo prefixo.
- O arquivo baixado é guardado com o nome que o loader procura: `libvolengine.so` no Linux e `libvolengine.dylib` no
  macOS (SONAME e `install_name` `@rpath/libvolengine.dylib` do build), e o nome do asset no Windows (o `.lib` do 6a foi
  gerado com `/name:volengine-native-win-x64.dll`).
- `VOLENGINE_NATIVE_LIB` (variável de ambiente) aponta para a biblioteca local e desliga o download. No Windows os
  arquivos `volengine.def` e `libvolengine.dll.a` têm de estar ao lado da DLL (o build local MinGW e o `_lib` do wheel
  têm os dois).
- Fora de uma tag exata e sem `VOLENGINE_NATIVE_LIB`, o configure falha com mensagem clara. `VOLENGINE_VERSION` (cache)
  vem de `git describe --tags --exact-match` ou é passada.
- Windows local: compilar a biblioteca e os testes pelo PowerShell com `C:/msys64/ucrt64/bin` primeiro no PATH (o `ld`
  chamado pelo Bash do Claude grava `--output-def` vazio, e a libstdc++ do Git mingw64 não tem os símbolos).
- Push e criação de tag só com o ok do autor. Sem `Co-Authored-By` nos commits.

## Interface pública (decisões que os testes usam)

Headers em `cpp/include/volengine/` (7 arquivos, abaixo do limite de 8): `error.hpp`, `input.hpp`, `fit.hpp`,
`simulation.hpp`, `dcc.hpp`, `context.hpp`, `volengine.hpp` (inclui os outros). Tudo `inline`/template, nada a compilar.

- `error.hpp`: `enum class Status : int32_t` (`ok`, `invalid_argument`, `invalid_handle`, `unsupported`,
  `out_of_memory`, `internal_error`); `struct Error { Status status; std::string message; }`;
  `Result<T>`; `detail::check(ve_status) -> Result<void>`; `detail::check_abi_version(int32_t native, int32_t expected
  = VE_ABI_VERSION) -> Result<void>`; `Result<void> check_abi()`; `detail::Owned<T, Free>` (unique_ptr com deleter).
- `input.hpp`:
  - enums `Model`, `Dist`, `VarInit`; `struct Spec { Model model; int p, q, arma_ar, arma_ma; Dist dist; bool
    include_mean; int mxreg, vxreg, archm; bool arfima, variance_targeting; std::vector<std::pair<std::string, double>>
    fixed, start; Result<void> validate() const; }`; `struct FitOptions { int max_iter; double tol; VarInit var_init;
    bool standard_errors; }`;
  - `class Panel` (visão emprestada): `static Result<Panel> create(std::span<const double>, std::size_t rows,
    std::size_t cols)` (ordem C); `static Result<Panel> create_strided(std::span<const double>, rows, cols,
    std::ptrdiff_t stride_rows, std::ptrdiff_t stride_cols)`; `Result<void> set_ranges(std::span<const std::size_t>
    starts, std::span<const std::size_t> lengths)`; `ve_panel* native() const noexcept`;
  - `class Batch`: `static Result<Batch> create()`; `set_specs(std::span<const Spec>)`, `set_panel(const Panel&)`,
    `set_jobs(std::span<const std::size_t> spec, std::span<const std::size_t> series)`, `set_options(const
    FitOptions&)`, `set_regressors(std::span<const Panel* const> mxreg, std::span<const Panel* const> vxreg)` (cada um
    `Result<void>`); `Result<std::size_t> job_count() const`.
- `fit.hpp` (todos `Result<...>` quando chamam a C):
  - `class Fit` (move-only): `status() -> FitStatus`, `nobs()`, `param_count()`, `param_names() ->
    std::vector<std::string_view>`, `estimates()`, `std_errors(SeKind)`, `t_values(SeKind)`, `p_values(SeKind)` (`enum
    class SeKind { hessian, robust }`), `loglik()`, `aic/bic/hqic/sic(IcScale = IcScale::per_obs)`, `sigma()`,
    `residuals()`, `std_residuals()`, `fitted()` (`std::span<const double>`);
  - diagnósticos: `ljung_box(std::size_t lag, bool squared = false, std::size_t fitdf = 0) -> Result<Test>`,
    `arch_lm(lag, fitdf)`, `sign_bias() -> Result<SignBias>`, `gof(std::size_t groups = 20)`, `nyblom() ->
    Result<Nyblom>`, e a função livre `nyblom_critical(std::size_t parameters) -> Result<std::array<double, 3>>`.
    `struct Test { double statistic, p_value; }`, `struct SignBias { Test sign, negative, positive, joint; }`,
    `struct Nyblom { std::vector<double> individual; double joint; std::vector<std::int32_t> estimated; }`;
  - `forecast(std::size_t horizon, const Panel* mxreg = nullptr, const Panel* vxreg = nullptr) -> Result<Forecast>`;
    `class Forecast`: `horizon()`, `sigma()`, `mean()`.
- `simulation.hpp`: `struct SimOptions { std::size_t horizon; std::size_t paths; std::uint64_t seed; bool correlations
  = false; const Panel* mxreg = nullptr; const Panel* vxreg = nullptr; }`; `class Simulation`: `paths()`, `horizon()`,
  `returns()`, `sigma()` (paths x horizon, linha = caminho).
- `dcc.hpp`: enums `DccModel`, `DccDist`, `Copula`, `StaticMethod`, `DccCriterion`; `struct DccSpec { DccModel model;
  int p, q; DccDist dist; std::vector<std::int32_t> groups; fixed, start; Copula copula; StaticMethod static_method;
  }`; `struct DccOptions { bool standard_errors = true; }`; `class DccFit` (move-only): `status()`, `nobs()`,
  `series()`, `param_count()`, `param_names()`, `estimates()`, `std_errors/t_values/p_values(SeKind)`, `loglik()`,
  `criterion(DccCriterion, IcScale = per_obs)`, `correlation()`, `covariance()` (T x m x m), `forecast(std::size_t
  horizon, const DccFuture* = nullptr) -> Result<DccForecast>`; `class DccFuture`: `static Result<DccFuture>
  create(series, horizon)`, `set(series, const Panel* mxreg, const Panel* vxreg)`; `class DccForecast`: `horizon()`,
  `series()`, `correlation()`, `covariance()`, `sigma()`, `mean()`; `class DccSimulation`: `paths()`, `horizon()`,
  `series()`, `returns()`, `sigma()`, `correlation()`, `covariance()`.
- `context.hpp`: `class Context` (move-only): `static Result<Context> create(int threads = 0)` (confere a ABI),
  `set_threads(int)`, `threads()`; `fit(const Batch&) -> Result<std::vector<Fit>>`; `fit(std::span<const Spec>, const
  Panel&, const FitOptions& = {})` (todas as specs sobre todas as séries, índice `spec * series + série`); `fit(const
  Spec&, std::span<const double> series, const FitOptions& = {}) -> Result<Fit>`; `simulate(const Fit&, const
  SimOptions&)`, `simulate(const Spec&, const SimOptions&)`, `simulate(const DccFit&, const SimOptions&, const
  DccFuture* = nullptr) -> Result<DccSimulation>`; `fit_dcc(std::span<const DccSpec>, std::span<const Fit>, const
  DccOptions& = {}) -> Result<std::vector<DccFit>>`; `fit_dcc(const DccSpec&, std::span<const Fit>, const DccOptions&
  = {}) -> Result<DccFit>`; `standard_errors(const DccFit&, std::span<const Fit>) -> Result<DccFit>`.

## Foco da revisão

1. **Visão que sobrevive ao dono:** o `Forecast`, a `Simulation` e o `DccForecast` têm de seguir válidos depois de
   liberar o `Fit`, o `DccFit` e as univariadas; um `DccFit` prevê depois de liberar as `Fit` (Task 4 e 6).
2. **Erro que vaza ou perde a mensagem:** uma spec inválida (`p = 0`) devolve `Error` com mensagem não vazia, sem
   handle pendurado; a mensagem copiada continua certa depois de chamadas bem-sucedidas na mesma thread. Num lote com
   erro nenhum `Fit` sai (Task 1 e 2).
3. **Entrada degenerada:** painel sem linhas, série com NaN, série curta demais (`FitStatus::insufficient_data` não é
   erro), `fitdf` ou `lag` fora do intervalo nos diagnósticos e erros-padrão pedidos com `standard_errors = false`
   voltam como `Error` ou como status, nunca como falha de segmentação (Task 2 e 3).
4. **DLL que não carrega no Windows:** o executável sai sem `main` quando a DLL não está ao lado. Todo executável do
   pacote usa `volengine_copy_native_library`, e o consumidor instalado roda de verdade (Task 1 e 7).
5. **Biblioteca de outra versão da ABI:** `Context::create` falha com mensagem que diz as duas versões, em vez de seguir
   com um layout que não confere (Task 1).

---

### Task 1: esqueleto do pacote e `error.hpp`

**Arquivos:**
- Criar: `cpp/CMakeLists.txt`, `cpp/cmake/FetchNativeLib.cmake`, `cpp/cmake/CopyNativeLibrary.cmake`,
  `cpp/include/volengine/error.hpp`, `cpp/include/volengine/context.hpp` (só `Context::create`, `set_threads`,
  `threads` por enquanto), `cpp/include/volengine/volengine.hpp`
- Criar: `cpp/tests/CMakeLists.txt`, `cpp/tests/error_test.cpp`, `cpp/tests/asset_names.cmake`

**Interfaces:**
- Produz: alvos `volengine::volengine` (INTERFACE, inclui `cpp/include` e `../include`, `cxx_std_23`) e
  `volengine::native` (IMPORTED); função CMake `volengine_copy_native_library(target)`; função CMake
  `volengine_asset_name(os arch out_variable)`; `ve::Error`, `ve::Result`, `ve::detail::check`,
  `ve::detail::check_abi_version`, `ve::Context` básico.
- Opções: `VOLENGINE_BUILD_TESTS` (padrão OFF, GoogleTest por `FetchContent`), `VOLENGINE_VERSION` (cache).

- [ ] **Passo 1: Escrever os testes que falham.**
  - `asset_names.cmake` (roda com `cmake -P`, registrado como teste `asset_names`): para as 7 combinações
    (`win x64`, `osx arm64`, `osx x64`, `linux x64`, `linux arm64`, `linux-musl x64`, `linux-musl arm64`) confere que
    `volengine_asset_name` devolve exatamente `volengine-native-<os>-<arch>.<dll|dylib|so>`; a lista esperada vem do
    §5.1 do 6a (9 assets menos `.dll.a` e `.lib`).
  - `error_test.cpp`:
    - `CheckOkIsSuccess`: `ve::detail::check(VE_OK).has_value()`.
    - `CheckErrorCarriesStatusAndMessage`: `ve_spec_create(999, &spec)` (modelo inválido) passado a `check` dá
      `Status::invalid_argument` e `message` não vazia.
    - `MessageSurvivesLaterCalls`: depois do erro acima, uma chamada bem-sucedida (`ve_abi_version()`) não muda a
      `message` já copiada (foco 2).
    - `AbiVersionMismatchNamesBothVersions`: `check_abi_version(7, 0)` falha e a mensagem contém `7` e `0` (foco 5).
    - `ContextCreateAndThreads`: `Context::create(2)`; `threads()` vale 2; `set_threads(0)` aceito; `set_threads(-1)`
      devolve `Status::invalid_argument`.
- [ ] **Passo 2: Rodar e ver falhar** (configurar `cpp/` com `VOLENGINE_NATIVE_LIB` e compilar: falta header).
- [ ] **Passo 3: Implementar.**
  - `FetchNativeLib.cmake`: função `volengine_fetch_native_lib()` e `volengine_asset_name`. Detecta o SO (musl por
    `file(GLOB /lib/ld-musl-*.so.1)`) e a arquitetura (`CMAKE_SYSTEM_PROCESSOR`). Com `VOLENGINE_NATIVE_LIB`, usa o
    caminho (absolutizado). Sem ela, baixa
    `https://github.com/GabrielMarquezMatte/VolEngine/releases/download/${VOLENGINE_VERSION}/<asset>` para
    `${CMAKE_BINARY_DIR}/volengine-native/` com o nome que o loader procura (restrições globais) e falha com
    `FATAL_ERROR` se o download falhar.
  - No Windows: baixa também o `.lib` (MSVC) ou o `.dll.a` (MinGW) e usa como `IMPORTED_IMPLIB`. Com
    `VOLENGINE_NATIVE_LIB` no MSVC, gera o `.lib` com `lib.exe /def:` sobre o `volengine.def` vizinho, com uma linha
    `LIBRARY <nome da DLL>` na frente (como o ExcelReader faz); no MinGW usa o `.dll.a` vizinho.
  - `IMPORTED_SONAME` recebe `libvolengine.so` ou `@rpath/libvolengine.dylib`, para o executável gravar o nome certo.
  - `error.hpp`, `Context::create` (chama `check_abi()` antes de `ve_context_create`) e o `check` com
    `ve_last_error()`.
  - `cpp/CMakeLists.txt`: `VOLENGINE_VERSION` por `git describe --tags --exact-match`; fora de tag e sem a variável
    de ambiente, `FATAL_ERROR` dizendo as duas saídas. Os testes entram com `add_subdirectory(tests)` quando
    `VOLENGINE_BUILD_TESTS`.
- [ ] **Passo 4: Rodar os testes.**
  - Linux e macOS: a biblioteca do build local, `VOLENGINE_NATIVE_LIB=<caminho>`.
  - Windows local: `python python/scripts/build_native.py` pelo PowerShell,
    `VOLENGINE_NATIVE_LIB=python/src/volengine/_lib/libvolengine.dll`, `cmake -S cpp -B build/cpp -G Ninja
    -DVOLENGINE_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release` com g++ do ucrt64, depois `cmake --build build/cpp` e
    `ctest --test-dir build/cpp --output-on-failure`.
  - Esperado: `asset_names` e os 5 casos de `error_test` passam. O teste do executável só roda no Windows se a DLL
    foi copiada (`volengine_copy_native_library`).
- [ ] **Passo 5: Commit** — `feat(cpp): package skeleton, error type and context creation`

### Task 2: entradas, lote e ajuste

**Arquivos:**
- Criar: `cpp/include/volengine/input.hpp`, `cpp/include/volengine/fit.hpp` (`Fit` só com `status`, `nobs`,
  `param_count`, `param_names`, `estimates`, `loglik`)
- Modificar: `cpp/include/volengine/context.hpp` (`fit` nas três formas), `volengine.hpp`
- Testar: `cpp/tests/fit_test.cpp`, `cpp/tests/fixture_data.hpp` (leitor de CSV dos fixtures, sem dependência)

**Interfaces:**
- Consome: Task 1.
- Produz: `Spec`, `FitOptions`, `Panel`, `Batch`, `Fit` (parte), `Context::fit` (`Interface pública`).
- `fixture_data.hpp`: `std::vector<std::vector<double>> read_csv_columns(const std::string& path)` (cabeçalho pulado);
  o caminho dos dados entra por `VOLENGINE_FIXTURE_DIR`, definido no CMake dos testes como `../tests/reference/data`.

- [ ] **Passo 1: Escrever os testes que falham.**
  - `SpecValidateRejectsOrderZero`: `Spec{.p = 0}.validate()` dá `invalid_argument` com mensagem (foco 2).
  - `FitSingleSeriesMatchesRugarch`: série `t5000` de `sgarch11_norm.csv`, `Spec{}` (sGARCH(1,1) norm), `FitOptions` com
    `tol = 1e-8`: `status() == converged`; `param_names()` é `{mu, omega, alpha1, beta1}`; `loglik()` e `estimates()`
    batem com o `rugarch.json` do caso `t5000/sgarch11_norm` (`loglik` -5058.359626076076; `mu` 0.045689656391604716,
    `omega` 0.010692733778622033, `alpha1` 0.07516720858429206, `beta1` 0.9030746410118694) na tolerância do teste
    Python do mesmo caso (`python/tests/oracles/test_oracles.py`).
  - `PanelFitIndexesSpecMajor`: 2 specs (norm e std) sobre as 3 colunas de `sgarch11_norm.csv`; 6 fits; o fit
    `[1 * 3 + 2]` é o da spec 1 na série 2 (conferir a distribuição pelo nome `shape` em `param_names`).
  - `BatchJobsSelectSubset`: lote com 2 specs e 3 séries e 2 jobs `(0, 2)` e `(1, 0)`; `job_count() == 2`; 2 fits.
  - `BatchWithBadSpecReturnsNoFits`: um lote com a spec `p = 0` devolve `Error` com mensagem que começa com
    `spec 0:`; nada vaza (rodar sob o sanitizer do Passo 4).
  - `ShortSeriesIsStatusNotError`: série de 5 pontos: o ajuste volta `Result` com valor e
    `FitStatus::insufficient_data` (foco 3). `NaNInSeriesIsError`: uma série com NaN dá `Error::invalid_argument`.
  - `EmptyPanelIsError`: `Panel::create({}, 0, 0)` é aceito pela C (`t * n == 0`); ajustar sobre ele dá `Error`.
  - `StridedPanelEqualsContiguous`: a mesma matriz em ordem Fortran com `create_strided` dá o mesmo `loglik`.
- [ ] **Passo 2: Rodar e ver falhar.**
- [ ] **Passo 3: Implementar.**
  - `Spec` vira `ve_spec` por `detail::make_spec(const Spec&) -> Result<detail::Owned<ve_spec, ve_spec_free>>`
    (cria, aplica cada setter com `check`, `fixed` e `start` por nome). `validate()` usa `ve_spec_validate`.
  - `Batch` guarda os handles; `set_specs` cria um `ve_spec*` por spec e passa o vetor de ponteiros; a C copia.
  - `Context::fit(const Batch&)` dimensiona a saída com `ve_batch_job_count` e embrulha cada `ve_fit*` em `Fit`.
  - As duas sobrecargas de conveniência montam o `Batch` (todos os jobs `spec x série`, ou uma spec e uma série).
  - Cada função com no máximo 5 parâmetros e 60 linhas: dividir `make_spec` em `apply_orders`, `apply_regressors`,
    `apply_named_values`.
- [ ] **Passo 4: Rodar os testes** (como na Task 1, mais um build com `-fsanitize=address,undefined` no Linux). Esperado:
  todos verdes, sem relatório do sanitizer.
- [ ] **Passo 5: Commit** — `feat(cpp): specs, panels, batches and univariate fits`

### Task 3: resultado do ajuste: erros-padrão, critérios, séries e diagnósticos

**Arquivos:**
- Modificar: `cpp/include/volengine/fit.hpp`
- Testar: `cpp/tests/fit_test.cpp` (mesmo arquivo), `cpp/tests/diagnostics_test.cpp`

**Interfaces:**
- Consome: Task 2. Produz: o resto do `Fit` (`Interface pública`) e `nyblom_critical`.

- [ ] **Passo 1: Escrever os testes que falham.**
  - `StdErrorsMatchRugarch`: no ajuste de `t5000/sgarch11_norm`, `std_errors(SeKind::hessian)` bate com `se` do
    `rugarch.json` e `std_errors(SeKind::robust)` com `qml_robust_se`, na tolerância do teste Python.
  - `TAndPValuesFollowStdErrors`: `t_values[i] == estimates[i] / std_errors[i]` e `p_values` em `[0, 1]`.
  - `CriteriaScales`: `aic(IcScale::total) == aic(IcScale::per_obs) * nobs` e `bic` maior que `aic` para `nobs > 7`.
  - `SeriesViewsHaveNobsLength`: `sigma()`, `residuals()`, `std_residuals()` e `fitted()` têm `nobs` elementos, e
    `sigma()` é positivo.
  - `StandardErrorsOffMakesSeAccessorsFail`: com `standard_errors = false`, `std_errors`, `t_values`, `p_values` e
    `nyblom` dão `invalid_argument`, e `estimates` e `loglik` seguem iguais (foco 3).
  - `LjungBoxMatchesRugarch` e `SignBiasMatchesRugarch`: contra `rugarch_diagnostics.json` (um caso de cada, a escolha
    fica com o implementador, registrada no ledger). `LjungBoxLagOutOfRange`: `lag = nobs` dá `invalid_argument`
    (foco 3). `ArchLmRequiresFitdfBelowLag`: `fitdf = lag` dá erro.
  - `NyblomCriticalValuesIncrease`: `nyblom_critical(1)` é `{0.353, 0.470, 0.748}`, `nyblom_critical(20)` é `{4.22, 4.52, 5.13}`
    (os valores de `tests/unit/capi/results/diagnostics_test.cpp`) e `nyblom_critical(0)` e `(21)` dão erro. `NyblomReportsEstimatedFlags`: `estimated` tem `param_count`
    entradas, todas 1 num ajuste livre.
  - `GofReturnsStatisticAndP`: `gof(20)` devolve `p_value` em `[0, 1]`.
- [ ] **Passo 2: Rodar e ver falhar. Passo 3: implementar** os acessores com um auxiliar privado que lê um vetor de
  `param_count` doubles (`values(function, se_kind)`) e outro que lê as visões `(data, len)` num `std::span`.
- [ ] **Passo 4: Rodar os testes.** Esperado: verdes. **Passo 5: Commit** — `feat(cpp): fit results and diagnostics`

### Task 4: previsão e simulação univariadas

**Arquivos:**
- Criar: `cpp/include/volengine/simulation.hpp`
- Modificar: `cpp/include/volengine/fit.hpp` (`forecast`, `Forecast`), `cpp/include/volengine/context.hpp`
  (`simulate` de `Fit` e de `Spec`), `volengine.hpp`
- Testar: `cpp/tests/future_test.cpp`

**Interfaces:**
- Consome: Task 3. Produz: `Forecast`, `SimOptions`, `Simulation`, `Context::simulate(const Fit&|const Spec&, ...)`.

- [ ] **Passo 1: Escrever os testes que falham.**
  - `ForecastMatchesRugarch`: previsão de horizonte 5 do ajuste `t5000/sgarch11_norm` contra o primeiro caso de
    `rugarch_forecast.json` desse ajuste (`sigma` e `mean`, tolerância do teste Python).
  - `ForecastOutlivesFit` (foco 1): o `Fit` sai de escopo e `sigma()` do `Forecast` ainda tem os mesmos valores.
  - `ForecastRejectsZeroHorizon`: horizonte 0 dá `invalid_argument`.
  - `SimulateFitIsReproducibleAndPrefixStable`: a mesma semente dá os mesmos caminhos; os 10 primeiros caminhos de uma
    simulação de 50 são os de uma de 10.
  - `SimulationOutlivesFitAndContext` (foco 1): a `Simulation` segue válida depois de liberar o `Fit`, o `Context` e as
    opções.
  - `SimulateSpecNeedsAllParametersFixed`: uma `Spec` com parâmetros livres dá `invalid_argument` com mensagem; com
    `mu, omega, alpha1, beta1` fixos simula, e `returns()` tem `paths * horizon` elementos.
  - `SimulationRegressorsMustMatchSpec`: `SimOptions` com `mxreg` para uma spec sem regressor dá erro.
- [ ] **Passo 2: Rodar e ver falhar. Passo 3: implementar.** `SimOptions` vira `ve_sim_options` dentro de
  `simulate`; `Simulation` guarda o handle `ve_sim` e expõe as visões.
- [ ] **Passo 4: Rodar os testes. Passo 5: Commit** — `feat(cpp): univariate forecast and simulation`

### Task 5: DCC e Copula-DCC

**Arquivos:**
- Criar: `cpp/include/volengine/dcc.hpp` (`DccSpec`, `DccOptions`, `DccFit`; `DccFuture`, `DccForecast`,
  `DccSimulation` e `DccFit::forecast` entram na Task 6)
- Modificar: `cpp/include/volengine/context.hpp` (`fit_dcc`, `standard_errors`), `volengine.hpp`
- Testar: `cpp/tests/dcc_test.cpp`

**Interfaces:**
- Consome: Task 3. Produz: `DccSpec`, `DccOptions`, `DccFit` (sem `forecast`), `Context::fit_dcc`,
  `Context::standard_errors`.

- [ ] **Passo 1: Escrever os testes que falham.** Painel `dcc3.csv` (colunas `A`, `B`, `C`); univariadas sGARCH(1,1) norm
  com `start` igual a `panels.dcc3.univariate` do `rmgarch_dcc.json` e `FitOptions{.tol = 1}` (como o teste Python).
  - `DccMatchesRmgarch`: `DccSpec{.model = DccModel::dcc, .dist = DccDist::mvnorm}` com `standard_errors = false`:
    `status() == converged`, `loglik()` perto de -6082.940870895449 e `estimates()` perto de `dcca1 = 0.04316409231673219`,
    `dccb1 = 0.914269645079578`, na tolerância do teste Python de `dcc3_DCC_mvnorm` (`test_dcc_oracles.py`).
  - `CorrelationIsRowMajorTimesMByM`: `correlation().size() == nobs * 3 * 3`, diagonal 1 e simetria em todo `t`.
  - `CovarianceIsComputedOnFirstAccess`: `covariance()` tem o mesmo tamanho e diagonal igual a `sigma^2` das univariadas.
  - `GridKeepsSpecOrder`: 3 specs (dcc, adcc, fdcc com `groups = {1, 1, 2}`) com mvnorm devolvem 3 fits na ordem.
  - `DccStandardErrorsFromFitsMatchOption`: `standard_errors(dcc_fit, fits)` dá `std_errors` iguais aos de um ajuste
    com `DccOptions{.standard_errors = true}` (rel 1e-8).
  - `DccRejectsSingleSeries`: `fit_dcc` com uma `Fit` só dá `invalid_argument`.
  - `DccStandardErrorsOffMakesSeAccessorsFail`: erro em `std_errors` (foco 3).
  - `CopulaDccAndStaticFit`: `copula = Copula::parametric`, `dist = DccDist::mvt` converge; `model = DccModel::static_`
    com `static_method = StaticMethod::kendall` e copula parametric converge. `DccCriterionMatchesLoglik`: `criterion(bic,
    IcScale::total) == -2 * loglik + k * ln(nobs)`, com `k = param_count`.
- [ ] **Passo 2: Rodar e ver falhar. Passo 3: implementar** (`DccBatch` interno: `ve_dcc_batch_*`; a saída é
  dimensionada com `ve_dcc_batch_spec_count`). **Passo 4: Rodar os testes.**
- [ ] **Passo 5: Commit** — `feat(cpp): DCC, aDCC, FDCC, static and Copula-DCC fits`

### Task 6: previsão e simulação DCC

**Arquivos:**
- Modificar: `cpp/include/volengine/dcc.hpp`, `cpp/include/volengine/context.hpp` (`simulate(const DccFit&, ...)`)
- Testar: `cpp/tests/dcc_future_test.cpp`

**Interfaces:**
- Consome: Tasks 4 e 5. Produz: `DccFuture`, `DccForecast`, `DccSimulation`, `DccFit::forecast`.

- [ ] **Passo 1: Escrever os testes que falham.**
  - `DccForecastMatchesRmgarch`: horizonte 3 do `dcc3_DCC_mvnorm` contra `rmgarch_dcc_forecast.json` (`R`, `sigma`,
    `mean`), tolerância do teste Python.
  - `DccForecastOutlivesUnivariateFits` (foco 1): liberar as `Fit` univariadas e o `DccFit` original e ler o
    `DccForecast`; e prever de um `DccFit` depois de liberar as `Fit`.
  - `DccForecastShapes`: `correlation()` e `covariance()` têm `horizon * m * m`, `sigma()` e `mean()` `horizon * m`.
  - `DccFutureRejectsWrongSeriesCount`: `DccFuture` com séries diferentes das do fit dá erro com o número da série.
  - `DccSimulationReproducible`: mesma semente, mesmos `returns()`; `paths * horizon * m` elementos.
  - `DccSimulationCorrelationsOnlyWhenAsked`: `correlation()` dá `invalid_argument` sem `SimOptions::correlations`, e
    com ele tem `paths * horizon * m * m`.
  - `DccSimulationRefusesOptionRegressors`: `SimOptions::mxreg` preenchido dá `invalid_argument`.
- [ ] **Passo 2: Rodar e ver falhar. Passo 3: implementar. Passo 4: Rodar os testes.**
- [ ] **Passo 5: Commit** — `feat(cpp): DCC forecast and simulation`

### Task 7: cobertura da ABI e achados

**Arquivos:**
- Criar: `cpp/scripts/check_abi_coverage.py`, `docs/superpowers/notes/2026-11-15-abi-findings-6b.md`
- Modificar: `cpp/tests/CMakeLists.txt` (teste `abi_coverage`)

**Interfaces:**
- Produz: `check_abi_coverage.py [--header PATH] [--wrapper DIR]` sai com 0 ou 1. Funções: `declared(header_text:
  str) -> set[str]` (os nomes `VE_API`), `used(wrapper_text: str) -> set[str]` (os `ve_*` chamados) e
  `UNUSED_ALLOWED: frozenset[str]` (as funções que o wrapper decide não chamar, com o motivo em comentário).

- [ ] **Passo 1: Escrever os testes que falham** (`python/tests/packaging/test_abi_coverage.py`, carregando o script por
  caminho como o `test_packaging_scripts.py`).
  - `test_declared_finds_every_api_function`: um texto de exemplo com 3 declarações `VE_API` (uma em duas linhas) dá os
    3 nomes; `VE_API void ve_x_free(ve_x* x);` entra.
  - `test_used_ignores_comments_and_strings`: `// ve_fake(` e `"ve_fake("` não contam.
  - `test_unused_not_allowed_fails`: uma declarada sem uso e fora da lista dá saída 1 e imprime o nome.
  - `test_real_header_is_covered`: sobre o `volengine.h` e os headers do wrapper reais, saída 0.
- [ ] **Passo 2: Rodar e ver falhar. Passo 3: implementar** o script. Registrar como teste do `ctest` da pasta `cpp`.
  Esperado no repositório: as 133 funções cobertas, `UNUSED_ALLOWED` vazio ou só com função justificada.
- [ ] **Passo 4: Rodar os testes.** Esperado: verdes.
- [ ] **Passo 5: Achados da ABI.** Escrever o `docs/superpowers/notes/...abi-findings-6b.md` com cada atrito encontrado
  nas Tasks 1 a 6 (por exemplo: `ve_fit_param_name` devolve ponteiro com validade presa ao `Fit`; `ve_dcc_fit_batch`
  não diz o tamanho de `out`; o `.def` MinGW não carrega `LIBRARY`). Uma linha por achado: função, o problema, o que o
  wrapper fez, se pede mudança de ABI no 6c. Sem achado, o arquivo diz isso.
- [ ] **Passo 6: Commit** — `test(cpp): every VE_API function is called by the wrapper`

### Task 8: pacote instalado, consumidores e CI

**Arquivos:**
- Modificar: `cpp/CMakeLists.txt` (`VOLENGINE_INSTALL`, config e versão do pacote)
- Criar: `cpp/cmake/volengineConfig.cmake.in`, `cpp/tests/package/CMakeLists.txt`, `cpp/tests/package/main.cpp`,
  `.github/workflows/cpp.yml`
- Testar: `cpp/tests/package/` (projeto consumidor), `cpp/tests/CMakeLists.txt`

**Interfaces:**
- Consome: Tasks 1 a 7. Produz: `find_package(volengine 0.9)` com `volengine::volengine` e
  `volengine_copy_native_library`; workflow `cpp.yml`.

- [ ] **Passo 1: Escrever o consumidor mínimo (falha primeiro).** `main.cpp` cria `Context`, ajusta sGARCH(1,1) norm
  numa série simulada e devolve 0 só se `status == converged` e `loglik` for finito. O `CMakeLists.txt` do consumidor
  tem a opção `VOLENGINE_CONSUMER_MODE` com `fetch_content` (declara `FetchContent` com `SOURCE_DIR` na raiz do
  repositório e `SOURCE_SUBDIR cpp`, e `find_package` não) e `find_package` (`find_package(volengine 0.9 REQUIRED)`).
  Os dois modos chamam `volengine_copy_native_library` no executável.
- [ ] **Passo 2: Rodar e ver falhar:** o modo `find_package` sem `cmake --install` não acha o pacote.
- [ ] **Passo 3: Implementar a instalação.** Instala `cpp/include/volengine/*.hpp` e `include/volengine/volengine.h` em
  `include/volengine`, a biblioteca (`bin` no Windows, `lib` nos outros), o import lib no Windows (`.lib` com MSVC,
  `.dll.a` com MinGW) e `volengineConfig.cmake` mais `volengineConfigVersion.cmake` (`SameMajorVersion`, versão sem o
  `v`; fora de tag, `0.0.0`) mais `CopyNativeLibrary.cmake`. `VOLENGINE_INSTALL` fica ON só quando `cpp/` é a raiz. O
  `.in` recria os dois alvos apontando para a cópia instalada, sem rede.
- [ ] **Passo 4: Rodar os dois modos.** Esperado: os dois executam e saem com 0.
  - Local (Windows, PowerShell): modo `fetch_content` com g++; modo `find_package` depois de `cmake --install
    build/cpp --prefix build/cpp-prefix` e `-DCMAKE_PREFIX_PATH=build/cpp-prefix`.
  - Repetir o modo `find_package` com MSVC (Developer PowerShell, `-G Ninja -DCMAKE_CXX_COMPILER=cl`) sobre a DLL
    MinGW: é o caso do foco 4.
- [ ] **Passo 5: Escrever o `.github/workflows/cpp.yml`.**
  - Gatilhos: `push` em master e `pull_request` que toquem `cpp/**`, `include/**`, `src/**`, `CMakeLists.txt`,
    `python/scripts/build_native.py` ou o próprio workflow.
  - Jobs `linux` (ubuntu-24.04, `g++-14` pelo apt), `macos` (macos-latest) e `windows` (windows-latest).
  - Cada job compila a biblioteca com `python python/scripts/build_native.py` (Linux e macOS com `uv run`; Windows no
    shell MSYS2 como o `wheels.yml`), exporta `VOLENGINE_NATIVE_LIB` e roda os testes e os dois modos do consumidor.
  - No Windows o wrapper compila com MSVC (`ilammy/msvc-dev-cmd@v1`, `cl`) sobre a DLL MinGW.
  - Não usar `--parallel` nos testes do repositório raiz (estourou a memória no `ci.yml`); aqui os testes são pequenos
    e o Ninja pode usar os núcleos.
- [ ] **Passo 6: actionlint** (`uvx --from actionlint-py actionlint .github/workflows/cpp.yml`), sem erros.
- [ ] **Passo 7: Push (pedir o ok do autor) e acompanhar** (`gh run watch`): os 3 jobs verdes. Cada correção é um commit
  `fix(ci): ...`.
- [ ] **Passo 8: Commit** (antes do push) — `ci: build and test the C++ wrapper on three platforms`

### Task 9: custo do wrapper

**Arquivos:**
- Criar: `cpp/benchmarks/CMakeLists.txt`, `cpp/benchmarks/wrapper_overhead.cpp`
- Modificar: `cpp/CMakeLists.txt` (`VOLENGINE_BUILD_BENCHMARKS`), `docs/performance/benchmarks.md` (seção "Plan 6b")

**Interfaces:**
- Produz: `BM_FitDirectC`, `BM_FitWrapper` (um sGARCH(1,1) norm, T = 3000) e `BM_BatchDirectC`, `BM_BatchWrapper` (12
  specs de sGARCH/gjrGARCH/eGARCH(1,1) com norm, std, ged, sstd sobre 21 séries, todas as threads, erros-padrão
  desligados).

- [ ] **Passo 1:** escrever os quatro benchmarks. A versão C chama `ve_*` direto, com os mesmos dados e opções. Cada um
  confere que o ajuste converge e que o `loglik` de uma iteração é igual nas duas versões (bit a bit), senão aborta.
- [ ] **Passo 2: Medir no Ryzen 7 5700X, mesma sessão, `--benchmark_repetitions=10`, alternando C e wrapper.**
  Esperado: razão wrapper/C dentro de 1 desvio. Fora disso é bug do wrapper (cópia de vetor, alocação por chamada).
- [ ] **Passo 3:** registrar no `benchmarks.md`: a tabela, a máquina, o commit e o compilador.
- [ ] **Passo 4: Commit** — `bench(cpp): the wrapper against direct C calls`

### Task 10: documentação e fechamento

**Arquivos:**
- Criar: `cpp/README.md`
- Modificar: `CONTRIBUTING.md` (seção dos testes C++), `CHANGELOG.md` (Unreleased), `README.md` (uma linha apontando
  para `cpp/README.md`)

- [ ] **Passo 1:** `cpp/README.md` em inglês: o `FetchContent` e o `find_package` do consumidor mínimo, a variável
  `VOLENGINE_NATIVE_LIB`, o que o wrapper copia e o que é visão, a tabela de plataformas do 6a. `CONTRIBUTING.md`: os
  comandos de build e teste do `cpp/`, e a nota do PowerShell no Windows. `CHANGELOG.md`: "Added: header-only C++23
  wrapper (`cpp/`)".
- [ ] **Passo 2: Verificar contra a release real.** Num diretório vazio fora do repositório, um consumidor `FetchContent`
  com `GIT_TAG v0.9.0` e `-DVOLENGINE_VERSION=v0.9.0` (sem `VOLENGINE_NATIVE_LIB`) baixa o asset do Windows, compila com
  MinGW e com MSVC e roda o ajuste. Esperado: `converged`. Conferir com `gh release view v0.9.0 --json assets` que os
  nomes que o `volengine_asset_name` gera existem nas 7 plataformas. O wrapper de `v0.9.0` não existe na tag: usar o
  `SOURCE_DIR` local no modo `FetchContent` e só baixar a biblioteca (`VOLENGINE_VERSION=v0.9.0`).
- [ ] **Passo 3: Commit** — `docs(cpp): wrapper README, changelog and contributing`
- [ ] **Passo 4:** atualizar a memória do roadmap (6b feito, próximo 6c) e listar ao autor os achados da ABI.
