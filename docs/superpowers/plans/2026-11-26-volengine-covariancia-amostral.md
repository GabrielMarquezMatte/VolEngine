# Covariância amostral e Ledoit-Wolf — plano de implementação

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `ve_cov_sample` e `ve_cov_ledoit_wolf` na C ABI, num módulo próprio (`src/cov/`), com bindings C++, C# e
Python e oráculo `numpy`/`sklearn`.

**Architecture:** funções puras sobre `PanelView` em `src/cov/`; a camada C valida o painel e copia o resultado para o
buffer do chamador só no sucesso. Os bindings criam um `ve_panel` da entrada e alocam a saída N × N.

**Tech Stack:** C++23 (GCC/MinGW, `-Wnrvo` como erro), GoogleTest, wrapper header-only, C# .NET 10 (LibraryImport,
PublicApiAnalyzers), Python ctypes + numpy (mypy), oráculo `numpy` + `scikit-learn` fora do ambiente de teste.

**Spec:** `docs/superpowers/specs/2026-11-26-volengine-covariancia-amostral-design.md`

## Global Constraints

- ABI 1 aditiva: `include/volengine/volengine.abi` só cresce (`python scripts/abi_contract.py write` e `check`).
- Assinaturas exatas: `VE_API ve_status ve_cov_sample(const ve_panel* panel, double* out);` e
  `VE_API ve_status ve_cov_ledoit_wolf(const ve_panel* panel, double* out, double* shrinkage);`. "Since 1.3.0."
- Painel nulo: `VE_INVALID_HANDLE`. `out`/`shrinkage` nulos, painel com intervalos (`ve_panel_set_ranges`), t < 2,
  n = 0, NaN ou infinito: `VE_INVALID_ARGUMENT`, e `out`/`shrinkage` não são tocados.
- Matrizes N × N row-major, simétricas bit a bit (triângulo superior calculado e espelhado).
- Amostral: ZᵀZ / (T − 1). Ledoit-Wolf: E = ZᵀZ / T, μ = tr(E)/N, β = (Σ_ij [(Z∘Z)ᵀ(Z∘Z)]_ij / T − ‖E‖²_F)/(N·T),
  δ = ‖E − μI‖²_F / N somado como quadrados da diferença, s = min(β, δ)/δ, s = 0 quando δ = 0; resultado
  (1 − s)E + sμI.
- STYLEGUIDE: funções ≤ 60 linhas, ≤ 5 parâmetros, aninhamento ≤ 3, sem `else`, ≤ 8 módulos por pasta
  (`tests/style/check_style.py`, teste ctest `style`). Pastas já cheias: `src/capi`, `cpp/tests`, `python/tests`,
  `python/src/volengine`, `csharp/src/VolEngine/Interop`.
- Sem `Co-Authored-By` nos commits. Sem push.

## Review Focus

1. Painel com strides (Fortran order, coluna fatiada `x[:, ::2]`, `ve_panel_set_strides`): mesmo resultado do contíguo.
2. Série com média enorme diante da variância (preço ~1e6 com variação ~1e-2): a centragem em duas passadas mantém
   a covariância dentro de 1e-9 relativo da série sem o deslocamento.
3. N > T (matriz singular): o Ledoit-Wolf continua definido, com intensidade em [0, 1], e bate com o sklearn.
4. Erro depois de validar parte do painel (NaN na última linha): `out` e `shrinkage` intactos.
5. Entrada Python 1-D, lista ou inteiros: tratada como (T, 1) / convertida para float64, como os painéis atuais.

---

### Task 1: Núcleo `src/cov/`

**Files:**
- Create: `src/cov/estimators.hpp`, `src/cov/estimators.cpp`
- Modify: `CMakeLists.txt` (fonte em `volengine_core`, teste em `volengine_tests`)
- Test: `tests/unit/cov/estimators_test.cpp`

**Interfaces:**
- Produces:
  ```cpp
  namespace ve::cov {
  std::vector<double> sample(const PanelView& panel);            // N * N, row-major
  struct Shrunk { std::vector<double> covariance; double shrinkage = 0.0; };
  Shrunk ledoit_wolf(const PanelView& panel);
  }
  ```
  Pré-condição (garantida pela camada C): painel retangular, t ≥ 2, n ≥ 1, valores finitos.

- [ ] **Step 1: testes que falham** em `tests/unit/cov/estimators_test.cpp`:
  - `CovSample.HandValues`: painel 3 × 2 {1, 2; 3, 5; 5, 11} dá {4, 9; 9, 21} (médias 3 e 6; ZᵀZ/2).
  - `CovSample.ExactlySymmetric`: painel aleatório 200 × 7 (semente fixa); `out[i*n+j] == out[j*n+i]` bit a bit.
  - `CovSample.StridedEqualsContiguous`: o mesmo painel lido com `stride_t = 1, stride_n = t` (column-major) dá o
    mesmo vetor bit a bit.
  - `CovSample.LargeOffset`: somar 1e6 a uma série muda cada entrada em ≤ 1e-9 relativo ao máximo |S|.
  - `CovLedoitWolf.SingleSeries`: n = 1 dá `shrinkage == 0.0` e covariância = Σz²/T.
  - `CovLedoitWolf.ConstantSeries`: todas as séries constantes dão `shrinkage == 0.0` e matriz nula.
  - `CovLedoitWolf.MatchesFormula`: painel 50 × 4 com o resultado calculado no teste pela fórmula das Global
    Constraints, tolerância 1e-14 relativa ao máximo; `shrinkage` em [0, 1].
  - `CovLedoitWolf.ExactlySymmetric`: como o da amostral.
- [ ] **Step 2:** `cmake --build --preset gcc-release` falha (`cov/estimators.hpp` não existe).
- [ ] **Step 3: implementar** em `src/cov/estimators.cpp`: um auxiliar que copia o painel centrado para colunas
  contíguas (média da esquerda para a direita), um auxiliar de produtos cruzados no triângulo superior com espelho,
  e as duas funções. `-Wnrvo`: um único local nomeado por caminho de retorno.
- [ ] **Step 4:** `ctest --preset gcc-release -R "Cov|style"` passa.
- [ ] **Step 5: commit** `feat(cov): sample and Ledoit-Wolf covariance estimators`.

### Task 2: C ABI `ve_cov_*`

**Files:**
- Create: `src/capi/cov/cov_api.cpp`
- Modify: `include/volengine/volengine.h` (bloco novo no fim, antes do `#ifdef __cplusplus` final, com o comentário da
  spec), `include/volengine/volengine.abi` (via script), `CMakeLists.txt`, `CHANGELOG.md` (`[Unreleased]` Added, bullet
  da C ABI)
- Test: `tests/unit/capi/cov/cov_test.cpp`

**Interfaces:**
- Consumes: Task 1.
- Produces: as duas funções das Global Constraints.

- [ ] **Step 1: testes que falham** (`CapiCov.*`):
  - `SampleMatchesCore` e `LedoitWolfMatchesCore`: resultado igual bit a bit a `ve::cov::*` no mesmo painel.
  - `NullPanel` → `VE_INVALID_HANDLE`; `NullOut`, `NullShrinkage` → `VE_INVALID_ARGUMENT`.
  - `RangesRefused`, `TooFewRows` (t = 1), `NoSeries` (n = 0), `NanRefused` (NaN na última linha), `InfRefused` →
    `VE_INVALID_ARGUMENT`, com `out` pré-preenchido com 7.0 e `shrinkage` com 7.0 intactos, e mensagem em
    `ve_last_error` não vazia.
- [ ] **Step 2:** build falha (símbolos ausentes).
- [ ] **Step 3: implementar** `cov_api.cpp` com `guard`/`set_error` como os demais; validação num auxiliar
  (`has_ranges`, t, n, finitude via `PanelView::at`), cálculo num vetor e cópia para `out` só no sucesso. Atualizar o
  header, `python scripts/abi_contract.py write`, o CHANGELOG.
- [ ] **Step 4:** `ctest --preset gcc-release -j 8` inteiro passa (inclui `abi` e `style`);
  `python scripts/abi_contract.py check` passa.
- [ ] **Step 5: commit** `feat(capi): ve_cov_sample and ve_cov_ledoit_wolf`.

### Task 3: Oráculo e binding Python

**Files:**
- Create: `tests/reference/oracle_sklearn_cov.py`, `tests/reference/fixtures/sklearn_cov.json`,
  `python/src/volengine/cov/__init__.py`, `python/tests/cov/test_cov.py`, `python/tests/cov/test_cov_oracles.py`
- Modify: `python/src/volengine/_native.py` (protótipos), `python/src/volengine/__init__.py` (`from . import cov`),
  `tests/reference/README.md` (linhas da tabela), `docs/guide/validation.md` (seção curta `numpy` e `sklearn`, com
  as versões), `docs/guide/python.md` (seção "Covariance estimators"), `CHANGELOG.md` (bullet Python)

**Interfaces:**
- Consumes: Task 2.
- Produces: `volengine.cov.sample(returns: ArrayLike) -> NDArray[np.float64]` (N, N) e
  `volengine.cov.ledoit_wolf(returns: ArrayLike) -> tuple[NDArray[np.float64], float]`; entrada 1-D vira (T, 1), 2-D
  qualquer stride (`panel_array` + `open_panel`); erros como `VolEngineError`.
- Fixture: `{"numpy_version", "sklearn_version", "results": [{"id", "returns": T x N (linhas), "sample": N x N,
  "ledoit_wolf": N x N, "shrinkage"}]}` com os casos `copula3` (o arquivo inteiro), `single` (coluna A de
  `copula3`), `wide` (T = 60, N = 40) e `wider` (T = 30, N = 50), os dois últimos normais com semente 20261126 e
  correlação vinda de cargas aleatórias, 17 dígitos significativos.

- [ ] **Step 1:** escrever `oracle_sklearn_cov.py` (rodado com
  `uv run --no-project --with numpy --with scikit-learn python tests/reference/oracle_sklearn_cov.py`) e gerar a
  fixture. Conferir que `wide` e `wider` têm intensidade entre 0,05 e 0,95.
- [ ] **Step 2: testes que falham:**
  - `test_cov_oracles.py`: para cada caso, `sample` e `ledoit_wolf` batem com a fixture com
    `abs(a − b) ≤ 1e-12 · max|b|` por entrada; `shrinkage` com 1e-12 relativo (absoluto 1e-15 no caso `single`, que
    é 0).
  - `test_cov.py`: Fortran order e `x[:, ::2]` iguais ao contíguo bit a bit; entrada 1-D dá (1, 1); lista de
    inteiros aceita; NaN, uma linha só e array 3-D dão erro (`VolEngineError`; 3-D `ValueError` de `panel_array`);
    resultado simétrico bit a bit.
- [ ] **Step 3:** `VOLENGINE_LIB=... uv run --project python pytest python/tests/cov -q` falha
  (`module 'volengine' has no attribute 'cov'`).
- [ ] **Step 4: implementar** o subpacote, os protótipos, o guia (bloco `python` executável que usa
  `ve.cov.ledoit_wolf` sobre um array gerado no próprio bloco), `validation.md`, README da referência, CHANGELOG.
- [ ] **Step 5:** pytest inteiro e `uv run --project python mypy python/src` passam; `ctest -R style` passa.
- [ ] **Step 6: commit** `feat(python): covariance estimators`.

### Task 4: Wrapper C++

**Files:**
- Create: `cpp/include/volengine/cov.hpp`, `cpp/tests/cov/cov_test.cpp`
- Modify: `cpp/include/volengine/volengine.hpp` (include), `cpp/tests/CMakeLists.txt`, `docs/guide/cpp.md` (seção
  "Covariance estimators", depois de "Cumulative forecast"), `CHANGELOG.md` (bullet C++)

**Interfaces:**
- Consumes: Task 2; `ve::Panel` de `input.hpp`.
- Produces: `namespace ve::cov { struct LedoitWolf { std::vector<double> covariance; double shrinkage; };
  Result<std::vector<double>> sample(const Panel&); Result<LedoitWolf> ledoit_wolf(const Panel&); }`. `Panel` precisa
  dar o `ve_panel*` ao `cov.hpp` (acessor `native()` ou equivalente já existente; usar o que `context.hpp` usa).

- [ ] **Step 1: testes que falham:** `Cov.SampleMatchesCapi` e `Cov.LedoitWolfMatchesCapi` (igual bit a bit a
  chamar a C ABI no mesmo painel `copula3.csv`), `Cov.RangesAreAnError` (`Panel::set_ranges` e depois `sample` dá
  `Status::invalid_argument` com mensagem).
- [ ] **Step 2:** build do wrapper falha (`volengine/cov.hpp` não existe).
- [ ] **Step 3: implementar** `cov.hpp` no padrão de `detail::check`/`detail::failure`.
- [ ] **Step 4:** `PATH=/c/msys64/ucrt64/bin:$PATH VOLENGINE_NATIVE_LIB=$PWD/build/gcc-release/libvolengine.dll bash
  cpp/scripts/ci_check.sh` passa.
- [ ] **Step 5: commit** `feat(cpp): covariance estimators`.

### Task 5: Binding C#

**Files:**
- Create: `csharp/src/VolEngine/Covariance/Covariances.cs`, `csharp/src/VolEngine/Covariance/LedoitWolfResult.cs`,
  `csharp/src/VolEngine/Interop/Native/NativeMethods.Covariance.cs`,
  `csharp/tests/VolEngine.Tests/Covariance/CovarianceTests.cs`
- Modify: `csharp/src/VolEngine/PublicAPI.Unshipped.txt`, `docs/guide/csharp.md` (seção "Covariance estimators",
  copiada também em `GuideSnippetsTests.cs` entre `BEGIN`/`END`), `CHANGELOG.md` (bullet C#)

**Interfaces:**
- Consumes: Task 2; fixture `sklearn_cov.json` da Task 3 (via `ReferenceData.Result("sklearn_cov.json", "id", ...)`).
- Produces: `public static class Covariances { public static double[][] Sample(double[,] returns);
  public static LedoitWolfResult LedoitWolf(double[,] returns); }` e
  `public sealed record LedoitWolfResult(double[][] Covariance, double Shrinkage);`. Entrada T × N (datas × séries);
  `null` → `ArgumentNullException`; status de erro → `VolEngineException`.

- [ ] **Step 1: testes que falham:** `Sample_matches_numpy` e `LedoitWolf_matches_sklearn` nos casos `copula3` e
  `wide` da fixture (mesma tolerância da Task 3); `Nan_throws` (`VolEngineException`, status `InvalidArgument`);
  `Null_throws`; `Result_is_symmetric`.
- [ ] **Step 2:** `dotnet test` falha na compilação (`Covariances` não existe).
- [ ] **Step 3: implementar** com `fixed` sobre o `double[,]`, `ve_panel_create`, saída num `double[n*n]` copiado em
  linhas; guia e snippet.
- [ ] **Step 4:** `PATH="/c/msys64/ucrt64/bin:$PATH" VOLENGINE_LIB=... dotnet test --solution csharp/VolEngine.slnx -c
  Release` passa sem avisos.
- [ ] **Step 5: commit** `feat(csharp): covariance estimators`.

## Ordem

Tasks 1 e 2 em sequência. Tasks 3, 4 e 5 dependem só da 2; a 5 lê a fixture da 3, então a 3 vem antes da 5 (ou a 5
gera a fixture primeiro, se rodarem em paralelo, com o mesmo script).
