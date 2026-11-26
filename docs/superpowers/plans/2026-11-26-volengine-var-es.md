# VaR e ES — plano de implementação

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `ve_risk_empirical` e `ve_risk_parametric` na C ABI, num módulo próprio (`src/risk/`), com bindings C++, C#
e Python e oráculo rugarch.

**Architecture:** funções puras em `src/risk/` (empírico sobre `PanelView` + pesos; paramétrico sobre
`dist::Dist` + `dist::ShapeParameters`, usando `dist::quantile` e uma média de cauda por fórmula fechada ou
quadratura); a camada C valida e escreve `*var`/`*es` só no sucesso.

**Tech Stack:** C++23 (GCC/MinGW, `-Wnrvo` como erro), Boost.Math header-only (`boost/math/quadrature`), GoogleTest,
wrapper header-only, C# .NET 10, Python ctypes + numpy, R 4.6.2 + rugarch 1.5.6 (`C:\Program Files\R\R-4.6.2\bin\Rscript.exe`)
para o oráculo.

**Spec:** `docs/superpowers/specs/2026-11-26-volengine-var-es-design.md`

## Global Constraints

- ABI 1 aditiva (`python scripts/abi_contract.py write`/`check`); as constantes `VE_RETURNS_LOG 0` e
  `VE_RETURNS_SIMPLE 1` e as duas assinaturas exatamente como na seção "C ABI" da spec, comentário incluído, com
  "Since 1.3.0".
- VaR e ES positivos como perda; p = 1 − confidence; empírico com interpolação linear na posição (T − 1)·p sobre os
  valores ordenados e ES = −média dos R ≤ q; paramétrico VaR = −(μ + σ q_p), ES = −(μ + σ m_p).
- Recusas (`VE_INVALID_ARGUMENT`, `*var`/`*es` intactos): `weights`/`var`/`es` nulos, `kind` fora de {0, 1},
  confidence fora de (0, 1) ou NaN, painel com intervalos, t = 0, n = 0, valor ou peso não finito, `dist` fora de
  0..7, parâmetros fora do domínio (`dist::in_domain`), μ não finito, σ < 0 ou não finito. Painel nulo:
  `VE_INVALID_HANDLE`.
- STYLEGUIDE: funções ≤ 60 linhas, ≤ 5 parâmetros (código interno; a C ABI segue a spec), aninhamento ≤ 3, sem `else`,
  ≤ 8 módulos por pasta. Pastas cheias: `src/capi`, `src/dist`, `cpp/include/volengine` (8 com `cov.hpp`),
  `cpp/tests`, `python/tests`, `python/src/volengine`, `csharp/src/VolEngine/Interop`.
- C#: o enum de distribuição já se chama `Distribution`; o record dos parâmetros se chama `ShockDistribution`.
- Sem `Co-Authored-By` nos commits. Sem push.

## Review Focus

1. Caudas extremas (confidence 0,999 e 0,9999) em nig, ghyp e jsu: quadratura sem perda de precisão nem NaN.
2. Distribuições muito pesadas (std com ν = 2,1; ged com ν = 0,6): ES finito e ≥ VaR.
3. Empírico com T = 1 e com p·(T − 1) inteiro exato: quantil = o valor, ES = −esse valor.
4. Painel de caminhos lido do acumulado DCC por strides (stride_t = K·m, deslocamento k·m): igual a copiar a fatia.
5. Pesos negativos (carteira vendida) e que não somam 1: aceitos, sem normalização.

---

### Task 1: Núcleo `src/risk/`

**Files:**
- Create: `src/risk/measures.hpp`, `src/risk/empirical.cpp`, `src/risk/parametric.cpp`
- Modify: `CMakeLists.txt`
- Test: `tests/unit/risk/measures_test.cpp`

**Interfaces:**
- Consumes: `ve::dist::quantile`, `ve::dist::make_density`/`log_density`, `ve::PanelView`.
- Produces (namespace `ve::risk`):
  ```cpp
  struct Measure { double var = 0.0; double es = 0.0; };
  enum class ReturnKind { log, simple };
  // Rectangular panel, t >= 1, n >= 1, finite values, weights.size() == n, confidence in (0, 1): checked by the caller.
  Measure empirical(const PanelView& returns, std::span<const double> weights, ReturnKind kind, double confidence);
  // E[z | z <= quantile(p)] of the standardized density; p in (0, 1), parameters in the domain.
  double tail_mean(dist::Dist dist, const dist::ShapeParameters<double>& parameters, double p);
  struct Location { double mean = 0.0; double sigma = 0.0; };
  Measure parametric(dist::Dist dist, const dist::ShapeParameters<double>& parameters, Location location, double confidence);
  ```

- [ ] **Step 1: testes que falham** (`tests/unit/risk/measures_test.cpp`):
  - `RiskEmpirical.HandValues`: retornos simples {−0,05, 0,01, 0,02, −0,02, 0,03}, peso {1}, confidence 0,8:
    p = 0,2, posição 0,8 entre −0,05 e −0,02 → q = −0,026, VaR = 0,026, ES = 0,05 (só −0,05 ≤ q).
  - `RiskEmpirical.SingleRow`: T = 1 dá VaR = ES = −R.
  - `RiskEmpirical.LogEqualsSimple`: log-retornos r e simples e^r − 1 dão o mesmo resultado (1e-15).
  - `RiskEmpirical.WeightsNotNormalized`: pesos {2, −1} dão a carteira 2 s₁ − s₂ (comparada à conta à mão).
  - `RiskEmpirical.StridedEqualsContiguous`: fatia de um buffer caminhos × K × m com stride_t = K·m, deslocamento
    k·m, igual à cópia contígua bit a bit.
  - `RiskParametric.NormalClosedForm`: confidence 0,975, μ = 0,001, σ = 0,02: VaR = −(0,001 − 0,02·1,959963984540054)
    (1e-14), ES = −(0,001 − 0,02·φ(1,96…)/0,025) (1e-14).
  - `RiskParametric.StudentClosedForm`: ν = 5, confidence 0,99, contra a fórmula fechada da spec calculada no teste
    com o quantil t de Boost (1e-13 relativo).
  - `RiskParametric.QuadratureMatchesStudent`: a rotina de quadratura usada para as outras distribuições, aplicada à
    std (ν ∈ {3, 8, 30}), bate com a fórmula fechada em 1e-10 relativo. Expor a quadratura como
    `double tail_mean_by_quadrature(dist::Dist, const dist::ShapeParameters<double>&, double p)` no header.
  - `RiskParametric.EsAtLeastVar`: os casos de `quantile_cases` (copiar a lista para o teste) × confidence {0,95,
    0,99, 0,999, 0,9999}: ES ≥ VaR, ambos finitos.
  - `RiskParametric.HeavyTails`: std ν = 2,1 e ged ν = 0,6 a 0,999: finitos e ES ≥ VaR.
  - `RiskParametric.ZeroSigma`: σ = 0 dá VaR = ES = −μ.
- [ ] **Step 2:** o build falha (`risk/measures.hpp` não existe).
- [ ] **Step 3: implementar.** Empírico: monta R por linha, ordena uma cópia, interpola, soma a cauda. Paramétrico:
  `tail_mean` despacha para a fórmula fechada (norm, std) ou para `tail_mean_by_quadrature`, que integra
  z·f(z) em (−∞, q] com `boost::math::quadrature::exp_sinh` (ou `tanh_sinh` no intervalo após trocar variável) e
  divide por p. Deixar escrito num comentário o limite de erro escolhido.
- [ ] **Step 4:** `ctest --preset gcc-release -R "Risk|style"` passa.
- [ ] **Step 5: commit** `feat(risk): empirical and parametric VaR and ES`.

### Task 2: C ABI `ve_risk_*`

**Files:**
- Create: `src/capi/risk/risk_api.cpp`, `tests/unit/capi/risk/risk_test.cpp`
- Modify: `include/volengine/volengine.h` (bloco da spec, depois do bloco `ve_cov_*`), `include/volengine/volengine.abi`
  (script), `CMakeLists.txt`, `CHANGELOG.md` (bullet da C ABI)

**Interfaces:**
- Consumes: Task 1.
- Produces: as assinaturas da spec.

- [ ] **Step 1: testes que falham** (`CapiRisk.*`): `EmpiricalMatchesCore` e `ParametricMatchesCore` (bit a bit);
  uma recusa por linha da lista de Global Constraints, com `*var` e `*es` pré-preenchidos com 7,0 e intactos e
  `ve_last_error` não vazio; painel nulo `VE_INVALID_HANDLE`; um parâmetro ignorado (shape NaN na normal) aceito.
- [ ] **Step 2:** build falha (símbolos ausentes).
- [ ] **Step 3: implementar** com `guard`/`set_error`; `dist` convertido como o spec já faz.
- [ ] **Step 4:** `ctest --preset gcc-release -j 8` inteiro passa; `python scripts/abi_contract.py check` passa.
- [ ] **Step 5: commit** `feat(capi): ve_risk_empirical and ve_risk_parametric`.

### Task 3: Oráculo e Python

**Files:**
- Create: `tests/reference/oracle_rugarch_risk.R`, `tests/reference/fixtures/rugarch_risk.json`,
  `python/src/volengine/risk/__init__.py`, `python/tests/risk/test_risk.py`, `python/tests/risk/test_risk_oracles.py`
- Modify: `python/src/volengine/_native.py` (protótipos), `python/src/volengine/__init__.py` (`from . import risk`,
  `__all__`), `tests/reference/README.md`, `docs/guide/validation.md` (seção "Risk measures" e versão do oráculo),
  `docs/guide/python.md` (seção "Risk measures"), `CHANGELOG.md` (bullet Python)

**Interfaces:**
- Consumes: Task 2; `DISTRIBUTIONS` de `_spec.py`.
- Produces: `volengine.risk.empirical(returns: ArrayLike, weights: ArrayLike, *, kind: Literal["log", "simple"] =
  "log", confidence: float) -> tuple[float, float]` (1-D `returns` é uma série, como em `ve.cov`) e
  `volengine.risk.parametric(mean: float, sigma: float, *, confidence: float, dist: str = "norm", skew: float | None =
  None, shape: float | None = None, ghlambda: float | None = None) -> tuple[float, float]`.
- Fixture: `{"rugarch_version", "r_version", "results": [{"dist", "skew", "shape", "ghlambda", "confidence",
  "quantile", "tail_mean"}]}`, 8 distribuições (casos de `quantile_cases`, com os nomes do rugarch: skew e shape na
  parametrização do motor, que é a do rugarch) × confidence {0,95, 0,99, 0,999}; `quantile` = `qdist(dist, p, ...)`,
  `tail_mean` = `integrate(function(z) z * ddist(dist, z, ...), -Inf, quantile, rel.tol = 1e-12)$value / p`.

- [ ] **Step 1:** escrever e rodar o script R (`& "C:\Program Files\R\R-4.6.2\bin\Rscript.exe"
  tests/reference/oracle_rugarch_risk.R`), conferir a fixture.
- [ ] **Step 2: testes que falham:**
  - `test_risk_oracles.py`: para cada caso, `parametric(0.0, 1.0, confidence=c, ...)` dá VaR = −quantile e
    ES = −tail_mean, com a tolerância derivada da observada (anotar no `validation.md`; não mais frouxa que 1e-8
    relativo sem justificativa escrita); com μ = 0,001 e σ = 0,02, VaR = −(μ + σ q).
  - `test_risk.py`: `empirical` igual a `-np.quantile(R, 1 - c)` e ao ES calculado com numpy em painéis aleatórios
    (1e-15 relativo); a fatia `sim.cumulative[:, k, :]` de uma simulação DCC com checkpoints passa sem cópia e bate
    com a cópia; 1-D é uma série; `kind="simple"`; pesos de tamanho errado, NaN, confidence 1.0 e `kind` inválido dão
    erro (`VolEngineError` ou `ValueError`); `dist="std"` sem `shape` dá `VolEngineError`.
- [ ] **Step 3:** `pytest python/tests/risk` falha (`module 'volengine' has no attribute 'risk'`).
- [ ] **Step 4: implementar** o subpacote, os protótipos, o guia (bloco `python` executável com `empirical` sobre o
  acumulado de uma simulação e `parametric` com a t), `validation.md`, README da referência, CHANGELOG.
- [ ] **Step 5:** pytest inteiro (o teste de cobertura da ABI fica vermelho até a Task 4, como no item 4: ledger),
  mypy e `ctest -R style` passam.
- [ ] **Step 6: commit** `feat(python): VaR and ES`.

### Task 4: Wrapper C++

**Files:**
- Move: `cpp/include/volengine/cov.hpp` → `cpp/include/volengine/risk/covariance.hpp` (ainda não lançado; a pasta
  tem 8 módulos)
- Create: `cpp/include/volengine/risk/measures.hpp`, `cpp/tests/risk/measures_test.cpp`
- Modify: `cpp/include/volengine/volengine.hpp`, `cpp/tests/CMakeLists.txt`, `cpp/tests/cov/cov_test.cpp` (nada
  muda se incluir `volengine.hpp`), `docs/guide/cpp.md` (seção "Risk measures"), `CHANGELOG.md` (bullet C++ novo e o
  caminho `volengine/risk/covariance.hpp` no bullet do item 4)

**Interfaces:**
- Consumes: Task 2; `ve::Panel`, `ve::Dist`.
- Produces (namespace `ve::risk`): `struct RiskMeasure { double var; double es; };`, `enum class ReturnKind :
  std::int32_t { log = VE_RETURNS_LOG, simple = VE_RETURNS_SIMPLE };`, `struct Distribution { Dist dist = Dist::norm;
  double skew = NaN; double shape = NaN; double ghlambda = NaN; };`, `Result<RiskMeasure> empirical(const Panel&,
  std::span<const double> weights, ReturnKind, double confidence)` e `Result<RiskMeasure> parametric(const
  Distribution&, double mean, double sigma, double confidence)`. Pesos de tamanho diferente de `cols()` dão
  `Status::invalid_argument` no wrapper, antes da C ABI.

- [ ] **Step 1: testes que falham:** `Risk.EmpiricalMatchesCapi`, `Risk.ParametricMatchesCapi` (bit a bit),
  `Risk.WrongWeightCountIsAnError`, `Risk.CumulativeDccPathsByStrides` (`DccSimulation::cumulative()` lido com
  `Panel::create_strided` na fatia do checkpoint k, igual à cópia).
- [ ] **Step 2:** build do wrapper falha.
- [ ] **Step 3: implementar**; mover `cov.hpp`.
- [ ] **Step 4:** `bash cpp/scripts/ci_check.sh` (com PATH do ucrt64 e `VOLENGINE_NATIVE_LIB`) passa; pytest
  inteiro passa (cobertura da ABI verde).
- [ ] **Step 5: commit** `feat(cpp): VaR and ES`.

### Task 5: Binding C#

**Files:**
- Create: `csharp/src/VolEngine/Risk/Risk.cs`, `csharp/src/VolEngine/Risk/RiskTypes.cs` (`RiskMeasure`,
  `ShockDistribution`, `ReturnKind`), `csharp/src/VolEngine/Interop/Native/NativeMethods.Risk.cs`,
  `csharp/tests/VolEngine.Tests/Risk/RiskTests.cs`
- Modify: `csharp/src/VolEngine/PublicAPI.Unshipped.txt`, `docs/guide/csharp.md` (seção "Risk measures") e
  `GuideSnippetsTests.cs` (snippet entre `BEGIN`/`END`), `CHANGELOG.md` (bullet C#)

**Interfaces:**
- Consumes: Task 2; fixture da Task 3.
- Produces: `public static class Risk { public static RiskMeasure Empirical(double[,] returns, double[] weights,
  ReturnKind kind, double confidence); public static RiskMeasure Parametric(ShockDistribution distribution, double
  mean, double sigma, double confidence); }`, `public sealed record RiskMeasure(double Var, double Es);`,
  `public sealed record ShockDistribution(Distribution Distribution, double Skew = double.NaN, double Shape =
  double.NaN, double GhLambda = double.NaN);`, `public enum ReturnKind { Log, Simple }`. Nulos →
  `ArgumentNullException`; pesos de tamanho errado → `ArgumentException`; status de erro → `VolEngineException`.

- [ ] **Step 1: testes que falham:** `Parametric_matches_rugarch` (os casos de confidence 0,99 da fixture),
  `Empirical_hand_values` (o caso da Task 1), `Wrong_weight_count_throws`, `Invalid_confidence_throws`,
  `Null_throws`.
- [ ] **Step 2:** a compilação falha (`Risk` não existe).
- [ ] **Step 3: implementar**; guia e snippet.
- [ ] **Step 4:** `dotnet test --solution csharp/VolEngine.slnx -c Release` passa sem avisos.
- [ ] **Step 5: commit** `feat(csharp): VaR and ES`.

## Ordem

Tasks 1 e 2 em sequência; 3, 4 e 5 dependem da 2; a 5 lê a fixture da 3; a cobertura da ABI do pytest só fica verde
depois da 4.
