# VolEngine — Plano 4c: forecast e simulação do DCC e da Copula-DCC

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `d.forecast(h)` e `ctx.simulate(d, ...)` sobre qualquer `DccFit` dos planos 4a e 4b:
- forecast de Rₜ₊ₕ, Hₜ₊ₕ, σ e média, como o `dccforecast` do `rmgarch` 1.4.3 (`n.roll = 0`);
- caminhos multivariados como o `dccsim`/`cgarchsim`, com o Philox e a regra de semente do 3b.

**Architecture:**
- **Estado no ajuste:** o lote passa a guardar no `DccFitResult`:
  - o estado final de cada série (spec, estimativa, `ForecastState`), compartilhado entre as specs via `shared_ptr`;
  - o estado final da recursão de correlação de cada spec (`DccEnd`): últimos Q, últimos escores, Q̄, N̄ e θ.
  Nenhum número existente muda.
- **Subpasta nova `src/dcc/future/`:** `src/dcc/` tem 7 módulos. Ela traz 3:
  - `state.*`: `DccEnd` e o `CorrelationStepper`, que avança Q/R um passo;
  - `forecast.*`;
  - `simulate.*`.
- **Univariado:** forecast e simulação de cada série reusam o 3a e o 3b:
  - `forecast::forecast` ganha uma sobrecarga sobre (spec, estimativa, estado);
  - o passo do 3b vira uma classe `sim::Path` que recebe o choque de fora.
- **Cópula:** o choque sai da cópula (mvnorm ou mvt) e vira zᵢ pela função quantil da marginal. A função quantil é
  nova e interna, em `dist/cdf.*`, porque `src/dist/` está no limite de 8 módulos.

**Tech Stack:** C++23, CMake + presets, GoogleTest v1.17.0, Google Benchmark v1.9.1, Boost.Math 1.87 (standalone,
só em `.cpp`: quantis `normal`/`students_t`, `gamma_q_inv`, `ibeta_inv`, `toms748_solve`), Python ≥ 3.11 + numpy +
pytest, R 4.6.2 + `rugarch` 1.5.6 + `rmgarch` 1.4.3.

**Spec:** `docs/superpowers/specs/2026-11-13-volengine-04-dcc-design.md`, seções 3 "4c", 8a e **11**, sobre a
spec-mãe `docs/superpowers/specs/2026-11-10-volengine-design.md`. Leia também os planos 3a, 3b, 4a e 4b
(`docs/superpowers/plans/2026-11-13-volengine-0{3a,3b,4a,4b}-*.md`): as regras de forecast, simulação, convenções e
cópula de lá valem aqui sem repetir.

## Roteiro

- **1, 2a–2f, 3a–3c, 4a e 4b (feitos).**
- **4c (este):** forecast e simulação do DCC e da Copula-DCC. Fecha o plano 4.

Depois: 5 performance; 6 wrapper C++, wheels, 1.0.

## Global Constraints

Valem todas as Global Constraints dos planos 4a e 4b, o `STYLEGUIDE.md` e o `ARCHITECTURE.md`. As que mais pesam
aqui:

- **Interface C:**
  - ≤ 5 parâmetros por função; configuração nova é função nova;
  - handles imutáveis; visões zero-copy válidas enquanto o handle existir;
  - nenhuma exceção atravessa a fronteira.
- **Nenhum número existente muda:**
  - todo `DccFit`, `Fit`, forecast e simulação dos planos 3a, 3b, 4a e 4b fica bit a bit igual;
  - todos os testes existentes passam sem mudar constantes nem tolerâncias.
- **Bit-identidade** da simulação entre números de threads (1, 3 e 16), como `Simulate.SameForAnyThreadCount` do 3b.
- **Semente:** o caminho p lê só `Stream(seed, p)`, reprodutível só dentro da mesma versão (CHANGELOG).
- **Boost em `double`:** todo quantil e toda CDF novos usam a `DoublePolicy` (`promote_double<false>`) de
  `dist/cdf.cpp`. Sem ela, o resultado depende da precisão x87 da thread (MinGW).
- **Licença:** o `rmgarch` (GPL) é lido para entender convenções e conferido numericamente. Nenhum código dele é
  copiado ou traduzido.
- **Estilo:**
  - nunca `else`;
  - funções ≤ 60 linhas, ≤ 5 parâmetros, aninhamento ≤ 3;
  - ≤ 8 módulos por pasta, também em `tests/` e `python/`. Estão no limite: `src/dist/`, `tests/unit/dcc/` e
    `python/src/volengine/` (8 cada);
  - Boost só em `.cpp`.
- **Ambiente e commits:**
  - commits sem `Co-Authored-By`; LF;
  - **testes só em release** (`cmake --build --preset gcc-release && ctest --preset gcc-release`);
  - executáveis e pytest com `/c/msys64/ucrt64/bin` primeiro no PATH e `VOLENGINE_LIB` apontando para
    `build/gcc-release/libvolengine.dll`;
  - `python tests/style/check_style.py` e `mypy --strict python/src` limpos ao fim de cada task que toca código.

### Regras do forecast (spec 11.3 e 11.4)

Notação: m séries; o ajuste vai até a observação T; h = 1..H.

- **Univariado:** σᵢ e médiaᵢ de cada série são o `forecast::forecast` do 3a, bit a bit iguais ao `Fit.forecast`
  da série, com as linhas futuras dela.
- **Escores e momentos:**
  - os escores sₜ são os zₜ do painel no DCC e os εₜ da cópula (4b, `copula_panel` na forma estimada);
  - Q̄ e N̄ são os do `Stage::Reported` da convenção da spec (`stage_moments`): são os da recursão que produziu o
    `fit.r`;
  - R̄ = diag(Q̄)^−½ Q̄ diag(Q̄)^−½, com diagonal 1 exata.
- **h = 1:** a recursão exata do modelo (4a, `CorrelationRecursion`) um passo depois de T, com os Q e escores
  observados.
- **h ≥ 2:**
  - DCC e aDCC: Rₜ₊ₕ = (1 − k^{h−1}) R̄ + k^{h−1} Rₜ₊₁, com k = Σa + Σb. No aDCC, g não entra (E[nn′] ≈ N̄ cancela
    o −gN̄). É a fórmula do `rmgarch`, conferida com erro 0 em DCC(1,1) e DCC(2,1);
  - FDCC: a mesma por elemento, com Kᵢⱼ = αᵢαⱼ + βᵢβⱼ (os pesos da recursão);
  - estática (cópula): R fixo, a matriz do ajuste.
- **H:** Hₜ₊ₕ = Dₕ Rₜ₊ₕ Dₕ, com Dₕ = diag(σᵢ,ₕ) do forecast univariado (o `sigmaFor`, como o `rmgarch`).
- **[Rmgarch]:** as mesmas fórmulas com os Q̄/N̄ e a pré-amostra da convenção `Rmgarch`. O FDCC h = 1 do `rmgarch` usa
  Q̄ = cov(z), que é o `panel.qbar` do `Stage::Reported`.

### Regras da simulação (spec 11.4)

Por caminho p e passo h, nesta ordem:
1. Rₕ = `CorrelationStepper::next()`: a recursão exata a partir do estado do caminho. Na estática é sempre o R do
   ajuste.
2. **Sorteios do caminho, em ordem fixa:** x₁..xₘ ~ N(0,1) (`sim::normal`) e depois, se houver, a variável de
   mistura.
3. **Choque** ε = c · L x, com L o fator de Cholesky próprio de Rₕ (`linalg::cholesky_factor`):
   - mvnorm: c = 1;
   - mvt (forma ν): c = √((ν − 2)/(2G)), G ~ Gamma(ν/2) (`sim::gamma`), a mesma forma da `std` do 3b;
   - mvlaplace: c = √W, W ~ Exp(1) = Gamma(1). É a mistura gaussiana da densidade de `dcc/likelihood.cpp`, com
     covariância Rₕ.
4. **zᵢ da série i:**
   - DCC: zᵢ = εᵢ;
   - cópula: uᵢ = F(εᵢ), com F = Φ na mvnorm e a CDF da `std` padronizada com forma ν na mvt (`dist::cdf`). Depois
     uᵢ é limitado a [`kUniformFloor`, 1 − `kUniformFloor`] e vira zᵢ = Fᵢ⁻¹(uᵢ):
     - paramétrica: `dist::quantile` da distribuição do ajuste i;
     - empírica: o quantil empírico dos zᵢₜ do ajuste, z₍ₖ₎ com k = min(⌈uᵢ (T + 1)⌉, T), a inversa de
       rank/(T + 1).
5. **Passo univariado:** `sim::Path::step(row, zᵢ)` de cada série, em ordem i = 1..m. A saída é rᵢ e σᵢ.
6. **Avanço da recursão:** `stepper.push(s)` com s = z (DCC) ou ε (cópula).
7. Com `correlations`, R[p, h] = Rₕ e H[p, h] = D Rₕ D, com D = diag(σ do passo).

- **Saídas:**
  - `returns` e `sigma`: (paths, horizon, m), row-major;
  - `R` e `H`: (paths, horizon, m, m), só com `correlations`.
- **[Rmgarch]:** a convenção só muda Q̄/N̄ e a pré-amostra, que já estão no `DccEnd`. O caminho não tem oráculo.

## Review Focus

1. **`DccFit` com status `Failed` mas estimativa finita** (busca de linha falhou) prevê e simula, como o 3a faz com
   um `Fit`. Com estimativa NaN, dá `VE_INVALID_ARGUMENT` com "no finite point". Testes na Task 3 e na Task 6.
2. **O `DccFit` devolvido por `standard_errors(fits)`** dá o mesmo forecast bit a bit, e o forecast funciona depois
   que os `Fit` univariados são liberados (`del fits; gc.collect()`). Teste na Task 6.
3. **Regressores mistos:** uma série com mxreg e as outras com `None` funciona. Uma sequência com outro
   comprimento, ou linhas que não batem com a série, dá erro que nomeia a série (a partir de 1). Testes na Task 4 e
   na Task 6.
4. **paths × horizon × m² estoura `size_t`** com `correlations`: dá `VE_INVALID_ARGUMENT`, nunca `bad_alloc` nem
   crash. Sem `correlations`, o limite é paths × horizon × m. Teste na Task 6.
5. **Séries ARFIMA e csGARCH no painel:** o estado da ARFIMA é o histórico inteiro, e o csGARCH tem `permanent`. O
   forecast de cada série é bit a bit o `Fit.forecast` dela, e a simulação roda. Testes na Task 4 e na Task 5.

---

### Task 1: Função quantil das distribuições univariadas

**Files:**
- Modify: `src/dist/cdf.hpp`, `src/dist/cdf.cpp`
- Test: `tests/unit/dist/cdf_test.cpp`

**Interfaces:**
- Consumes: `dist::cdf`, `make_density`, `ShapeParameters`, os `LogDensity`/`SkewedLogDensity`/hyperbolic/johnson
  de `densities.hpp`.
- Produces:
  ```cpp
  namespace ve::dist {
  // The z with cdf(dist, parameters, z) == u, u in (0, 1), of the standardized density of make_density.
  // Internal (plan 4c): the copula simulation's margins; no C or Python surface.
  double quantile(Dist dist, const ShapeParameters<double>& parameters, double u);
  }
  ```

- [ ] **Step 1: Escrever os testes**

```cpp
TEST(DistQuantile, InvertsTheCdf) {
  // Para cada dist (norm, std, ged, sstd, sged, nig, ghyp, jsu), nas formas dos testes de CDF do 3c (incluindo
  // as extremas que o 3b usa em Samplers.StandardizedMoments) e u em {1e-12, 1e-6, 0.01, 0.3, 0.5, 0.7, 0.99,
  // 1 - 1e-6, 1 - 1e-12}: |cdf(quantile(u)) - u| <= 1e-12 * min(u, 1 - u) + 1e-15
}
TEST(DistQuantile, KnownValues) {
  // norm u = 0.975: 1.959963984540054 (1e-14); std nu 5 u = 0.975: t_5^-1(0.975) * sqrt(3/5) (1e-14);
  // quantile(0.5) == 0 nas simétricas (norm, std, ged, nig/ghyp com beta 0, jsu com skew 0)
}
TEST(DistQuantile, Monotone) {
  // 1000 u crescentes em (0, 1): quantile estritamente crescente em cada dist
}
TEST(DistQuantile, SameBitsOnAnyThread) {
  // quantile da std e da sged calculado na thread principal e numa std::thread: bits iguais
}
```

- [ ] **Step 2: Rodar e ver falhar** — `ctest --preset gcc-release -R DistQuantile` → não compila.
- [ ] **Step 3: Implementar `quantile` em `cdf.cpp`.**
  - norm: `boost::math::quantile(normal)`;
  - std e ged: o quantil do kernel em forma fechada (`ibeta_inv`/`gamma_q_inv`, a inversa das fórmulas de
    `kernel_cdf`), com a escala do kernel;
  - sstd e sged: `skewed_cdf` invertida por partes em y = 0. P(y < 0) = 1/(1 + ξ²); o kernel é invertido de cada
    lado, e z = (y − shift)/stretch;
  - jsu: forma fechada pela transformação sinh do normal;
  - nig e ghyp: `toms748_solve` sobre `cdf(z) − u`, com intervalo expandido a partir da média ± desvios até mudar de
    sinal, 60 iterações e tolerância de 2⁻⁴⁸ relativa. É determinístico, portanto bit-idêntico entre threads, desde
    que a CDF seja. A CDF da nig/ghyp herda o item aberto do Bessel K x87 (memória do 4b); o teste de threads usa
    std e sged.
- [ ] **Step 4: Rodar e ver passar** — suíte release inteira PASS.
- [ ] **Step 5: Commit** — `git commit -m "feat: add univariate quantile functions"`

---

### Task 2: Forecast e passo de simulação univariados sobre (spec, estimativa, estado)

**Files:**
- Modify: `src/forecast/forecast.hpp`, `src/forecast/forecast.cpp`, `src/sim/start.hpp`, `src/sim/start.cpp`,
  `src/sim/simulate.cpp`, `CMakeLists.txt`
- Create: `src/sim/path.hpp`, `src/sim/path.cpp`
- Test: `tests/unit/forecast/forecast_test.cpp`, `tests/unit/sim/simulate_test.cpp`

**Interfaces:**
- Consumes: `forecast::detail::first_step`, `Inputs`, `moments_of`; `sim::Start`, `Stream`, `Sampler`.
- Produces:
  ```cpp
  namespace ve::forecast {
  // The forecast of a fit from what it keeps; the FitResult overload calls it with fit.spec, fit.estimate, fit.state.
  Forecast forecast(const Spec& spec, std::span<const double> estimate, const ForecastState& state,
                    std::size_t horizon, const FutureRegressors& future);
  }
  namespace ve::sim {
  // start_of_fit(fit) == start_of(fit.spec, fit.estimate, fit.state).
  Start start_of(const Spec& spec, std::span<const double> estimate, const forecast::ForecastState& state);
  // One path of one series: each step forecasts one step from the state and takes the shock z (e = sigma z).
  class Path {
  public:
      explicit Path(const Start& start);
      // The presample of a spec start (start.presample residuals = presample_sigma * draws), as plan 3b.
      void draw_presample(Stream& stream, const Sampler& sampler);
      struct Draw { double r; double sigma; };
      Draw step(const forecast::FutureRegressors& row, double z);
  };
  }
  ```
- `simulate_path` do 3b passa a ser `Path` + `draw_presample` + `step(row, sampler.draw(stream))`, na mesma ordem de
  sorteios. O `trim` e o `push` de estado moram em `Path`.

- [ ] **Step 1: Escrever os testes**

```cpp
TEST(Forecast, FromKeptState) {
  // Para cada fit dos testes de forecast do 3a (incluindo ARFIMA, csGARCH e um com mxreg/vxreg): forecast(fit, h,
  // future) e forecast(fit.spec, fit.estimate, fit.state, h, future) dão sigma e mean bit a bit iguais (h = 10)
}
TEST(Simulate, PathStepMatchesSimulate) {
  // sGARCH(1,1)-std e csGARCH de fit: Path(start_of_fit(fit)) com step(row, sampler.draw(Stream(seed, 0))) por 60
  // passos dá r e sigma bit a bit iguais à linha 0 de simulate(...)
}
```

- [ ] **Step 2: Rodar e ver falhar** — `ctest --preset gcc-release -R "Forecast.FromKeptState|Simulate.PathStep"` →
  não compila.
- [ ] **Step 3: Implementar.** Mover o corpo de `forecast(const FitResult&...)` para a sobrecarga nova; extrair
  `Path` de `simulate_path`. Os testes existentes de `tests/unit/sim/` e `tests/unit/forecast/`, sem mudança, são a
  prova de que nenhum número do 3a/3b mudou.
- [ ] **Step 4: Rodar e ver passar** — suíte release inteira PASS.
- [ ] **Step 5: Commit** — `git commit -m "refactor: forecast and step univariate paths from a kept state"`

---

### Task 3: Estado final do DCC no ajuste e o `CorrelationStepper`

**Files:**
- Create: `src/dcc/future/state.hpp`, `src/dcc/future/state.cpp`
- Modify: `src/dcc/fit.hpp`, `src/dcc/fit.cpp`, `src/dcc/recursion.hpp` (só um acessor), `CMakeLists.txt`,
  `ARCHITECTURE.md`
- Test: `tests/unit/dcc/future/state_test.cpp`

**Interfaces:**
- Consumes: `CorrelationRecursion`, `stage_moments`, `weights_of`, `Stage::Reported` (`recursion.hpp`);
  `full_parameters`, `layout_of` (`parameter_map.hpp`); `copula::copula_panel`, `CopulaInputs::uniforms_of`.
- Produces:
  ```cpp
  namespace ve::dcc::future {
  // A series as forecast and simulation read it (plan 4c): its spec, estimate and final state.
  struct SeriesEnd { Spec spec; std::vector<double> estimate; forecast::ForecastState state; };
  // The correlation recursion after observation rows - 1, at Stage::Reported of the spec's convention:
  // theta every DCC parameter (natural, layout order, fixed ones included); qbar and nbar packed (nbar aDCC only);
  // q the last spec.q matrices Q, packed, newest first; scores the last spec.p rows of the scores, newest first;
  // static_r the m x m R of a static copula (q and scores empty). Empty theta: the fit has no finite point (its r
  // is null). An all-fixed spec has an empty table but a full theta.
  struct DccEnd {
      std::vector<double> theta, qbar, nbar, q, scores, static_r;
  };
  // rows <= scores.t (default all): the end after the first rows observations, for the tests.
  DccEnd dcc_end(const StandardizedPanel& scores, const DccSpec& spec, std::span<const double> theta,
                 std::size_t rows);
  // Q and R one step after an end, then the next ones as scores arrive.
  class CorrelationStepper {
  public:
      CorrelationStepper(const DccEnd& end, const DccSpec& spec, std::size_t m);
      std::span<const double> next();               // R of the next step, m x m row-major
      void push(std::span<const double> scores);    // the scores of that step (m)
  };
  }
  namespace ve::dcc {
  struct DccFitResult {
      // ... 4a/4b fields ...
      // Filled by fit_dcc_batch (null/empty from fit_dcc alone, and when the estimate is NaN).
      std::shared_ptr<const std::vector<future::SeriesEnd>> series_end;
      future::DccEnd end;
  };
  }
  ```
- **No lote** (`fit_in_batch`), depois do ajuste e dos erros-padrão: `series_end` é um só por lote, montado dos
  `fits`, e `end = dcc_end(...)`:
  - DCC: sobre `batch.panel`;
  - cópula dinâmica: sobre `copula_panel(copula.uniforms_of(spec), *copula.univariate, spec.dist, shape estimado)`;
  - estática: só `static_r`, do `result.r`.
- **Somas do stepper na ordem do `CorrelationRecursion`:** intercepto; depois, por lag, a·zz′ e g·nn′; depois b·Q
  por lag. Assim os bits batem com o `fit.r`.

- [ ] **Step 1: Escrever os testes**

```cpp
TEST(DccEnd, StepperContinuesTheFit) {
  // dcc3, para DCC(1,1), DCC(2,1), aDCC e FDCC (grupos 1,1,2), mvnorm, nas duas convenções: end =
  // dcc_end(panel, spec, theta, T - 5); stepper; para k = 0..4: next() == R_{T-5+k} de fit.r (bit a bit) e
  // push(z da linha T-5+k)
}
TEST(DccEnd, FilledByTheBatch) {
  // fit_dcc_batch no dcc3 com DCC e copula DCC mvt paramétrica: series_end tem 3 entradas com spec/estimate/state
  // iguais aos dos fits e é o mesmo ponteiro nos dois resultados; end.theta == full_parameters(...); para a
  // cópula, stepper(end).next() é finito com diagonal 1. Static mvnorm kendall: static_r == as m x m primeiras
  // de r
}
TEST(DccEnd, NoFiniteEstimate) {
  // Review Focus 1: o spec fora do domínio de DccFit.FailsOutsideTheDomain (dccb1 fixo em 1.2) no lote: Failed,
  // r null, end.theta vazio; um spec com todos os parâmetros fixos: tabela vazia, end.theta == os fixos
}
TEST(DccEnd, NumbersUnchanged) {
  // loglik, estimate, r e se de um lote do dcc3 iguais bit a bit aos de antes (as constantes dos testes do 4a
  // passam sem mudança; este teste só confere que end não altera result.r)
}
```

- [ ] **Step 2: Rodar e ver falhar** — `ctest --preset gcc-release -R DccEnd` → não compila.
- [ ] **Step 3: Implementar.** `CorrelationRecursion` ganha só `std::span<const T> q() const` (o Qₜ corrente,
  packed); `dcc_end` roda a recursão até `rows` e guarda o anel dos últimos Q e escores. O teste novo vai em
  `tests/unit/dcc/future/`, porque `tests/unit/dcc/` está no limite.
- [ ] **Step 4: Rodar e ver passar** — suíte release inteira PASS.
- [ ] **Step 5: Commit** — `git commit -m "feat: keep the final DCC state in DCC fits"`

---

### Task 4: Forecast do DCC e oráculo `dccforecast`

**Files:**
- Create: `src/dcc/future/forecast.hpp`, `src/dcc/future/forecast.cpp`, `tests/reference/dcc_forecast_rmgarch.R`,
  `tests/reference/fixtures/rmgarch_dcc_forecast.json`
- Modify: `CMakeLists.txt`, `tests/reference/README.md`
- Test: `tests/unit/dcc/future/forecast_test.cpp`, `tests/unit/dcc/future/forecast_oracle_test.cpp`

**Interfaces:**
- Consumes: `DccEnd`, `CorrelationStepper`, `SeriesEnd` (Task 3); `forecast::forecast(spec, estimate, state, ...)`
  (Task 2).
- Produces:
  ```cpp
  namespace ve::dcc::future {
  // One FutureRegressors per series, horizon rows each (empty for a series without regressors).
  using SeriesRegressors = std::vector<forecast::FutureRegressors>;
  // h = 1..horizon: r and h (horizon x m x m), sigma and mean (horizon x m), row-major.
  struct DccForecast { std::size_t horizon = 0, series = 0; std::vector<double> r, h, sigma, mean; };
  // Why fit cannot be forecast with future, or nullopt: no finite point ("no finite point"), no series_end
  // (fit_dcc outside a batch), future with another number of series, or a series whose rows do not match its spec
  // ("series i: ..." from 1, with the messages of the 3a regressor checks).
  std::optional<std::string> forecast_problem(const DccFitResult& fit, std::size_t horizon,
                                              const SeriesRegressors& future);
  DccForecast forecast_dcc(const DccFitResult& fit, std::size_t horizon, const SeriesRegressors& future);
  }
  ```
- **Regras:** as do "Regras do forecast" acima. R̄ sai de `end.qbar`.

- [ ] **Step 1: Gerar o oráculo.** `tests/reference/dcc_forecast_rmgarch.R`, no padrão de `dcc_rmgarch.R`, sobre o
  `dcc3`:
  - univariados sGARCH(1,1)-norm, como `panel_fit`;
  - casos DCC(1,1) mvnorm, DCC(2,1) mvnorm e DCC(1,1) mvt com `n.ahead = 10`;
  - aDCC mvnorm e FDCC mvnorm (grupos 1,1,2) com `n.ahead = 1`;
  - por caso: os parâmetros DCC estimados (17 dígitos), `rcor`, `rcov` e `sigma` do forecast para h = 1..H.

  Grava `fixtures/rmgarch_dcc_forecast.json` e documenta a regeneração em `tests/reference/README.md`. A sonda da
  spec 11.3 é o ponto de partida; o script não copia código do `rmgarch`.
- [ ] **Step 2: Escrever os testes**

```cpp
TEST(DccForecastOracle, MatchesDccforecast) {
  // Valores copiados do JSON (padrão de oracles/oracle_test.cpp). dcc3_fits() (univariados fixos nos de rugarch);
  // spec do caso com todos os parâmetros DCC fixos nos do rmgarch e convention = Rmgarch; fit_dcc_batch;
  // forecast_dcc(h = H): r e h com |erro| <= 1e-8, sigma com erro relativo <= 1e-8. Se um caso não fechar, a causa
  // vai para validation.md e só aquele caso ganha tolerância maior, justificada
}
TEST(DccForecast, Formula) {
  // dcc3 Standard, DCC(1,1) e aDCC: r[h] == (1 - k^(h-1)) Rbar + k^(h-1) r[1] (1e-15), k = a + b (sem g);
  // FDCC: o mesmo por elemento com K_ij; h = 1 == stepper.next() bit a bit; h[h] == D r[h] D com D do sigma
}
TEST(DccForecast, OneGroupFdccIsDcc) {
  // FDCC com um grupo, fdcca1 = 0.2 e fdccb1 = sqrt(0.9) fixos (pesos 0.04 e 0.9) vs DCC com dcca1 = 0.04 e
  // dccb1 = 0.9 fixos: r iguais (1e-14)
}
TEST(DccForecast, SeriesAreTheirOwnForecast) {
  // Review Focus 5: painel com uma série ARFIMA, uma csGARCH e uma sGARCH com mxreg (T comum): sigma e mean da
  // série i == forecast(fits[i], H, future_i) bit a bit
}
TEST(DccForecast, Copula) {
  // copula DCC mvt paramétrica e empírica no copula3: r[1] == stepper(end).next(); h >= 2 pela fórmula; static
  // mvnorm kendall: r[h] == static_r em todo h
}
TEST(DccForecast, Problems) {
  // Review Focus 1 e 3: o Failed de DccEnd.NoFiniteEstimate -> "no finite point"; uma cópia de um resultado
  // convergido com status = Failed (o caso da busca de linha que falha com ponto finito) -> forecast normal, igual
  // ao do original; future com 2 entradas para m = 3 -> mensagem com "series"; série 2 com mxreg sem linhas ->
  // "series 2"
}
```

- [ ] **Step 3: Rodar e ver falhar** — `ctest --preset gcc-release -R DccForecast` → não compila.
- [ ] **Step 4: Implementar `forecast_problem` e `forecast_dcc`.**
- [ ] **Step 5: Rodar e ver passar** — suíte release inteira PASS.
- [ ] **Step 6: Commit** — `git commit -m "feat: forecast DCC and copula DCC fits"`

---

### Task 5: Simulação do DCC e da Copula-DCC

**Files:**
- Create: `src/dcc/future/simulate.hpp`, `src/dcc/future/simulate.cpp`
- Modify: `CMakeLists.txt`
- Test: `tests/unit/dcc/future/simulate_test.cpp`

**Interfaces:**
- Consumes: `CorrelationStepper`, `DccEnd`, `SeriesEnd` (Task 3); `SeriesRegressors`, `forecast_problem`,
  `forecast_dcc` (Task 4, os testes); `sim::Path`, `sim::start_of` (Task 2); `dist::quantile` (Task 1);
  `sim::normal`, `sim::gamma`, `Stream`; `linalg::cholesky_factor`; `run_in_blocks`, `Context::with_pool`.
- Produces:
  ```cpp
  namespace ve::dcc::future {
  struct DccRequest {
      std::size_t horizon = 0, paths = 0;
      std::uint64_t seed = 0;
      bool correlations = false;
      SeriesRegressors future{};
  };
  // returns and sigma paths x horizon x m; r and h paths x horizon x m x m, empty without correlations.
  struct DccSimulation { std::size_t paths = 0, horizon = 0, series = 0; std::vector<double> returns, sigma, r, h; };
  // Path p reads only Stream(seed, p): the same for any thread count. The caller checked
  // forecast_problem(fit, request.horizon, request.future).
  DccSimulation simulate_dcc(Context& context, const DccFitResult& fit, const DccRequest& request);
  }
  ```
- **Regras:** as do "Regras da simulação" acima. Os zᵢₜ ordenados da marginal empírica saem de `fit.panel->z`, uma
  vez por chamada.

- [ ] **Step 1: Escrever os testes** (sementes fixas, determinísticos; "5 SE" como no 3b)

```cpp
TEST(DccSimulate, SameForAnyThreadCount) {
  // DCC mvt, aDCC mvnorm, mvlaplace e copula DCC mvt paramétrica com margens std: 1, 3 e 16 threads, bit a bit
  // (returns, sigma, r, h)
}
TEST(DccSimulate, Prefixes) {
  // os primeiros k caminhos de um run maior e os primeiros passos de um horizonte maior são iguais
}
TEST(DccSimulate, FirstStepShocks) {
  // dcc3 DCC mvnorm, mvt e mvlaplace, 4e5 caminhos, h = 1: média de z_i z_j (z = e / sigma) == r[1] do forecast
  // em 5 SE; sigma do passo 1 == sigma do forecast bit a bit
}
TEST(DccSimulate, MeansFollowTheForecast) {
  // dcc3 DCC(1,1) e aDCC, 2e4 caminhos, h = 20, correlations: média de R[p, h] dentro de 5 SE + 0.01 do forecast
  // (a aproximação de Engle-Sheppard tem viés pequeno); média de sigma^2 em 5 SE do sigma^2 do forecast
}
TEST(DccSimulate, CopulaMargins) {
  // copula3, copula DCC mvnorm paramétrica: u = F_i(z) dos caminhos com média 0.5 e variância 1/12 (5 SE);
  // empírica: todo z simulado é um dos z_it do ajuste
}
TEST(DccSimulate, CorrelationsOff) {
  // sem correlations: r e h vazios, returns e sigma iguais bit a bit aos do run com correlations
}
TEST(DccSimulate, ArfimaAndComponentSeries) {
  // Review Focus 5: o painel da Task 4 com ARFIMA e csGARCH: 100 caminhos x 30 passos finitos
}
```

- [ ] **Step 2: Rodar e ver falhar** — `ctest --preset gcc-release -R DccSimulate` → não compila.
- [ ] **Step 3: Implementar `simulate_dcc`.** O stepper é por caminho e os `Path` são por série; os sorteios seguem a
  ordem do passo 2 das regras.
- [ ] **Step 4: Rodar e ver passar** — suíte release inteira PASS.
- [ ] **Step 5: Commit** — `git commit -m "feat: simulate DCC and copula DCC paths"`

---

### Task 6: Interface C e Python

**Files:**
- Create: `src/capi/dcc/dcc_forecast_api.cpp`, `src/capi/dcc/dcc_sim_api.cpp`,
  `python/tests/dcc/test_dcc_forecast.py`, `python/tests/dcc/test_dcc_simulate.py`
- Modify: `include/volengine/volengine.h`, `src/capi/handles.hpp`, `src/capi/results/sim_api.cpp`, `CMakeLists.txt`,
  `tests/unit/capi/dcc_test.cpp`, `tests/unit/capi/c_header_check.c`, `tests/unit/capi/abi_test.cpp` (se ele lista
  os símbolos), `python/src/volengine/_dcc.py`, `_forecast.py`, `_simulation.py`, `api.py`, `__init__.py`,
  `python/tests/typing_cases.py`
- Test: os acima

**Interfaces:**
- Consumes: Tasks 3 a 5; `ve::capi::store_rows`/`check_stored` (`regressor_rows.hpp`).
- Produces (C):
  ```c
  typedef struct ve_dcc_future ve_dcc_future;      /* regressors of the horizon, per series */
  typedef struct ve_dcc_forecast ve_dcc_forecast;
  typedef struct ve_dcc_sim ve_dcc_sim;
  VE_API ve_status ve_dcc_future_create(size_t series, size_t horizon, ve_dcc_future** out);
  /* series from 0; NULL panels: none for that series. */
  VE_API ve_status ve_dcc_future_set(ve_dcc_future* x, size_t series, const ve_panel* mxreg, const ve_panel* vxreg);
  VE_API void ve_dcc_future_free(ve_dcc_future* x);
  /* x NULL: no series has regressors. */
  VE_API ve_status ve_dcc_forecast_create(const ve_dcc_fit* f, size_t horizon, const ve_dcc_future* x,
                                          ve_dcc_forecast** out);
  VE_API ve_status ve_dcc_forecast_horizon(const ve_dcc_forecast* fc, size_t* out);
  VE_API ve_status ve_dcc_forecast_series(const ve_dcc_forecast* fc, size_t* out);
  VE_API ve_status ve_dcc_forecast_correlation(const ve_dcc_forecast* fc, const double** data, size_t* len);
  VE_API ve_status ve_dcc_forecast_covariance(const ve_dcc_forecast* fc, const double** data, size_t* len);
  VE_API ve_status ve_dcc_forecast_sigma(const ve_dcc_forecast* fc, const double** data, size_t* len);
  VE_API ve_status ve_dcc_forecast_mean(const ve_dcc_forecast* fc, const double** data, size_t* len);
  VE_API void ve_dcc_forecast_free(ve_dcc_forecast* fc);
  /* Keep R and H per path in a DCC simulation; ve_simulate_fit and ve_simulate_spec refuse options with it on. */
  VE_API ve_status ve_sim_options_set_correlations(ve_sim_options* o, int32_t on);
  /* o's own regressors must be unset (VE_INVALID_ARGUMENT): a DCC takes them per series in x. */
  VE_API ve_status ve_simulate_dcc(ve_context* ctx, const ve_dcc_fit* f, const ve_sim_options* o,
                                   const ve_dcc_future* x, ve_dcc_sim** out);
  VE_API ve_status ve_dcc_sim_paths(const ve_dcc_sim* sim, size_t* out);
  VE_API ve_status ve_dcc_sim_horizon(const ve_dcc_sim* sim, size_t* out);
  VE_API ve_status ve_dcc_sim_series(const ve_dcc_sim* sim, size_t* out);
  VE_API ve_status ve_dcc_sim_returns(const ve_dcc_sim* sim, const double** data, size_t* len);
  VE_API ve_status ve_dcc_sim_sigma(const ve_dcc_sim* sim, const double** data, size_t* len);
  /* VE_INVALID_ARGUMENT unless the options had correlations on. */
  VE_API ve_status ve_dcc_sim_correlation(const ve_dcc_sim* sim, const double** data, size_t* len);
  VE_API ve_status ve_dcc_sim_covariance(const ve_dcc_sim* sim, const double** data, size_t* len);
  VE_API void ve_dcc_sim_free(ve_dcc_sim* sim);
  ```
- Produces (Python):
  ```python
  class DccFit:
      def forecast(self, horizon: int, *, mxreg: Sequence[ArrayLike | None] | None = None,
                   vxreg: Sequence[ArrayLike | None] | None = None) -> DccForecast: ...
  class DccForecast:   # in _forecast.py; read-only zero-copy views
      R: NDArray[np.float64]; H: NDArray[np.float64]   # (horizon, N, N)
      sigma: NDArray[np.float64]; mean: NDArray[np.float64]   # (horizon, N)
  class DccSimulation:  # in _simulation.py
      returns: NDArray[np.float64]; sigma: NDArray[np.float64]   # (paths, horizon, N)
      R: NDArray[np.float64]; H: NDArray[np.float64]   # (paths, horizon, N, N); VolEngineError without correlations
  class Context:
      @overload
      def simulate(self, source: Fit | Spec, *, horizon: int, paths: int, seed: int,
                   **regressors: Unpack[FutureKeywords]) -> Simulation: ...
      @overload
      def simulate(self, source: DccFit, *, horizon: int, paths: int, seed: int, correlations: bool = False,
                   mxreg: Sequence[ArrayLike | None] | None = None,
                   vxreg: Sequence[ArrayLike | None] | None = None) -> DccSimulation: ...
  ```
- **Limites em `ve_simulate_dcc`:** paths × horizon × m × 2 doubles e, com `correlations`, paths × horizon × m² × 2
  doubles abaixo de `SIZE_MAX`. Caso contrário, `VE_INVALID_ARGUMENT` "too large".
- **Docstrings:**
  - `DccFit.forecast` e `simulate` dizem que o `DccFit` guarda o estado final de cada série e não precisa dos `Fit`;
  - `standard_errors` diz que precisa deles porque refiltra a amostra inteira;
  - `simulate` repete a regra de semente do 3b.

- [ ] **Step 1: Escrever os testes**

```text
tests/unit/capi/dcc_test.cpp:
  DccForecastApi: dimensões e visões == forecast_dcc; erros: f NULL (VE_INVALID_HANDLE), horizon 0, x com outro
    series ou horizon, fit sem ponto finito (mensagem "no finite point")
  DccSimApi: returns/sigma == simulate_dcc; correlation sem a opção -> VE_INVALID_ARGUMENT; o com mxreg próprio ->
    VE_INVALID_ARGUMENT; ve_simulate_fit com correlations -> VE_INVALID_ARGUMENT;
    Review Focus 4: paths = SIZE_MAX / 2, horizon 2, m = 3 com correlations -> VE_INVALID_ARGUMENT "too large"
python/tests/dcc/test_dcc_forecast.py:
  formato (H, N, N) e (H, N); read-only; h >= 2 pela fórmula com o R[0] do próprio forecast e R̄ de np.cov(z)
    normalizado (convenção pública); sigma[:, i] == fits[i].forecast(H).sigma
  Review Focus 2: d.standard_errors(fits).forecast(H) == d.forecast(H) (array_equal); del fits; gc.collect();
    d.forecast(H) funciona
  Review Focus 3: mxreg=[None, x, None] numa série com regressor; comprimento 2 -> VolEngineError com "series"
python/tests/dcc/test_dcc_simulate.py:
  formatos; mesma seed -> arrays iguais; correlations=False -> R levanta VolEngineError; copula DCC simula
python/tests/typing_cases.py:
  reveal/assert_type: ctx.simulate(fit, ...) -> Simulation; ctx.simulate(dcc_fit, ...) -> DccSimulation;
  dcc_fit.forecast(5) -> DccForecast
```

- [ ] **Step 2: Rodar e ver falhar** — `ctest --preset gcc-release -R "DccForecastApi|DccSimApi"` e
  `pytest python/tests/dcc` → falham.
- [ ] **Step 3: Implementar.** Os handles novos ficam em `handles.hpp`, e os acessores seguem o padrão de
  `sim_api.cpp`. Os módulos Python novos não cabem (`python/src/volengine/` tem 8): `DccForecast` entra em
  `_forecast.py` e `DccSimulation` em `_simulation.py`. O `ve_dcc_future` guarda `StoredRows` por série; a checagem
  contra a spec de cada série é a de `forecast_problem`.
- [ ] **Step 4: Rodar e ver passar** — suíte release, `pytest python/tests`, `check_style.py`, `mypy --strict` →
  PASS.
- [ ] **Step 5: Commit** — `git commit -m "feat: expose DCC forecast and simulation through the C API and Python"`

---

### Task 7: Benchmarks e documentação

**Files:**
- Modify: `tests/benchmarks/fit_benchmarks.cpp`, `docs/performance/benchmarks.md`, `docs/guide/validation.md`,
  `ARCHITECTURE.md`, `CHANGELOG.md`, `README.md` (se ele mostra o DCC)

- [ ] **Step 1: Benchmarks.** Na carga do 4a (m = 21, T = 5000, univariados ajustados fora do laço):
  - `BM_DccForecast/{dcc,adcc}_mvnorm/m:21/h:60`;
  - `BM_DccSimulate/{dcc_mvnorm,adcc_mvnorm,copula_dcc_mvt_param}/paths:1000/h:60/correlations:{0,1}/threads:{1,all}`.
  A cópula mvt usa `mshape` fixo em 8, para não pagar o perfil de ~37 s no setup.
- [ ] **Step 2: Rodar.** Rodar 5 repetições, como nos planos anteriores, e preencher
  `docs/performance/benchmarks.md` "## DCC forecast and simulation (plan 4c)". Se o autor adiar a rodada, a seção
  fica marcada **Pending** com o comando, como a do 4b.
- [ ] **Step 3: Documentar.**
  - `validation.md` ganha "### `rmgarch`, DCC forecast" (11.3 e 11.5: o que bate, a tolerância, a persistência
    total nas ordens (a, b), o aDCC h > 1 com E[nn′] ≈ N̄ sem oráculo, o FDCC h ≥ 2 tratado como bug, o FDCC h = 1
    com Q̄ = cov(z), a cópula sem `cgarchforecast`) e "## DCC simulation" (choques, cópula, quantis, os testes
    estatísticos);
  - `ARCHITECTURE.md`: `src/dcc/future/`, `sim/path.*`, os dois `capi/dcc/*_api.cpp`;
  - `CHANGELOG.md`: a API nova e a regra de semente;
  - README: um exemplo curto de `d.forecast(h).H`, se a seção do DCC existir.
- [ ] **Step 4: Rodar** — suíte release, `pytest python/tests`, `check_style.py` → PASS.
- [ ] **Step 5: Commit** — `git commit -m "bench: add DCC forecast and simulation benchmarks and document plan 4c"`
