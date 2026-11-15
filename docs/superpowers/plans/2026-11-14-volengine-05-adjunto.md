# Gradiente adjunto (fase 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Calcular ℓ e ∂ℓ/∂θ por adjunto escrito à mão (forward `loglik<double>`, reverse por componente) para
sGARCH/iGARCH/gjrGARCH/eGARCH × norm/std/ged × ARMA, e usá-lo no objetivo do L-BFGS e no gradiente da Hessiana
numérica; todo o resto continua no `Dual`.

**Architecture:** Cada componente ganha um `backward` logo abaixo do forward que deriva (densidade em
`dist/symmetric.hpp`, ARMA em `mean/arma.hpp`, recursões em `garch/*.hpp`). Um arquivo novo
`model/loglik_adjoint.{hpp,cpp}` roda `loglik<double>` gravando e e σ², depois os `backward` na ordem inversa de
`evaluate`. `has_adjoint` é a lista explícita do que está coberto; `fit/fit.cpp` usa o adjunto quando ela diz sim.

**Tech Stack:** C++23, GoogleTest, Google Benchmark; nenhuma dependência nova.

**Spec:** `docs/superpowers/specs/2026-11-14-volengine-05-adjoint-design.md` (design-mãe:
`docs/superpowers/specs/2026-11-14-volengine-05-performance-design.md`).

## Global Constraints

- Interface C não muda; nenhuma assinatura pública existente muda.
- Bit-identidade entre números de threads continua valendo.
- O forward é sempre `model::loglik<double>`; nenhum segundo forward. ℓ do adjunto é bit-idêntico a `loglik<double>`.
- Teste de propriedade: |adjunto − Dual| ≤ 1e-10 · max(1, ‖∇ do Dual‖∞) por componente.
- Derivadas em pontos não diferenciáveis seguem o `Dual`: `abs` com sinal 0 em 0; indicador do gjr com derivada 0;
  `pow` com a regra de `ad::detail::pow_slope`.
- `STYLEGUIDE.md` inteiro (funções ≤ 60 linhas, ≤ 5 parâmetros, aninhamento ≤ 3, sem `else`), verificado por
  `tests/style/check_style.py` no ctest.
- Testes só no preset `gcc-release` (`cmake --build --preset gcc-release && ctest --preset gcc-release`); medidas
  no `gcc-bench`.
- Código e docs do repo em inglês; docs em `docs/superpowers` em português.
- Commits sem trailer `Co-Authored-By`.

## Review Focus

1. **Resíduo exatamente zero com ged** (retorno zero, ARMA(0,0), μ = 0): com ν < 1 o `Dual` dá derivada não finita
   ou zero pela regra do `pow_slope`/`abs`; o adjunto tem de dar o mesmo veredito de finitude (e o mesmo valor quando
   finito). Teste na Task 3 e na Task 1.
2. **z exatamente zero no eGARCH** (|z| com sinal 0): mesmo critério. Teste na Task 5.
3. **Parâmetros fixos** (`Layout::fixed`): o gradiente em θ inclui os fixos e a cadeia por J os zera; o adjunto não
   pode depender de quais estão livres. Caso fixo no teste de propriedade (Task 3) e um ajuste com β fixo (Task 6).
4. **Série mais curta que held** (todo σ² é presample): o adjunto do presample recebe tudo. Caso T = 3 no teste de
   propriedade (Task 3).
5. **Ponto fora do domínio** (σ² ≤ 0 ou forma fora): o adjunto devolve ℓ não finito sem ler os buffers do reverse,
   o objetivo devolve +∞. Teste na Task 3.

---

### Task 1: Adjunto do termo das densidades simétricas

**Files:**
- Modify: `src/dist/symmetric.hpp`
- Test: `tests/unit/dist/densities_test.cpp`

**Interfaces:**
- Produces:
  ```cpp
  namespace ve::dist {
  // Partials of term = log f(e / sigma) - log(sigma2) / 2 under a LogDensity<double>.
  struct TermAdjoint { double e; double sigma2; double scale; double power; };
  TermAdjoint term_adjoint(const LogDensity<double>& density, double e, double sigma2);
  }
  ```
  (∂/∂log_constant é sempre 1 e não vai no struct.)

- [ ] **Step 1: Write the failing test** `DensityTerm.AdjointMatchesDual`: para norm, std (ν ∈ {2.5, 8, 30}) e ged
  (ν ∈ {0.8, 1.5, 3}), e ∈ {−1.3, 0.2, 2.7}, σ² ∈ {0.4, 1.7}: monte o termo em `Dual<4>` com lanes (e, σ², scale,
  power) — scale e power da `make_log_density<double>` semeados como variáveis, `log_constant` constante — pela
  mesma expressão da verossimilhança (std/norm via z² = e·e/σ²; ged via `abs(e)/sqrt(σ²)`), menos `0.5 * log(σ²)`.
  Compare cada parcial de `term_adjoint` com `EXPECT_NEAR(..., 1e-12 * max(1, |dual|))`.
  Acrescente o caso e = 0 com ged ν = 0.8 e ν = 1.5: `std::isfinite` de cada parcial igual ao do `Dual`, e valores
  iguais quando finitos.
- [ ] **Step 2: Run** `ctest --preset gcc-release -R DensityTerm` — Expected: FAIL (não compila: `term_adjoint` não
  existe).
- [ ] **Step 3: Implement `term_adjoint`** em `src/dist/symmetric.hpp`, abaixo de `log_density_at`, um ramo por
  `density.dist`, fórmula no comentário de cada um (x = e²/σ²):
  - norm: ∂e = −e/σ²; ∂σ² = ½(x − 1)/σ²; scale = power = 0.
  - std (s = scale, p = power, u = x/s, w = p/(1+u)): ∂e = −2we/(σ²s); ∂σ² = (wu − ½)/σ²; ∂s = wu/s;
    ∂p = −log1p(u).
  - ged (λ = scale, p = power, y = |e|/(√σ² λ)): ∂e = −½ p y^{p−1} sign(e)/(√σ² λ); ∂σ² = (¼ p y^p − ½)/σ²;
    ∂λ = ½ p y^p/λ; ∂p = −½ y^p log y (0 em y = 0). y^{p−1} pela mesma regra de `ad::detail::pow_slope`, sign(0) = 0.
- [ ] **Step 4: Run** `ctest --preset gcc-release -R DensityTerm` — Expected: PASS.
- [ ] **Step 5: Commit** `git commit -m "feat: adjoint of the symmetric density terms"`

### Task 2: Adjunto da recursão ARMA

**Files:**
- Modify: `src/mean/arma.hpp`
- Test: `tests/unit/mean/arma_test.cpp`

**Interfaces:**
- Produces:
  ```cpp
  namespace ve::mean {
  // d/d mu, d/d ar_i, d/d ma_j, accumulated (+=).
  struct ArmaGradient { double mu; std::span<double> ar; std::span<double> ma; };
  // The reverse of arma_residuals over d_t = r_t - mu. residual_adjoint holds d/d e_t on entry and is consumed
  // (overwritten) by the sweep.
  void arma_residuals_backward(std::span<const double> r, const ArmaParameters<double>& p,
                               std::span<const double> residuals, std::span<double> residual_adjoint,
                               ArmaGradient& gradient);
  }
  ```

- [ ] **Step 1: Write the failing test** `Arma.BackwardMatchesDual`: r = `simulate_garch_series(3, 200)`; para
  ARMA (0,0), (1,1), (2,2), (3,1): pesos w_t fixos (por exemplo sin(t)); L = Σ w_t e_t com `arma_residuals` em
  `Dual<8>` (lanes μ, ar…, ma…); compare com `arma_residuals_backward` a partir de residual_adjoint = w, tolerância
  1e-12 relativa ao maior |∂|.
- [ ] **Step 2: Run** `ctest --preset gcc-release -R Arma.BackwardMatchesDual` — Expected: FAIL (não compila).
- [ ] **Step 3: Implement** em `src/mean/arma.hpp`, abaixo de `arma_residuals`. Varredura t = T−1…0 com
  g = residual_adjoint[t] e m = |ar|:
  - t < m: μ −= g;
  - senão: μ += g(Σ ar − 1); ar_i −= g·d_{t−i}; para j ≤ min(|ma|, t): ma_j −= g·e_{t−j} e
    residual_adjoint[t−j] −= g·ma_j.
- [ ] **Step 4: Run** o teste — Expected: PASS.
- [ ] **Step 5: Commit** `git commit -m "feat: adjoint of the ARMA recursion"`

### Task 3: Composição, sGARCH e iGARCH, teste de propriedade

**Files:**
- Create: `src/model/model_parameters.hpp` (internal: as visões de θ que `likelihood.cpp` monta hoje)
- Create: `src/model/loglik_adjoint.hpp`, `src/model/loglik_adjoint.cpp`
- Modify: `src/model/likelihood.cpp`, `src/model/likelihood.hpp`, `src/garch/sgarch.hpp`, `CMakeLists.txt`
- Test: `tests/unit/model/loglik_adjoint_test.cpp` (novo, registrado em `CMakeLists.txt`)

**Interfaces:**
- Consumes: `dist::term_adjoint` (Task 1), `mean::arma_residuals_backward` (Task 2).
- Produces:
  ```cpp
  namespace ve::model {
  // The presample loglik uses at theta: Sample from e0 (mean square, held = max orders), Backcast fixed by data.
  garch::Presample<double> presample_at(const Data& data, std::span<const double> e0,
                                        std::span<const double> theta, const Layout& layout, VarInit var_init);
  // True only for the combinations of the adjoint's phase 1, listed explicitly.
  bool has_adjoint(const Layout& layout, VarInit var_init);
  // ell and d ell / d theta (natural space) by the adjoint; buffers kept between calls. Requires
  // has_adjoint(layout, var_init). grad is unspecified when the result is not finite.
  class LoglikAdjoint {
  public:
      double evaluate(const Data& data, std::span<const double> theta, const Layout& layout, VarInit var_init,
                      std::span<double> grad);
  private: /* residuals, sigma2, residual_adjoint, sigma2_adjoint */
  };
  }
  namespace ve::garch {
  // d/d omega, alpha_i, beta_j (+=) and the presample's adjoint; the residual and sigma2 adjoints as in ARMA.
  struct SGarchAdjoint { std::span<double> residuals; std::span<double> sigma2; double omega;
                         std::span<double> alpha; std::span<double> beta; double presample; };
  void sgarch_variance_backward(std::span<const double> residuals, const SGarchParameters<double>& p,
                                const Presample<double>& presample, std::span<const double> sigma2,
                                SGarchAdjoint& adjoint);
  }
  ```
  `has_adjoint` nesta task: modelo ∈ {SGarch, IGarch}, dist ∈ {Norm, Std, Ged}, sem mxreg, vxreg, archm, ARFIMA nem
  variance targeting; qualquer `VarInit`; com ou sem média.

- [ ] **Step 1: Write the failing test** `tests/unit/model/loglik_adjoint_test.cpp`:
  - `LoglikAdjointGrid` parametrizado por modelo {SGarch, IGarch} × (p, q) ∈ {(1,1), (2,1), (1,2), (3,3)} ×
    ARMA ∈ {(0,0), (1,1), (2,2)} × {Norm, Std, Ged} × {Sample, Backcast} × com/sem média; série
    `simulate_garch_series(11, 600)`.
  - Pontos por caso: interior (μ 0.04, ar 0.2/i, ma −0.15/j, ω 0.02, α 0.05/p cada, β 0.9/q cada, ν 8 std e 1.5
    ged); persistência 0.999; std ν = 2.5 e ν = 30 (ramo Stirling); ged ν = 0.8 e 3.
  - `TEST_P(LoglikAdjointGrid, MatchesDual)`: ℓ do adjunto `EXPECT_EQ` em bits ao `loglik<double>`; gradiente contra
    `loglik<Dual<kMaxParameterCount>>` com a tolerância global.
  - `TEST(LoglikAdjoint, SeriesShorterThanPresample)`: T = 3, sGARCH(3,3)-ARMA(2,2)-std, Sample; mesmo critério.
  - `TEST(LoglikAdjoint, FixedParameterDoesNotChangeGradient)`: layout com β fixo (`Spec::fixed`) dá o mesmo
    gradiente em θ que sem fixo.
  - `TEST(LoglikAdjoint, OutsideDomainIsNotFinite)`: ω = −1 → resultado não finito, igual ao `loglik<double>`.
  - `TEST(LoglikAdjoint, ZeroResidualWithGedMatchesDual)`: r com um zero exato, ARMA(0,0) sem média, ged ν 0.8 e
    1.5: finitude de cada componente igual à do `Dual`, valores iguais quando finitos.
  - `TEST(LoglikAdjoint, CoversPhaseOneOnly)`: verdadeiro para SGarch e IGarch × norm/std/ged × Sample/Backcast;
    falso para ApArch, CsGarch, FGarchAllGarch, GjrGarch, EGarch (até as Tasks 4–5), Sstd, Jsu, mxreg = 1,
    vxreg = 1, archm = 1, ARFIMA, variance targeting.
- [ ] **Step 2: Run** `ctest --preset gcc-release -R LoglikAdjoint` — Expected: FAIL (não compila).
- [ ] **Step 3: Refactor** `block`, `arma_parameters`, `sgarch_parameters`, `gjrgarch_parameters`,
  `egarch_parameters` e o cálculo de β_q do iGARCH de `likelihood.cpp` para `model/model_parameters.hpp`, sem mudar
  comportamento; declare e defina `presample_at` (envolve o `presample<double>` existente). Rode
  `ctest --preset gcc-release -R Likelihood` — Expected: PASS (refactor puro).
- [ ] **Step 4: Implement `sgarch_variance_backward`** abaixo de `sgarch_variance`. Varredura t = T−1…held com
  g = sigma2_adjoint[t], b = presample.value:
  - ω += g;
  - para cada i: se i > t, α_i += g·b e presample += g·α_i; senão α_i += g·e²_{t−i} e
    residuals[t−i] += 2g·α_i·e_{t−i};
  - para cada j: se j > t, β_j += g·b e presample += g·β_j; senão β_j += g·σ²_{t−j} e sigma2[t−j] += g·β_j.
  - Depois, presample += Σ_{t<held} sigma2_adjoint[t].
- [ ] **Step 5: Implement `has_adjoint` e `LoglikAdjoint::evaluate`**, nesta ordem:
  1. ℓ = `loglik<double>(data, θ, layout, var_init, {.residuals, .sigma2})`; se não finito, devolve ℓ.
  2. Densidade: `make_log_density<double>(dist, ν)`; para cada t, `term_adjoint` → residual_adjoint[t],
     sigma2_adjoint[t], e somas de ∂scale e ∂power. Forma: `make_log_density<ad::Dual<1>>(dist, variable<1>(ν, 0))`
     dá c′, scale′, power′; ∂ν = T·c′ + Σ∂scale·scale′ + Σ∂power·power′ (norm: sem forma).
  3. Variância: sGARCH direto; iGARCH roda o backward com o β efetivo (β_q = 1 − Σα − Σ_{j<q} β_j) e depois
     α_i −= ∂β_q, β_j −= ∂β_q (j < q), ∂β_q armazenado = 0.
  4. Presample: com Sample, residual_adjoint[t] += presample_adjoint · 2e_t / T; com Backcast, nada.
  5. ARMA: `arma_residuals_backward` com d_t = r_t − μ (μ = 0 sem média, e então ∂μ = 0).
- [ ] **Step 6: Run** `ctest --preset gcc-release -R "LoglikAdjoint|Likelihood|Style"` — Expected: PASS.
- [ ] **Step 7: Commit** `git commit -m "feat: adjoint log-likelihood gradient for sGARCH and iGARCH"`

### Task 4: gjrGARCH

**Files:**
- Modify: `src/garch/gjrgarch.hpp`, `src/model/loglik_adjoint.cpp`
- Test: `tests/unit/model/loglik_adjoint_test.cpp`

**Interfaces:**
- Produces: `garch::GjrGarchAdjoint` (os campos de `SGarchAdjoint` mais `std::span<double> gamma`) e
  `gjrgarch_variance_backward(residuals, const GjrGarchParameters<double>&, presample, sigma2, GjrGarchAdjoint&)`.

- [ ] **Step 1: Extend the test**: GjrGarch entra no `LoglikAdjointGrid` (γ 0.05/p cada) e passa a verdadeiro em
  `CoversPhaseOneOnly`.
- [ ] **Step 2: Run** `ctest --preset gcc-release -R LoglikAdjoint` — Expected: FAIL (`has_adjoint` falso / sem
  backward).
- [ ] **Step 3: Implement** abaixo de `gjrgarch_variance`. O termo i de t:
  - se i > t: (α_i + γ_i/2)·b, logo α_i += g·b, γ_i += g·b/2 e presample += g(α_i + γ_i/2);
  - se e_{t−i} < 0: (α_i + γ_i)e², logo α_i += g·e², γ_i += g·e² e resíduo += 2g(α_i + γ_i)e;
  - senão: α_i·e², logo α_i += g·e² e resíduo += 2gα_i·e.
  - β como no sGARCH.
  - Ligue o modelo em `has_adjoint` e em `evaluate`.
- [ ] **Step 4: Run** o mesmo comando — Expected: PASS.
- [ ] **Step 5: Commit** `git commit -m "feat: adjoint log-likelihood gradient for gjrGARCH"`

### Task 5: eGARCH

**Files:**
- Modify: `src/garch/egarch.hpp`, `src/model/loglik_adjoint.cpp`, possivelmente `src/dist/moments.cpp` (instanciação
  `Dual<1>` de `abs_moment`, se o link pedir)
- Test: `tests/unit/model/loglik_adjoint_test.cpp`

**Interfaces:**
- Produces: `garch::EGarchAdjoint` (`SGarchAdjoint` + `gamma` + `double abs_moment`; aqui `sigma2` guarda o adjunto
  de h_t = log σ²_t e `presample` o de log b) e
  `egarch_variance_backward(residuals, const EGarchParameters<double>&, presample, sigma2, EGarchAdjoint&)`.

- [ ] **Step 1: Extend the test**: EGarch no grid (α −0.05, γ 0.1/p cada, β 0.9/q cada) e verdadeiro em
  `CoversPhaseOneOnly`; acrescente `TEST(LoglikAdjoint, EgarchZeroShockMatchesDual)`: r com zero exato, ARMA(0,0)
  sem média — finitude e valores como no caso ged.
- [ ] **Step 2: Run** `ctest --preset gcc-release -R LoglikAdjoint` — Expected: FAIL.
- [ ] **Step 3: Implement**:
  - Antes da varredura: em `evaluate`, o adjunto de h é gh_t = sigma2_adjoint_t · σ²_t. A varredura recalcula
    h_t = log σ²_t, ou h_t = log b para t < held.
  - Varredura t = T−1…held com g = gh_t:
    - ω += g;
    - para cada i, se i > t: γ_i += g(√(2/π) − E) e ∂E −= g·γ_i;
    - para cada i, senão, com z = e_{t−i}·exp(−½h_{t−i}):
      - α_i += g·z; γ_i += g(|z| − E); ∂E −= g·γ_i;
      - gz = g(α_i + γ_i·sign z), com sign 0 = 0;
      - resíduo_{t−i} += gz·exp(−½h_{t−i}); gh_{t−i} += gz·(−½z);
    - para cada j: se j > t, β_j += g·log b e ∂log b += g·β_j; senão β_j += g·h_{t−j} e gh_{t−j} += g·β_j.
  - Depois da varredura:
    - ∂log b += Σ_{t<held} gh_t; o adjunto de b é ∂log b / b, que segue para o passo 4 da Task 3;
    - ∂ν += ∂E · E′(ν), com E′(ν) de `dist::abs_moment<ad::Dual<1>>(dist, shape, 1)`; norm sem forma.
- [ ] **Step 4: Run** o mesmo comando — Expected: PASS.
- [ ] **Step 5: Commit** `git commit -m "feat: adjoint log-likelihood gradient for eGARCH"`

### Task 6: Usar o adjunto no ajuste

**Files:**
- Modify: `src/fit/fit.cpp`
- Test: `tests/unit/fit/fit_test.cpp`

**Interfaces:**
- Consumes: `model::has_adjoint`, `model::LoglikAdjoint` (Task 3).

O caminho só troca a forma de calcular o mesmo gradiente, então não há teste que fique vermelho antes. A prova é o
teste novo abaixo (cobre a entrada pública `fit::loglik_gradient`), a suíte inteira, os oráculos e a comparação da
grade na Task 7.

- [ ] **Step 1: Write the test** `FitGradient.AdjointPathMatchesDual`: para sGARCH(1,1)-ARMA(1,1)-std, eGARCH(1,1)-ged
  e gjrGARCH(2,1)-norm em `simulate_garch_series(5, 800)`, `fit::loglik_gradient` contra
  `loglik<Dual<kMaxParameterCount>>`, tolerância global; e `FitGradient.FixedBetaFitConverges`: sGARCH(1,1)-std com
  β fixo em 0.85 (`Spec::fixed`) termina `Converged`, β estimado = 0.85 exato e ℓ igual ao
  `loglik<double>` no estimado.
- [ ] **Step 2: Run** `ctest --preset gcc-release -R FitGradient` — Expected: PASS (mesmo gradiente pelo `Dual`).
- [ ] **Step 3: Implement**:
  - **`NegativeMeanLoglik<N>`**: com `has_adjoint(layout, var_init)`, guardado no construtor, o objetivo segue estes
    passos:
    1. semeia u e roda `to_natural<Number>` como hoje;
    2. copia os valores para θ;
    3. calcula ℓ e ∂ℓ/∂θ por `LoglikAdjoint::evaluate`;
    4. se ℓ não for finito, devolve +∞ com gradiente zero, como hoje;
    5. senão, grad_i = −Σ_j ∂ℓ/∂θ_j · θ_j.grad[i] / n.

    Sem `has_adjoint`, o caminho atual não muda. Extraia funções para ficar dentro do styleguide.
  - **`LoglikGradient<N>`**: com `has_adjoint`, `evaluate` direto em θ. Validade e finitude como hoje.
- [ ] **Step 4: Run** `ctest --preset gcc-release` (suíte inteira) e `pytest python/tests` — Expected: tudo verde;
  oráculos rugarch/arch/rmgarch nas tolerâncias atuais. Um oráculo que mude vai ao autor com os números.
- [ ] **Step 5: Commit** `git commit -m "perf: use the adjoint gradient in the optimizer and the numerical Hessian"`

### Task 7: Medida e documentação

**Files:**
- Modify: `tests/benchmarks/fit_benchmarks.cpp`, `docs/performance/profile-5a.md` (seção nova "Adjoint gradient"),
  `docs/performance/benchmarks.md`, `docs/performance/python-vs-arch.md`, `CHANGELOG.md`,
  `docs/superpowers/specs/2026-11-14-volengine-05-adjoint-design.md` (§10 "feito")

- [ ] **Step 1: Add** a variante `BM_Loglik/<case>/adjoint` para os casos de `named_specs` com `has_adjoint`.
- [ ] **Step 2: Measure A/B na mesma sessão** (procedimento de `docs/performance/measuring.md`, base = commit antes
  da Task 6):
  - grade de 1080 com 1 e 16 threads, erros-padrão on/off;
  - `compare-profile` da grade: status e ℓ;
  - quick suite: `BM_Loglik`, `BM_FitOne`, `BM_TccGridSample`;
  - `BM_DccGrid/m:21/T:5000/threads:1`, que não deve mover, e `BM_DccStandardErrors`;
  - Python contra `arch`.

  Expected: grade ~0,55 do tempo com 1 thread; todos `CONVERGED`; ajustes que trocarem de ótimo local vão listados
  ao autor.
- [ ] **Step 3: Write** os números nos docs listados; CHANGELOG em "Performance".
- [ ] **Step 4: Run** `ctest --preset gcc-release -R Style` — Expected: PASS.
- [ ] **Step 5: Commit** `git commit -m "docs: record the adjoint gradient benchmarks"`
