# VolEngine — Plano 4a: núcleo DCC

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Estimar o passo 2 de um DCC sobre N ajustes univariados escolhidos pelo usuário, como o `dccfit` do `rmgarch`. Cobre DCC e aDCC com ordens (a, b) e FDCC com grupos, com as distribuições mvnorm, mvt e mvlaplace, em lote de specs. Devolve:
- ℓ e critérios na convenção do `rmgarch`;
- Rₜ e Hₜ;
- erros-padrão em dois estágios, por opção no ajuste ou por função explícita.

Tudo exposto pela interface C e pelo Python (`ctx.fit_dcc`) e validado contra o `rmgarch` 1.4.3.

**Architecture:**
- **Módulo:** um módulo novo, `src/dcc/`. Ele recebe os zₜ = eₜ/σₜ dos `FitResult` univariados (que já guardam e e σ) e calcula Q̄ e N̄ uma vez por conjunto, que todas as specs do lote compartilham.
- **Recursão e verossimilhança:** são templates em `T` (`double` ou `Dual`), como os univariados, e o L-BFGS existente estima os parâmetros do DCC numa reparametrização que impõe as restrições.
- **Erros-padrão:** são uma etapa separada. Ela refiltra os univariados em pontos perturbados e por isso precisa dos `Fit` de origem: ou no próprio ajuste (opção ligada), ou depois, pela função explícita que devolve um `DccFit` novo, com os erros-padrão, que compartilha Rₜ com o original.

**Tech Stack:** C++23, CMake + presets, GoogleTest v1.17.0, Google Benchmark v1.9.1, Python ≥ 3.11 + numpy + pytest, R 4.6.2 + `rugarch` 1.5.6 + `rmgarch` 1.4.3.

**Spec:** `docs/superpowers/specs/2026-11-13-volengine-04-dcc-design.md` (seções 3 "4a", 4–9), sobre a spec-mãe `docs/superpowers/specs/2026-11-10-volengine-design.md` (seção 3 "Multivariado", seções 5–7).

## Roteiro

- **1, 2a–2f, 3a–3c (feitos).**
- **4a (este):** núcleo DCC.
- **4b:** Copula-DCC.
- **4c:** forecast e simulação do DCC.

Depois: 5 performance; 6 wrapper C++, wheels, 1.0.

## Global Constraints

Valem todas as Global Constraints dos planos anteriores, o `STYLEGUIDE.md` e o `ARCHITECTURE.md`. As que mais pesam aqui:

- **Interface C:**
  - ≤ 5 parâmetros por função; configuração nova é função nova;
  - handles imutáveis, legíveis de várias threads;
  - visões zero-copy válidas enquanto o handle existir;
  - nenhuma exceção atravessa a fronteira.
- **Nenhum número univariado muda:** `fit_one`, erros-padrão, forecast, simulação e diagnósticos ficam iguais, e todos os testes existentes passam sem mudar constantes nem tolerâncias.
- **Bit-identidade** entre números de threads no lote DCC.
- **Licença:** o `rmgarch` (GPL) foi lido para entender convenções e conferido numericamente contra objetos ajustados. Nenhum código dele é copiado ou traduzido. As convenções abaixo são descritas em matemática. O detalhe completo, com os números de referência, está em `docs/superpowers/plans/2026-11-13-volengine-04a-rmgarch-notes.md`.
- **Estilo:**
  - nunca `else`;
  - funções ≤ 60 linhas, ≤ 5 parâmetros, aninhamento ≤ 3;
  - ≤ 8 módulos por pasta (`src/dcc/` nasce com 7);
  - Boost só em `.cpp`.
- **Ambiente e commits:**
  - commits sem `Co-Authored-By`; LF;
  - **testes só em release**;
  - executáveis com `/c/msys64/ucrt64/bin` primeiro no PATH.

### Convenções do `rmgarch` 1.4.3 (lidas e conferidas a ~1e-15 contra ajustes)

Notação: m séries, T observações, zₜ = eₜ/σₜ por série, dos `Fit` (sem re-padronizar nem centrar).

**Q̄ e N̄.**
- Q̄ é a covariância amostral de z: centrada, com divisor T − 1, e **não** é uma matriz de correlação.
- No aDCC, nₜ = 1(zₜ < 0) ∘ zₜ, e N̄ é a covariância amostral de n. No DCC, N̄ = 0 e n = 0.

**Recursão DCC/aDCC** de ordem (P, Q), com Rₜ = diag(Qₜ)^−½ Qₜ diag(Qₜ)^−½:

    Qₜ = (1 − Σa − Σb) Q̄ − (Σg) N̄ + Σⱼ aⱼ zₜ₋ⱼ zₜ₋ⱼᵀ + Σⱼ gⱼ nₜ₋ⱼ nₜ₋ⱼᵀ + Σₖ bₖ Qₜ₋ₖ

**FDCC** (Billio–Caporin): G grupos, ordem (1, 1). α = S a e β = S b, em que S é a seleção por grupo:

    Qₜ = (1 − ααᵀ − ββᵀ) ∘ Q̄ + ααᵀ ∘ zₜ₋₁zₜ₋₁ᵀ + ββᵀ ∘ Qₜ₋₁

Com um grupo, (a_F, b_F) equivale ao DCC com a = a_F² e b = b_F².

**Duas pré-amostras.** Qₛ = Q̄ para s ≤ 0, e o rmgarch usa dois zₛ diferentes conforme o caso:

| Uso | zₛ, nₛ para s ≤ 0 | Q̄ do FDCC |
| --- | --- | --- |
| Objetivo que o otimizador minimiza | 0 | zᵀz/T (sem centrar) |
| Tudo que é reportado: ℓ, ℓₜ, Rₜ, Qₜ, Hₜ, scores e Hessiana dos erros-padrão | (1, …, 1) | cov(z) |

As estimativas maximizam o objetivo, e a ℓ reportada usa a outra pré-amostra. **O VolEngine reproduz isso** para que ℓ e BIC batam (decisão do Claude, pela regra "convenção padrão segue o oráculo"), e `validation.md` registra.

**Verossimilhança reportada.**

    ℓₜ = log f(zₜ; Rₜ) − Σᵢ log σᵢₜ    (a densidade de εₜ com Hₜ = Dₜ Rₜ Dₜ, Dₜ = diag(σₜ))

Com qₜ = zₜᵀ Rₜ⁻¹ zₜ e Lₜ = log|Rₜ|:
- **mvnorm:** −½ (m log 2π + Lₜ + qₜ).
- **mvt** (t padronizada, uma forma ν para todas as séries):

      lnΓ((ν+m)/2) − lnΓ(ν/2) − (m/2) log(π(ν−2)) − ½ Lₜ − ((ν+m)/2) log(1 + qₜ/(ν−2))

- **mvlaplace,** com v = 1 − m/2 e o `log_bessel_k` existente:

      log 2 − (m/2) log 2π − ½ Lₜ + (v/2) log(qₜ/2) + log K_v(√(2qₜ))

A ℓ reportada é Σₜ ℓₜ. **Não** é a soma das ℓ univariadas mais um termo de correlação.

**Objetivo do otimizador** (pré-amostra zero, sem Σ log σ):
- mvnorm: ½ Σ (Lₜ + qₜ), sem constantes;
- mvt e mvlaplace: −Σ log f, com todas as constantes.

**Parâmetros:**
- **Ordem:** a₁..a_P, b₁..b_Q, g₁..g_P (aDCC), forma (mvt). No FDCC: a[g=1..G], b[g=1..G], forma.
- **Nomes:** `dcca1`, `dccb1`, `dccg1`, `mshape`, `fdcca1..G` e `fdccb1..G`.
- **Valores iniciais:**

  | Parâmetro | Início |
  | --- | --- |
  | aⱼ | 0,05/P |
  | bₖ | 0,9/Q |
  | gⱼ | 0,05/P |
  | forma | 5 |
  | FDCC a | 0,01 |
  | FDCC b | 0,9 |

- **Restrições:**
  - DCC: aⱼ, bₖ > 0 e Σa + Σb < 1;
  - aDCC: Σa + Σb + δ Σg < 1, com δ o maior autovalor de Q̄^−½ N̄ Q̄^−½;
  - forma em (4, 50);
  - FDCC: (maxₘ a_g)² + (maxₘ b_g)² < 1, que é o conjunto que o rmgarch impõe com G⁴ desigualdades quando G > 1.
- **O bug do rmgarch com G = 1** (impõe a + b ≤ 1) **não é reproduzido.** O VolEngine usa a restrição do modelo, o FDCC de um grupo fica fora do oráculo, e `validation.md` registra.
- **Fixos e iniciais** como nos univariados (`fixed`/`start` por nome). Um parâmetro fixo sai da tabela e do k.

**Erros-padrão em dois estágios** (Engle–Sheppard; o rmgarch tem uma única tabela, o sanduíche abaixo). θ = (θ₁, …, θₘ, θ_D), com os parâmetros univariados estimados de cada série e depois os n_D do DCC:

- **A** (n_p × n_p):
  - os blocos univariados Aᵢᵢ são a Hessiana de −ℓᵢ do ajuste univariado;
  - os outros blocos univariados e o bloco (linha univariada, coluna DCC) são 0;
  - as linhas do DCC são a Hessiana da **ℓ total reportada (+ℓ)** em todo θ. Cada avaliação refiltra os univariados no θᵢ perturbado e recalcula z, Q̄ e N̄. São diferenças progressivas com hᵢ = ε^⅓ max(|θᵢ|, 1) e Hᵢⱼ = [f(θ + hᵢeᵢ + hⱼeⱼ) − f(θ + hᵢeᵢ) − f(θ + hⱼeⱼ) + f(θ)] / (hᵢhⱼ), simetrizada.
- **S** (T × n_p):
  - as colunas univariadas são os scores de −ℓᵢₜ do ajuste univariado;
  - as colunas DCC são diferenças centrais de +ℓₜ, com hₖ = max(|θₖ|/2, 0,01) ε^⅓.
- **Sanduíche:** B = cov(S), centrada e com divisor T − 1, e Ā = A/T. Então V = Ā⁻¹ B Ā⁻¹ / T, **sem transposta**, e o erro-padrão é √|diag V|.
- **Tabela "robust"** do VolEngine: as linhas do DCC de V, que é o que o `matcoef` do rmgarch mostra.
- **Tabela "hessian"** do VolEngine (o rmgarch não tem): √diag((−H_DD)⁻¹), em que H_DD é o bloco DCC de A. Como A é bloco-triangular, esse é o bloco DCC de A⁻¹. É decisão do Claude, porque o design pede as duas tabelas.
- **t e p:** t = θ/se e p = 2(1 − Φ(|t|)).

**Critérios** (por observação, como os univariados):
- k = parâmetros univariados estimados de todas as séries + parâmetros do DCC estimados + m(m − 1)/2;
- AIC = (−2ℓ + 2k)/T, BIC = (−2ℓ + k log T)/T, HQIC = (−2ℓ + 2k log log T)/T e SIC = −2ℓ/T + log((T + 2k)/T).
- Com `per_obs = False`, os mesmos sem dividir por T, como nos univariados.

**Saídas:**
- Rₜ e Hₜ em layout T × m × m, com o tempo como primeira dimensão, contíguo por instante. Isso dá `(T, m, m)` no numpy sem cópia. O `rcor` do rmgarch é [m, m, T].
- Hₜ = Dₜ Rₜ Dₜ, com o σ dos univariados.
- R₁ é o da pré-amostra de uns.

**Entrada e erros** (decisão do Claude):
- É preciso m ≥ 2 `Fit` com ponto finito e o mesmo T, além de T ≥ 100 (como no rmgarch).
- No FDCC, `groups` tem m entradas em 1..G, e cada grupo tem pelo menos uma série. O rmgarch não confere isso: um rótulo pulado zera a série.
- Tudo isso é `VE_INVALID_ARGUMENT`, com mensagem.
- A falha de convergência de uma spec do lote vira o status dela, sem derrubar o lote.
- **Erros-padrão com univariados que têm `mxreg`/`vxreg`:** a refiltragem precisaria dos regressores, que o `FitResult` não guarda. Isso dá `VE_UNSUPPORTED` com mensagem. `ponytail:` guardar os regressores quando alguém pedir. O TCC não usa regressores.

## Review Focus

1. **Univariados de tamanhos diferentes, ou um `FAILED` sem ponto finito, ou m = 1:** erro de argumento citando a série. → teste na Task 6.
2. **Grade com specs que não convergem** (aDCC com P = 2 num par quase descorrelacionado): status `max_iter`/`failed` por spec, nenhuma exceção, as outras specs iguais às de um lote sozinhas (bit a bit). → teste na Task 4.
3. **Rₜ válida em todo t:** diagonal exatamente 1, simétrica, definida positiva, mesmo com a mvt na borda (forma → 50) e com a + b perto de 1. → teste na Task 2.
4. **Erros-padrão pela função explícita com `Fit` diferentes dos do ajuste** (outro conjunto, ordem trocada, um `Fit` reajustado): erro de argumento, nunca um número. → teste na Task 5.
5. **Memória da grade:** 9 specs com m = 21 e T = 5.000 guardam Rₜ, mas Hₜ só quando pedido (calculado uma vez, seguro entre threads). Liberar os `Fit` não invalida nada. → testes nas Tasks 4 e 6.

---

## Mapa de arquivos

```
src/linalg/symmetric_eigen.*   largest_eigenvalue (power iteration; novo)
src/dcc/spec.hpp               DccModel, DccDist, DccSpec (novo)
src/dcc/panel.hpp, .cpp        StandardizedPanel: z (T x m), sigma, Q̄, N̄, Q̄ do FDCC, delta (novo)
src/dcc/recursion.hpp          correlation_path<T>(...) — Qₜ/Rₜ com a pré-amostra pedida (novo)
src/dcc/likelihood.hpp, .cpp   dcc_terms<T>, objective<T>, reported loglik (novo)
src/dcc/parameter_map.hpp,.cpp layout, nomes, reparametrização, inícios (novo)
src/dcc/fit.hpp, .cpp          DccFitResult, fit_dcc, fit_dcc_batch (novo)
src/dcc/standard_errors.*      dcc_standard_errors (novo)
src/capi/dcc/                  dcc_spec_api.cpp, dcc_fit_api.cpp (nova subpasta: src/capi já tem 8 módulos)
include/volengine/volengine.h  seção DCC
python/src/volengine/_dcc.py   DccSpec, DccFit, DccOptions (novo); api.py Context.fit_dcc; _native.py; __init__.py
tests/unit/dcc/*, tests/unit/capi/dcc_test.cpp, python/tests/test_dcc.py
tests/reference/dcc_rmgarch.R, data/dcc3.csv, fixtures/rmgarch_dcc.json, python/tests/oracles/test_dcc_oracles.py
```

---

### Task 1: Painel padronizado, Q̄/N̄ e autovalor

**Files:**
- Create: `src/linalg/symmetric_eigen.hpp`, `.cpp`, `src/dcc/spec.hpp`, `src/dcc/panel.hpp`, `src/dcc/panel.cpp`
- Modify: `CMakeLists.txt`
- Test: `tests/unit/linalg/symmetric_eigen_test.cpp`, `tests/unit/dcc/panel_test.cpp`

**Interfaces:**
- Produces:
  ```cpp
  namespace ve::linalg {
  // The largest eigenvalue of L^-1 N L^-T, L the Cholesky factor of the symmetric positive definite Q (the spectrum
  // of Q^-1/2 N Q^-1/2), N symmetric positive semidefinite; power iteration to 1e-14 relative. n x n row-major.
  double largest_relative_eigenvalue(std::span<const double> q, std::span<const double> n, std::size_t size);
  }
  namespace ve::dcc {
  enum class DccModel : int32_t { Dcc = 0, ADcc = 1, FDcc = 2 };
  enum class DccDist : int32_t { MvNorm = 0, MvT = 1, MvLaplace = 2 };
  struct DccSpec { DccModel model = DccModel::Dcc; int p = 1; int q = 1; DccDist dist = DccDist::MvNorm;
                   std::vector<int> groups{}; std::vector<ParameterValue> fixed{}, start{}; };
  // z = e / sigma of each fit (T x m row-major), sigma (T x m), and what every spec shares.
  struct StandardizedPanel {
      std::size_t t = 0, m = 0;
      std::vector<double> z, sigma;
      std::vector<double> qbar, nbar, qbar_raw;   // cov(z), cov(n), z'z/T (FDCC estimation); m x m
      double delta = 0.0;                         // largest eigenvalue of Qbar^-1/2 Nbar Qbar^-1/2
  };
  StandardizedPanel standardize(std::span<const FitResult* const> fits);   // the caller checked them
  std::optional<std::string> panel_problem(std::span<const FitResult* const> fits);  // regra "Entrada e erros"
  }
  ```

- [ ] **Step 1: Escrever os testes**

```cpp
TEST(SymmetricEigen, MatchesKnownSpectrum) {
  // Q = I: autovalor de N = [[2,1],[1,2]] → 3 (1e-14); Q = diag(4, 1), N = diag(1, 1) → 1; N = 0 → 0
}
TEST(DccPanel, CovariancesLikeR) {
  // z de 3 séries fixas (5 linhas): qbar == covariância centrada com divisor T-1 (1e-15), nbar idem sobre
  // n = 1(z<0) z (z == 0 conta 0), qbar_raw == z'z/T
}
TEST(DccPanel, ZFromTheFits) { /* 2 fit_one simulados: z[t*m + i] == residuals/sigma do fit i (bit a bit) */ }
TEST(DccPanel, Problems) {
  // m = 1; Ts diferentes (cita "series 1"); fit sem ponto finito (cita "finite"); T = 99 (cita "100")
}
```

- [ ] **Step 2: Rodar e ver falhar** — `ctest --preset gcc-release -R "SymmetricEigen|DccPanel"` → não compila.
- [ ] **Step 3: Implementar.**
- [ ] **Step 4: Rodar e ver passar** — suíte release inteira PASS.
- [ ] **Step 5: Commit** — `git commit -m "feat: standardize univariate fits into a DCC panel"`

---

### Task 2: Recursão e verossimilhança

**Files:**
- Create: `src/dcc/recursion.hpp`, `src/dcc/likelihood.hpp`, `src/dcc/likelihood.cpp`
- Test: `tests/unit/dcc/recursion_test.cpp`, `tests/unit/dcc/likelihood_test.cpp`

**Interfaces:**
- Consumes: Task 1; `linalg::cholesky_*`; `dist::log_bessel_k` e `log_bessel_k_slope`, este para o `Dual`.
- Produces:
  ```cpp
  namespace ve::dcc {
  enum class Presample { Zero, Ones };   // regra "Duas pré-amostras"
  // Natural DCC parameters in the order of the "Parâmetros" rule (shape last when mvt).
  template <class T> struct DccParameters { std::span<const T> a, b, g; T shape; };
  // Writes R_t (T x m x m) when r is not empty; returns per-observation log f(z_t; R_t) (without -sum log sigma)
  // into terms. Uses qbar_raw when FDCC and presample == Zero, as rmgarch does.
  template <class T>
  void correlation_terms(const StandardizedPanel& panel, const DccSpec& spec, const DccParameters<T>& theta,
                         Presample presample, std::span<T> terms, std::span<double> r = {});
  double reported_loglik(const StandardizedPanel&, const DccSpec&, std::span<const double> theta,
                         std::span<double> r = {});        // sum_t (log f - sum_i log sigma_it), Ones presample
  template <class T> T objective(const StandardizedPanel&, const DccSpec&, std::span<const T> theta);  // Zero presample
  }
  ```
- `tests/reference/data/dcc3.csv` já está commitado, junto com o plano. São os dados das notas
  `docs/superpowers/plans/2026-11-13-volengine-04a-rmgarch-notes.md` (seção 10: DCC simulado, seed 42, T = 1500,
  m = 3). As notas trazem os coeficientes univariados das três séries e os números de referência usados abaixo.

- [ ] **Step 1: Escrever os testes**

```cpp
// dcc3.csv + univariados sGARCH(1,1)-norm fixos nos coeficientes do rugarch (as notas, seção 10):
// A (-0.00891999125885, 0.01947914006937, 0.07282178352751, 0.90657081832066), B (...), C (...).
TEST(DccLikelihood, MatchesRmgarchAtItsPoint) {
  // DCC mvnorm a = 0.0431630001835819, b = 0.9142701826310855:
  // reported_loglik == -6082.94087772754 (rel 1e-10); 3 primeiros l_t: -3.87527571695479, -2.9806184777153,
  // -5.51904233840933 (1e-10); R_1: r12 0.497294779040641, r13 0.323589762766861, r23 0.368658387124292 (1e-12);
  // objective == 1911.09985833825 (rel 1e-10)
}
TEST(DccLikelihood, OtherDistributionsAndModelsAtRmgarchPoints) {
  // nos pontos da tabela "Other variants" das notas: DCC/mvt -6083.64532606; aDCC/mvnorm -6082.91639426;
  // FDCC grupos (1,1,2) -6082.56656677; DCC/mvlaplace -6364.40615222; DCC(2,1) -6082.93058943 (rel 1e-9)
}
TEST(DccLikelihood, FdccOneGroupIsDcc) {
  // as três distribuições: FDCC 1 grupo (a_F, b_F) == DCC (a_F^2, b_F^2) em loglik e R_t (rel 1e-13)
}
TEST(DccLikelihood, DualGradientMatchesDifferences) { /* objective<Dual> vs diferença central, rel 1e-6, 3 dists */ }
TEST(DccRecursion, CorrelationsAreValid) {
  // Review Focus 3: a + b = 0.999, forma 49.999: diag(R_t) == 1 exato, simétrica, Cholesky ok em todo t
}
```

- [ ] **Step 2: Rodar e ver falhar** — `ctest --preset gcc-release -R "Dcc(Likelihood|Recursion)"` → não compila.
- [ ] **Step 3: Implementar.** Rₜ⁻¹ e log|Rₜ| saem do Cholesky existente, uma vez por t.
- [ ] **Step 4: Rodar e ver passar** — suíte release inteira PASS.
- [ ] **Step 5: Commit** — `git commit -m "feat: add the DCC, aDCC and FDCC likelihoods"`

---

### Task 3: Parâmetros e estimação de uma spec

**Files:**
- Create: `src/dcc/parameter_map.hpp`, `src/dcc/parameter_map.cpp`, `src/dcc/fit.hpp`, `src/dcc/fit.cpp`
- Test: `tests/unit/dcc/parameter_map_test.cpp`, `tests/unit/dcc/fit_test.cpp`

**Interfaces:**
- Consumes: Tasks 1–2; `optim::Lbfgs` (como `fit.cpp` univariado); `ad::Dual`.
- Produces:
  ```cpp
  namespace ve::dcc {
  struct DccLayout { /* counts, fixed mask and values, started */ };
  DccLayout layout_of(const DccSpec& spec, std::size_t m);
  std::vector<std::string> parameter_names(const DccLayout&);
  std::optional<std::string> spec_problem(const DccSpec& spec, std::size_t m);  // orders 1..6, FDCC (1,1), groups
  // Unconstrained u <-> natural theta enforcing the "Restrições" rule (delta from the panel for aDCC).
  template <class T> void to_natural(std::span<const T> u, std::span<T> theta, const DccLayout&, double delta);
  void to_unconstrained(std::span<const double> theta, std::span<double> u, const DccLayout&, double delta);
  struct DccFitResult {
      FitStatus status; DccSpec spec; std::vector<std::string> names; std::vector<double> estimate;
      std::vector<double> se_hessian, se_robust;      // NaN until standard errors are computed (Task 5)
      double loglik; std::size_t nobs, series, univariate_parameters;
      std::shared_ptr<const std::vector<double>> r;   // T x m x m, shared with copies (Task 5)
      std::shared_ptr<const StandardizedPanel> panel; // sigma for H_t; shared by the specs of a batch
      std::vector<double> univariate_estimates;       // concatenated, the identity checked by Task 5
  };
  DccFitResult fit_dcc(std::shared_ptr<const StandardizedPanel> panel, const DccSpec& spec, const FitConfig& config);
  }
  ```
- A reparametrização segue o padrão de `model/weight_terms.hpp`:
  - os pesos são termos ≥ 0 que somam a persistência P, abaixo de 1 − 1e-8, com a parte g do aDCC escalada por δ;
  - o FDCC usa o raio de (max a, max b) abaixo de 1 − 1e-8;
  - a forma é logística em (4, 50).

  A estimativa minimiza `objective`. O status e a ℓ reportada seguem as regras.

- [ ] **Step 1: Escrever os testes**

```cpp
TEST(DccParameterMap, RoundTripAndConstraints) {
  // u aleatórios: to_natural satisfaz as restrições de cada modelo; to_unconstrained(to_natural(u)) == u (1e-12)
}
TEST(DccParameterMap, NamesAndProblems) {
  // DCC(1,1) mvt: dcca1 dccb1 mshape; aDCC(2,1): dcca1 dcca2 dccb1 dccg1 dccg2; FDCC 2 grupos: fdcca1 fdcca2
  // fdccb1 fdccb2; problemas: FDCC (2,1), groups de tamanho errado, grupo vazio, ordem 7
}
TEST(DccFit, MatchesRmgarchEstimates) {
  // dcc3 + univariados do rugarch: DCC mvnorm; objetivo do VolEngine <= o do rmgarch no ponto dele + 1e-7;
  // a, b dentro de rel 1e-3 de (0.0431630001835819, 0.9142701826310855); loglik reportada rel 1e-7
}
TEST(DccFit, RecoversSimulatedParameters) { /* m = 4, T = 4000 simulado DCC(0.06, 0.9) mvnorm: |erro| < 0.02 */ }
TEST(DccFit, FixedAndStart) { /* mshape fixo 8: names sem mshape; loglik == rmgarch -6132.56175655 (rel 1e-6) */ }
```

- [ ] **Step 2: Rodar e ver falhar** — `ctest --preset gcc-release -R "DccParameterMap|DccFit"` → não compila.
- [ ] **Step 3: Implementar.**
- [ ] **Step 4: Rodar e ver passar** — suíte release inteira PASS.
- [ ] **Step 5: Commit** — `git commit -m "feat: estimate DCC models by maximum likelihood"`

---

### Task 4: Lote, critérios, Rₜ e Hₜ

**Files:**
- Modify: `src/dcc/fit.hpp`, `src/dcc/fit.cpp`
- Test: `tests/unit/dcc/batch_test.cpp`

**Interfaces:**
- Consumes: Task 3; `Context::with_pool`.
- Produces:
  ```cpp
  namespace ve::dcc {
  // One panel, many specs: the panel is built once; the specs are split among the threads.
  std::vector<DccFitResult> fit_dcc_batch(Context& context, std::span<const FitResult* const> fits,
                                          std::span<const DccSpec> specs, const DccOptions& options);
  struct DccOptions { FitConfig config{}; bool standard_errors = true; };  // standard_errors: Task 5
  double dcc_criterion(const DccFitResult& fit, Criterion which, bool per_obs);  // regra "Critérios"
  // H_t = D_t R_t D_t, computed on first call and kept (std::call_once in the C handle).
  std::vector<double> covariance_path(const DccFitResult& fit);
  }
  ```

- [ ] **Step 1: Escrever os testes**

```cpp
TEST(DccBatch, CriteriaMatchRmgarch) {
  // DCC mvnorm em dcc3: k = 12 + 2 + 3 = 17; AIC 8.13325450364, BIC 8.19347100136, SIC 8.13300143181,
  // HQIC 8.15568733467 no ponto do rmgarch (rel 1e-10); FDCC 2 grupos: k = 19
}
TEST(DccBatch, SameAsAloneForAnyThreadCount) {
  // Review Focus 2: grade 3 modelos x 3 dists (FDCC 2 grupos) + aDCC(2,1) num par quase descorrelacionado;
  // 1, 3 e 16 threads: estimate, loglik, r bit a bit iguais às specs ajustadas sozinhas; status por spec
}
TEST(DccBatch, CovarianceFromSigma) { /* H_t == D_t R_t D_t (rel 1e-15); r compartilhado pelo panel; fits liberados */ }
```

- [ ] **Step 2: Rodar e ver falhar.**
- [ ] **Step 3: Implementar.**
- [ ] **Step 4: Rodar e ver passar** — suíte release inteira PASS.
- [ ] **Step 5: Commit** — `git commit -m "feat: fit DCC specs in batches with criteria and covariances"`

---

### Task 5: Erros-padrão em dois estágios

**Files:**
- Create: `src/dcc/standard_errors.hpp`, `src/dcc/standard_errors.cpp`
- Modify: `src/dcc/fit.cpp` (opção ligada), `src/fit/fit.hpp`, `src/fit/fit.cpp` (expor a reavaliação univariada; sem mudar números)
- Test: `tests/unit/dcc/standard_errors_test.cpp`

**Interfaces:**
- Consumes: Tasks 1–4. Para cada série, a ℓ, os scores e a Hessiana univariados em θᵢ. O problema é reconstruído do `FitResult`: r = fitted + residuals, spec, config e estimativa.
- Produces:
  ```cpp
  namespace ve::fit {
  // The univariate problem of a fit without regressors, rebuilt from its result; nullopt with mxreg/vxreg.
  std::optional<LikelihoodProblem> problem_of(const FitResult& fit, std::vector<double>& storage);
  }
  namespace ve::dcc {
  // Regra "Erros-padrão em dois estágios": se_hessian and se_robust of the DCC rows.
  struct DccStandardErrors { std::vector<double> hessian, robust; };
  std::expected<DccStandardErrors, std::string> dcc_standard_errors(const DccFitResult& fit,
                                                                      std::span<const FitResult* const> fits);
  // The fits must be those of the fit: same m, T and univariate estimates (bit for bit); else the message.
  std::optional<std::string> fits_problem(const DccFitResult& fit, std::span<const FitResult* const> fits);
  }
  ```
- Só as linhas estimadas entram; as fixas ficam NaN. Univariados com regressores dão a mensagem que vira `VE_UNSUPPORTED` na Task 6.

- [ ] **Step 1: Escrever os testes**

```cpp
TEST(DccStandardErrors, MatchRmgarchSandwich) {
  // dcc3, univariados estimados no ponto do rugarch (start + tol 1, como no oráculo do 3c), DCC no ponto do rmgarch:
  // robust: se(dcca1) 0.00627467301294802, se(dccb1) 0.0132675580840498 (rel 2e-3: Hessianas univariadas
  // analíticas vs numéricas do rugarch); hessian: 0.00684873, 0.01556525 (rel 2e-3)
}
TEST(DccStandardErrors, OtherVariants) { /* mvt, aDCC, FDCC 2 grupos, mvlaplace: robust das "Other variants" (rel 5e-3) */ }
TEST(DccStandardErrors, RejectsOtherFits) {
  // Review Focus 4: outro conjunto, ordem trocada, um fit reajustado com outra spec: mensagem; com vxreg: "regressors"
}
TEST(DccStandardErrors, OptionMatchesExplicit) { /* opção ligada == função explícita (bit a bit); desligada: NaN */ }
```

- [ ] **Step 2: Rodar e ver falhar.**
- [ ] **Step 3: Implementar.** A Hessiana univariada sai pelo mesmo caminho que `standard_errors.cpp` usa. Os univariados que não mudam numa perturbação não são refiltrados.
- [ ] **Step 4: Rodar e ver passar** — suíte release inteira PASS, sem mudar nenhum teste univariado.
- [ ] **Step 5: Commit** — `git commit -m "feat: add two-stage standard errors to DCC fits"`

---

### Task 6: Interface C e Python

**Files:**
- Create: `src/capi/dcc/dcc_spec_api.cpp`, `src/capi/dcc/dcc_fit_api.cpp`, `python/src/volengine/_dcc.py`
- Modify: `include/volengine/volengine.h`, `src/capi/handles.hpp`, `python/src/volengine/api.py`, `_native.py`, `__init__.py`, `CMakeLists.txt`
- Test: `tests/unit/capi/dcc_test.cpp`, `python/tests/test_dcc.py`

**Interfaces:**
- Produces (nomes finais podem seguir o estilo vizinho; a forma é esta):
  ```c
  typedef struct ve_dcc_spec ve_dcc_spec; typedef struct ve_dcc_options ve_dcc_options; typedef struct ve_dcc_fit ve_dcc_fit;
  #define VE_DCC_MODEL_DCC 0 / ADCC 1 / FDCC 2;   #define VE_DCC_DIST_MVNORM 0 / MVT 1 / MVLAPLACE 2
  VE_API ve_status ve_dcc_spec_create(int32_t model, ve_dcc_spec** out);
  VE_API ve_status ve_dcc_spec_set_order(ve_dcc_spec* s, int32_t p, int32_t q);
  VE_API ve_status ve_dcc_spec_set_dist(ve_dcc_spec* s, int32_t dist);
  VE_API ve_status ve_dcc_spec_set_groups(ve_dcc_spec* s, const int32_t* groups, size_t count);
  VE_API ve_status ve_dcc_spec_set_fixed(ve_dcc_spec* s, const char* name, double value);   /* and _set_start */
  VE_API void ve_dcc_spec_free(ve_dcc_spec* s);
  VE_API ve_status ve_dcc_options_create(ve_dcc_options** out);   /* standard errors on by default */
  VE_API ve_status ve_dcc_options_set_standard_errors(ve_dcc_options* o, int32_t on);
  VE_API void ve_dcc_options_free(ve_dcc_options* o);
  typedef struct ve_dcc_batch ve_dcc_batch;
  VE_API ve_status ve_dcc_batch_create(ve_dcc_batch** out);
  VE_API ve_status ve_dcc_batch_set_specs(ve_dcc_batch* b, const ve_dcc_spec* const* specs, size_t n);
  VE_API ve_status ve_dcc_batch_set_fits(ve_dcc_batch* b, const ve_fit* const* fits, size_t m);  /* the m univariate fits */
  VE_API ve_status ve_dcc_batch_set_options(ve_dcc_batch* b, const ve_dcc_options* o);
  VE_API void ve_dcc_batch_free(ve_dcc_batch* b);
  /* out: n_specs handles, one per spec, in spec order. */
  VE_API ve_status ve_dcc_fit_batch(ve_context* ctx, const ve_dcc_batch* b, ve_dcc_fit** out);
  ```
  Os acessores espelham os de `ve_fit`:
  - `ve_dcc_fit_status`, `_nobs`, `_series`, `_param_count`, `_param_name`, `_estimates`, `_std_errors(se_kind)`, `_t_values`, `_p_values`, `_loglik`, `_criterion(which, per_obs)`;
  - `_correlation(d, &data, &len)` e `_covariance(d, &data, &len)`;
  - `_free`;
  - `ve_dcc_fit_standard_errors(ctx, d, fits, m, &out)`, que devolve um `ve_dcc_fit` novo.
- Python:
  ```python
  class DccSpec: model: Literal["dcc","adcc","fdcc"]; order: tuple[int,int]=(1,1); dist: Literal["mvnorm","mvt","mvlaplace"];
                 groups: Sequence[int] | None; fixed: Mapping[str,float]; start: Mapping[str,float]
  class DccFit: status, nobs, series, names; coef(se="hessian"|"robust") -> CoefTable; loglik; aic()/bic()/hqic()/sic();
                R, H: read-only (T, m, m); def standard_errors(self, univariate: Sequence[Fit]) -> DccFit
  Context.fit_dcc(spec: DccSpec | Sequence[DccSpec], univariate: Sequence[Fit], *, standard_errors: bool = True)
      -> DccFit | list[DccFit]
  ```
- Erros: as regras "Entrada e erros"; `VE_UNSUPPORTED` para erros-padrão com regressores.

- [ ] **Step 1: Escrever os testes**

```cpp
TEST(CapiDcc, MatchesCore) { /* lote de 3 specs: acessores == fit_dcc_batch (bit a bit), R e H zero-copy */ }
TEST(CapiDcc, OutlivesInputs) { /* libera fits, specs, batch e contexto: views válidas e iguais */ }
TEST(CapiDcc, InvalidArguments) { /* Review Focus 1 e 4; handles NULL; groups errados; out NULL */ }
TEST(CapiDcc, StandardErrorsLater) { /* opção off → NaN; ve_dcc_fit_standard_errors → igual à opção on */ }
```

```python
def test_fit_dcc_grid_and_bic(ctx, series): ...   # lista de 9 specs → 9 DccFit; bic() finito; R (T, 3, 3) read-only
def test_dcc_standard_errors_later(ctx, series): ...
def test_dcc_bad_inputs_raise(ctx, series): ...    # ValueError / VolEngineError com a mensagem
```

- [ ] **Step 2: Rodar e ver falhar.**
- [ ] **Step 3: Implementar.**
- [ ] **Step 4: Rodar e ver passar** — suíte release inteira, `pytest`, `check_style.py`, `mypy --strict` (limpo).
- [ ] **Step 5: Commit** — `git commit -m "feat: expose DCC fits through the C API and Python"`

---

### Task 7: Oráculo `rmgarch`

**Files:**
- Create: `tests/reference/dcc_rmgarch.R`, `tests/reference/fixtures/rmgarch_dcc.json`, `python/tests/oracles/test_dcc_oracles.py`
- Modify: `tests/reference/README.md`, `docs/guide/validation.md`

**Interfaces:**
- O script R:
  - lê `data/dcc3.csv`, sem regerar, e um segundo conjunto com m = 5, tirado das colunas de `grid.csv`;
  - ajusta os univariados com `multifit`;
  - roda o `dccfit` (solnp) em DCC, aDCC e DCC(2,1) com mvnorm, mvt e mvlaplace, FDCC com 2 grupos e mvnorm, e DCC-mvt com `mshape` fixo;
  - grava para cada caso `{id, univariate coefs, dcc coefs, llh, criteria, se, R at t = 1, T/2, T}`.
- No Python:
  - **no ponto do rmgarch:** univariados por `start` + `tol=1` nos coeficientes do rugarch e DCC com os parâmetros fixos no ponto do rmgarch. ℓ, critérios e R batem a rel 1e-10, e os erros-padrão `robust` a rel 2e-3;
  - **estimado:** objetivo ≤ o do rmgarch + 1e-7, parâmetros a rel 1e-3 e ℓ a rel 1e-7.
- `validation.md` ganha a seção "`rmgarch`, DCC", com:
  - as convenções acima: as duas pré-amostras, Q̄ = cov, a ℓ completa, k com m(m − 1)/2 e o sanduíche sem transposta;
  - as diferenças: o bug do FDCC com G = 1 não reproduzido, a tabela "hessian" que o rmgarch não tem, e o FDCC-mvt/mvlaplace sem oráculo (validado pela identidade de um grupo = DCC);
  - a tabela de diferenças observadas.

- [ ] **Step 1: Gerar a fixture.**
- [ ] **Step 2: Escrever o teste** `test_dcc_matches_rmgarch[<id>]` e rodar → PASS. Uma divergência passa por systematic-debugging, e um xfail exige um parágrafo.
- [ ] **Step 3: Documentar.**
- [ ] **Step 4: Rodar** — `pytest python/tests` PASS.
- [ ] **Step 5: Commit** — `git commit -m "test: validate DCC fits against rmgarch"`

---

### Task 8: Benchmarks e documentação

**Files:**
- Modify: `tests/benchmarks/fit_benchmarks.cpp`, `docs/performance/benchmarks.md`, `README.md`, `CHANGELOG.md`, `ARCHITECTURE.md`

**Interfaces:**
- **Grade do TCC:**
  - `BM_DccGrid/m:21/T:5000/threads:<1|hw>`: 21 séries simuladas, univariados sGARCH(1,1)-std fora da medição, e a grade DCC, aDCC e FDCC (3 grupos) × mvnorm, mvt e mvlaplace sem erros-padrão;
  - `BM_DccFit/<model>_<dist>/m:21`: um ajuste de cada;
  - `BM_DccStandardErrors/dcc_mvnorm/m:21`: os erros-padrão de um ajuste (spec, seção 7).

  Cinco repetições, publicadas em `benchmarks.md`.
- **Documentação:**
  - `README.md`: exemplo da grade do TCC, com `ctx.fit_dcc(specs, fits, standard_errors=False)`, a escolha pelo `bic()`, depois `best.standard_errors(fits).coef()` e `best.R`;
  - `CHANGELOG.md`: entrada "Added";
  - `ARCHITECTURE.md`: `src/dcc/`, `src/capi/dcc/`, `linalg/symmetric_eigen` e a dependência `fit` <- `dcc` <- `capi`.

- [ ] **Step 1: Escrever os benchmarks** e rodar com `--benchmark_filter="BM_Dcc"` → termina.
- [ ] **Step 2: Medir** e publicar.
- [ ] **Step 3: Documentar.**
- [ ] **Step 4: Rodar** — suíte release, `pytest`, `check_style.py` → PASS.
- [ ] **Step 5: Commit** — `git commit -m "bench: add DCC benchmarks and document plan 4a"`
