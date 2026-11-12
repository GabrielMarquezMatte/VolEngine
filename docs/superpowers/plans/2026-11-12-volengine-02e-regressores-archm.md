# VolEngine — Plano 2e: regressores externos (`mxreg`/`vxreg`) e ARCH-in-mean

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Sobre o 2d, aceitar regressores externos na média (`mxreg`) e na variância (`vxreg`) e o ARCH-in-mean (`archm`, com σ ou σ²) em todos os modelos e distribuições, como no `rugarch`. Os regressores entram no contrato de entrada da interface C, como matrizes T×k por job do lote. A validação é contra o `rugarch` em todos os casos e contra o `arch` na média com regressores e sem AR.

**Architecture:** A verossimilhança passa a receber um `model::Data`: a série e duas matrizes de regressores, uma para a média e outra para a variância, alinhadas às linhas da série. A média vira uma recursão em dₜ = rₜ − mₜ. Com `archm`, mₜ depende de σₜ, e média e variância andam juntas a cada t. Para isso cada recursão de variância ganha a forma de passo (`step(t)`), e o laço de série inteira é esse mesmo passo, sem mudar a aritmética. Os regressores de variância somam ao intercepto, na escala de cada modelo. O mapa irrestrito ↔ natural escala os coeficientes novos pelas escalas da série e dos regressores, para que o ajuste não dependa das unidades.

**Tech Stack:** C++23, CMake + presets, GoogleTest v1.17.0, Google Benchmark v1.9.1, BS::thread_pool v5.0.0, Boost.Math `boost-1.87.0`, Python ≥ 3.11 + numpy + pytest, `arch` 8.0.0, R 4.6.2 + `rugarch` 1.5.6.

**Spec:** `docs/superpowers/specs/2026-11-10-volengine-design.md`, seções 3 ("regressores externos na média e na variância (`mxreg`/`vxreg`), ARCH-in-mean (`archm`/`archpow`) … Os regressores fazem parte do contrato de entrada desde o início (matrizes T×k com strides por trabalho do lote)") e 5. Os planos 2a–2d registram as convenções que este plano estende.

## Roteiro

- **2a–2d (feitos):** motor genérico, variantes, distribuições assimétricas, parâmetros fixos, valores iniciais e VT.
- **2e (este):** `mxreg`/`vxreg` e `archm`/`archpow`.
- **2f:** ARFIMA.

Depois: 3 diagnósticos/forecast/simulação; 4 DCC; 5 performance; 6 wrapper C++, wheels, 1.0.

## Global Constraints

Valem todas as Global Constraints dos planos 1, 2a–2d, o `STYLEGUIDE.md` e o `ARCHITECTURE.md`. As que mais pesam aqui:

- A interface C é o único contrato estável. Enums `typedef int32_t` + `#define`, só acréscimos; ≤ 5 parâmetros por função; configuração nova é função nova; nenhuma exceção atravessa a fronteira C.
- Ajustes bit-idênticos para qualquer número de threads na mesma máquina e ISA. **Uma spec sem regressor e sem `archm` dá os mesmos números do 2d:** todos os testes existentes passam sem mudar constantes nem tolerâncias. Isso vale também para o laço de variância reescrito em forma de passo.
- Convenção padrão segue o `rugarch`. O código dele (GPL) pode ser lido para entender convenções, nunca copiado ou traduzido. **Uma dúvida de convenção se resolve lendo a fonte**: papers, a vinheta e os manuais do `rugarch`, a documentação e o código do `arch` (NCSA). A referência vai para `validation.md`.
- Anti-objetivo: não fazer mais do que o necessário. Ficam fora o `archex` do `rugarch` (regressores da média multiplicados por σₜ), os regressores futuros para forecast (o plano 3 decide) e a omissão da restrição de estacionariedade quando há `vxreg` (o `rugarch` a desliga; aqui P < 1 vale sempre, ver "Variância").
- Estilo: nunca `else`; funções ≤ 60 linhas; ≤ 5 parâmetros; aninhamento ≤ 3; ≤ 8 módulos por pasta; comentários só para restrição externa ou invariante; medições em `docs/`. `-Wnrvo` como erro no GCC.
- Commits sem trailer `Co-Authored-By`; LF. Python no venv `python/.venv`; R em `C:/Program Files/R/R-4.6.2/bin/Rscript.exe`.

### Novas regras deste plano

**Convenções do `rugarch`.** Foram confirmadas antes do plano, com `ugarchfilter` (parâmetros fixos) contra uma implementação independente em R, com diferença de ℓ ≤ 3e-12 e de σ ≤ 2e-16 em sGARCH com ARMA(0,0), (1,0), (0,1) e (1,1), archpow 1 e 2, `mxreg` e `vxreg`. A vinheta do `rugarch` (seções 2.1 e 2.2, eqs. 1, 2, 9 e 10) descreve o mesmo modelo. A Task 8 refaz a confirmação com ajustes.

**Média.** Com mₜ = μ + Σₖ δₖ xₜ,ₖ + ξ σₜᵖ (o termo em ξ só com `archm`; p = `archpow` ∈ {1, 2}) e dₜ = rₜ − mₜ:

- eₜ = dₜ para t < m (m = ordem AR, como hoje);
- eₜ = dₜ − Σᵢ φᵢ dₜ₋ᵢ − Σⱼ θⱼ eₜ₋ⱼ a partir de t = m. O AR atua sobre rₜ − mₜ, com os regressores e o termo archm dentro;
- sem regressor e sem `archm`, dₜ = rₜ − μ, e a recursão é a de hoje, bit a bit.

**Resíduos de pré-amostra.** e⁰ são os resíduos da média com ξ = 0, isto é, a mesma recursão sem o termo archm. Eles alimentam a média amostral da pré-amostra (`VE_VARINIT_SAMPLE`) e o alvo do VT. Sem `archm`, e⁰ = e. É o que o `ugarchfilter` faz. No `ugarchfit`, o `rugarch` só faz isso quando o `archm` recebe valor inicial; sem valor inicial, ele roda antes um ajuste auxiliar sem `archm` e usa o σ desse ajuste. A Task 8 passa `start.pars = list(archm = 0)` e registra isso em `validation.md`.

**Variância.** Com cₜ = Σₖ ζₖ vₜ,ₖ (o vₜ do próprio t; "passed pre-lagged" na vinheta), o intercepto ω vira ω + cₜ na escala de cada modelo: σ²ₜ (sGARCH, iGARCH, gjrGARCH), log σ²ₜ (eGARCH), σₜ^δ (apARCH, fGARCH). No csGARCH, cₜ entra em qₜ = ω + cₜ + ρqₜ₋₁ + φ(ε²ₜ₋₁ − σ²ₜ₋₁), e a pré-amostra de q segue em ω/(1 − ρ), sem cₜ. Os valores de pré-amostra não leem cₜ. Um σ² não positivo leva o `loglik` a −inf, como hoje. A persistência continua limitada a P < 1 − 1e-8 pelo mapa. O `rugarch` desliga essa restrição quando há `vxreg`; aqui não, e `validation.md` registra a diferença.

**VT com `vxreg`:** ω = alvo − Σₖ ζₖ v̄ₖ, com v̄ₖ a média da coluna k nas linhas da série e o alvo do 2d (m₂(1 − P), log(m₂)(1 − Σβ) ou m_δ(1 − P)). Confirmado com regressor constante nos cinco modelos com VT.

**Backcast (`VE_VARINIT_BACKCAST`) com `mxreg`:** o `arch` calcula o backcast nos resíduos do valor inicial da média, que é OLS (`pinv(x)·y`, `arch/univariate/base.py` e `mean.py`). Aqui, o backcast usa os resíduos do OLS de r em [1 (se há média), X]. Sem `mxreg`, isso reduz a r − r̄ (ou r), a regra do 2a. O ARMA continua fora do OLS, como no 2a.

**Parâmetros e nomes** (ordem do `rugarch`): `mu`, `ar1..`, `ma1..`, `archm`, `mxreg1..mxregK`, `omega`, `alpha..`, `beta..`, `gamma..`, `eta1..`, `eta2..`, `delta`/`lambda`, `vxreg1..vxregJ`, `skew`, `shape`, `ghlambda`. `#define VE_MAX_REGRESSORS 8`, então K, J ≤ 8. `kMaxParameterCount = 6·kMaxOrder + 6 + 1 + 2·kMaxRegressors = 59`. O `Dual` mais largo é `Dual<kMaxParameterCount>`, e os buckets ficam 8, 16, 32, 59.

**Escalas e mapa** (u irrestrito → natural; s = desvio padrão da série; s_x,k e s_v,k = desvio padrão populacional da coluna k nas linhas da série, ou 1 quando esse desvio é 0):

| Parâmetro | Mapa | `ParameterKind` | escala natural (passo da Hessiana) |
|---|---|---|---|
| `archm`, p = 1 | ξ = u | `Location` | 1 |
| `archm`, p = 2 | ξ = u/s | `Location` | 1/s |
| `mxreg_k` | δ = u·s/s_x,k | `Location` | s/s_x,k |
| `vxreg_k` | ζ = u·S_ω/s_v,k, com S_ω = s² (sGARCH, iGARCH, gjrGARCH, csGARCH), 1 (eGARCH) e s^δ (apARCH, fGARCH, com δ o natural corrente) | `Location` | S_ω/s_v,k |
| `mu` | sem mudança (r̄ + s·m) | `Location` | s |

O passo da Hessiana dos parâmetros `Location` passa a ser `kRelativeStep·max(|θ|, natural_scale)`, onde `natural_scale` é a coluna da direita. Para `mu` isso é o passo de hoje.

**Valores iniciais.** Com `mxreg`, `mu` e δ vêm do OLS de r em [1 (se há média), X], o mesmo do backcast. `archm` começa em 0 e `vxreg` também. O ω padrão do 2d segue sem mudança. Uma coluna de X colinear com outra, ou com a constante, faz o OLS cair no mínimo de norma (pseudo-inversa), e o início continua finito.

## Review Focus

1. **Coluna de regressor constante ou colinear com `mu`** (uma coluna de uns, ou duas colunas iguais): s_x = 1, o OLS de mínima norma dá início finito, o ajuste termina finito e, se a Hessiana for singular, os erros-padrão saem NaN sem exceção. → testes nas Tasks 2, 5 e 6.
2. **`vxreg` que leva σ² a ≤ 0 em algum t** (ζ negativo grande): `loglik` = −inf, a busca recua, o ajuste termina finito, e nunca sai +inf nem NaN. → testes nas Tasks 3 e 6.
3. **Regressores mal alinhados no lote**: linhas diferentes de T, colunas diferentes do k da spec, NaN numa linha usada, painel de regressor com ranges próprios, regressor dado a uma spec com k = 0 ou ausente com k > 0. Todos dão `VE_INVALID_ARGUMENT`, a mensagem cita o job, e nenhum handle é criado. → teste na Task 7.
4. **Mudança de unidades** (r × 100 e X × 1000): os ajustes são equivariantes, com δ escalado por 100/1000, ξ igual (p = 1) ou dividido por 100 (p = 2), ζ coerente com S_ω e ℓ deslocado por T·log 100. → teste na Task 6.
5. **Painel irregular com regressores e 4 threads**: o job j usa as linhas startⱼ..startⱼ + lenⱼ − 1 dos seus regressores; o resultado é bit-idêntico ao `fit_one` nas fatias. → teste na Task 7.

---

## Mapa de arquivos

```
src/core/spec.hpp                  Spec::mxreg, Spec::vxreg, Spec::archm; kMaxRegressors; kMaxParameterCount = 59
src/linalg/cholesky.*              movido de src/fit/ (camada abaixo de model)
src/linalg/least_squares.*         OLS de mínima norma (k ≤ 9 colunas) por equações normais com pivô
src/model/data.hpp                 model::Data, RegressorMatrix
src/model/layout.*                 blocos archm, mxreg, vxreg; nomes; kinds
src/mean/arma.hpp                  recursão em d_t; passo por t
src/garch/*.hpp                    cada recursão com passo por t e intercepto variável opcional
src/model/likelihood.*             loglik(Data, ...) e laço conjunto com archm; pré-amostra em e⁰; VT com v̄; backcast por OLS
src/model/parameter_map.*          mapas de archm, mxreg, vxreg; Standardization com escalas dos regressores; natural_scale
src/model/start_values.*           início por OLS
src/fit/*                          Data no fit_one e no lote; passo da Hessiana por natural_scale; Dual<59>
src/capi/*, include/volengine/volengine.h   set_mxreg, set_vxreg, set_archm, ve_batch_set_regressors
python/src/volengine/*             Spec(mxreg=, vxreg=, archm=); Context.fit(..., mxreg=, vxreg=)
tests/...                          espelhando src/; dados, casos e fixtures novos em tests/reference/
```

---

### Task 1: Spec, Layout e larguras

**Files:**
- Modify: `src/core/spec.hpp`, `src/model/layout.hpp`, `src/model/layout.cpp`, `src/fit/fit.cpp` (buckets), as instâncias `Dual<42>` → `Dual<kMaxParameterCount>` em `src/model/*.cpp` e nos comentários "Instantiated only for"
- Test: `tests/unit/model/layout_test.cpp`

**Interfaces:**
- Produces:
  ```cpp
  namespace ve {
  constexpr int kMaxRegressors = 8;
  constexpr std::size_t kMaxParameterCount = 6 * kMaxOrder + 6 + 1 + 2 * kMaxRegressors;   // 59
  struct Spec { /* campos do 2d, e: */ int mxreg = 0; int vxreg = 0; int archm = 0; };       // archm: 0, 1 (σ) ou 2 (σ²)
  }
  namespace ve::model {
  struct Layout { /* campos do 2d, e: */ int archm = 0; int mxreg = 0; int vxreg = 0;
      bool has_archm() const; std::size_t archm_index() const;
      std::size_t mxreg_begin() const; std::size_t vxreg_begin() const; };
  }
  ```
- Ordem: a da regra "Parâmetros e nomes". `omega()` passa a vir depois de `mxreg`, e `skew()` depois de `vxreg`. `kind` dá `Location` para `archm`, `mxreg` e `vxreg`. `spec_problem` rejeita `mxreg`/`vxreg` fora de 0..8 e `archm` fora de {0, 1, 2}; o `ve_spec_set_*` da Task 7 também rejeita, mas a spec pode chegar ao núcleo pelo C++.

- [ ] **Step 1: Escrever os testes**

```cpp
TEST(Layout, RegressorBlocksFollowRugarchOrder) {
  // sGARCH(1,1), ARMA(1,1), archm 1, mxreg 2, vxreg 1, ghyp:
  // mu ar1 ma1 archm mxreg1 mxreg2 omega alpha1 beta1 vxreg1 skew shape ghlambda
  // apARCH: ... beta1 gamma1 delta vxreg1 ...; csGARCH: ... eta11 eta21 vxreg1 ...
}
TEST(Layout, WidestFitsWidestDual) { /* ALLGARCH, ARMA(6,6), (6,6), archm, mxreg 8, vxreg 8, ghyp: count() == 59 == kMaxParameterCount */ }
TEST(Layout, RegressorKinds) { /* archm, mxreg_k, vxreg_k: Location */ }
TEST(Layout, RegressorSpecProblems) { /* mxreg 9, vxreg -1, archm 3 → mensagens com "mxreg", "vxreg", "archm" */ }
TEST(Layout, PlainSpecUnchanged) { /* sem os blocos: índices e count() do 2d em todo layout */ }
```

- [ ] **Step 2: Rodar e ver falhar** — `cmake --build --preset gcc-debug && ctest --preset gcc-debug -R Layout` → não compila.
- [ ] **Step 3: Implementar.**
- [ ] **Step 4: Rodar e ver passar** — suíte inteira PASS, sem mudar constantes.
- [ ] **Step 5: Commit** — `git commit -m "feat: add the archm, mxreg and vxreg blocks to the layout and widen the duals to 59"`

---

### Task 2: Dados, média com regressores e backcast por OLS

**Files:**
- Move: `src/fit/cholesky.{hpp,cpp}` → `src/linalg/cholesky.{hpp,cpp}`, com o namespace `ve::linalg`. Atualize os includes em `src/fit/` e o CMake.
- Create: `src/linalg/least_squares.hpp`, `src/linalg/least_squares.cpp`, `src/model/data.hpp`
- Modify: `src/mean/arma.hpp`, `src/model/likelihood.hpp`, `src/model/likelihood.cpp`, `ARCHITECTURE.md` (a camada `linalg` entre `core` e `model`)
- Test: `tests/unit/linalg/least_squares_test.cpp`, `tests/unit/mean/arma_test.cpp`, `tests/unit/model/likelihood_test.cpp`

**Interfaces:**
- Produces:
  ```cpp
  namespace ve::model {
  struct RegressorMatrix {                          // T × columns, row-major, as linhas da série
      std::span<const double> values; std::size_t columns = 0;
      double at(std::size_t t, std::size_t k) const { return values[t * columns + k]; }
  };
  struct Data { std::span<const double> r; RegressorMatrix mean{}; RegressorMatrix variance{}; };
  template <class T> T loglik(const Data& data, std::span<const T> theta, const Layout&, VarInit, const LoglikOutputs<T>&);
  // a sobrecarga de hoje, loglik(std::span<const double> r, ...), continua e chama a nova com Data{r}
  }
  namespace ve::linalg {
  // Solução de norma mínima de min ||y − A b||, A com rows × columns (row-major), columns ≤ 9.
  std::vector<double> least_squares(std::span<const double> a, std::size_t columns, std::span<const double> y);
  }
  namespace ve::mean {
  template <class T> struct ArmaParameters { T mu; std::span<const T> ar; std::span<const T> ma; };   // sem mudança
  // eₜ da regra "Média" sobre d (d tem o tamanho de residuals)
  template <class T> void arma_residuals_of_deviations(std::span<const T> d, const ArmaParameters<T>& p, std::span<T> residuals);
  }
  ```
- Regras "Média" (ainda sem archm), "Resíduos de pré-amostra" e "Backcast".
- `least_squares` usa equações normais com Cholesky (`linalg::cholesky`). Se a matriz AᵀA não for positiva definida, cai num pseudo-inverso: decomposição espectral de Jacobi de AᵀA, com autovalores abaixo de 1e-12·max cortados.

- [ ] **Step 1: Escrever os testes**

```cpp
TEST(LeastSquares, MatchesClosedForm) { /* y = 1 + 2x + ruído fixo, A = [1, x]: b == (XᵀX)⁻¹Xᵀy (1e-12) */ }
TEST(LeastSquares, MinimumNormWhenCollinear) {
  // Review Focus 1: A = [1, 1, x] (duas colunas de uns): b finito, b₀ == b₁ (1e-12), A b ajusta como no caso não colinear (1e-10)
}
TEST(Arma, DeviationsReduceToToday) { /* d = r − μ: arma_residuals_of_deviations == arma_residuals bit a bit em ARMA(0..2, 0..2) */ }
TEST(Likelihood, MeanRegressorsHandCase) {
  // r e X de 6 pontos, ARMA(1,0), mxreg 2, sGARCH(1,1)-norm, sample init: e₀ = d₀, eₜ = dₜ − φ dₜ₋₁,
  // dₜ = rₜ − μ − δ·xₜ; σ² e ℓ pela recursão à mão (rel 1e-14)
}
TEST(Likelihood, BackcastFromOls) { /* backcast com mxreg == EWMA do 2a nos resíduos do OLS de r em [1, X] (1e-15) */ }
// GradientMatchesFiniteDifferences, DoubleAndDualAgreeOnValue e TermsSumToTotalAndScoresSumToGradient ganham
// mxreg ∈ {0, 2} em SGarch, EGarch, ApArch, CsGarch × ARMA {(0,0),(1,1)} × {sample, backcast} × {norm, std}.
```

- [ ] **Step 2: Rodar e ver falhar** — `ctest --preset gcc-debug -R "LeastSquares|Arma|Likelihood"` → não compila.
- [ ] **Step 3: Implementar.**
- [ ] **Step 4: Rodar e ver passar** — suíte inteira PASS; `Likelihood.MatchesPlanOneSimulatedValues` sem mudar constantes.
- [ ] **Step 5: Commit** — `git commit -m "feat: take mean regressors in the likelihood and the backcast from their least squares"`

---

### Task 3: Regressores na variância

**Files:**
- Modify: `src/garch/sgarch.hpp`, `egarch.hpp`, `gjrgarch.hpp`, `power_garch.hpp`, `csgarch.hpp`, `src/model/likelihood.cpp`
- Test: `tests/unit/garch/*_test.cpp`, `tests/unit/model/likelihood_test.cpp`

**Interfaces:**
- Consumes: `Data::variance` (Task 2); `Layout::vxreg_begin` (Task 1); `targeted_omega` do 2d.
- Cada struct de parâmetros (`SGarchParameters`, `EGarchParameters`, `GjrGarchParameters`, `PowerGarchParameters`, `CsGarchParameters`) ganha `std::span<const T> intercept_shift{}`, com o tamanho da série (cₜ) ou vazio. Vazio é a soma de hoje, sem o termo; não pode haver `+ 0` no caminho sem regressor.
- Regras "Variância" e "VT com `vxreg`". `targeted_omega` subtrai Σζₖ v̄ₖ quando há `vxreg`.

- [ ] **Step 1: Escrever os testes**

```cpp
TEST(VarianceRegressors, ConstantRegressorShiftsOmega) {
  // v ≡ 0.7, ζ = 0.03: σ² com vxreg == σ² sem vxreg e ω + 0.021, bit a bit em sGARCH, iGARCH, eGARCH, gjrGARCH, apARCH e fGARCH-ALLGARCH;
  // no csGARCH, a recursão com cₜ em qₜ e pré-amostra ω/(1 − ρ) (à mão, 1e-14), diferente do ω deslocado
}
TEST(VarianceRegressors, TimeVaryingHandCase) { /* sGARCH(1,1), vₜ = |cos(t/25)|, 5 pontos à mão (1e-14) */ }
TEST(VarianceRegressors, TargetingSubtractsRegressorMeans) { /* VT sGARCH e eGARCH: ω == alvo − ζ v̄ (1e-15) */ }
TEST(VarianceRegressors, NonPositiveVarianceIsInvalid) { /* Review Focus 2: ζ = −10 com v ≥ 0.5: loglik == −inf, sem exceção */ }
// os três testes da grade ganham vxreg ∈ {0, 1} nos mesmos modelos da Task 2, com e sem VT
```

- [ ] **Step 2: Rodar e ver falhar** — `ctest --preset gcc-debug -R "VarianceRegressors|Likelihood"` → FAIL.
- [ ] **Step 3: Implementar.**
- [ ] **Step 4: Rodar e ver passar** — suíte inteira PASS, sem mudar constantes.
- [ ] **Step 5: Commit** — `git commit -m "feat: shift the variance intercept by the variance regressors"`

---

### Task 4: ARCH-in-mean

**Files:**
- Modify: `src/garch/*.hpp` (passo por t), `src/mean/arma.hpp` (passo por t), `src/model/likelihood.cpp`
- Test: `tests/unit/garch/*_test.cpp`, `tests/unit/model/likelihood_test.cpp`

**Interfaces:**
- Consumes: Tasks 1–3.
- Cada recursão de variância ganha um passo `T <model>_step(..., std::size_t t)` que devolve σ²ₜ (no csGARCH também qₜ) a partir de valores anteriores a t. A função de série inteira de hoje vira o laço desse passo, com a mesma aritmética. A média ganha `T arma_step(std::span<const T> d, std::span<const T> residuals, const ArmaParameters<T>&, std::size_t t)`.
- Com `layout.has_archm()`, o `loglik` calcula primeiro e⁰ (regra "Resíduos de pré-amostra"), depois a pré-amostra e o alvo do VT a partir de e⁰. Em seguida, para cada t: σ²ₜ pelo passo, mₜ com ξ√σ²ₜ (p = 1) ou ξσ²ₜ (p = 2), dₜ e eₜ pelo passo da média. Sem `archm`, o caminho de hoje.

- [ ] **Step 1: Escrever os testes**

```cpp
TEST(Garch, StepLoopsEqualSeriesFunctions) { /* para cada modelo: o laço do passo == a função de série inteira de hoje, bit a bit */ }
TEST(Likelihood, ArchInMeanHandCase) {
  // r de 6 pontos, ARMA(1,1), mxreg 1, sGARCH(1,1)-norm, archm p ∈ {1, 2}, sample init: pré-amostra = média de e⁰²,
  // σ², m, d, e e ℓ à mão pela regra "Média" (rel 1e-14)
}
TEST(Likelihood, ArchInMeanZeroIsPlain) { /* ξ = 0: loglik == loglik da spec sem archm, bit a bit */ }
// os três testes da grade ganham archm ∈ {0, 1, 2} em SGarch, EGarch, GjrGarch, ApArch, CsGarch × ARMA {(0,0),(1,1)} × {sample, backcast}
```

- [ ] **Step 2: Rodar e ver falhar** — `ctest --preset gcc-debug -R "Garch|Likelihood"` → FAIL.
- [ ] **Step 3: Implementar.**
- [ ] **Step 4: Rodar e ver passar** — suíte inteira PASS; nos `BM_Loglik` de hoje (preset `gcc-bench`, rodada curta), registre antes e depois no relatório da task. O passo não pode piorar o caminho sem archm de forma visível; se piorar, pare e reporte.
- [ ] **Step 5: Commit** — `git commit -m "feat: add ARCH-in-mean with the mean and variance stepped together"`

---

### Task 5: Mapa, escalas e valores iniciais

**Files:**
- Modify: `src/model/parameter_map.hpp`, `src/model/parameter_map.cpp`, `src/model/start_values.hpp`, `src/model/start_values.cpp`
- Test: `tests/unit/model/parameter_map_test.cpp`, `tests/unit/model/start_values_test.cpp`

**Interfaces:**
- Consumes: Tasks 1–2 (`Data`, `least_squares`).
- Produces:
  ```cpp
  namespace ve::model {
  struct Standardization { double location = 0.0; double scale = 1.0;
      std::array<double, kMaxRegressors> mean_regressor_scale{};       // s_x,k; 1 onde não há coluna
      std::array<double, kMaxRegressors> variance_regressor_scale{}; };  // s_v,k
  Standardization standardization_of(const Data& data);   // a de hoje, standardization_of(r), continua para Data{r}
  // A escala natural da tabela "Escalas e mapa" para um parâmetro Location; s para mu.
  double natural_scale(const Layout&, const Standardization&, std::span<const double> theta, std::size_t index);
  std::vector<double> start_values(const Data& data, const Layout& layout);   // a de r continua
  }
  ```
- Regras "Escalas e mapa" e "Valores iniciais". O u do `vxreg` lê o δ natural corrente, como o ω do 2a. Os fixos do 2d continuam valendo, e um `mxreg` ou `vxreg` fixo não é sobrescrito pelo OLS.

- [ ] **Step 1: Escrever os testes**

```cpp
TEST(ParameterMap, PlainSpecsUnchanged) { /* testes do 2a–2d sem mudança */ }
TEST(ParameterMap, RegressorRoundTrip) { /* 14 modelos × archm {0,1,2} × mxreg {0,2} × vxreg {0,1}: θ → u → θ' (rel 1e-12) */ }
TEST(ParameterMap, RegressorScalesMakeMapUnitFree) {
  // Review Focus 4: r·100 e X·1000: u iguais (1e-12) para θ escalados por 100/1000 (δ), 1/100 (ξ, p = 2), S_ω/s_v (ζ)
}
TEST(ParameterMap, ConstantColumnScaleIsOne) { /* Review Focus 1: coluna constante → mean_regressor_scale == 1 */ }
TEST(StartValues, OlsStart) {
  // mxreg 2: mu e δ == least_squares([1, X], r) (1e-14); archm == 0, vxreg == 0;
  // Review Focus 1: [1, 1, x] colinear → início finito e is_valid
  // fixed {mxreg1: 0.5}: mxreg1 == 0.5 e os outros do OLS com mxreg1 fixo (1e-12)
}
```

- [ ] **Step 2: Rodar e ver falhar** — `ctest --preset gcc-debug -R "ParameterMap|StartValues"` → FAIL.
- [ ] **Step 3: Implementar.** Com mxreg1 fixo, o OLS dos outros é feito em r − δ₁x₁.
- [ ] **Step 4: Rodar e ver passar** — suíte inteira PASS.
- [ ] **Step 5: Commit** — `git commit -m "feat: map and start the regressor and archm coefficients in the units of the data"`

---

### Task 6: Ajuste e erros-padrão

**Files:**
- Modify: `src/fit/fit.hpp`, `src/fit/fit.cpp`, `src/fit/standard_errors.hpp`, `src/fit/standard_errors.cpp`
- Modify: `tests/unit/simulated_garch.hpp`. Acrescente séries com regressores e archm, simuladas de forma independente da biblioteca, com os processos da tabela da Task 8 e X, V gerados como lá.
- Test: `tests/unit/fit/fit_test.cpp`, `tests/unit/fit/standard_errors_test.cpp`

**Interfaces:**
- Consumes: Tasks 1–5.
- Produces: `FitResult fit_one(const model::Data& data, const Spec&, const FitConfig&)`; a sobrecarga de r continua. `fit::LikelihoodProblem` passa a guardar `model::Data`. `standard_errors(gradient, theta, scores, layout, const model::Standardization&)` troca o `double series_scale`, e o passo dos `Location` usa `natural_scale`. `has_insufficient_data` não muda.

- [ ] **Step 1: Escrever os testes**

```cpp
// T = 3000, séries da Task 8 pelo helper
TEST(FitRegressors, BeatsTrueParameters) {
  // as quatro séries com o próprio modelo: Converged e ℓ(θ̂) ≥ ℓ(θ verdadeiro) − 1e-6|ℓ|; δ ±0.05, ζ ±0.02, ξ ±0.05
}
TEST(FitRegressors, UnitEquivariance) {
  // Review Focus 4: arx_sgarch11_norm com r·100 e X·1000: Converged, δ̂' == δ̂·100/1000 (rel 1e-4), ℓ̂' == ℓ̂ − T log 100 (rel 1e-8)
  // archm2_egarch11_std com r·100: ξ̂' == ξ̂/100 (rel 1e-4)
}
TEST(FitRegressors, CollinearColumnFinite) { /* Review Focus 1: X = [x, x]: status finito, sem exceção; SE NaN ou finitos */ }
TEST(FitRegressors, NegativeVarianceRegion) { /* Review Focus 2: sgarch11_vx_std com start {vxreg1: −5}: termina finito, sem exceção */ }
TEST(StandardErrors, RegressorStepsFollowNaturalScale) { /* SE de δ com X·1000 == SE·1/1000 (rel 1e-4) */ }
```

- [ ] **Step 2: Rodar e ver falhar** — `ctest --preset gcc-debug -R "FitRegressors|StandardErrors"` → FAIL.
- [ ] **Step 3: Implementar** os helpers e o driver. Se uma tolerância de recuperação não fechar, registre o valor observado no relatório antes de mexer.
- [ ] **Step 4: Rodar e ver passar** — suíte C++ inteira PASS, `-Wnrvo` sem aviso.
- [ ] **Step 5: Commit** — `git commit -m "feat: fit with mean and variance regressors and ARCH-in-mean"`

---

### Task 7: Interface C, lote e Python

**Files:**
- Modify: `include/volengine/volengine.h`, `src/capi/spec_api.cpp`, `src/capi/batch_api.cpp`, `src/capi/handles.hpp`, `src/fit/batch.hpp`, `src/fit/batch.cpp`, `python/src/volengine/_spec.py`, `python/src/volengine/_native.py`, `python/src/volengine/api.py`
- Test: `tests/unit/capi/spec_test.cpp`, `tests/unit/capi/batch_test.cpp`, `python/tests/test_fit.py`

**Interfaces:**
- Produces (C, só acréscimos):
  ```c
  #define VE_MAX_REGRESSORS 8
  VE_API ve_status ve_spec_set_mxreg(ve_spec* s, int32_t count);     /* 0 .. VE_MAX_REGRESSORS */
  VE_API ve_status ve_spec_set_vxreg(ve_spec* s, int32_t count);     /* 0 .. VE_MAX_REGRESSORS */
  VE_API ve_status ve_spec_set_archm(ve_spec* s, int32_t power);     /* 0 sem, 1 sigma, 2 variância */
  /* mxreg e vxreg têm n_jobs entradas, alinhadas com os jobs; NULL no array inteiro = nenhum job tem; uma entrada
     NULL = esse job não tem. Painéis emprestados, T linhas (as do painel de dados) × k colunas, com strides. */
  VE_API ve_status ve_batch_set_regressors(ve_batch* b, const ve_panel* const* mxreg, const ve_panel* const* vxreg, size_t n_jobs);
  ```
  O `check_batch` valida, por job, na ordem: n_jobs igual ao número de jobs; regressor presente se e só se o k da spec > 0; colunas == k; linhas == T do painel de dados; painel de regressor sem ranges; nenhum NaN ou inf nas linhas que a série usa. A mensagem é "job <i>: ...". O `run_job` copia as linhas da série (startⱼ..startⱼ + lenⱼ − 1) de cada regressor para matrizes row-major e chama `fit_one(Data, ...)`. O header documenta as regras "Média", "Variância", "VT com `vxreg`", "Backcast" e a diferença de estacionariedade.
- Python: `Spec(..., mxreg: int = 0, vxreg: int = 0, archm: int = 0)`, com a docstring das regras. `Context.fit(spec, data, *, options=None, starts=None, lengths=None, mxreg=None, vxreg=None)`. Com `data` (T,), os regressores têm forma (T, k); com `data` (T, N), têm forma (T, N, k). Toda spec com k > 0 recebe a fatia da sua série, sem cópia (painel com strides), e toda spec com k = 0 recebe NULL. Se o k de uma spec for diferente de `shape[-1]`, ou faltar o array, é `ValueError`.

- [ ] **Step 1: Escrever os testes**

```cpp
TEST(CapiSpec, RegressorSetters) { /* mxreg/vxreg 0..8 OK, 9 e −1 → INVALID_ARGUMENT; archm 0,1,2 OK, 3 → INVALID_ARGUMENT */ }
TEST(CapiFit, RegressorChecks) {
  // Review Focus 3: cada caso de check_batch acima → INVALID_ARGUMENT, mensagem com "job <i>", fits_out todo NULL
}
TEST(CapiFit, RaggedRegressorsMatchSlices) {
  // Review Focus 5: 3 séries irregulares, mxreg 2 e vxreg 1, threads 1 e 4: bit-idêntico a ve::fit_one(Data das fatias)
}
TEST(CapiFit, RegressorParamNames) { /* ... archm mxreg1 mxreg2 omega ... vxreg1 ... via ve_fit_param_name */ }
```
```python
def test_mean_regressors_1d(ctx, series): ...          # Spec(mxreg=2), X (T, 2): names com mxreg1, mxreg2; Converged
def test_regressors_panel_slices(ctx): ...             # data (T, 3), X (T, 3, 2): cada fit == fit da coluna com X[:, j, :]
def test_regressor_shape_mismatch_raises(ctx): ...     # k da spec ≠ shape[-1], ou array ausente → ValueError
def test_archm_names(ctx, series): ...                 # Spec(archm=2): "archm" depois de ma e antes de mxreg
```

- [ ] **Step 2: Rodar e ver falhar** — `ctest --preset gcc-debug -R "CapiSpec|CapiFit"` e pytest → FAIL.
- [ ] **Step 3: Implementar.**
- [ ] **Step 4: Rodar e ver passar** — suítes C++ e Python inteiras PASS; header C11-limpo; `mypy --strict` PASS.
- [ ] **Step 5: Commit** — `git commit -m "feat: expose regressors and ARCH-in-mean through the C API and Python"`

---

### Task 8: Oráculos

**Files:**
- Modify: `tests/reference/generate_data.py`, `tests/reference/cases.json`, `tests/reference/oracle_rugarch.R`, `tests/reference/oracle_arch.py`, `tests/reference/README.md`, `python/tests/test_oracles.py`, `docs/guide/validation.md`
- Create (gerados, commitados): `tests/reference/data/grid_2e.csv` (as séries) e `tests/reference/data/regressors_2e.csv` (x1, x2, v1); fixtures novas em `tests/reference/fixtures/`

**Interfaces:**
- Consumes: pacote Python (Task 7). Pré-requisitos: R 4.6.2 + `rugarch` 1.5.6 e o venv com `arch` 8.0.0; se faltar algum, pare e peça ao usuário.
- Regressores (T = 3000, um só `numpy.random.default_rng(20261114)`): x1ₜ = sin(t/40), x2ₜ ~ N(0, 1), v1ₜ = |cos(t/25)|, com t = 1..T. Séries (σ² inicial 1, 1000 de aquecimento descartadas, com os regressores do aquecimento gerados pela mesma regra em t ≤ 0):

| Série | Processo |
|---|---|
| `arx_sgarch11_norm` | AR(1) φ = 0.3, μ = 0.05, mxreg (x1, x2) δ = (0.3, −0.2); sGARCH(1,1) ω = 0.02, α = 0.08, β = 0.9; norm |
| `sgarch11_vx_std` | μ = 0.05; sGARCH(1,1) ω = 0.01, vxreg v1 ζ = 0.05, α = 0.08, β = 0.88; std ν = 6 |
| `archm_gjrgarch11_norm` | μ = 0.02, archm p = 1 ξ = 0.1; gjrGARCH(1,1) ω = 0.02, α = 0.03, γ = 0.09, β = 0.9; norm |
| `archm2_egarch11_std` | μ = 0.02, archm p = 2 ξ = 0.05; eGARCH(1,1) ω = −0.01, α = −0.08, γ = 0.15, β = 0.97; std ν = 6 |

- Casos (fixados):
  - `rugarch`: cada série com o próprio modelo; mais `sgarch11_vx_std` em csGARCH(1,1)-std, apARCH(1,1)-std e eGARCH(1,1)-std com vxreg 1; `arx_sgarch11_norm` com ARMA(1,1), archm p = 1 e mxreg 2; `sgarch11_vx_std` com VT e vxreg 1. Com `archm`, o script passa `start.pars = list(archm = 0)` (regra "Resíduos de pré-amostra").
  - `arch` (backcast): `arch_model(y, x=X, mean="LS", vol="GARCH", p=1, q=1)` com norm na série `arx_sgarch11_norm`, contra a spec ARMA(0,0) + mxreg 2. O caso não tem AR; o `ARX` do `arch` aplica o AR em y, não em y − mₜ, então AR com regressores não é comparável.
- Regra de interior do 2b–2d. Caso que falhe é trocado, com o motivo em "Case selection".
- Tolerâncias: as do 2a, por grandeza, sem afrouxar.
- Convenções a confirmar e documentar em `validation.md`, com a fonte: as regras "Média", "Resíduos de pré-amostra", "Variância" (inclusive o csGARCH), "VT com `vxreg`" e "Backcast"; os nomes e a ordem; a diferença de estacionariedade com `vxreg`; e o ajuste auxiliar do `rugarch` para o `archm` sem valor inicial. Se o oráculo divergir, a correção vai no núcleo, com os testes de gradiente mantidos, e a mudança é registrada.

- [ ] **Step 1: Gerar dados e fixtures** — `python/.venv/Scripts/python tests/reference/generate_data.py && Rscript tests/reference/oracle_rugarch.R && python/.venv/Scripts/python tests/reference/oracle_arch.py` → um resultado por caso e oráculo; as fixtures antigas não mudam (só acréscimos).
- [ ] **Step 2: Estender o teste** — `case_spec` e o `fit` dos casos passam `mxreg`, `vxreg` e `archm` e os regressores de `regressors_2e.csv`; formato do caso com `"mxreg": [colunas]`, `"vxreg": [colunas]` e `"archm": p`, documentado no README.
- [ ] **Step 3: Rodar** — `VOLENGINE_LIB=<abs> python/.venv/Scripts/python -m pytest python/tests/test_oracles.py -v` → PASS; depois de cada correção no núcleo, as suítes C++ e Python inteiras.
- [ ] **Step 4: Commit** — `git commit -m "test: validate regressors and ARCH-in-mean against rugarch and arch"`

---

### Task 9: Benchmarks e documentação

**Files:**
- Modify: `tests/benchmarks/fit_benchmarks.cpp`, `docs/performance/benchmarks.md`, `README.md`, `CHANGELOG.md`, `ARCHITECTURE.md`

**Interfaces:**
- Benchmarks novos (os existentes não mudam): `BM_Loglik/<serie>/{double,dual}` e `BM_FitOne/<serie>`, T = 5000, nas quatro séries da Task 8 com o próprio modelo. Nesta máquina só compilar e rodar curto: `--benchmark_min_time=0.01s --benchmark_filter=BM_Loglik`. A medição fica para a máquina de benchmark, e `benchmarks.md` traz a nota de casos ainda não medidos e a comparação antes/depois da Task 4.
- README: um exemplo com `Spec(mxreg=2, archm=1)` e `ctx.fit(spec, y, mxreg=X)`, e as regras curtas, com o link para `validation.md`. CHANGELOG (`[Unreleased]`). ARCHITECTURE: `linalg/`, `model::Data`, recursões em forma de passo e laço conjunto com `archm`.

- [ ] **Step 1: Escrever** — `cmake --preset gcc-bench && cmake --build --preset gcc-bench` sem aviso; a rodada curta termina.
- [ ] **Step 2: Verificar** — `ctest --preset gcc-release` e pytest inteiros PASS; `check_style.py .` sai 0.
- [ ] **Step 3: Commit** — `git commit -m "bench: add regressor and ARCH-in-mean benchmarks and document plan 2e"`
