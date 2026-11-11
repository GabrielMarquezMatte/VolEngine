# VolEngine — Plano 2a: motor genérico, sGARCH(p,q), ARMA, norm/std/ged e benchmarks

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Trocar o caminho analítico do sGARCH(1,1)-norm do plano 1 por um motor genérico com diferenciação automática (forward-mode), e por cima dele entregar sGARCH(p,q), média ARMA(m,n) e as distribuições norm, std e ged — validados contra `rugarch` e `arch` e medidos por benchmarks em C++ e Python.

**Architecture:** A verossimilhança vira um template sobre o tipo escalar (`double` ou `ad::Dual<N>`). O mesmo código dá o valor, o gradiente exato no espaço irrestrito (objetivo do L-BFGS) e os scores por observação no espaço natural (erros-padrão). Um `Layout` derivado da `Spec` define blocos de parâmetros, offsets e nomes no formato do `rugarch`. A interface C só cresce: novas distribuições como valores de enum, um setter para ARMA, ordens até `VE_MAX_ORDER`.

**Tech Stack:** C++23, CMake + presets, GoogleTest v1.17.0, BS::thread_pool v5.0.0, Google Benchmark v1.9.1 (novo), Python ≥ 3.11 + numpy + pytest, `arch` 8.0.0 e R 4.6.2 + `rugarch` 1.5.6 (oráculos e benchmark Python).

**Spec:** `docs/superpowers/specs/2026-11-10-volengine-design.md` (e o plano 1, `docs/superpowers/plans/2026-11-10-volengine-01-fundacao.md`, como histórico das convenções)

## Roteiro

O plano 2 do roteiro original foi dividido (decisão do usuário, 2026-11-11):

- **2a (este):** motor genérico com AD, sGARCH(p,q), ARMA(m,n), norm/std/ged, benchmarks C++ e Python vs `arch`.
- **2b:** eGARCH, gjrGARCH, apARCH, iGARCH, csGARCH, fGARCH (e submodelos).
- **2c:** sstd, sged, nig, ghyp, jsu (Bessel K via Boost.Math).
- **2d:** parâmetros fixos, valores iniciais, variance targeting, `mxreg`/`vxreg`, `archm`/`archpow`, ARFIMA.

Depois, como antes: 3 diagnósticos/forecast/simulação; 4 DCC; 5 performance (L-BFGS em lote com SIMD); 6 wrapper C++, wheels, 1.0.

## Global Constraints

Todas as Global Constraints e a seção "Estilo" do plano 1 continuam valendo (texto completo em `STYLEGUIDE.md` e no plano 1). As que mais pesam aqui:

- A interface C é o único contrato estável. Nenhuma struct pública; configuração por setters; todo `ve_*` que pode falhar retorna `ve_status`; enums `typedef int32_t` + `#define`, só crescem; ≤ 5 parâmetros por função; nenhuma exceção atravessa a fronteira C.
- Sem estado global. Resultados imutáveis. Falha de convergência de um ajuste não falha o lote; erro de argumento falha a chamada.
- Ajustes bit-idênticos para qualquer número de threads na mesma máquina e ISA.
- Convenção padrão segue `rugarch` (`VE_VARINIT_SAMPLE`); a do `arch` é `VE_VARINIT_BACKCAST`.
- Critérios e diagnósticos são funções sobre o resultado, nunca campos.
- Só `double` nos dados e resultados.
- Código do `rugarch`/`rmgarch` (GPL-3) pode ser lido para entender convenções, nunca copiado ou traduzido; em zona cinzenta, paper e documentação. O `arch` é NCSA: idem, e código portado exige aviso em `THIRD-PARTY-NOTICES.md`.
- Anti-objetivo: não fazer mais do que o necessário.
- Estilo: nunca `else`; funções ≤ 60 linhas; ≤ 5 parâmetros; aninhamento ≤ 3; ≤ 8 módulos por pasta; sem `utils`/`helpers`; comentários só para restrição externa ou invariante; medições vão para `docs/`.
- Commits sem trailer `Co-Authored-By`; LF.
- Windows/MinGW: a DLL linka `libstdc++`, `libgcc` e `winpthread` estaticamente. Testes Python no venv `python/.venv` (Python 3.14 do Windows), nunca no Python do MSYS2.
- Convenções já fixadas no plano 1 (não mudar sem registro em `docs/guide/validation.md`): `p` = ordem ARCH (termos α), `q` = ordem GARCH (termos β); SE robusto = sanduíche QML H⁻¹SᵀSH⁻¹; backcast fixo nos resíduos da média amostral (`r − r̄`, ou `r` sem média); `VE_FIT_FAILED` carrega o melhor ponto finito ou NaN; threads ≤ `VE_MAX_THREADS` (1024); otimização no espaço padronizado (μ e ω escalados por sd(r), objetivo −ℓ/n − log s).

### Novas regras deste plano

- **Camadas:** `core ← ad ← mean, garch, dist ← model ← fit ← capi`, e `optim ← fit`. `optim` fica genérico (só o L-BFGS); a reparametrização sai de `optim/transform` para `model/parameter_map`. `ARCHITECTURE.md` e `STYLEGUIDE.md` são atualizados na Task 3.
- **Ordens:** `1 ≤ p, q ≤ VE_MAX_ORDER` e `0 ≤ m, n ≤ VE_MAX_ORDER`, com `#define VE_MAX_ORDER 6`. Com isso k ≤ 1 + 6 + 6 + 1 + 6 + 6 + 1 = 27 parâmetros, e o maior `Dual` é `Dual<32>`. Relaxar o limite depois não quebra nada.
- **Nomes e ordem dos parâmetros** (os do `rugarch`): `mu`, `ar1..arm`, `ma1..man`, `omega`, `alpha1..alphap`, `beta1..betaq`, `shape`. `mu` só com média; `shape` só para std e ged.
- **Distribuições** (todas padronizadas com média 0 e variância 1, como no `rugarch` e no `arch`); ℓₜ = log f(zₜ) − ½ log σ²ₜ, com zₜ = εₜ/σₜ:
  - norm: log f(z) = −½ log 2π − ½ z².
  - std (ν > 2): log f(z) = lgamma((ν+1)/2) − lgamma(ν/2) − ½ log(π(ν−2)) − ((ν+1)/2) log(1 + z²/(ν−2)).
  - ged (ν > 0): λ = sqrt(2^(−2/ν) Γ(1/ν) / Γ(3/ν)); log f(z) = log ν − log λ − (1 + 1/ν) log 2 − lgamma(1/ν) − ½ |z/λ|^ν.
- **Média ARMA** (hipótese a confirmar contra o `rugarch` na Task 7): εₜ = (rₜ − μ) − Σᵢ φᵢ (rₜ₋ᵢ − μ) − Σⱼ θⱼ εₜ₋ⱼ, com termos de índice negativo iguais a zero; a verossimilhança soma todos os t.
- **Variância sGARCH(p,q):** σ²ₜ = ω + Σᵢ₌₁ᵖ αᵢ ε²ₜ₋ᵢ + Σⱼ₌₁^q βⱼ σ²ₜ₋ⱼ.
  - `VE_VARINIT_SAMPLE` (rugarch): σ²ₜ = m̄ = média de ε² na amostra inteira (com os parâmetros de média correntes) para t < max(p, q); recursão com lags reais a partir de t = max(p, q). Para (1,1) reproduz o plano 1. A Task 7 confirma para p, q > 1.
  - `VE_VARINIT_BACKCAST` (arch): ε² e σ² de índice negativo valem b (backcast do plano 1: τ = min(75, T), pesos 0.94ⁱ, sobre `r − r̄`, fixo); recursão desde t = 0.
- **Reparametrização** (espaço irrestrito → natural; s = sd(r), r̄ = média de r):
  - μ = r̄ + s·m;
  - φᵢ, θⱼ identidade, sem restrição de estacionariedade ou invertibilidade, como o `rugarch` por padrão;
  - ω = s²·exp(u);
  - persistência P = logistic(v)·(1 − 1e-8); pesos (α₁..α_p, β₁..β_q) = P·softmax(w₁, …, w_{p+q−1}, 0). Para (1,1) isso é exatamente o plano 1 (α = P·logistic(w), β = P·(1 − logistic(w)));
  - std: ν = 2 + exp(x); ged: ν = exp(x).
- **Valores iniciais:**
  - média: μ = r̄, φ = θ = 0;
  - variância: ω = var(r)·(1 − 0.95), αᵢ = 0.05/p, βⱼ = 0.90/q;
  - forma: ν = 8 para std, ν = 2 para ged.
- **Dados insuficientes:** comprimento < k + 2, ou ≤ max(m, n, p, q), resulta em `INSUFFICIENT_DATA`.
- **Erros-padrão perto da fronteira:** parâmetros αᵢ ou βⱼ com valor < 1e-8 são marcados como de fronteira. Eles saem da Hessiana e da matriz de scores, ficam com SE NaN, e os demais são calculados sobre o bloco restante. Se o bloco restante não for positivo-definido, todos ficam NaN.
- **Critérios:** k = número de parâmetros estimados (todos, neste plano).
- **Benchmarks:** seguem a seção "Tests and Benchmarks" do `STYLEGUIDE.md`:
  - todo número publicado diz a máquina;
  - comparação com outra biblioteca é de trabalho igual, ou diz explicitamente que não é;
  - nada de razão entre resultados medidos em máquinas diferentes.

## Review Focus

1. **ARMA quase não-invertível ou explosiva durante a otimização** (MA com |θ| ≥ 1, AR com raiz unitária): o ajuste termina finito, convergido ou `FAILED`, sem crash e sem NaN vazando para outros ajustes do lote. → teste na Task 5.
2. **Série mais curta que a ordem** (comprimento 3 com p = q = 3, ou ARMA(6,0) com 7 observações): `INSUFFICIENT_DATA`, sem ler fora do array. → teste na Task 5.
3. **std em dados praticamente normais** (ν̂ → ∞): o ajuste converge com ν grande e finito; SE de `shape` pode ser NaN ou enorme, mas μ, ω, α e β têm SE finito. → teste na Task 5.
4. **Lote misturando especificações de tamanhos diferentes** (sGARCH(1,1)-norm com k = 4 e ARMA(2,2)-sGARCH(3,3)-std com k = 14, que caem em buckets de `Dual` diferentes): cada ajuste é bit-idêntico ao `fit_one` isolado, com qualquer número de threads. → teste na Task 6.
5. **ged com ν < 1 e retorno exatamente igual à média** (z = 0, onde |z|^ν tem derivada infinita): loglik e gradiente finitos. → teste na Task 3.

---

## Mapa de arquivos

```
src/ad/dual.hpp                   ve::ad::Dual<N>: valor + N derivadas; aritmética, log, exp, sqrt, pow, abs
src/ad/special.hpp/.cpp           digamma(double); lgamma(Dual) via digamma
src/mean/arma.hpp                 resíduos ARMA(m,n) templados
src/garch/sgarch.hpp              variância sGARCH(p,q) templada (substitui o sgarch analítico do plano 1)
src/dist/densities.hpp            log-densidades norm, std, ged templadas
src/model/layout.hpp/.cpp         Layout (blocos, offsets, k, nomes) a partir da Spec
src/model/likelihood.hpp/.cpp     ℓ e ℓₜ templados; instâncias para double e Dual<8|16|32>
src/model/parameter_map.hpp/.cpp  irrestrito ↔ natural templado; Standardization; validade
src/model/start_values.hpp/.cpp   valores iniciais
src/fit/                          fit.cpp, standard_errors.cpp generalizados (mesmos módulos do plano 1)
src/optim/transform.*             removidos (substituídos por model/parameter_map)
tests/unit/ad/, mean/, garch/, dist/, model/   testes espelhando src/
tests/benchmarks/                 Google Benchmark (C++)
python/benchmarks/bench_vs_arch.py
tests/reference/cases.json        grade de casos dos oráculos
docs/performance/benchmarks.md, docs/performance/python-vs-arch.md
```

---

### Task 1: Números duais e funções especiais

**Files:**
- Create: `src/ad/dual.hpp`, `src/ad/special.hpp`, `src/ad/special.cpp`
- Modify: `CMakeLists.txt` (special.cpp no `volengine_core`; testes)
- Test: `tests/unit/ad/dual_test.cpp`, `tests/unit/ad/special_test.cpp`

**Interfaces:**
- Produces:
  ```cpp
  namespace ve::ad {
  template <std::size_t N> struct Dual { double value = 0.0; std::array<double, N> grad{}; };
  template <std::size_t N> Dual<N> variable(double value, std::size_t index);   // grad = e_index
  template <std::size_t N> Dual<N> constant(double value);
  // operadores + − * / (Dual⊗Dual, Dual⊗double, double⊗Dual), unário −, comparações por value (<, >, <=, >=)
  // funções: log, exp, sqrt, abs (derivada sign(x), 0 em x = 0), pow(Dual, double), pow(Dual, Dual), lgamma
  double value_of(double x);  template <std::size_t N> double value_of(const Dual<N>& x);   // para código templado
  double digamma(double x);   // x > 0
  }
  ```
- `digamma`: recorrência ψ(x) = ψ(x+1) − 1/x até x ≥ 6, depois série assintótica até o termo x⁻¹⁰. `lgamma(Dual)` = `std::lgamma(value)` com derivada `digamma(value)·grad`.

- [ ] **Step 1: Escrever os testes**

```cpp
TEST(Dual, ArithmeticAndChainRule) {
  auto x = ve::ad::variable<4>(1.5, 0), y = ve::ad::variable<4>(-0.7, 1);
  auto f = ve::ad::log(x * x + ve::ad::exp(y)) / (y - 2.0);
  // compara f.value e f.grad[0..1] com diferenças centrais h = 1e-6 sobre a mesma expressão em double: |Δ| < 1e-8
  EXPECT_EQ(f.grad[2], 0.0); EXPECT_EQ(f.grad[3], 0.0);
}
TEST(Dual, PowAndAbs) { /* pow(Dual,Dual), pow(Dual,double), abs em x<0, x>0 e x=0 (grad 0) contra diferenças centrais 1e-7 */ }
TEST(Dual, ComparisonsUseValue) { /* variable<2>(1,0) < constant<2>(2) */ }
TEST(Special, DigammaKnownValues) {
  EXPECT_NEAR(ve::ad::digamma(1.0), -0.57721566490153286, 1e-14);
  EXPECT_NEAR(ve::ad::digamma(0.5), -1.9635100260214235, 1e-13);
  EXPECT_NEAR(ve::ad::digamma(10.0), 2.2517525890667211, 1e-14);
  EXPECT_NEAR(ve::ad::digamma(1e-3), -1000.4237549404110, 1e-9);
}
TEST(Special, LgammaDualDerivative) { /* lgamma(variable(x)) para x ∈ {0.3, 1, 4.5, 50}: grad[0] == digamma(x) (1e-14) */ }
```

- [ ] **Step 2: Rodar e ver falhar** — `cmake --build --preset gcc-debug && ctest --preset gcc-debug -R "Dual|Special"` → não compila.
- [ ] **Step 3: Implementar** os headers e `special.cpp`.
- [ ] **Step 4: Rodar e ver passar** — mesmo comando → PASS; `ctest --preset gcc-debug` inteiro (inclui `style`) PASS.
- [ ] **Step 5: Commit** — `git commit -m "feat: add forward-mode dual numbers and digamma"`

---

### Task 2: Spec estendida e Layout dos parâmetros

**Files:**
- Modify: `src/core/spec.hpp`
- Create: `src/model/layout.hpp`, `src/model/layout.cpp`
- Test: `tests/unit/model/layout_test.cpp`

**Interfaces:**
- Consumes: `ve::Spec`, `ve::Dist` do plano 1.
- Produces:
  ```cpp
  namespace ve {
  enum class Dist : int32_t { Norm = 0, Std = 1, Ged = 2 };
  constexpr int kMaxOrder = 6;
  struct Spec { Model model = Model::SGarch; int p = 1; int q = 1; int ar = 0; int ma = 0;
                Dist dist = Dist::Norm; bool include_mean = true; };
  }
  namespace ve::model {
  struct Layout {
    bool include_mean; int ar; int ma; int arch; int garch; Dist dist;
    std::size_t mu() const;            // índice de mu (só válido com include_mean)
    std::size_t ar_begin() const; std::size_t ma_begin() const;
    std::size_t omega() const; std::size_t alpha_begin() const; std::size_t beta_begin() const;
    bool has_shape() const; std::size_t shape() const;
    std::size_t count() const;         // k
  };
  Layout layout_of(const Spec& spec);  // arch = spec.p, garch = spec.q
  std::vector<std::string> parameter_names(const Layout& layout);
  }
  ```

- [ ] **Step 1: Escrever os testes**

```cpp
TEST(Layout, PlanOneShape) { auto l = ve::model::layout_of({}); EXPECT_EQ(l.count(), 4u);
  EXPECT_EQ(ve::model::parameter_names(l), (std::vector<std::string>{"mu","omega","alpha1","beta1"})); }
TEST(Layout, FullShapeFollowsRugarchOrder) {
  ve::Spec s; s.p = 2; s.q = 3; s.ar = 1; s.ma = 2; s.dist = ve::Dist::Std;
  auto l = ve::model::layout_of(s);
  EXPECT_EQ(ve::model::parameter_names(l), (std::vector<std::string>{
    "mu","ar1","ma1","ma2","omega","alpha1","alpha2","beta1","beta2","beta3","shape"}));
  EXPECT_EQ(l.count(), 11u); EXPECT_EQ(l.omega(), 4u); EXPECT_EQ(l.beta_begin(), 7u); EXPECT_EQ(l.shape(), 10u);
}
TEST(Layout, NoMeanNoShape) { /* include_mean=false, Norm: nomes começam em "ar1" ou "omega"; has_shape()==false */ }
TEST(Layout, LargestShapeFitsDual32) { /* p=q=ar=ma=kMaxOrder, Ged → count()==27 */ }
```

- [ ] **Step 2: Rodar e ver falhar** — `ctest --preset gcc-debug -R Layout` → não compila.
- [ ] **Step 3: Implementar.** As offsets são calculadas na ordem dos blocos da regra "Nomes e ordem".
- [ ] **Step 4: Rodar e ver passar** — suíte inteira PASS (os usos de `Spec` do plano 1 continuam compilando: campos novos têm padrão).
- [ ] **Step 5: Commit** — `git commit -m "feat: add ARMA orders, std and ged to Spec and a parameter layout"`

---

### Task 3: Verossimilhança genérica (ARMA + sGARCH(p,q) + norm/std/ged)

**Files:**
- Create: `src/mean/arma.hpp`, `src/dist/densities.hpp`, `src/model/likelihood.hpp`, `src/model/likelihood.cpp`
- Rewrite: `src/garch/sgarch.hpp` (recursão templada; `sgarch.cpp` removido)
- Delete: o caminho analítico `sgarch_norm_loglik` e seus testes analíticos (os casos de valor à mão migram)
- Modify: `ARCHITECTURE.md`, `STYLEGUIDE.md` (linha das camadas), `CMakeLists.txt`
- Test: `tests/unit/model/likelihood_test.cpp`, `tests/unit/mean/arma_test.cpp`, `tests/unit/garch/sgarch_test.cpp` (reescrito), `tests/unit/dist/densities_test.cpp`

**Interfaces:**
- Consumes: `ad::Dual`, `ad::lgamma`, `ad::abs`, `ad::pow` (Task 1); `model::Layout` (Task 2); `ve::VarInit`.
- Produces:
  ```cpp
  namespace ve::model {
  // Cada span é vazio (não calcular) ou tem tamanho T.
  template <class T> struct LoglikOutputs { std::span<T> terms{}; std::span<T> residuals{}; std::span<T> sigma2{}; };
  // theta no espaço natural, na ordem do Layout. Retorna Σₜ ℓₜ; não finito se algum σ²ₜ ≤ 0 ou não finito,
  // se ν fora do domínio, ou se algum termo não for finito. Saídas não especificadas nesse caso.
  template <class T>
  T loglik(std::span<const double> r, std::span<const T> theta, const Layout& layout, VarInit var_init,
           const LoglikOutputs<T>& out);
  // Instâncias explícitas em likelihood.cpp para T = double, ad::Dual<8>, ad::Dual<16>, ad::Dual<32>.
  }
  ```
- Fórmulas: as regras "Média ARMA", "Variância sGARCH(p,q)" e "Distribuições" das Global Constraints. O backcast b é calculado em `double` (não depende de θ).

- [ ] **Step 1: Escrever os testes**

```cpp
TEST(Likelihood, PlanOneHandCaseStillHolds) {
  // o caso à mão do plano 1: r = {0.1,-0.2,0.3}, θ = {0, 0.1, 0.1, 0.8}, sample init
  // σ²₀ = 0.14/3; σ²₁ = 0.1 + 0.1·0.01 + 0.8·σ²₀; σ²₂ = 0.1 + 0.1·0.04 + 0.8·σ²₁ (1e-12), loglik normal (1e-12)
}
TEST(Likelihood, MatchesPlanOneSimulatedValues) {
  // sGARCH(1,1)-norm, série de 500 obs com mt19937_64(42), θ = {0.05, 0.01, 0.08, 0.9}: os valores de loglik
  // nas duas inits, calculados pelo plano 1, gravados como constantes no teste ANTES de apagar o código analítico
  // (rodar o código antigo uma vez para obter os números) — |Δ| < 1e-10·|ℓ|
}
TEST(Likelihood, SampleInitHoldsFirstMaxOrderVariances) {
  /* p=2, q=3: sigma2[0..2] == m̄; sigma2[3] segue a recursão completa (1e-12) */ }
TEST(Likelihood, BackcastFeedsPresample) { /* p=2,q=2: sigma2[0] == ω + (α₁+α₂+β₁+β₂)·b (1e-12) */ }
TEST(Arma, ResidualsHandCase) {
  // ARMA(1,1), μ=0.1, φ=0.5, θ=0.3, r = {0.2, 0.4, -0.1}:
  // ε₀ = 0.1; ε₁ = 0.3 − 0.5·0.1 − 0.3·0.1 = 0.22; ε₂ = −0.2 − 0.5·0.3 − 0.3·0.22 = −0.416 (1e-14)
}
TEST(Densities, IntegrateToOneWithUnitVariance) {
  // norm, std(ν=5), std(ν=2.5), ged(ν=0.8), ged(ν=1.5), ged(ν=2): ∫f = 1 e ∫z²f = 1 por quadratura (trapézio em
  // [-60, 60], 2e5 pontos) com tolerância 1e-6; ged(ν=2) == norm em z ∈ {-3,-0.5,0,2} (1e-14)
}
TEST(Likelihood, GradientMatchesFiniteDifferences) {
  // grade: {(1,1),(2,1),(1,2),(3,3)} × ARMA {(0,0),(1,1),(2,0)} × {norm,std,ged} × {sample,backcast} × mean {on,off}
  // série de 600 obs (mt19937_64(11)), θ interior; Dual<32>: |grad_i − CD_i| ≤ 1e-6·max(1,|grad_i|), hᵢ = 1e-6·max(1,|θᵢ|)
}
TEST(Likelihood, TermsSumToTotalAndScoresSumToGradient) { /* Σ terms == ℓ (1e-10·|ℓ|) e Σ terms[t].grad == ℓ.grad (1e-9) */ }
TEST(Likelihood, DoubleAndDualAgreeOnValue) { /* value_of(loglik<Dual<16>>) == loglik<double> bit a bit */ }
TEST(Likelihood, GedZeroResidualHasFiniteGradient) {
  // Review Focus 5: ged ν = 0.7, include_mean, r[10] == μ exatamente → ℓ e todas as derivadas finitas
}
TEST(Likelihood, InvalidDomainGivesNonFinite) { /* ω < 0; std ν = 1.5; ged ν = 0 → !isfinite */ }
```

- [ ] **Step 2: Rodar e ver falhar** — `ctest --preset gcc-debug -R "Likelihood|Arma|Densities"` → não compila. Antes de apagar o código analítico, rode-o uma vez para gerar as constantes de `MatchesPlanOneSimulatedValues` e registre no relatório.
- [ ] **Step 3: Implementar.** As funções templadas ficam nos headers, e as instâncias em `likelihood.cpp`, para que os headers não sejam reinstanciados em todo canto. O caminho do ajuste (`fit.cpp`, e `standard_errors.cpp`, que hoje chama `sgarch_norm_loglik` para o gradiente da Hessiana) passa temporariamente a chamar `loglik<double>` e `loglik<Dual<8>>` com o layout de (1,1), e continua passando nos testes do plano 1 até a Task 5. Atualize as camadas em `STYLEGUIDE.md` e `ARCHITECTURE.md`.
- [ ] **Step 4: Rodar e ver passar** — suíte C++ inteira PASS, inclusive `fit`, `capi` e `style`. Testes Python e de oráculo (`VOLENGINE_LIB=…`) continuam PASS.
- [ ] **Step 5: Commit** — `git commit -m "feat: add templated ARMA-sGARCH(p,q) likelihood with norm, std and ged"`

---

### Task 4: Mapa de parâmetros e valores iniciais

**Files:**
- Create: `src/model/parameter_map.hpp`, `src/model/parameter_map.cpp`, `src/model/start_values.hpp`, `src/model/start_values.cpp`
- Delete: `src/optim/transform.hpp`, `src/optim/transform.cpp`, `tests/unit/optim/transform_test.cpp` (casos úteis migram)
- Test: `tests/unit/model/parameter_map_test.cpp`, `tests/unit/model/start_values_test.cpp`

**Interfaces:**
- Consumes: `Layout` (Task 2), `ad::Dual` (Task 1).
- Produces:
  ```cpp
  namespace ve::model {
  struct Standardization { double location = 0.0; double scale = 1.0; };   // movido de optim
  template <class T> void to_natural(std::span<const T> u, std::span<T> theta, const Layout&, const Standardization&);
  void to_unconstrained(std::span<const double> theta, std::span<double> u, const Layout&, const Standardization&);
  bool is_valid(std::span<const double> theta, const Layout& layout);   // ω>0, α,β≥0, Σα+Σβ<1−1e-8, ν no domínio
  std::vector<double> start_values(std::span<const double> r, const Layout& layout);   // natural
  }
  ```
- Fórmulas: regras "Reparametrização" e "Valores iniciais" das Global Constraints. Em `to_unconstrained`, os pesos usam log(peso/peso_ref) com piso `DBL_MIN`, como no plano 1.

- [ ] **Step 1: Escrever os testes**

```cpp
TEST(ParameterMap, OneOneMatchesPlanOneTransform) {
  // u = {0.2, -1.0, 2.0, 0.5}, Standardization{0.01, 1.3}: θ == {0.01+1.3·0.2, 1.69·e^-1, P·σ(0.5), P·(1−σ(0.5))}
  // com P = σ(2.0)·(1−1e-8) (1e-15)
}
TEST(ParameterMap, RoundTripAllShapes) { /* grade da Task 3: θ interior → u → θ' (rel 1e-12) */ }
TEST(ParameterMap, AlwaysValid) { /* 2000 u aleatórios por layout (mt19937_64(5), N(0,3)) → is_valid */ }
TEST(ParameterMap, TinyWeightsInvertFinite) { /* α₂ = 1e-14, β = 0: u finito */ }
TEST(ParameterMap, DualMatchesDouble) { /* value_of(to_natural<Dual<32>>) == to_natural<double> bit a bit */ }
TEST(StartValues, FollowRule) {
  // p=2,q=3,ARMA(1,1),std: μ=r̄, ar=ma=0, ω=var·0.05, α=0.025 cada, β=0.3 cada, shape=8; ged → shape=2
  // e is_valid(start_values) para todo layout
}
```

- [ ] **Step 2: Rodar e ver falhar** — `ctest --preset gcc-debug -R "ParameterMap|StartValues"` → não compila.
- [ ] **Step 3: Implementar.** `fit.cpp` passa a usar `model::to_natural` e `start_values` (ainda só (1,1)). O teste de invariância de escala do plano 1 continua passando sem mudar tolerâncias.
- [ ] **Step 4: Rodar e ver passar** — suíte inteira PASS.
- [ ] **Step 5: Commit** — `git commit -m "refactor: move reparametrization into a layout-driven parameter map"`

---

### Task 5: Driver de ajuste genérico e erros-padrão com fronteira

**Files:**
- Modify: `src/fit/fit.cpp`, `src/fit/fit.hpp`, `src/fit/fit_result.hpp`, `src/fit/standard_errors.hpp`, `src/fit/standard_errors.cpp`
- Test: `tests/unit/fit/fit_test.cpp` (estendido), `tests/unit/fit/standard_errors_test.cpp` (novo)

**Interfaces:**
- Consumes: Tasks 1–4.
- Produces:
  ```cpp
  namespace ve {
  struct FitResult { /* como no plano 1, com */ std::vector<std::string> names; /* (era string_view) */ };
  FitResult fit_one(std::span<const double> r, const Spec& spec, const FitConfig& config);   // mesma assinatura
  }
  namespace ve::fit {
  // gradient(theta, grad): gradiente natural de ℓ em theta; false se theta é inválido ou ℓ não é finito.
  using GradientFn = std::function<bool(std::span<const double>, std::span<double>)>;
  StandardErrors standard_errors(const GradientFn& gradient, std::span<const double> theta,
                                 std::span<const double> scores, const model::Layout& layout, double series_scale);
  }
  ```
- Despacho de bucket: k ≤ 8 → `Dual<8>`, ≤ 16 → `Dual<16>`, senão `Dual<32>`. Objetivo: −ℓ(θ(u))/n − log s com gradiente AD em u. Scores: ℓₜ com `Dual` semeado em θ natural.
- Hessiana: diferenças centrais do gradiente AD natural, com passos relativos do plano 1 (μ: 1e-5·max(|μ|, s); positivos: 1e-5·θ, com piso 1e-11 para α, β; φ, θ_MA: 1e-5·max(|θ|, 1e-3); ν: 1e-5·ν). Mantém a regra de encolher o passo quando o ponto é inválido, e a regra de fronteira das Global Constraints.

- [ ] **Step 1: Escrever os testes**

```cpp
// Séries simuladas no próprio teste (helper em tests/unit/simulated_garch.hpp estendido para ARMA, p,q e
// inovações std/ged), T = 5000.
TEST(FitGeneric, RecoversSGarch21Std) { /* α=(0.05,0.04), β=0.88, ν=6, seed 21: Converged; |αᵢ−α|<0.03; |β−0.88|<0.04; |ν−6|<1.5 */ }
TEST(FitGeneric, RecoversArma11SGarch11Ged) { /* μ=0.05, φ=0.5, θ=-0.2, α=0.08, β=0.9, ν=1.5, seed 22: |φ−0.5|<0.05; |θ+0.2|<0.06; |ν−1.5|<0.15 */ }
TEST(FitGeneric, RecoversSGarch12Norm) { /* α=0.08, β=(0.5,0.4), seed 23: Converged; |Σβ−0.9|<0.03 */ }
TEST(FitGeneric, NamesAreGenerated) { /* p=2,q=1,ARMA(1,0),std → {"mu","ar1","omega","alpha1","alpha2","beta1","shape"} */ }
TEST(FitGeneric, ScaleInvarianceStillHolds) { /* o teste do plano 1 agora também para ARMA(1,1)-sGARCH(2,1)-std, mesmas tolerâncias */ }
TEST(FitGeneric, ShortSeriesInsufficientData) {
  // Review Focus 2: p=q=3 com 3 obs; ARMA(6,0) com 7 obs; ARMA(0,0) sGARCH(1,1) com k+1 = 5 obs → INSUFFICIENT_DATA
}
TEST(FitGeneric, ExplosiveArmaEndsFinite) {
  // Review Focus 1: série simulada AR(1) com φ = 0.999 e ruído GARCH, ARMA(1,1): status ∈ {Converged, MaxIter, Failed},
  // sem exceção, e se Converged então estimate e loglik finitos
}
TEST(FitGeneric, NearNormalDataWithStd) {
  // Review Focus 3: dados normais sGARCH(1,1), dist std: status Converged ou MaxIter; ν̂ > 20; SEs de mu, omega,
  // alpha1, beta1 finitos e > 0
}
TEST(StandardErrors, BoundaryParameterGetsNaNOthersFinite) {
  // sGARCH(2,1) ajustado a uma série sGARCH(1,1) (α₂ verdadeiro = 0) cuja estimativa cai na fronteira: força θ com
  // α₂ = 0 e chama standard_errors diretamente → se_hessian[alpha2] NaN; os demais finitos e > 0
}
TEST(FitGeneric, DeterministicAcrossBuckets) { /* fit_one duas vezes → bit-idêntico, para k=4 (Dual<8>), 11 (16), 27 (32) */ }
```

- [ ] **Step 2: Rodar e ver falhar** — `ctest --preset gcc-debug -R "FitGeneric|StandardErrors"` → falha (sem suporte genérico).
- [ ] **Step 3: Implementar** o despacho por bucket, o objetivo AD, os scores AD, a Hessiana e a regra de fronteira. Remova o código específico do (1,1).
- [ ] **Step 4: Rodar e ver passar** — suíte C++ inteira e Python (incluindo oráculos do plano 1, sem mudar tolerâncias) PASS. Se os números observados em `validation.md` mudarem, atualize-os.
- [ ] **Step 5: Commit** — `git commit -m "feat: fit ARMA-sGARCH(p,q) with norm, std and ged through automatic differentiation"`

---

### Task 6: Interface C e Python para as novas especificações

**Files:**
- Modify: `include/volengine/volengine.h`, `src/capi/spec_api.cpp`, `src/capi/fit_api.cpp`, `src/capi/handles.hpp`
- Modify: `python/src/volengine/_spec.py`, `python/src/volengine/_native.py`, `python/src/volengine/api.py`, `python/pyproject.toml` (`mypy` no extra `test`)
- Test: `tests/unit/capi/spec_test.cpp`, `tests/unit/capi/batch_test.cpp` (estendidos), `python/tests/test_fit.py` (estendido), `python/tests/typing_cases.py` + `python/tests/test_typing.py` (novos)

**Interfaces:**
- Consumes: `ve::Spec` (Task 2), `fit_one` (Task 5).
- Produces (C, só acréscimos):
  ```c
  #define VE_DIST_STD 1
  #define VE_DIST_GED 2
  #define VE_MAX_ORDER 6
  /* 1 <= p, q <= VE_MAX_ORDER; p = ARCH order (alpha terms), q = GARCH order (beta terms). */
  VE_API ve_status ve_spec_set_order(ve_spec* s, int32_t p, int32_t q);              /* agora aceita até VE_MAX_ORDER */
  /* 0 <= m, n <= VE_MAX_ORDER: ARMA(m, n) conditional mean. Default ARMA(0, 0). */
  VE_API ve_status ve_spec_set_arma(ve_spec* s, int32_t m, int32_t n);
  ```
- Python: `Spec(model="sgarch", *, p=1, q=1, arma=(0, 0), dist="norm", include_mean=True)`, com `DISTRIBUTIONS = {"norm": 0, "std": 1, "ged": 2}`. Uma `arma` que não seja um par de inteiros gera `ValueError`, e os limites são validados pelo C (`VolEngineError`).
- `Context.fit` ganha `@typing.overload`s para a inferência de tipo do retorno. O corpo e o comportamento em runtime não mudam:
  ```python
  Series1D = Sequence[float] | np.ndarray[tuple[int], np.dtype[np.float64]]
  @overload
  def fit(self, spec: Spec, data: Series1D, *, options=..., starts=..., lengths=...) -> Fit: ...
  @overload
  def fit(self, spec: Spec, data: ArrayLike, *, ...) -> Fit | NDArray[np.object_]: ...   # dimensão desconhecida estaticamente
  @overload
  def fit(self, spec: Sequence[Spec], data: ArrayLike, *, ...) -> NDArray[np.object_]: ...
  ```
  O padrão vale para toda API Python futura com retorno dependente do argumento (`fit_dcc` no plano 4).
- `ve_fit_param_name` devolve os nomes gerados, válidos até `ve_fit_free`.
- Pendências do plano 1, só texto no header:
  - "as funções `ve_context_*` podem ser chamadas concorrentemente, exceto `ve_context_free`";
  - a referência ao `rugarch` passa a ser "`garchOrder = c(alpha order, beta order)`" (o `rugarch` chama essas ordens de q e p ao contrário). A mesma correção vai na docstring do `Spec`.

- [ ] **Step 1: Escrever os testes**

```cpp
TEST(CapiSpec, OrdersUpToMax) { /* set_order(6,6) OK; (7,1) e (1,7) → INVALID_ARGUMENT com "VE_MAX_ORDER" ou "6" na mensagem; (0,1) INVALID */ }
TEST(CapiSpec, Arma) { /* set_arma(0,0), (6,6) OK; (-1,0), (7,0) INVALID; spec inalterada no erro */ }
TEST(CapiSpec, NewDistributions) { /* set_dist(1), (2) OK; (3) INVALID */ }
TEST(CapiFit, MixedBucketsInOneBatch) {
  // Review Focus 4: specs {sGARCH(1,1)-norm, ARMA(2,2)-sGARCH(3,3)-std, ARMA(1,0)-sGARCH(2,1)-ged} × 3 séries,
  // threads 1 e 4: estimates/loglik/sigma bit-idênticos a ve::fit_one direto
}
TEST(CapiFit, GeneratedParamNames) { /* ARMA(1,1)-sGARCH(2,1)-std: nomes via ve_fit_param_name na ordem do Layout */ }
```
```python
def test_arma_std_spec_fits(ctx, series):
    f = ctx.fit(ve.Spec(p=2, arma=(1, 1), dist="std"), series[:, 0])
    assert f.names == ("mu", "ar1", "ma1", "omega", "alpha1", "alpha2", "beta1", "shape")
@pytest.mark.parametrize("arma", [(1,), (1, 2, 3), (1.5, 0), "11"])
def test_bad_arma_raises_value_error(arma): ...
def test_order_above_max_raises(): ...   # Spec(p=7) → VolEngineError
```
```python
# python/tests/typing_cases.py — só checado estaticamente, nunca executado
def cases(ctx: ve.Context, spec: ve.Spec, one: list[float], two: NDArray[np.float64]) -> None:
    assert_type(ctx.fit(spec, one), ve.Fit)
    assert_type(ctx.fit([spec, spec], one), NDArray[np.object_])
    assert_type(ctx.fit([spec], two), NDArray[np.object_])
    assert_type(ctx.fit(spec, two), ve.Fit | NDArray[np.object_])
# python/tests/test_typing.py
def test_fit_overloads_infer_return_types():
    stdout, stderr, status = mypy.api.run(["--strict", str(TYPING_CASES)])
    assert status == 0, stdout + stderr
```

- [ ] **Step 2: Rodar e ver falhar** — `ctest --preset gcc-debug -R "CapiSpec|CapiFit"` e pytest → FAIL.
- [ ] **Step 3: Implementar.**
- [ ] **Step 4: Rodar e ver passar** — suítes C++ e Python inteiras PASS; o header continua C11-limpo.
- [ ] **Step 5: Commit** — `git commit -m "feat: expose ARMA orders, std and ged through the C API and Python"`

---

### Task 7: Oráculos para a nova grade

**Files:**
- Create: `tests/reference/cases.json`
- Modify: `tests/reference/generate_data.py`, `tests/reference/oracle_rugarch.R`, `tests/reference/oracle_arch.py`, `tests/reference/README.md`, `python/tests/test_oracles.py`, `docs/guide/validation.md`
- Create (gerados, commitados): novas séries em `tests/reference/data/`, fixtures em `tests/reference/fixtures/`

**Interfaces:**
- Consumes: pacote Python (Task 6).
- `cases.json`: lista de `{"id": str, "series": str, "p": int, "q": int, "arma": [m, n], "include_mean": bool, "dist": "norm"|"std"|"ged", "oracles": ["rugarch", "arch"]}`. Formato do fixture do plano 1, com `"case": id` em cada resultado e os parâmetros com os nomes do VolEngine (o script do `arch` converte `alpha[1]`→`alpha1`, `nu`→`shape`, etc.).
- Grade (fixada):
  - séries do plano 1 + 3 novas, T = 3000, `default_rng(20261111)`: sGARCH(2,1)-std ν=6; ARMA(1,1)(φ=0.5, θ=−0.2)-sGARCH(1,1)-ged ν=1.5; sGARCH(1,2)-norm β=(0.5,0.4);
  - casos `rugarch`: (1,1), (2,1), (1,2), (2,2) × {norm, std, ged} sobre a série gerada pelo processo correspondente, mais ARMA(1,1)-(1,1)-{norm, ged} e ARMA(2,0)-(1,1)-norm;
  - casos `arch`: média constante, (1,1), (2,1), (1,2) × {normal, t, ged}. O `arch` não tem MA nem a convenção de AR com t completo, então ARMA é validado só contra o `rugarch` (decisão do plano).
  - Toda estimativa do oráculo precisa ser interior (α, β > 1e-4, persistência < 0.999). Caso contrário, o caso é trocado e o motivo vai para `validation.md`.
- Tolerâncias: as do plano 1, por grandeza, inclusive `shape`. Para `shape` a SE do `rugarch` vem da Hessiana numérica dele; se a tolerância de 5e-2 não for atingida, registre o motivo e o valor observado, e não afrouxe sem esse registro.
- Convenções a confirmar e documentar: a inicialização da variância para p, q > 1 (regra "Variância sGARCH(p,q)"), o pré-amostral do ARMA (regra "Média ARMA"), e o `rugarch` usando `m̄` dos resíduos ARMA. Se o oráculo divergir, a correção vai no núcleo (Task 3), com os testes de gradiente mantidos exatos, e a mudança é registrada em `validation.md`.

- [ ] **Step 1: Gerar dados e fixtures** — `python/.venv/Scripts/python tests/reference/generate_data.py && "C:/Program Files/R/R-4.6.2/bin/Rscript.exe" tests/reference/oracle_rugarch.R && python/.venv/Scripts/python tests/reference/oracle_arch.py` → fixtures com um resultado por caso e oráculo; os scripts param com erro em não-convergência ou estimativa de fronteira.
- [ ] **Step 2: Estender o teste** — `test_matches_oracle` parametrizado sobre (oráculo, caso), com as mesmas asserções do plano 1 (status, loglik unilateral e relativa, parâmetros, SE Hessiana, SE robusto — para o `rugarch` contra `qml_robust_se` —, σ e critérios na escala do oráculo).
- [ ] **Step 3: Rodar** — `VOLENGINE_LIB=<abs> python/.venv/Scripts/python -m pytest python/tests/test_oracles.py -v` → PASS; se uma convenção divergir, corrigir no núcleo como descrito, rodar a suíte C++ e Python inteiras.
- [ ] **Step 4: Commit** — `git commit -m "test: validate ARMA-sGARCH(p,q) with norm, std and ged against rugarch and arch"`

---

### Task 8: Benchmarks C++

**Files:**
- Create: `tests/benchmarks/fit_benchmarks.cpp`, `tests/benchmarks/simulated_panel.hpp`, `docs/performance/benchmarks.md`
- Modify: `CMakeLists.txt` (opção `VE_BUILD_BENCHMARKS`, padrão OFF; ON no preset novo `gcc-bench`), `CMakePresets.json`, `CONTRIBUTING.md`

**Interfaces:**
- Consumes: interface C (benchmark pelo contrato, como um usuário) e `ve::fit_one` (custo de um ajuste sem a camada C).
- Google Benchmark `v1.9.1` via FetchContent, por URL do tarball com `URL_HASH SHA256` (o executor calcula o hash ao baixar e registra no relatório). `BENCHMARK_ENABLE_TESTING OFF`.
- Benchmarks (nomes fixados):
  - `BM_Loglik/<spec>`: uma avaliação de `loglik<double>` e uma de `loglik<Dual<N>>` para T = 5000, nos specs `s11_norm`, `s21_std`, `arma11_s11_ged`, `arma22_s33_std`. Mede o custo da AD.
  - `BM_FitOne/<spec>`: `ve::fit_one` nos mesmos specs, T = 5000.
  - `BM_Batch/threads:{1,2,4,8,hw}`: `ve_fit_batch` com 20 séries × 1 spec `s11_norm`.
  - `BM_TccGrid/threads:{1,hw}`: a carga do TCC, 20 séries × 5000 obs × grade {p, q ∈ 1..3} × {ARMA(0,0), ARMA(1,1)} × {norm, std, ged} = 1080 ajustes numa única chamada `ve_fit_batch`. É a referência da meta do spec (< 10 s numa CPU de 8 núcleos, objeto do plano 5; aqui só é medida).
- Cada benchmark falha alto, com `state.SkipWithError`, se algum ajuste não terminar `CONVERGED` ou `MAX_ITER`, ou se a grade gerar zero trabalho. Os contadores reportam ajustes por segundo e a contagem por status.
- `docs/performance/benchmarks.md`: tabela com os resultados de `gcc-bench` (Release, `-O3`). Registra a máquina (CPU, núcleos, RAM, SO, compilador; via `Get-CimInstance Win32_Processor` e `systeminfo`), a data, o commit e o comando para reproduzir. Nenhuma razão entre máquinas diferentes.

- [ ] **Step 1: Escrever os benchmarks** e o preset `gcc-bench` (Ninja, Release, `VE_BUILD_BENCHMARKS=ON`, `VE_BUILD_TESTS=OFF`).
- [ ] **Step 2: Rodar** — `cmake --preset gcc-bench && cmake --build --preset gcc-bench && build/gcc-bench/volengine_benchmarks --benchmark_out=build/gcc-bench/bench.json --benchmark_out_format=json --benchmark_repetitions=5 --benchmark_report_aggregates_only=true` → todos sem erro.
- [ ] **Step 3: Escrever `docs/performance/benchmarks.md`** a partir do JSON (mediana e desvio, ajustes/s, máquina).
- [ ] **Step 4: Verificar** que `ctest --preset gcc-debug` continua PASS e que o CI compila os benchmarks sem rodar (um passo `cmake --build` com `-DVE_BUILD_BENCHMARKS=ON` no job Linux).
- [ ] **Step 5: Commit** — `git commit -m "bench: add C++ benchmarks for likelihood, single fits, batches and the TCC grid"`

---

### Task 9: Benchmark Python vs `arch` e documentação

**Files:**
- Create: `python/benchmarks/bench_vs_arch.py`, `docs/performance/python-vs-arch.md`
- Modify: `README.md`, `CHANGELOG.md`, `ARCHITECTURE.md`, `CONTRIBUTING.md`

**Interfaces:**
- Consumes: pacote Python (Task 6); `arch` 8.0.0 no venv.
- `bench_vs_arch.py`:
  - **Trabalho igual (seção principal):**
    - mesmas 20 séries simuladas (`default_rng(7)`, T = 5000) e mesmos specs suportados pelos dois: média constante × {(1,1), (2,1), (1,2)} × {normal/norm, t/std, ged};
    - `volengine` com `Context(threads=1)` e `var_init="backcast"`; `arch` com `rescale=False`, `fit(disp="off")`, sequencial;
    - mede o tempo total de parede (`time.perf_counter`, 5 repetições, mediana) e confere que cada par de log-verossimilhanças concorda em 1e-4 relativo (um par fora disso é falha do benchmark, não um número publicado).
  - **Fora do trabalho igual, rotulada assim:** `volengine` com `threads=0` (todos os núcleos). O `arch` não paraleliza um ajuste, então essa linha mede o paralelismo do lote e não a comparação de algoritmo.
  - Sai uma tabela markdown, com a máquina, as versões (`volengine`, `arch`, `numpy`, Python) e a data, para `docs/performance/python-vs-arch.md`.
  - O texto diz que os otimizadores diferem (L-BFGS no VolEngine, SLSQP no `arch`), de modo que o tempo inclui iterações diferentes; a coluna de loglik mostra que os dois chegam ao mesmo ótimo.
  - Segue o `STYLEGUIDE.md` para Python (funções ≤ 60 linhas etc.); é verificado pelo `check_style.py` se a pasta estiver no escopo dele, e se não estiver, inclua `python/benchmarks/` no escopo.
- Documentação:
  - README: specs suportados (sGARCH(p,q), ARMA, norm/std/ged) e o exemplo com `arma` e `dist`;
  - CHANGELOG: `[Unreleased]`;
  - ARCHITECTURE: as novas pastas e o fluxo AD;
  - CONTRIBUTING: como rodar os dois benchmarks.

- [ ] **Step 1: Escrever o script.**
- [ ] **Step 2: Rodar** — `VOLENGINE_LIB=<abs>/build/gcc-release/libvolengine.dll python/.venv/Scripts/python python/benchmarks/bench_vs_arch.py --out docs/performance/python-vs-arch.md` → tabela gerada; nenhum par de loglik fora de 1e-4.
- [ ] **Step 3: Atualizar a documentação.**
- [ ] **Step 4: Verificar** — `ctest --preset gcc-release` e pytest inteiros PASS; `check_style.py .` sai 0.
- [ ] **Step 5: Commit** — `git commit -m "bench: compare the Python package with arch on matched work and document plan 2a"`
