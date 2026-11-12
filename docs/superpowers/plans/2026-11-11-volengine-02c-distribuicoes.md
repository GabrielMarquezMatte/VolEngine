# VolEngine — Plano 2c: sstd, sged, nig, ghyp e jsu

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Sobre os 14 modelos do plano 2b, entregar as distribuições sstd, sged, nig, ghyp e jsu (padronizadas, parametrização do `rugarch`), com as restrições dos modelos generalizadas para choques assimétricos, validadas contra `rugarch` (todas) e `arch` (sstd), expostas pela interface C e pelo Python, e com benchmarks.

**Architecture:** Cada família de densidade vira um tipo com os constantes calculados uma vez por avaliação (`dist/skewed.hpp`, `dist/hyperbolic.hpp`, `dist/johnson.hpp`), e `dist::Density<T>` é um `std::variant` delas com a densidade simétrica do 2a. O `loglik` visita a variante uma vez, fora do laço temporal. Os parâmetros de distribuição passam a ser até três (`skew`, `shape`, `ghlambda`), lidos do `Layout` como `dist::ShapeParameters<T>`. Os momentos usados nas restrições (E|z|, κ do apARCH/fGARCH, P(z < 0) do gjrGARCH) ficam em forma fechada para norm/std/ged e vêm da quadratura do 2b para as assimétricas. O Bessel K vem do Boost.Math, embrulhado em `dist/bessel` com derivadas para o `Dual`.

**Tech Stack:** C++23, CMake + presets, GoogleTest v1.17.0, Google Benchmark v1.9.1, BS::thread_pool v5.0.0, Boost.Math `boost-1.87.0` (standalone, só headers), Python ≥ 3.11 + numpy + scipy + pytest, `arch` 8.0.0, R 4.6.2 + `rugarch` 1.5.6. Medição fica para a máquina de benchmark, como no 2b.

**Spec:** `docs/superpowers/specs/2026-11-10-volengine-design.md`. Os planos 2a e 2b (`docs/superpowers/plans/2026-11-11-volengine-02a-motor-generico.md`, `...-02b-variantes-garch.md`) são o histórico das convenções que este plano estende.

## Roteiro

- **2a (feito):** motor genérico com AD, sGARCH(p,q), ARMA(m,n), norm/std/ged, benchmarks.
- **2b (feito):** eGARCH, gjrGARCH, apARCH, iGARCH, csGARCH, fGARCH e submodelos.
- **2c (este):** sstd, sged, nig, ghyp, jsu.
- **2d:** parâmetros fixos, valores iniciais, variance targeting, `mxreg`/`vxreg`, `archm`/`archpow`, ARFIMA.

## Global Constraints

Valem todas as Global Constraints dos planos 1, 2a e 2b, o `STYLEGUIDE.md` e o `ARCHITECTURE.md`. As que mais pesam aqui:

- A interface C é o único contrato estável. Enums `typedef int32_t` + `#define`, só acréscimos; ≤ 5 parâmetros por função; nenhuma exceção atravessa a fronteira C.
- Ajustes bit-idênticos para qualquer número de threads na mesma máquina e ISA. **norm, std e ged dão os mesmos números do 2b:** todos os testes existentes passam sem mudar constantes nem tolerâncias. Em particular, o gjrGARCH simétrico usa π = 0.5 exato, e as fórmulas generalizadas reduzem às antigas bit a bit (multiplicar e dividir por 0.5 é exato).
- Convenção padrão segue o `rugarch`. **Sem opção de parametrização do `arch` para o sstd** (decisão do usuário, 2026-11-11): o skew-t de Hansen do `arch` é a mesma família, com ξ = √((1 + λ)/(1 − λ)) e a mesma log-verossimilhança. A conversão (e o SE por delta method, dξ/dλ = 1/(ξ(1 − λ)²)) fica documentada no header, na docstring do Python e em `validation.md`; o usuário converte. A Task 9 atualiza a seção 9 do spec.
- Código do `rugarch`, `fGarch` e `gamlss` (GPL) pode ser lido para entender convenções, nunca copiado ou traduzido. As densidades vêm dos papers (Fernandez & Steel 1998; Barndorff-Nielsen 1977 e Prause 1999 para a parametrização (ρ, ζ); Johnson 1949 e Rigby & Stasinopoulos 2005 para a jsu) e das fórmulas abaixo.
- Boost.Math: licença BSL-1.0, só headers, buscado por FetchContent. Nenhum outro componente do Boost.
- Anti-objetivo: nada de CDF, quantil ou momentos das distribuições como API pública (o plano 3 decide o que o forecast e o GOF precisam). `ghst` do `rugarch` não está no escopo.
- Estilo: nunca `else`; funções ≤ 60 linhas; ≤ 5 parâmetros; aninhamento ≤ 3; ≤ 8 módulos por pasta; comentários só para restrição externa ou invariante; medições em `docs/`. `-Wnrvo` como erro no GCC.
- Commits sem trailer `Co-Authored-By`; LF. Python no venv `python/.venv` (já existe nesta máquina); scipy vem como dependência do `arch`.

### Novas regras deste plano

**Distribuições (C e `ve::Dist`, só acréscimos):**

| Valor | C | `ve::Dist` | Python | Parâmetros (ordem) |
|---|---|---|---|---|
| 0–2 | `VE_DIST_NORM`, `_STD`, `_GED` | como hoje | como hoje | — / `shape` / `shape` |
| 3 | `VE_DIST_SSTD` | `Sstd` | `"sstd"` | `skew` ξ > 0, `shape` ν > 2 |
| 4 | `VE_DIST_SGED` | `Sged` | `"sged"` | `skew` ξ > 0, `shape` ν > 0 |
| 5 | `VE_DIST_NIG` | `Nig` | `"nig"` | `skew` ρ ∈ (−1, 1), `shape` ζ > 0 |
| 6 | `VE_DIST_GHYP` | `Ghyp` | `"ghyp"` | `skew` ρ ∈ (−1, 1), `shape` ζ > 0, `ghlambda` λ ∈ ℝ |
| 7 | `VE_DIST_JSU` | `Jsu` | `"jsu"` | `skew` ν ∈ ℝ, `shape` τ > 0 |

Os parâmetros de distribuição ficam no fim do vetor, na ordem `skew`, `shape`, `ghlambda` (a do `rugarch`). O maior layout passa a ser fGARCH-ALLGARCH, ARMA(6,6), (6,6), ghyp: `kMaxParameterCount = 6·kMaxOrder + 6 = 42`. O `Dual<40>` vira `Dual<42>` (o mais largo é sempre `kMaxParameterCount`), e os buckets ficam 8, 16, 32, 42.

**Densidades** (todas com média 0 e variância 1; z é o resíduo padronizado):

- **sstd, sged** (Fernandez–Steel, padronizada como no `rugarch`): g é o núcleo simétrico do 2a (std ou ged com o mesmo ν), m₁ = E|x| sob g (o `abs_moment(·, 1)` do 2b),
  μ_ξ = m₁(ξ − 1/ξ), σ_ξ² = (1 − m₁²)(ξ² + 1/ξ²) + 2m₁² − 1, y = μ_ξ + σ_ξ z,
  log f(z) = log(2/(ξ + 1/ξ)) + log σ_ξ + log g(y/ξ) para y ≥ 0, e log g(yξ) para y < 0.
  O bico fica em z* = −μ_ξ/σ_ξ. Com ξ = 1 a densidade é o núcleo.
- **nig, ghyp** (hiperbólica generalizada na parametrização (ρ, ζ), padronizada): com Rλ = K_{λ+1}(ζ)/K_λ(ζ) e Qλ = K_{λ+2}(ζ)/K_λ(ζ),
  δ = [Rλ/ζ + ρ²/(1 − ρ²)·(Qλ − Rλ²)]^{−1/2}, α = ζ/(δ√(1 − ρ²)), β = ρα, μ = −βδ²Rλ/ζ,
  log f(z) = log a + ((λ − ½)/2)·log(δ² + (z − μ)²) + log K_{λ−½}(α√(δ² + (z − μ)²)) + β(z − μ),
  log a = (λ/2)·log(α² − β²) − ½ log 2π − (λ − ½) log α − λ log δ − log K_λ(ζ).
  nig é ghyp com λ = −½ **fixo em `double`**: a ordem K₋₁ = K₁ não carrega derivada.
- **jsu** (Johnson SU, na parametrização de Rigby & Stasinopoulos): w = exp(1/τ²), ω = −ν/τ, c = [½(w − 1)(w cosh 2ω + 1)]^{−1/2},
  x = (z − c√w sinh ω)/c, r = −ν + τ asinh x,
  log f(z) = log τ − log c − ½ log(1 + x²) − ½ log 2π − r²/2.
  Hipótese: `skew` = ν e `shape` = τ do `rugarch`; a Task 9 confirma (inclusive o sinal de ν).

**Bessel K** (`dist/bessel`): `log K_ν(x)` usa `boost::math::cyl_bessel_k(|ν|, x)` (K é par em ν) com uma policy sem exceção (`overflow_error`, `underflow_error`, `domain_error` e `evaluation_error` todos `ignore_error`), de modo que um estouro dá ±inf ou NaN e nunca lança. Para x ≥ 700, onde o Boost entra em underflow, usa a expansão assintótica log K_ν(x) = ½ log(π/(2x)) − x + log(1 + Σ_{k≥1} aₖ), aₖ = aₖ₋₁·(4ν² − (2k − 1)²)/(8kx), até |aₖ| < 1e-17 ou 30 termos. Derivadas:
- em x: ∂ log K_ν/∂x = −K_{ν−1}(x)/K_ν(x) − ν/x, com a razão por diferença de logs;
- em ν (só ghyp): diferença central em log K com h = 1e-5·max(1, |ν|). É a única derivada aproximada do motor; o teste de gradiente a cobre com a tolerância de sempre.

**Momentos dos choques** (generalizam a regra do 2b):

- norm, std, ged: as formas fechadas do 2b, sem mudança.
- Assimétricas: E|z|^p, κ = E(|z| − η₁z)^λ e κ deslocado vêm todos de `shifted_moment` (η₂ = 0 nos dois primeiros). A quadratura parte em η₂ e no bico da densidade: 0 para norm, std, ged, nig, ghyp e jsu (como hoje), z* para sstd e sged.
- **P(z < 0)** (`probability_below_zero`): 0.5 exato nas simétricas; nas assimétricas, a mesma quadratura sobre (−∞, 0) com integrando f(z).
- Momento infinito: std **e sstd** com ν ≤ p (a cauda da sstd é a da t). O `tail_exponent` algébrico vale para as duas. nig, ghyp, sged e jsu têm todos os momentos.

**gjrGARCH com choque assimétrico** (π = P(z < 0)):

- Persistência P = Σα + Σβ + π Σγ.
- Pesos, nas posições alpha, beta, gamma: ((1 − π)α₁.., β.., π(α₁ + γ₁)..), e de volta αᵢ = w_{αᵢ}/(1 − π), γᵢ = w_{γᵢ}/π − αᵢ. Com π = 0.5 é a regra do 2b, bit a bit.
- Backcast: 1[ε < 0]ε² continua lendo b/2 para qualquer distribuição. É o que o `arch` faz (a constante não depende da distribuição), e a Task 9 confirma com o skew-t.

**eGARCH:** E|z| da distribuição da spec (quadratura nas assimétricas). O backcast continua lendo |z| = √(2/π), a convenção do `arch` registrada no 2b.

**Reparametrização** (irrestrito → natural):

| Parâmetro | Mapa | `ParameterKind` |
|---|---|---|
| `skew` sstd, sged | ξ = exp(x) | `Positive` |
| `skew` nig, ghyp | ρ = 2·logistic(x) − 1 | `Real` |
| `skew` jsu | identidade | `Real` |
| `shape` sstd | 2 + exp(x) | `Positive` |
| `shape` sged, nig, ghyp, jsu | exp(x) | `Positive` |
| `ghlambda` | identidade | `Real` |

Os mapas de distribuição rodam antes dos pesos, porque κ e π dependem deles.

**Valores iniciais:** sstd (ξ = 1, ν = 8), sged (ξ = 1, ν = 2), nig (ρ = 0, ζ = 2), ghyp (ρ = 0, ζ = 2, λ = −0.5), jsu (ν = 0, τ = 2). Todos começam na versão simétrica, e nig/ghyp com a curtose de uma t₈.

**Fronteira** (acréscimos à regra do 2b): |ρ| > 1 − 1e-8 marca `skew` de nig e ghyp; 1/shape < 1e-8 marca `shape` em std, sstd, nig, ghyp e jsu (os limites normais), e no ghyp marca também `ghlambda`, que deixa de ser identificado ali.

**Séries simuladas** (iguais no gerador Python e no helper C++; σ² inicial 1, 1000 observações de aquecimento descartadas, μ = 0.05):

| Série | Processo | Choques |
|---|---|---|
| `gjrgarch11_sstd` | gjrGARCH(1,1): ω = 0.01, α = 0.03, γ = 0.09, β = 0.9 | sstd ξ = 0.85, ν = 6 |
| `aparch11_sged` | apARCH(1,1): ω = 0.02, α = 0.08, γ = 0.4, β = 0.9, δ = 1.3 | sged ξ = 0.9, ν = 1.4 |
| `egarch11_jsu` | eGARCH(1,1): ω = −0.01, α = −0.08, γ = 0.15, β = 0.97 | jsu ν = −0.5, τ = 2 |
| `sgarch11_nig` | sGARCH(1,1): ω = 0.02, α = 0.08, β = 0.9 | nig ρ = −0.3, ζ = 1.5 |
| `sgarch11_ghyp` | sGARCH(1,1): ω = 0.02, α = 0.08, β = 0.9 | ghyp ρ = −0.3, ζ = 1.5, λ = 1 |

Sorteio dos choques: sstd e sged por |x| do núcleo com sinal +ξ (probabilidade ξ²/(1 + ξ²)) ou −1/ξ, depois z = (y − μ_ξ)/σ_ξ; jsu por z = c(√w sinh ω + sinh((N + ν)/τ)); nig como mistura normal–inversa gaussiana (C++: Michael–Schucany–Haas sobre o Box–Muller portável; Python: `numpy.random.Generator.wald`); ghyp só no Python, por `scipy.stats.genhyperbolic` com os (α, β, δ, μ) acima. O helper C++ não simula ghyp: os testes C++ do ghyp usam a série nig.

## Review Focus

1. **ghyp/nig em dados quase normais** (ζ̂ cresce sem limite): o argumento do Bessel passa de 700, o caminho assintótico assume, e o ajuste termina finito (`Converged`, `MaxIter` ou `Failed`), sem exceção. → testes nas Tasks 2 e 7.
2. **Bessel que estoura** (K_ν(x) para x minúsculo e ν grande, ou ghlambda derivando para |λ| ~ 50): `log_bessel_k` devolve +inf ou NaN sem lançar, o `loglik` sai não finito e a line search recua. Um termo +inf nunca vira melhora. → testes nas Tasks 2 e 5.
3. **sstd/sged muito assimétricas** (ξ ∈ {0.05, 20}): π, E|z| e κ seguem finitos e no intervalo certo (0 < π < 1), e o mapa ida e volta fecha. → testes nas Tasks 4 e 6.
4. **apARCH/fGARCH com sstd e ν ≤ λ:** κ infinito, como no std; `loglik` não finito e ajuste finito. → testes nas Tasks 4 e 5.
5. **Lote com k = 4 e k = 42** (ALLGARCH, ARMA(6,6), (6,6), ghyp) num só `ve_fit_batch`: bit-idêntico ao `fit_one` isolado, com 1 e 4 threads. → teste na Task 8.

---

## Mapa de arquivos

```
CMakeLists.txt                     Boost.Math (FetchContent, alvo INTERFACE SYSTEM), bessel.cpp
src/core/spec.hpp                  Dist com 8 valores; kMaxParameterCount = 42
src/ad/dual.hpp                    asinh, sinh, cosh
src/dist/symmetric.hpp             o conteúdo atual de densities.hpp (norm, std, ged), sem mudança de comportamento
src/dist/densities.hpp             ShapeParameters, Density (variant), make_density, log_density, in_domain
src/dist/skewed.hpp                sstd, sged
src/dist/hyperbolic.hpp            nig, ghyp
src/dist/johnson.hpp               jsu
src/dist/bessel.hpp/.cpp           log K_nu(x) com derivadas
src/dist/moments.hpp/.cpp          momentos sobre ShapeParameters; probability_below_zero; bico z*
src/model/layout.*                 blocos skew, shape, ghlambda; shape_parameters_of
src/model/likelihood.cpp           visita da Density; Dual<42>
src/model/parameter_map.*          mapas de distribuição; pesos do gjr com π; fronteira
src/model/start_values.cpp         valores iniciais das distribuições
src/fit/fit.cpp                    kWidestDual = kMaxParameterCount
src/capi/spec_api.cpp, include/volengine/volengine.h   distribuições novas
python/src/volengine/_spec.py      DISTRIBUTIONS
tests/...                          espelhando src/; séries, casos e fixtures novos
```

---

### Task 1: Distribuições no `Dist`, blocos de parâmetros no `Layout` e `Dual<42>`

**Files:**
- Modify: `src/core/spec.hpp`, `src/model/layout.hpp`, `src/model/layout.cpp`, `src/fit/fit.cpp`, e as instâncias `Dual<40>` → `Dual<42>` em `src/model/likelihood.cpp`, `src/model/parameter_map.cpp` e nos comentários "Instantiated only for"
- Create: `src/dist/symmetric.hpp` (move o conteúdo de `src/dist/densities.hpp`)
- Modify: `src/dist/densities.hpp` (passa a incluir `symmetric.hpp` e definir `ShapeParameters`)
- Test: `tests/unit/model/layout_test.cpp`

**Interfaces:**
- Produces:
  ```cpp
  namespace ve {
  enum class Dist : int32_t { Norm = 0, Std = 1, Ged = 2, Sstd = 3, Sged = 4, Nig = 5, Ghyp = 6, Jsu = 7 };
  constexpr std::size_t kMaxParameterCount = 6 * kMaxOrder + 6;
  }
  namespace ve::dist {
  template <class T> struct ShapeParameters { T skew{0.0}; T shape{0.0}; T ghlambda{0.0}; };   // ausentes = 0
  }
  namespace ve::model {
  struct Layout {   // campos do 2b, e:
      bool has_skew() const; std::size_t skew() const;          // sstd, sged, nig, ghyp, jsu
      bool has_shape() const; std::size_t shape() const;        // todas menos norm
      bool has_ghlambda() const; std::size_t ghlambda() const;  // ghyp
  };
  template <class T> dist::ShapeParameters<T> shape_parameters_of(std::span<const T> theta, const Layout&);
  }
  ```
- `shape_of` sai; quem o usava passa a ler `shape_parameters_of(...).shape`. `count()` soma os três blocos. `kWidestDual = kMaxParameterCount`, e o `static_assert` continua.

- [ ] **Step 1: Escrever os testes**

```cpp
TEST(Layout, DistributionBlocksFollowRugarchOrder) {
  // sGARCH(1,1) com média: sstd, sged, nig, jsu → mu omega alpha1 beta1 skew shape; ghyp → ... skew shape ghlambda
  // norm, std, ged: nomes do 2b sem mudança
}
TEST(Layout, WidestFitsWidestDual) { /* ALLGARCH, ARMA(6,6), (6,6), ghyp: count() == kMaxParameterCount == 42 */ }
TEST(Layout, DistributionKinds) {
  // skew: sstd, sged Positive; nig, ghyp, jsu Real. shape Positive em todas. ghlambda Real.
}
TEST(Layout, ShapeParametersOf) { /* ghyp: lê skew, shape, ghlambda das posições; norm: os três 0 */ }
```

- [ ] **Step 2: Rodar e ver falhar** — `cmake --build --preset gcc-debug && ctest --preset gcc-debug -R Layout` → não compila.
- [ ] **Step 3: Implementar.** `kind` ganha os casos da tabela "Reparametrização".
- [ ] **Step 4: Rodar e ver passar** — suíte inteira PASS, sem mudar nenhuma constante dos testes existentes.
- [ ] **Step 5: Commit** — `git commit -m "feat: add the skewed distributions to Dist, their parameter blocks and Dual<42>"`

---

### Task 2: Boost.Math e `log_bessel_k`

**Files:**
- Modify: `CMakeLists.txt`
- Create: `src/dist/bessel.hpp`, `src/dist/bessel.cpp`
- Modify: `src/ad/dual.hpp` (`asinh`, `sinh`, `cosh`, para a Task 3)
- Test: `tests/unit/dist/bessel_test.cpp`, `tests/unit/ad/dual_test.cpp`

**Interfaces:**
- CMake: `FetchContent_Declare(boost_math GIT_REPOSITORY https://github.com/boostorg/math GIT_TAG boost-1.87.0 SOURCE_SUBDIR do-not-build)`, e um alvo `INTERFACE` `ve_boost_math` com `${boost_math_SOURCE_DIR}/include` como `SYSTEM` e `BOOST_MATH_STANDALONE`. Só `volengine_core` o liga, como `PRIVATE`.
- Produces:
  ```cpp
  namespace ve::dist {
  double log_bessel_k(double order, double x);                               // regra "Bessel K"; nunca lança
  template <class T> T log_bessel_k(double order, const T& x);               // derivada só em x
  template <class T> T log_bessel_k(const T& order, const T& x);             // derivadas em order e x
  }
  namespace ve::ad { template <std::size_t N> Dual<N> asinh(const Dual<N>&); /* sinh, cosh idem */ }
  ```
- Só `bessel.cpp` inclui o Boost; o header declara as funções e define os templates sobre a de `double`.

- [ ] **Step 1: Escrever os testes**

```cpp
TEST(Bessel, MatchesClosedForms) {
  // K_{1/2}(x) = sqrt(pi/(2x)) e^-x; K_{3/2}(x) = K_{1/2}(x)(1 + 1/x); x ∈ {1e-3, 0.5, 3, 40, 699, 701, 5000}, ordem ± (rel 1e-13 em log)
  // K_0(1) = 0.42102443824070834, K_1(1) = 0.60190723019723457 (rel 1e-14)
}
TEST(Bessel, AsymptoticJoinsBoost) {
  // ordem ∈ {0, 1, 2.3, 6}: log K(700 + 1e-6) - log K(700 - 1e-6) fica a 1e-11 de 2e-6·∂ₓ log K(700)
}
TEST(Bessel, DerivativesMatchFiniteDifferences) {
  // Dual<8> em (order, x), order ∈ {-1, 0.3, 2.5}, x ∈ {0.2, 3, 900}: |grad - CD| ≤ 1e-6·max(1, |grad|)
  // a sobrecarga de ordem double dá grad em order == 0
}
TEST(Bessel, NeverThrows) {
  // Review Focus 2: log_bessel_k(50, 1e-300), (300, 1e-5), (0, 0), (1, -1), (NaN, 1): sem exceção; resultado +inf ou NaN
}
TEST(Dual, HyperbolicFunctions) { /* asinh, sinh, cosh: valor == std (bit a bit) e derivadas 1/sqrt(1+x^2), cosh, sinh (1e-15) */ }
```

- [ ] **Step 2: Rodar e ver falhar** — `ctest --preset gcc-debug -R "Bessel|Dual"` → não compila.
- [ ] **Step 3: Implementar.**
- [ ] **Step 4: Rodar e ver passar** — suíte inteira PASS; os builds MSVC/Clang do CI compilam o Boost sem aviso nosso (os dele ficam calados por `SYSTEM`).
- [ ] **Step 5: Commit** — `git commit -m "feat: add log Bessel K with derivatives on Boost.Math"`

---

### Task 3: Densidades sstd, sged, nig, ghyp e jsu

**Files:**
- Create: `src/dist/skewed.hpp`, `src/dist/hyperbolic.hpp`, `src/dist/johnson.hpp`
- Modify: `src/dist/densities.hpp`
- Test: `tests/unit/dist/densities_test.cpp`, `tests/unit/dist/reference_quadrature.hpp`

**Interfaces:**
- Consumes: `ShapeParameters` (Task 1), `log_bessel_k`, `ad::asinh` (Task 2), `LogDensity` e `abs_moment` do 2b.
- Produces:
  ```cpp
  namespace ve::dist {
  template <class T> struct SkewedLogDensity { LogDensity<T> kernel; T log_constant; T shift; T stretch; T skew; };
  template <class T> struct HyperbolicLogDensity { T log_constant; T alpha; T beta; T delta2; T location; T order; bool free_order; };
  template <class T> struct JohnsonLogDensity { T log_constant; T center; T scale; T skew; T shape; };
  template <class T> using Density = std::variant<LogDensity<T>, SkewedLogDensity<T>, HyperbolicLogDensity<T>, JohnsonLogDensity<T>>;
  template <class T> Density<T> make_density(Dist dist, const ShapeParameters<T>& parameters);
  template <class T> T log_density(const Density<T>& density, const T& z);   // visita; o loglik visita fora do laço
  template <class T> T density_kink(const Density<T>& density);              // 0, ou z* nas Fernandez–Steel
  bool in_domain(Dist dist, const ShapeParameters<double>& parameters);       // tabela "Distribuições"
  }
  ```
- Fórmulas: regra "Densidades". `make_log_density` do 2b continua existindo e é o que `make_density` devolve para norm/std/ged.

- [ ] **Step 1: Escrever os testes**

```cpp
// Casos: sstd (0.7, 5), (1.4, 12); sged (0.8, 1.2), (1.3, 2.5); nig (-0.5, 0.8), (0.4, 3);
//        ghyp (-0.3, 1.5, 1), (0.6, 0.7, -2.5); jsu (-0.8, 1.5), (0.6, 3)
TEST_P(SkewedDensities, StandardizedMoments) {
  // trapézio de reference_quadrature (com cauda da t na sstd, partido no bico):
  // ∫f = 1, ∫z f = 0, ∫z² f = 1 (abs 1e-9)
}
TEST(SkewedDensities, ReduceToKnownForms) {
  // sstd(ξ=1, ν) == std(ν) e sged(ξ=1, ν) == ged(ν) em 50 pontos de [-8, 8] (rel 1e-14)
  // ghyp(ρ, ζ, λ=-0.5) == nig(ρ, ζ) (rel 1e-12); ρ = 0 e jsu ν = 0 simétricas: f(z) == f(-z) (rel 1e-14)
}
TEST(SkewedDensities, SkewSign) { /* ξ < 1, ρ < 0 e jsu ν > 0 põem mais massa abaixo de 0: ∫_{-∞}^0 f > 0.5 */ }
TEST(SkewedDensities, DerivativesMatchFiniteDifferences) {
  // log_density<Dual<8>> semeado em (z, skew, shape, ghlambda), nos casos acima e z ∈ {-3, -0.2, 0.4, 5}:
  // |grad - CD| ≤ 1e-6·max(1, |grad|)
}
TEST(SkewedDensities, Domains) { /* in_domain: sstd ν = 2 falso, 2.01 verdadeiro; ξ ≤ 0 falso; |ρ| ≥ 1 falso; τ ≤ 0 falso; ghlambda qualquer finito */ }
TEST(SkewedDensities, FarTailsStayFinite) {
  // Review Focus 1: nig (0, 2000) e ghyp (0.2, 5000, 3) em z ∈ {-40, 0, 40}: log_density finito (caminho assintótico)
}
```

- [ ] **Step 2: Rodar e ver falhar** — `ctest --preset gcc-debug -R SkewedDensities` → não compila.
- [ ] **Step 3: Implementar.** Constantes por avaliação ficam no struct; `log_density` faz só o que depende de z.
- [ ] **Step 4: Rodar e ver passar** — suíte inteira PASS.
- [ ] **Step 5: Commit** — `git commit -m "feat: add the sstd, sged, nig, ghyp and jsu densities"`

---

### Task 4: Momentos para choques assimétricos

**Files:**
- Modify: `src/dist/moments.hpp`, `src/dist/moments.cpp`
- Test: `tests/unit/dist/moments_test.cpp`

**Interfaces:**
- Consumes: `Density`, `make_density`, `density_kink` (Task 3).
- Produces (as três do 2b trocam `shape` por `ShapeParameters`; a forma fechada segue para norm/std/ged):
  ```cpp
  namespace ve::dist {
  template <class T> T abs_moment(Dist dist, const ShapeParameters<T>& shape, const T& power);
  template <class T> T asymmetric_moment(Dist dist, const ShapeParameters<T>& shape, const T& eta1, const T& power);
  template <class T> T shifted_moment(Dist dist, const ShapeParameters<T>& shape, const T& eta1, const T& eta2, const T& power);
  template <class T> T probability_below_zero(Dist dist, const ShapeParameters<T>& shape);
  }
  ```
- Regra "Momentos dos choques". O integrando do 2b passa a guardar `Density<T>` e visita uma vez por nó. A quadratura parte em η₂ e em `density_kink`; para norm/std/ged o ponto é 0, o de hoje, e os números do 2b não mudam.

- [ ] **Step 1: Escrever os testes**

```cpp
TEST(Moments, SymmetricUnchanged) { /* os testes do 2b passam sem mudar constantes (só a chamada troca de tipo) */ }
TEST(Moments, SkewedMatchReferenceQuadrature) {
  // os casos da Task 3 × power ∈ {0.8, 1, 1.5, 2} × eta1 ∈ {-0.5, 0, 0.7}: abs_moment e asymmetric_moment contra o
  // trapézio (rel 1e-6); abs_moment(power = 2) == 1 (rel 1e-10)
}
TEST(Moments, ProbabilityBelowZero) {
  // norm, std, ged: == 0.5 exato; casos da Task 3: contra o trapézio (abs 1e-9), e 0 < π < 1
  // Review Focus 3: sstd e sged com ξ ∈ {0.05, 20}: π, E|z| e κ(eta1=0.3, power=1.5) finitos, 0 < π < 1
}
TEST(Moments, SkewedStudentInfiniteAtOrAbovePower) { /* Review Focus 4: sstd ν = 2.5, power 2.7 → +inf em abs, asymmetric e shifted */ }
TEST(Moments, SkewedDerivativesMatchFiniteDifferences) {
  // Dual<8> em (skew, shape, ghlambda, eta1, eta2, power) para as quatro funções, sstd (0.8, 6), nig (-0.4, 1.2),
  // ghyp (0.3, 2, 1.5), jsu (0.5, 2): |grad - CD| ≤ 1e-6·max(1, |grad|)
}
```

- [ ] **Step 2: Rodar e ver falhar** — `ctest --preset gcc-debug -R Moments` → não compila.
- [ ] **Step 3: Implementar.** Se 64 nós não fecharem 1e-6 em alguma assimétrica, aumente os nós só para ela e registre em `docs/performance/benchmarks.md`; a tolerância não afrouxa.
- [ ] **Step 4: Rodar e ver passar** — suíte inteira PASS.
- [ ] **Step 5: Commit** — `git commit -m "feat: take shock moments and P(z < 0) under the skewed distributions"`

---

### Task 5: `loglik` com as distribuições novas

**Files:**
- Modify: `src/model/likelihood.cpp`
- Test: `tests/unit/model/likelihood_test.cpp`

**Interfaces:**
- Consumes: Tasks 1, 3, 4. Assinatura de `loglik` sem mudança.
- `sum_of_terms` visita a `Density` uma vez; norm/std/ged seguem pelos ramos de hoje (z² ou |z|), e as novas leem z = e/√σ². O guard de domínio usa `in_domain(dist, ShapeParameters)`. Um total não finito (inclusive +inf) vira `kInvalid`, checado uma vez no fim, sem desvio a mais no laço. eGARCH lê E|z| de `abs_moment`; apARCH/fGARCH leem κ de `asymmetric_moment`/`shifted_moment`; gjrGARCH não muda (π só entra na persistência).

- [ ] **Step 1: Escrever os testes**

```cpp
TEST(Likelihood, SkewedHandCase) {
  // r = {0.1, -0.2, 0.3}, sGARCH(1,1) sem média, sample init: loglik == Σ [log_density(z_t) - ½ log σ²_t],
  // com σ²_t da recursão à mão do 2a e log_density da Task 3, para as cinco distribuições (rel 1e-14)
}
TEST(Likelihood, NonFiniteTermIsInvalid) {
  // Review Focus 2: ghyp com ζ = 1e-300 e λ = 40, onde os log K estouram: loglik == -inf, nunca +inf nem NaN, sem exceção
}
TEST(Likelihood, PowerSkewedStudentNeedsShapeAbovePower) { /* Review Focus 4: apARCH δ = 2.5, sstd ν = 2.4 → !isfinite; ν = 3 → finito */ }
// GradientMatchesFiniteDifferences, DoubleAndDualAgreeOnValue e TermsSumToTotalAndScoresSumToGradient passam a percorrer
// as 8 distribuições × {SGarch, EGarch, GjrGarch, ApArch, FGarchAllGarch} × {(1,1)} × ARMA {(0,0),(1,1)} × {sample, backcast};
// θ à mão até a Task 6, depois pelo esquema to_natural(u) do 2b.
```

- [ ] **Step 2: Rodar e ver falhar** — `ctest --preset gcc-debug -R Likelihood` → FAIL.
- [ ] **Step 3: Implementar.**
- [ ] **Step 4: Rodar e ver passar** — suíte C++ inteira PASS; `Likelihood.MatchesPlanOneSimulatedValues` sem mudar constantes.
- [ ] **Step 5: Commit** — `git commit -m "feat: evaluate the likelihood under the skewed distributions"`

---

### Task 6: Reparametrização, validade, fronteira e valores iniciais

**Files:**
- Modify: `src/model/parameter_map.hpp`, `src/model/parameter_map.cpp`, `src/model/start_values.cpp`
- Test: `tests/unit/model/parameter_map_test.cpp`, `tests/unit/model/start_values_test.cpp`

**Interfaces:**
- Consumes: Tasks 1, 4. Mesmas assinaturas do 2b.
- Regras "Reparametrização", "gjrGARCH com choque assimétrico", "Valores iniciais" e "Fronteira". `is_valid` usa `in_domain(dist, ShapeParameters)` e a persistência com π. `shock_moment` passa os `ShapeParameters`.

- [ ] **Step 1: Escrever os testes**

```cpp
TEST(ParameterMap, SymmetricUnchanged) { /* testes do 2a e 2b sem mudança, inclusive DualMatchesDouble bit a bit */ }
TEST(ParameterMap, RoundTripAllDistributions) {
  // 14 modelos × (1,1) × ARMA {(0,0),(1,1)} × as 5 novas: θ interior → u → θ' (rel 1e-12)
  // Review Focus 3: sstd e sged com ξ ∈ {0.05, 20} em gjrGARCH e apARCH: idem
}
TEST(ParameterMap, AlwaysValidSkewed) { /* 2000 u por layout (mt19937_64(5), N(0,3)) → is_valid, as 5 novas × 14 modelos */ }
TEST(ParameterMap, GjrPersistenceUsesProbabilityBelowZero) {
  // sstd (0.7, 6): Σα + Σβ + π Σγ == logistic(v)(1 - 1e-8) (1e-14), com π = probability_below_zero
}
TEST(ParameterMap, DistributionBoundaries) {
  // nig ρ = 1 - 1e-9 marca skew; sstd ν = 1e9 marca shape; ghyp ζ = 1e9 marca shape e ghlambda; jsu τ = 1e9 marca shape
}
TEST(StartValues, Distributions) { /* tabela "Valores iniciais" (1e-15) e is_valid(start_values) em todo layout */ }
```

- [ ] **Step 2: Rodar e ver falhar** — `ctest --preset gcc-debug -R "ParameterMap|StartValues"` → FAIL.
- [ ] **Step 3: Implementar**, e trocar os θ à mão da Task 5 pelo esquema `to_natural(u)`.
- [ ] **Step 4: Rodar e ver passar** — suíte inteira PASS.
- [ ] **Step 5: Commit** — `git commit -m "feat: map, validate and start the skewed distribution parameters"`

---

### Task 7: Ajuste e erros-padrão com as distribuições novas

**Files:**
- Modify: `tests/unit/simulated_garch.hpp` (choques sstd, sged, nig, jsu, regra "Séries simuladas"; implementação independente da biblioteca)
- Modify (só se um teste pedir): `src/fit/fit.cpp`, `src/fit/standard_errors.cpp`
- Test: `tests/unit/fit/fit_test.cpp`, `tests/unit/fit/standard_errors_test.cpp`

**Interfaces:**
- Consumes: Tasks 1–6. `fit_one`, `standard_errors` e `FitResult` sem mudança de assinatura. Os passos da Hessiana saem de `Layout::kind` (Task 1).

- [ ] **Step 1: Escrever os testes**

```cpp
// T = 5000, seeds 41..44; ghyp ajustado na série nig
TEST_P(FitDistributions, BeatsTrueParameters) {
  // gjrgarch11_sstd, aparch11_sged, egarch11_jsu, sgarch11_nig com o próprio modelo e dist, e sGARCH-ghyp na série nig
  // (verdadeiro: λ = -0.5): Converged; ℓ(θ̂) ≥ ℓ(θ verdadeiro) - 1e-6·|ℓ|
}
TEST(FitDistributions, RecoversShapeParameters) {
  // ξ ±0.05, ν sstd ±1.5, ν sged ±0.15, ρ ±0.08, ζ ±0.5, jsu ν ±0.15, τ ±0.3; pesos e demais com as tolerâncias do 2b
}
TEST(FitDistributions, SymmetricLimitMatchesKernel) { /* série sGARCH(1,1)-std ν = 6 do helper (seed 45): ℓ̂_sstd ≥ ℓ̂_std - 1e-6·|ℓ| (sstd contém std) */ }
TEST(FitDistributions, NearNormalEndsFinite) {
  // Review Focus 1: nig e ghyp em série sGARCH-norm (simulate_portable_garch_series): status ∈ {Converged, MaxIter, Failed},
  // sem exceção; se Converged, estimate e loglik finitos
}
TEST(FitDistributions, DeterministicWidestBucket) { /* k = 42 (ALLGARCH, ARMA(6,6), (6,6), ghyp, T = 600, max_iter = 30): fit_one duas vezes bit-idêntico */ }
TEST(StandardErrors, SkewedFinite) { /* os ajustes de BeatsTrueParameters: se_hessian e se_robust finitos e > 0 em todos os parâmetros */ }
```

- [ ] **Step 2: Rodar e ver falhar** — `ctest --preset gcc-debug -R "FitDistributions|StandardErrors"` → FAIL (simuladores ausentes).
- [ ] **Step 3: Implementar** os simuladores e o que os testes pedirem no driver. Se uma tolerância de recuperação não fechar, registre o valor observado no relatório da task antes de mexer.
- [ ] **Step 4: Rodar e ver passar** — suíte C++ inteira PASS, `-Wnrvo` sem aviso.
- [ ] **Step 5: Commit** — `git commit -m "test: fit every model under the skewed distributions"`

---

### Task 8: Interface C e Python

**Files:**
- Modify: `include/volengine/volengine.h`, `src/capi/spec_api.cpp`, `python/src/volengine/_spec.py`
- Test: `tests/unit/capi/spec_test.cpp`, `tests/unit/capi/batch_test.cpp`, `python/tests/test_fit.py`

**Interfaces:**
- Produces (C, só acréscimos): `#define VE_DIST_SSTD 3` … `VE_DIST_JSU 7`, com `static_assert` contra `ve::Dist`. No header, junto de cada valor, os parâmetros, os domínios e a parametrização; no sstd, a conversão para o λ de Hansen do `arch` (regra das Global Constraints). `ve_spec_set_dist` aceita 0..7; fora disso, `VE_INVALID_ARGUMENT` com "dist" na mensagem.
- Python: `DISTRIBUTIONS = {"norm": 0, "std": 1, "ged": 2, "sstd": 3, "sged": 4, "nig": 5, "ghyp": 6, "jsu": 7}`; a docstring de `Spec` lista as distribuições, os nomes `skew`/`shape`/`ghlambda` e a conversão do sstd para o `arch`.

- [ ] **Step 1: Escrever os testes**

```cpp
TEST(CapiSpec, AllDistributions) { /* set_dist(0..7) OK; 8 e -1 → INVALID_ARGUMENT com "dist" */ }
TEST(CapiFit, MixedBucketsWithWidest) {
  // Review Focus 5 (estende MixedBucketsInOneBatch): specs {sGARCH(1,1)-norm k=4, gjrGARCH(1,1)-sstd, ALLGARCH ARMA(6,6)-(6,6)-ghyp k=42}
  // × 3 séries T=600, max_iter=30, threads 1 e 4: estimates/loglik/sigma bit-idênticos a ve::fit_one
}
TEST(CapiFit, DistributionParamNames) { /* ghyp(1,1): ... skew shape ghlambda via ve_fit_param_name */ }
```
```python
@pytest.mark.parametrize(("dist", "extra"), [("sstd", ("skew", "shape")), ("nig", ("skew", "shape")),
                                             ("ghyp", ("skew", "shape", "ghlambda")), ("jsu", ("skew", "shape"))])
def test_distribution_names(ctx, series, dist, extra): ...   # names == ("mu", "omega", "alpha1", "beta1", *extra)
def test_unknown_distribution(): ...                         # Spec(dist="ghst") → ValueError com a lista das 8
```

- [ ] **Step 2: Rodar e ver falhar** — `ctest --preset gcc-debug -R "CapiSpec|CapiFit"` e pytest → FAIL.
- [ ] **Step 3: Implementar.**
- [ ] **Step 4: Rodar e ver passar** — suítes C++ e Python inteiras PASS; header C11-limpo; `mypy --strict` dos casos de tipo PASS.
- [ ] **Step 5: Commit** — `git commit -m "feat: expose the skewed distributions through the C API and Python"`

---

### Task 9: Oráculos para as distribuições novas

**Files:**
- Modify: `tests/reference/generate_data.py`, `tests/reference/cases.json`, `tests/reference/oracle_rugarch.R`, `tests/reference/oracle_arch.py`, `tests/reference/README.md`, `python/tests/test_oracles.py`, `docs/guide/validation.md`, `docs/superpowers/specs/2026-11-10-volengine-design.md` (seção 9: o sstd não tem opção do `arch`, e por quê)
- Create (gerados, commitados): `tests/reference/data/grid_2c.csv`, fixtures novas em `tests/reference/fixtures/`

**Interfaces:**
- Consumes: pacote Python (Task 8). Pré-requisitos: R 4.6.2 + `rugarch` 1.5.6 e o venv com `arch` 8.0.0. Se faltar algum, pare e peça ao usuário.
- Dados: `grid_2c.csv` com as cinco séries da tabela "Séries simuladas", T = 3000, de um só `numpy.random.default_rng(20261113)`, na ordem da tabela.
- Casos (fixados):
  - `rugarch`: cada série com o próprio modelo em (1,1) × {sstd, sged, nig, ghyp, jsu} (25 casos); na série `aparch11_sged`, fGARCH-ALLGARCH(1,1) com sstd e com jsu (η₂ deslocado sob choque assimétrico).
  - `arch` (backcast), com `SkewStudent`: `GARCH(p=1, q=1)`, `GARCH(p=1, o=1, q=1)` e `EGARCH(p=1, o=1, q=1)` na série `gjrgarch11_sstd`. `oracle_arch.py` converte λ → ξ e o SE pelo delta method (regra das Global Constraints); no EGARCH, o ω como no 2b, com E|z| da sstd ajustada.
  - Regra de interior do 2b, mais: |ρ| < 0.999, ξ ∈ (0.05, 20), 1/ν, 1/ζ, 1/τ > 1e-4, |ghlambda| < 10. Caso que falhe é trocado, com o motivo em `validation.md` ("Case selection").
- Tolerâncias: as do 2a, por grandeza, sem afrouxar.
- Convenções a confirmar e documentar em `validation.md`:
  - a padronização Fernandez–Steel do `rugarch` (μ_ξ, σ_ξ) para sstd e sged;
  - a parametrização (ρ, ζ) de nig e ghyp e o nome `ghlambda`;
  - `skew` = ν e `shape` = τ da jsu, e o sinal de ν;
  - P(z < 0) na persistência do gjrGARCH, E|z| do eGARCH e κ do apARCH sob as assimétricas;
  - que o skew-t do `arch` dá o mesmo ℓ com ξ = √((1 + λ)/(1 − λ)), e o b/2 do backcast do GJR com ele.

  Se o oráculo divergir, a correção vai no núcleo (Tasks 3–6), com os testes de gradiente mantidos, e a mudança é registrada.

- [ ] **Step 1: Gerar dados e fixtures** — `python/.venv/Scripts/python tests/reference/generate_data.py && Rscript tests/reference/oracle_rugarch.R && python/.venv/Scripts/python tests/reference/oracle_arch.py` → um resultado por caso e oráculo; os scripts param com erro em não-convergência ou fronteira.
- [ ] **Step 2: Estender o teste** — `test_matches_oracle` e `test_fixture_estimates_are_interior` cobrem os casos novos, com as asserções do 2a.
- [ ] **Step 3: Rodar** — `VOLENGINE_LIB=<abs> python/.venv/Scripts/python -m pytest python/tests/test_oracles.py -v` → PASS; depois de cada correção no núcleo, as suítes C++ e Python inteiras.
- [ ] **Step 4: Commit** — `git commit -m "test: validate the skewed distributions against rugarch and arch"`

---

### Task 10: Benchmarks e documentação

**Files:**
- Modify: `tests/benchmarks/fit_benchmarks.cpp`, `python/benchmarks/bench_vs_arch.py`, `docs/performance/benchmarks.md`, `docs/performance/python-vs-arch.md` (nota de "não medido"), `README.md`, `CHANGELOG.md`, `ARCHITECTURE.md`

**Interfaces:**
- Benchmarks novos (nomes fixados; os existentes não mudam):
  - `BM_Loglik/<serie>/{double,dual}` e `BM_FitOne/<serie>`, T = 5000, para as cinco séries de "Séries simuladas" com o próprio modelo e dist (a ghyp na série nig, como na Task 7).
  - `BM_LogBesselK/{fixed_order,free_order}` sobre 5000 argumentos de uma série nig, para o plano 5 decidir sobre o Bessel próprio.
  - `BM_TccGridFull/threads:hw`: o painel do `BM_TccGrid` × 14 modelos × {p, q ∈ 1..3} × ARMA {(0,0), (1,1)} × as 8 distribuições = 40320 ajustes numa chamada `ve_fit_batch`, a carga inteira do TCC no univariado. Contadores: ajustes/s, por status e por distribuição. `SkipWithError` com zero trabalho ou se alguma distribuição tiver mais de 5% fora de `CONVERGED`/`MAX_ITER`.
  - Python: uma linha `sGARCH(1,1)-sstd` contra `arch` com `SkewStudent`, no mesmo formato das existentes.
- Nesta máquina só compilar e rodar curto: `--benchmark_min_time=0.01s --benchmark_filter="BM_Loglik|BM_LogBesselK"`, sem publicar nada.

- [ ] **Step 1: Escrever os benchmarks** — `cmake --preset gcc-bench && cmake --build --preset gcc-bench` → sem aviso; a rodada curta termina sem `SkipWithError`.
- [ ] **Step 2: Documentação** — README (distribuições suportadas e um exemplo com `dist="sstd"`), CHANGELOG (`[Unreleased]`), ARCHITECTURE (`dist/` com a variante `Density`, o Boost.Math só em `bessel.cpp`, `Dual<42>`), `benchmarks.md` com a nota de casos ainda não medidos.
- [ ] **Step 3: Verificar** — `ctest --preset gcc-release` e pytest inteiros PASS; `check_style.py .` sai 0; o CI compila os benchmarks.
- [ ] **Step 4: Commit** — `git commit -m "bench: add benchmarks for the skewed distributions and document plan 2c"`

---

## Pendente na máquina de benchmark

Na máquina do `docs/performance/benchmarks.md` (AMD Ryzen 7 5700X), depois deste plano e antes do plano 3, junto com o que ficou do 2b:

1. **Rodar** os benchmarks novos e o `BM_TccGridFull`, e atualizar `benchmarks.md` com máquina, data e commit. O tempo do `BM_TccGridFull` é a referência da meta de < 10 s do spec (univariado).
2. **Profile** — AMD uProf sobre `BM_TccGridFull` e `BM_FitOne` de nig e ghyp: quanto do tempo é Bessel, e quanto é a quadratura dos momentos.
3. **Decidir pelos dados** (plano 5 executa): Bessel K próprio se `BM_LogBesselK` e o profile mostrarem que domina; derivada em ν analítica se a diferença central do ghyp pesar.
