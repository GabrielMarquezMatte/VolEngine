# VolEngine — Plano 1: fundação e fatia vertical sGARCH(1,1)-norm

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Um único modelo (sGARCH(1,1), média constante, distribuição normal) estimado de ponta a ponta — núcleo C++23 → interface C → Python — e validado contra `rugarch` e `arch`, com o contrato, o lote, os painéis irregulares e o thread pool já na forma final.

**Architecture:** Núcleo C++23 interno em `src/`, exposto só pela interface C `include/volengine/volengine.h` (handles opacos, setters, `ve_status`, sem structs públicas). Python via ctypes sobre a DLL. O lote é uma lista de trabalhos (especificação, série) executada no thread pool do `ve_context`; cada ajuste é escalar e single-thread, então o resultado não depende do número de threads.

**Tech Stack:** C++23 (GCC ≥ 15, MSVC VS 2022 17.9+/VS 18, Clang com libc++ ≥ 18), CMake ≥ 3.28 + Ninja + `CMakePresets.json`, FetchContent, GoogleTest v1.17.0, BS::thread_pool v5.0.0, Python ≥ 3.11 + numpy + pytest, `arch` (oráculo), R + `rugarch` (oráculo).

**Spec:** `docs/superpowers/specs/2026-11-10-volengine-design.md`

## Roteiro dos planos

Este é o plano 1 de 6. Cada plano entrega software testável e é escrito quando o anterior termina.

1. **Fundação + sGARCH(1,1)-norm de ponta a ponta** (este).
2. Amplitude univariada: todas as variantes, ordens p,q, ARMA, todas as distribuições, opções do `rugarch` (fixos, iniciais, variance targeting, `mxreg`/`vxreg`, `archm`, ARFIMA).
3. Diagnósticos, forecast e simulação (Philox) univariados.
4. DCC, aDCC, FDCC, Copula-DCC: ajuste, erros-padrão em dois estágios, forecast e simulação.
5. Performance: L-BFGS em lote com SIMD (Highway), Bessel K, benchmark da carga do TCC.
6. `volengine.hpp` (wrapper C++23), wheels/PyPI, release, abidiff, preparação da 1.0.

## Global Constraints

- A interface C é o único contrato estável. Nenhuma struct pública; configuração só por setters sobre handles opacos.
- Prefixo `ve_` para funções e tipos, `VE_` para macros/constantes. Tipos de tamanho fixo (`int32_t`, `size_t`, `ptrdiff_t`, `double`).
- Enums da interface C são `typedef int32_t` + `#define`; valores só crescem por acréscimo, nunca são renumerados.
- Toda função que pode falhar retorna `ve_status`; mensagem detalhada em `ve_last_error()`, thread-local.
- Nenhuma exceção atravessa a interface C: toda função `extern "C"` captura tudo (`std::bad_alloc` → `VE_OUT_OF_MEMORY`, resto → `VE_INTERNAL_ERROR`).
- Sem estado global: thread pool e configuração de execução vivem no `ve_context`.
- Resultados (`ve_fit`) imutáveis; liberar um handle nunca invalida outro; a especificação é copiada no ajuste.
- Falha de convergência de um ajuste não falha o lote; erro de argumento falha a chamada inteira.
- NaN no meio de uma série é erro de argumento.
- Ajustes bit-idênticos para qualquer número de threads na mesma máquina e ISA.
- Convenção padrão segue `rugarch`; a do `arch` é opção.
- Critérios e diagnósticos são funções sobre o resultado, nunca campos.
- Só `double`.
- Anti-objetivo: não fazer mais do que o necessário. Nada fora do que este plano lista.
- O código do `rugarch` (GPL-3) pode ser lido para entender convenções, nunca copiado ou traduzido.
- Commits sem trailer `Co-Authored-By`.
- Windows/MinGW: a DLL linka `libstdc++`, `libgcc` e `winpthread` estaticamente, para que o Python do Windows a carregue sem DLLs do MSYS2 no PATH.
- Python dos testes: um venv em `python/.venv` criado com o Python 3.14 do Windows (`py -3.14 -m venv python/.venv`, ou `uv venv --python 3.14 python/.venv`), nunca com o Python do MSYS2 que está primeiro no PATH (`C:\msys64\ucrt64\bin\python.exe`, sem wheels de numpy). `arch` é instalado nesse venv.

## Review Focus

1. **Série curta demais** (comprimento < número de parâmetros + 2, inclusive 0 num painel irregular): o ajuste daquela série sai com `VE_FIT_INSUFFICIENT_DATA`, sem crash, e os outros ajustes do lote seguem normais. → teste em Task 7.
2. **Série constante** (variância zero): o ajuste sai com `VE_FIT_FAILED`, sem NaN propagando para outros ajustes nem crash. → teste em Task 5.
3. **Painel em ordem Fortran ou com stride negativo** (numpy transposto ou `[::-1]`): resultado idêntico ao do mesmo dado contíguo. → teste em Task 7 (C) e Task 8 (Python).
4. **Índices fora do alcance** (job aponta para spec ou série inexistente; início + comprimento além de T): `VE_INVALID_ARGUMENT` com mensagem, nenhum handle produzido. → teste em Task 7.
5. **Handle usado depois de liberar outro e leitura concorrente**: liberar a spec, as opções, o painel e o contexto não invalida os `ve_fit`; vários threads lendo o mesmo `ve_fit` ao mesmo tempo. → teste em Task 7.

---

## Mapa de arquivos

```
CMakeLists.txt                     alvo `volengine` (SHARED), testes, FetchContent
CMakePresets.json                  gcc-debug, gcc-release, msvc-release
.gitignore  .editorconfig
include/volengine/volengine.h      interface C (o contrato)
src/capi/export.cpp                todas as funções extern "C"; fronteira de exceções
src/capi/error.hpp/.cpp            last_error thread-local, set_error, guard
src/capi/handles.hpp               definições de ve_context/ve_spec/ve_fit_options/ve_panel/ve_fit
src/core/context.hpp/.cpp          ve::Context (thread pool)
src/core/spec.hpp                  ve::Spec, ve::FitConfig (valores copiáveis)
src/core/panel.hpp                 ve::PanelView (ponteiro, T, N, strides, ranges)
src/garch/sgarch.hpp/.cpp          recursão, loglik, gradiente, scores (normal)
src/optim/lbfgs.hpp                L-BFGS escalar (template)
src/optim/transform.hpp/.cpp       reparametrização restrito ↔ irrestrito + jacobiano
src/fit/fit_result.hpp             ve::FitResult
src/fit/fit.hpp/.cpp               driver do ajuste, Hessiana, erros-padrão
src/fit/criteria.hpp/.cpp          loglik, aic, bic, hqic, sic
src/fit/batch.hpp/.cpp             execução do lote no pool
tests/unit/*.cpp                   GoogleTest
tests/unit/c_header_check.c        compila o header como C11 puro
python/pyproject.toml
python/src/volengine/__init__.py, _native.py, api.py
python/tests/test_*.py
tests/reference/                   gerador de dados, scripts dos oráculos, fixtures
docs/guide/validation.md           tolerâncias e convenções
README.md ARCHITECTURE.md STYLEGUIDE.md CHANGELOG.md LICENSE CITATION.cff
.github/workflows/ci.yml
```

## Convenções numéricas deste plano (fixadas)

- Parâmetros sGARCH(1,1)-norm na ordem e com os nomes do `rugarch`: `mu`, `omega`, `alpha1`, `beta1` (sem `mu` se `include_mean = 0`).
- Resíduo: εₜ = rₜ − μ. Recursão: σ²ₜ = ω + α ε²ₜ₋₁ + β σ²ₜ₋₁.
- `VE_VARINIT_SAMPLE` (padrão, `rugarch`): m̄ = média de ε² sobre a amostra toda (com o μ corrente); σ²₀ = m̄; a loglik soma t = 0..T−1. O executor confirma no `rugarch` (`R/rugarch-sgarch.R` e `src/filters.c`, só leitura) se o termo t = 0 entra na soma e se ε²₋₁ = m̄; o fixture da Task 9 decide.
- `VE_VARINIT_BACKCAST` (`arch`): τ = min(75, T); wᵢ = 0.94ⁱ / Σ 0.94ʲ, i = 0..τ−1; b = Σ wᵢ ε²ᵢ; σ²₀ = ω + α b + β b.
- Densidade normal: ℓₜ = −½ (log 2π + log σ²ₜ + ε²ₜ / σ²ₜ).
- Critérios por observação (padrão, `rugarch`, `VE_IC_PER_OBS`), n = observações, k = parâmetros estimados:
  AIC = (−2ℓ + 2k)/n; BIC = (−2ℓ + k log n)/n; HQIC = (−2ℓ + 2k log log n)/n; SIC = −2ℓ/n + log((n + 2k)/n).
  `VE_IC_TOTAL` (`arch`): os mesmos multiplicados por n (SIC: −2ℓ + n log((n + 2k)/n)).
- p-valor: 2 (1 − Φ(|t|)).
- Erro-padrão Hessiana: raiz da diagonal de (−H)⁻¹, H = Hessiana da loglik no espaço natural. Robusto: H⁻¹ (SᵀS) H⁻¹, S = scores por observação (T×k).

---

### Task 1: Esqueleto de build, header C e versão da ABI

**Files:**
- Create: `CMakeLists.txt`, `CMakePresets.json`, `.gitignore`, `.editorconfig`
- Create: `include/volengine/volengine.h`
- Create: `src/capi/export.cpp`
- Test: `tests/unit/abi_test.cpp`, `tests/unit/c_header_check.c`

**Interfaces:**
- Produces: alvo CMake `volengine` (SHARED, `VE_BUILDING` definido ao compilar); alvo `volengine_tests` (GoogleTest, `gtest_discover_tests`); macro `VE_API` (dllexport/visibility default); `#define VE_ABI_VERSION 0`; `VE_API int32_t ve_abi_version(void);`

- [ ] **Step 1: Escrever os testes**

```cpp
// tests/unit/abi_test.cpp
#include <gtest/gtest.h>
#include <volengine/volengine.h>
TEST(Abi, RuntimeVersionMatchesHeader) { EXPECT_EQ(ve_abi_version(), VE_ABI_VERSION); }
```

```c
/* tests/unit/c_header_check.c — compilado como C11 num alvo OBJECT `volengine_c_header_check` */
#include <volengine/volengine.h>
int ve_c_header_check(void) { return ve_abi_version(); }
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `cmake --preset gcc-debug && cmake --build --preset gcc-debug`
Expected: falha de configuração/compilação (arquivos ausentes).

- [ ] **Step 3: Implementar**

CMake: `cmake_minimum_required(VERSION 3.28)`, `project(VolEngine VERSION 0.1.0 LANGUAGES C CXX)`, `CMAKE_CXX_STANDARD 23` obrigatório, `CMAKE_C_STANDARD 11`, warnings como erro (`-Wall -Wextra -Wpedantic -Werror` / `/W4 /WX`), `CMAKE_CXX_VISIBILITY_PRESET hidden`. FetchContent de GoogleTest `v1.17.0` só quando `VE_BUILD_TESTS` (padrão ON no top-level). Em MinGW: `target_link_options(volengine PRIVATE -static-libgcc -static-libstdc++ -Wl,-Bstatic,--whole-archive -lwinpthread -Wl,--no-whole-archive,-Bdynamic)`. Presets: `gcc-debug`, `gcc-release` (Ninja, `g++`/`gcc`), `msvc-release` (generator `Visual Studio 18 2026`); `buildPresets` e `testPresets` correspondentes. `.gitignore`: `build/`, `python/.venv/`, `__pycache__/`, `*.pyc`, `.pytest_cache/`. Header com guarda, `extern "C"` sob `__cplusplus`, `<stdint.h>`, `<stddef.h>`.

- [ ] **Step 4: Rodar e ver passar**

Run: `cmake --build --preset gcc-debug && ctest --preset gcc-debug`
Expected: `Abi.RuntimeVersionMatchesHeader` PASS; `c_header_check.c` compila sem aviso.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "build: add CMake skeleton, C header and ABI version"
```

---

### Task 2: Status, erro thread-local e `ve_context`

**Files:**
- Modify: `include/volengine/volengine.h`, `src/capi/export.cpp`, `CMakeLists.txt`
- Create: `src/capi/error.hpp`, `src/capi/error.cpp`, `src/capi/handles.hpp`, `src/core/context.hpp`, `src/core/context.cpp`
- Test: `tests/unit/context_test.cpp`

**Interfaces:**
- Produces (C):
  ```c
  typedef int32_t ve_status;
  #define VE_OK 0
  #define VE_INVALID_ARGUMENT 1
  #define VE_INVALID_HANDLE 2
  #define VE_UNSUPPORTED 3
  #define VE_OUT_OF_MEMORY 4
  #define VE_INTERNAL_ERROR 5
  typedef struct ve_context ve_context;
  VE_API const char* ve_last_error(void);          /* "" se não houve erro; válido até a próxima chamada ve_* nesta thread */
  VE_API ve_status ve_context_create(ve_context** out);
  VE_API ve_status ve_context_set_threads(ve_context* ctx, int32_t threads); /* 0 = hardware_concurrency; < 0 inválido */
  VE_API ve_status ve_context_threads(const ve_context* ctx, int32_t* out);
  VE_API void ve_context_free(ve_context* ctx);    /* NULL é no-op */
  ```
- Produces (C++ interno): `ve::capi::set_error(ve_status, std::string_view) -> ve_status`; `ve::capi::guard(F&&) -> ve_status` (limpa o erro, chama F, converte exceções); `ve::Context` com `BS::light_thread_pool& pool()` e `int threads() const`.

- [ ] **Step 1: Escrever os testes**

```cpp
TEST(Context, CreateDefaultsToHardwareThreads) {
  ve_context* c = nullptr;
  ASSERT_EQ(ve_context_create(&c), VE_OK);
  int32_t n = 0; ASSERT_EQ(ve_context_threads(c, &n), VE_OK);
  EXPECT_EQ(n, static_cast<int32_t>(std::max(1u, std::thread::hardware_concurrency())));
  ve_context_free(c);
}
TEST(Context, SetThreads) { /* set 3 → threads()==3; set 0 → hardware_concurrency */ }
TEST(Context, NegativeThreadsIsInvalidArgument) {
  /* ve_context_set_threads(c, -1) == VE_INVALID_ARGUMENT; ve_last_error() contém "threads" */
}
TEST(Context, NullOutIsInvalidArgument) { EXPECT_EQ(ve_context_create(nullptr), VE_INVALID_ARGUMENT); }
TEST(Context, NullHandleIsInvalidHandle) { int32_t n; EXPECT_EQ(ve_context_threads(nullptr, &n), VE_INVALID_HANDLE); }
TEST(Context, SuccessClearsLastError) { /* provoca erro, depois chamada OK → ve_last_error() == "" */ }
TEST(Context, LastErrorIsPerThread) { /* erro na thread A; std::thread B lê "" */ }
TEST(Context, FreeNullIsNoop) { ve_context_free(nullptr); }
```

- [ ] **Step 2: Rodar e ver falhar** — `ctest --preset gcc-debug -R Context` → não compila.

- [ ] **Step 3: Implementar**

FetchContent de `https://github.com/bshoshany/thread-pool` tag `v5.0.0` (só o include). `ve::Context` guarda `std::unique_ptr<BS::light_thread_pool>`; `set_threads` recria o pool. `ve_last_error` guarda `thread_local std::string`.

- [ ] **Step 4: Rodar e ver passar** — `ctest --preset gcc-debug -R Context` → PASS.

- [ ] **Step 5: Commit** — `git commit -m "feat: add status codes, thread-local errors and ve_context"`

---

### Task 3: Recursão sGARCH(1,1), loglik normal, gradiente e scores

**Files:**
- Create: `src/garch/sgarch.hpp`, `src/garch/sgarch.cpp`
- Test: `tests/unit/sgarch_test.cpp`

**Interfaces:**
- Produces:
  ```cpp
  namespace ve::garch {
  enum class VarInit { Sample, Backcast };
  struct SGarchConfig { bool include_mean = true; VarInit var_init = VarInit::Sample; };
  std::size_t sgarch_param_count(const SGarchConfig&);            // 4 ou 3
  // theta no espaço natural, na ordem mu?, omega, alpha1, beta1.
  // sigma2 (T), grad (k), scores (T×k, row-major): spans vazios = não calcular.
  // Retorna a loglik; não finita se σ²ₜ ≤ 0 em algum t.
  double sgarch_norm_loglik(std::span<const double> r, std::span<const double> theta,
                            const SGarchConfig& cfg, std::span<double> sigma2,
                            std::span<double> grad, std::span<double> scores);
  }
  ```

- [ ] **Step 1: Escrever os testes**

```cpp
TEST(SGarch, SampleInitRecursionMatchesHand) {
  std::vector<double> r{0.1, -0.2, 0.3}, s2(3);
  std::vector<double> th{0.0, 0.1, 0.1, 0.8};
  double ll = ve::garch::sgarch_norm_loglik(r, th, {}, s2, {}, {});
  EXPECT_NEAR(s2[0], 0.14 / 3, 1e-12);
  EXPECT_NEAR(s2[1], 0.1 + 0.1 * 0.01 + 0.8 * (0.14 / 3), 1e-12);
  EXPECT_NEAR(s2[2], 0.1 + 0.1 * 0.04 + 0.8 * s2[1], 1e-12);
  double expect = 0; for (int t = 0; t < 3; ++t)
    expect += -0.5 * (std::log(2 * std::numbers::pi) + std::log(s2[t]) + r[t] * r[t] / s2[t]);
  EXPECT_NEAR(ll, expect, 1e-12);
}
TEST(SGarch, BackcastInitMatchesArchFormula) { /* r de 100 obs fixas; σ²₀ == ω + (α+β)·b com b pela fórmula das convenções */ }
TEST(SGarch, NoMeanHasThreeParams) { EXPECT_EQ(ve::garch::sgarch_param_count({.include_mean = false}), 3u); }
TEST(SGarch, GradientMatchesCentralDifferences) {
  /* para Sample e Backcast, com e sem mean: série de 500 obs gerada com std::mt19937_64(42);
     |grad_i − (ℓ(θ+hᵢ)−ℓ(θ−hᵢ))/2hᵢ| ≤ 1e-6·max(1,|grad_i|), hᵢ = 1e-6·max(1,|θᵢ|) */
}
TEST(SGarch, ScoresSumToGradient) { /* Σₜ scores[t,i] == grad[i] com tolerância 1e-9·max(1,|grad_i|) */ }
TEST(SGarch, NonPositiveVarianceGivesNonFinite) { /* omega = -1 → !std::isfinite(ll) */ }
```

- [ ] **Step 2: Rodar e ver falhar** — `ctest --preset gcc-debug -R SGarch` → não compila.

- [ ] **Step 3: Implementar** a recursão direta e a recursão das derivadas ∂σ²ₜ/∂θ = ∂ω/∂θ + α ∂ε²ₜ₋₁/∂θ + ε²ₜ₋₁ ∂α/∂θ + β ∂σ²ₜ₋₁/∂θ + σ²ₜ₋₁ ∂β/∂θ, incluindo a dependência de σ²₀ (via m̄ ou b) em μ, ω, α, β. Uma passada sobre t.

- [ ] **Step 4: Rodar e ver passar** — `ctest --preset gcc-debug -R SGarch` → PASS.

- [ ] **Step 5: Commit** — `git commit -m "feat: add sGARCH(1,1) normal likelihood with analytic gradient"`

---

### Task 4: Reparametrização e L-BFGS escalar

**Files:**
- Create: `src/optim/transform.hpp`, `src/optim/transform.cpp`, `src/optim/lbfgs.hpp`
- Test: `tests/unit/optim_test.cpp`

**Interfaces:**
- Produces:
  ```cpp
  namespace ve::optim {
  enum class Stop { Converged, MaxIter, LineSearchFailed, NonFinite };
  struct LbfgsOptions { int max_iter = 500; int memory = 10; double gtol = 1e-6; double ftol = 1e-12; };
  struct LbfgsResult { Stop stop; int iterations; double f; };
  // f(x, g) devolve o valor e preenche g; minimiza in-place em x.
  template <class F> LbfgsResult lbfgs_minimize(F&& f, std::span<double> x, const LbfgsOptions&);
  // sGARCH: natural (mu?, ω, α, β) ↔ irrestrito (mu?, u, v, w).
  void sgarch_to_natural(std::span<const double> u, std::span<double> theta, bool include_mean);
  void sgarch_to_unconstrained(std::span<const double> theta, std::span<double> u, bool include_mean);
  // grad_u = Jᵀ grad_theta
  void sgarch_chain_gradient(std::span<const double> u, std::span<const double> grad_theta,
                             std::span<double> grad_u, bool include_mean);
  }
  ```
- Transformação (fixada): ω = exp(u); s = σ(v)·(1 − 1e-8) (persistência, σ = logística); a = σ(w); α = s·a; β = s·(1 − a). μ não é transformado.
- Convergência: ‖g‖∞ ≤ gtol·max(1, |f|), ou |fₖ − fₖ₊₁| ≤ ftol·max(1, |f|). Busca linear com condições de Wolfe fortes (c₁ = 1e-4, c₂ = 0.9).

- [ ] **Step 1: Escrever os testes**

```cpp
TEST(Lbfgs, MinimizesRosenbrock) { /* x0 = {-1.2, 1}; stop == Converged; |x−(1,1)|∞ < 1e-5 */ }
TEST(Lbfgs, MinimizesQuadratic10d) { /* f = Σ i·xᵢ²; converge em ≤ 50 iterações para x≈0 (1e-8) */ }
TEST(Lbfgs, ReportsMaxIter) { /* Rosenbrock, max_iter = 3 → stop == MaxIter, iterations == 3 */ }
TEST(Lbfgs, ReportsNonFinite) { /* f devolve NaN no ponto inicial → stop == NonFinite */ }
TEST(Transform, RoundTrip) { /* θ = {0.05, 0.01, 0.08, 0.9} → u → θ' ; |θ−θ'|∞ < 1e-12 */ }
TEST(Transform, AlwaysStationaryAndPositive) { /* 1000 u aleatórios: ω>0, α≥0, β≥0, α+β<1 */ }
TEST(Transform, ChainGradientMatchesFiniteDifferences) { /* g(u) = Σ cᵢ θᵢ(u); compara Jᵀc com diferenças centrais, 1e-7 */ }
```

- [ ] **Step 2: Rodar e ver falhar** — `ctest --preset gcc-debug -R "Lbfgs|Transform"` → não compila.
- [ ] **Step 3: Implementar** (two-loop recursion de Nocedal–Wright, Alg. 7.4; busca linear Alg. 3.5/3.6).
- [ ] **Step 4: Rodar e ver passar** — mesmo comando → PASS.
- [ ] **Step 5: Commit** — `git commit -m "feat: add scalar L-BFGS and sGARCH reparametrization"`

---

### Task 5: Driver de ajuste, erros-padrão e critérios

**Files:**
- Create: `src/core/spec.hpp`, `src/fit/fit_result.hpp`, `src/fit/fit.hpp`, `src/fit/fit.cpp`, `src/fit/criteria.hpp`, `src/fit/criteria.cpp`
- Test: `tests/unit/fit_test.cpp`

**Interfaces:**
- Consumes: `sgarch_norm_loglik`, `sgarch_param_count` (Task 3); `lbfgs_minimize`, transformações (Task 4).
- Produces:
  ```cpp
  namespace ve {
  enum class Model : int32_t { SGarch = 0 };
  enum class Dist : int32_t { Norm = 0 };
  struct Spec { Model model = Model::SGarch; int p = 1, q = 1; Dist dist = Dist::Norm; bool include_mean = true; };
  struct FitConfig { int max_iter = 500; double tol = 1e-6; garch::VarInit var_init = garch::VarInit::Sample; };
  enum class FitStatus : int32_t { Converged = 0, MaxIter = 1, Failed = 2, InsufficientData = 3 };
  struct FitResult {
    FitStatus status; Spec spec; FitConfig config;
    std::vector<std::string_view> names;           // literais estáticos
    std::vector<double> estimate, se_hessian, se_robust;   // k; NaN se Hessiana não invertível
    double loglik; std::size_t nobs;
    std::vector<double> sigma, residuals, std_residuals, fitted;  // T; σ, não σ²
  };
  FitResult fit_one(std::span<const double> r, const Spec&, const FitConfig&);
  enum class IcScale { PerObs, Total };
  double aic(const FitResult&, IcScale); double bic(const FitResult&, IcScale);
  double hqic(const FitResult&, IcScale); double sic(const FitResult&, IcScale);
  }
  ```
- Valores iniciais (fixados): μ = média(r); persistência 0.95 com α = 0.05, β = 0.90; ω = var(r)·(1 − 0.95).
- Hessiana: diferenças centrais do gradiente analítico no espaço natural, passo hᵢ = 1e-5·max(|θᵢ|, 1e-3), simetrizada. Comprimento < k + 2 → `InsufficientData` sem otimizar. Loglik não finita no ponto inicial ou parada `NonFinite`/`LineSearchFailed` sem ponto finito → `Failed`.

- [ ] **Step 1: Escrever os testes**

```cpp
// série simulada no próprio teste: sGARCH(1,1), μ=0.05, ω=0.01, α=0.08, β=0.9, T=5000, mt19937_64(7)
TEST(Fit, RecoversSimulatedParameters) { /* Converged; |α̂−0.08|<0.02; |β̂−0.9|<0.03; |μ̂−0.05|<0.03 */ }
TEST(Fit, NamesFollowRugarch) { /* names == {"mu","omega","alpha1","beta1"}; sem mean → {"omega","alpha1","beta1"} */ }
TEST(Fit, StandardErrorsPositiveAndRobustDiffers) { /* se_hessian > 0, se_robust > 0, não todos iguais */ }
TEST(Fit, FittedSeriesConsistent) {
  /* sigma.size()==T; residuals[t] == r[t]−μ̂; std_residuals[t]*sigma[t] == residuals[t] (1e-12); fitted[t]==μ̂ */
}
TEST(Fit, InsufficientData) { /* r com 5 obs → InsufficientData; estimate vazio ou NaN; sem crash */ }
TEST(Fit, ConstantSeriesFails) { /* r = 1000×0.3 → Failed; sem exceção */ }
TEST(Criteria, MatchFormulas) {
  /* FitResult sintético: loglik=-100, k=4, n=200 → aic(PerObs)=(200+8)/200; bic=(200+4·log 200)/200;
     hqic=(200+8·log log 200)/200; sic=200/200+log((200+8)/200); Total = PerObs·n (sic: 200+200·log(208/200)) */
}
TEST(Fit, Deterministic) { /* duas chamadas → estimate bit-idêntico */ }
```

- [ ] **Step 2: Rodar e ver falhar** — `ctest --preset gcc-debug -R "Fit|Criteria"` → não compila.
- [ ] **Step 3: Implementar** `fit_one` e os critérios em `src/fit/`.
- [ ] **Step 4: Rodar e ver passar** — mesmo comando → PASS.
- [ ] **Step 5: Commit** — `git commit -m "feat: add sGARCH fit driver, standard errors and information criteria"`

---

### Task 6: Interface C de especificação, opções e painel

**Files:**
- Modify: `include/volengine/volengine.h`, `src/capi/export.cpp`, `src/capi/handles.hpp`
- Create: `src/core/panel.hpp`
- Test: `tests/unit/capi_spec_test.cpp`

**Interfaces:**
- Consumes: `ve::Spec`, `ve::FitConfig` (Task 5); `guard`, `set_error` (Task 2).
- Produces (C):
  ```c
  #define VE_MODEL_SGARCH 0
  #define VE_DIST_NORM 0
  #define VE_VARINIT_SAMPLE 0
  #define VE_VARINIT_BACKCAST 1
  typedef struct ve_spec ve_spec;
  typedef struct ve_fit_options ve_fit_options;
  typedef struct ve_panel ve_panel;
  VE_API ve_status ve_spec_create(int32_t model, ve_spec** out);
  VE_API ve_status ve_spec_set_order(ve_spec* s, int32_t p, int32_t q);
  VE_API ve_status ve_spec_set_dist(ve_spec* s, int32_t dist);
  VE_API ve_status ve_spec_set_include_mean(ve_spec* s, int32_t include_mean); /* 0/1 */
  VE_API void ve_spec_free(ve_spec* s);
  VE_API ve_status ve_fit_options_create(ve_fit_options** out);
  VE_API ve_status ve_fit_options_set_max_iter(ve_fit_options* o, int32_t max_iter);  /* ≥ 1 */
  VE_API ve_status ve_fit_options_set_tolerance(ve_fit_options* o, double tol);       /* finito, > 0 */
  VE_API ve_status ve_fit_options_set_var_init(ve_fit_options* o, int32_t var_init);
  VE_API void ve_fit_options_free(ve_fit_options* o);
  /* Painel emprestado: não copia `data`; os dados precisam viver até o fim de cada ve_fit_batch que o usa.
     Elemento (t, j) em data[t*stride_t + j*stride_n]; strides em elementos, podem ser negativos. */
  VE_API ve_status ve_panel_create(const double* data, size_t t, size_t n,
                                   ptrdiff_t stride_t, ptrdiff_t stride_n, ve_panel** out);
  /* Opcional: série j ocupa as linhas starts[j] .. starts[j]+lengths[j]-1. Copia os arrays. */
  VE_API ve_status ve_panel_set_ranges(ve_panel* p, const size_t* starts, const size_t* lengths);
  VE_API void ve_panel_free(ve_panel* p);
  ```
- Neste plano, só `VE_MODEL_SGARCH`, `VE_DIST_NORM` e ordem (1,1) são aceitos: outros valores conhecidos do enum futuro ainda não existem, então qualquer outro valor é `VE_INVALID_ARGUMENT`, e ordem ≠ (1,1) com p,q ≥ 1 é `VE_UNSUPPORTED`.
- Validação no `ve_panel_set_ranges`: starts[j] + lengths[j] ≤ T, senão `VE_INVALID_ARGUMENT` e o painel não muda. NaN é verificado no `ve_fit_batch` (Task 7), sobre o intervalo de cada série usada.

- [ ] **Step 1: Escrever os testes**

```cpp
TEST(CapiSpec, CreateAndConfigure) { /* create SGARCH, set_order(1,1), set_dist(NORM), include_mean(0) → todos VE_OK */ }
TEST(CapiSpec, UnknownModelIsInvalid) { /* ve_spec_create(99, &s) == VE_INVALID_ARGUMENT; s == nullptr */ }
TEST(CapiSpec, OrderOtherThanOneOneIsUnsupported) { /* set_order(2,1) == VE_UNSUPPORTED; (0,1) == VE_INVALID_ARGUMENT */ }
TEST(CapiOptions, Validation) { /* max_iter 0 → INVALID; tol −1, NaN → INVALID; var_init 7 → INVALID */ }
TEST(CapiPanel, RangesOutOfBoundsRejected) { /* T=10, starts={5}, lengths={6} → INVALID_ARGUMENT */ }
TEST(CapiPanel, NullDataRejected) { /* data == nullptr com t·n > 0 → INVALID_ARGUMENT */ }
TEST(CapiHandles, NullHandleIsInvalidHandle) { /* ve_spec_set_dist(nullptr, 0) == VE_INVALID_HANDLE */ }
TEST(CapiHandles, FreeNullIsNoop) { ve_spec_free(nullptr); ve_fit_options_free(nullptr); ve_panel_free(nullptr); }
```

- [ ] **Step 2: Rodar e ver falhar** — `ctest --preset gcc-debug -R "Capi"` → não compila.
- [ ] **Step 3: Implementar.**
- [ ] **Step 4: Rodar e ver passar** — mesmo comando → PASS.
- [ ] **Step 5: Commit** — `git commit -m "feat: add C API for specs, fit options and panels"`

---

### Task 7: `ve_fit_batch` e acessores do resultado

**Files:**
- Modify: `include/volengine/volengine.h`, `src/capi/export.cpp`, `src/capi/handles.hpp`
- Create: `src/fit/batch.hpp`, `src/fit/batch.cpp`
- Test: `tests/unit/capi_fit_test.cpp`

**Interfaces:**
- Consumes: `ve::fit_one`, critérios (Task 5); `ve::Context` (Task 2); handles da Task 6.
- Produces (C):
  ```c
  #define VE_FIT_CONVERGED 0
  #define VE_FIT_MAX_ITER 1
  #define VE_FIT_FAILED 2
  #define VE_FIT_INSUFFICIENT_DATA 3
  #define VE_SE_HESSIAN 0
  #define VE_SE_ROBUST 1
  #define VE_IC_PER_OBS 0
  #define VE_IC_TOTAL 1
  typedef struct ve_fit ve_fit;
  /* Trabalho i ajusta specs[job_spec[i]] na série job_series[i]. options NULL = padrões.
     Em sucesso, fits_out[i] recebe um handle novo por trabalho; em erro, nenhum handle é criado. */
  VE_API ve_status ve_fit_batch(ve_context* ctx, const ve_spec* const* specs, size_t n_specs,
                                const ve_panel* panel, const size_t* job_spec, const size_t* job_series,
                                size_t n_jobs, const ve_fit_options* options, ve_fit** fits_out);
  VE_API ve_status ve_fit_status(const ve_fit* f, int32_t* out);
  VE_API ve_status ve_fit_nobs(const ve_fit* f, size_t* out);
  VE_API ve_status ve_fit_param_count(const ve_fit* f, size_t* out);
  VE_API ve_status ve_fit_param_name(const ve_fit* f, size_t i, const char** out); /* string estática */
  /* Arrays de param_count elementos; qualquer um pode ser NULL. */
  VE_API ve_status ve_fit_coef(const ve_fit* f, int32_t se_kind, double* estimate, double* std_error,
                               double* t_value, double* p_value);
  VE_API ve_status ve_loglik(const ve_fit* f, double* out);
  VE_API ve_status ve_aic(const ve_fit* f, int32_t scale, double* out);
  VE_API ve_status ve_bic(const ve_fit* f, int32_t scale, double* out);
  VE_API ve_status ve_hqic(const ve_fit* f, int32_t scale, double* out);
  VE_API ve_status ve_sic(const ve_fit* f, int32_t scale, double* out);
  /* Visões válidas enquanto o handle existir. */
  VE_API ve_status ve_fit_sigma(const ve_fit* f, const double** data, size_t* len);
  VE_API ve_status ve_fit_residuals(const ve_fit* f, const double** data, size_t* len);
  VE_API ve_status ve_fit_std_residuals(const ve_fit* f, const double** data, size_t* len);
  VE_API ve_status ve_fit_fitted(const ve_fit* f, const double** data, size_t* len);
  VE_API void ve_fit_free(ve_fit* f);
  ```
- Execução: copia cada série para um buffer contíguo, roda `fit_one` por trabalho em `pool.submit_blocks(0, n_jobs, …, n_blocks)` com n_blocks = min(n_jobs, 8 × threads), espera todos. Toda validação (índices, NaN no intervalo de cada série usada, ponteiros) acontece antes de submeter. Mensagens de erro (fixadas): índice inválido contém `"job"`; NaN contém `"NaN"`, a série e a linha.
- `ve_loglik`/critérios num ajuste com status `FAILED` ou `INSUFFICIENT_DATA` devolvem `VE_OK` com NaN.

- [ ] **Step 1: Escrever os testes**

```cpp
// Fixture: 3 séries simuladas (como na Task 5, seeds 1,2,3), T=3000, matriz C-order.
TEST(CapiFit, CartesianBatchMatchesFitOne) { /* 2 specs (com/sem mean) × 3 séries = 6 jobs; estimate de cada == ve::fit_one, bit-idêntico */ }
TEST(CapiFit, ResultIndependentOfThreadCount) { /* threads=1 vs threads=4 → estimate, loglik, sigma bit-idênticos */ }
TEST(CapiFit, FortranOrderAndNegativeStrideMatchContiguous) { /* mesmos dados em F-order e com stride_t negativo → bit-idêntico */ }
TEST(CapiFit, RaggedPanel) { /* starts={0,1000,2990}, lengths={3000,2000,10}: nobs 3000/2000/10; série 3 → INSUFFICIENT_DATA; outras CONVERGED */ }
TEST(CapiFit, ZeroLengthSeries) { /* lengths[j]=0 → INSUFFICIENT_DATA, sem crash */ }
TEST(CapiFit, JobIndexOutOfRange) { /* job_spec={5} → INVALID_ARGUMENT; fits_out[0] permanece nullptr; ve_last_error() menciona "job" */ }
TEST(CapiFit, NaNInsideUsedRangeRejected) { /* NaN em (100, 1) → INVALID_ARGUMENT; NaN fora do intervalo da série num painel irregular → OK */ }
TEST(CapiFit, AccessorsConsistent) {
  /* param_count == 4; nomes; coef com SE_HESSIAN: t == est/se, p == 2(1−Φ(|t|)) (1e-12);
     ve_aic(PER_OBS)·nobs == ve_aic(TOTAL) (1e-9); sigma len == nobs */
}
TEST(CapiFit, HandlesOutliveInputs) { /* libera spec, options, panel, contexto e o buffer de dados; depois lê sigma e coef do ve_fit */ }
TEST(CapiFit, ConcurrentReads) { /* 8 std::thread lendo coef/sigma do mesmo ve_fit 1000×; resultados iguais */ }
TEST(CapiFit, FailedFitAccessorsReturnNaN) { /* série constante → status FAILED; ve_loglik == VE_OK e isnan */ }
```

- [ ] **Step 2: Rodar e ver falhar** — `ctest --preset gcc-debug -R CapiFit` → não compila.
- [ ] **Step 3: Implementar.**
- [ ] **Step 4: Rodar e ver passar** — `ctest --preset gcc-debug` (suíte inteira) → PASS.
- [ ] **Step 5: Commit** — `git commit -m "feat: add batched fit and result accessors to the C API"`

---

### Task 8: Pacote Python

**Files:**
- Create: `python/pyproject.toml`, `python/src/volengine/__init__.py`, `python/src/volengine/_native.py`, `python/src/volengine/api.py`, `python/src/volengine/py.typed`
- Test: `python/tests/conftest.py`, `python/tests/test_fit.py`

**Interfaces:**
- Consumes: interface C das Tasks 2, 6, 7.
- Produces (Python):
  ```python
  class Context:      def __init__(self, threads: int = 0) -> None
                      def fit(self, spec: Spec | Sequence[Spec], data: ArrayLike, *, options: FitOptions | None = None,
                              starts: ArrayLike | None = None, lengths: ArrayLike | None = None) -> Fit | np.ndarray
  class Spec:         def __init__(self, model: str = "sgarch", *, p: int = 1, q: int = 1, dist: str = "norm",
                                   include_mean: bool = True) -> None
  class FitOptions:   def __init__(self, *, max_iter: int = 500, tol: float = 1e-6, var_init: str = "sample") -> None
  class Fit:          status: str            # "converged" | "max_iter" | "failed" | "insufficient_data"
                      nobs: int; names: tuple[str, ...]
                      sigma, residuals, std_residuals, fitted: np.ndarray   # views read-only, zero-copy
                      def coef(self, se: str = "hessian") -> CoefTable      # "hessian" | "robust"
  class CoefTable(NamedTuple): names: tuple[str, ...]; estimate: np.ndarray; std_error: np.ndarray
                               t_value: np.ndarray; p_value: np.ndarray
  class VolEngineError(Exception): status: int
  def loglik(fit: Fit) -> float
  def aic(fit: Fit, *, per_obs: bool = True) -> float   # idem bic, hqic, sic
  ```
- Formas de retorno de `Context.fit`: spec única + 1D → `Fit`; spec única + 2D (T, N) → `np.ndarray` de objetos (N,); lista + 1D → (n_specs,); lista + 2D → (n_specs, N), produto cartesiano.
- A DLL é carregada de `VOLENGINE_LIB` (caminho do arquivo) ou de `volengine/_lib/`. Arrays de entrada são `float64`; strides do numpy convertidos de bytes para elementos (não contíguo não é copiado). Arrays de saída são views sobre os ponteiros do `ve_fit`, com o objeto `Fit` mantido vivo pela base da view e `writeable = False`. Status ≠ `VE_OK` → `VolEngineError` com `ve_last_error()`.
- Ambiente: venv conforme Global Constraints; `python/.venv/Scripts/pip install -e python[test]` (extras `test` = `pytest`, `numpy`).

- [ ] **Step 1: Escrever os testes**

```python
def test_single_spec_1d_returns_fit(ctx, series): assert isinstance(ctx.fit(ve.Spec(), series[:, 0]), ve.Fit)
def test_grid_shape(ctx, series):  # series: (3000, 3)
    fits = ctx.fit([ve.Spec(), ve.Spec(include_mean=False)], series); assert fits.shape == (2, 3)
def test_views_zero_copy_and_readonly(ctx, series):
    f = ctx.fit(ve.Spec(), series[:, 0]); assert not f.sigma.flags.writeable; assert f.sigma.shape == (3000,)
def test_view_survives_fit_reference_drop(ctx, series):
    s = ctx.fit(ve.Spec(), series[:, 0]).sigma; import gc; gc.collect(); assert np.isfinite(s).all()
def test_fortran_and_reversed_match(ctx, series):
    a = ctx.fit(ve.Spec(), series); b = ctx.fit(ve.Spec(), np.asfortranarray(series))
    assert all(np.array_equal(x.coef().estimate, y.coef().estimate) for x, y in zip(a, b))
def test_ragged(ctx, series):
    fits = ctx.fit(ve.Spec(), series, starts=[0, 1000, 2990], lengths=[3000, 2000, 10])
    assert [f.nobs for f in fits] == [3000, 2000, 10]; assert fits[2].status == "insufficient_data"
def test_criteria_scale(ctx, series):
    f = ctx.fit(ve.Spec(), series[:, 0]); assert ve.aic(f, per_obs=False) == pytest.approx(ve.aic(f) * f.nobs)
def test_nan_raises(ctx, series):
    s = series.copy(); s[10, 0] = np.nan
    with pytest.raises(ve.VolEngineError, match="NaN"): ctx.fit(ve.Spec(), s)
def test_unknown_dist_raises(): 
    with pytest.raises(ValueError): ve.Spec(dist="cauchy")
```
`conftest.py`: `series` = 3 séries geradas como na Task 5 em numpy (`np.random.default_rng(1..3)`), `ctx` = `ve.Context(threads=2)`.

- [ ] **Step 2: Rodar e ver falhar** — `python/.venv/Scripts/python -m pytest python/tests -q` → ImportError.
- [ ] **Step 3: Implementar.** Em `_native.py`, declarar `argtypes`/`restype` de toda função usada; o comentário do topo diz que o arquivo espelha `include/volengine/volengine.h`.
- [ ] **Step 4: Rodar e ver passar** — `VOLENGINE_LIB=build/gcc-debug/libvolengine.dll python/.venv/Scripts/python -m pytest python/tests -q` → PASS.
- [ ] **Step 5: Commit** — `git commit -m "feat: add Python package over the C API"`

---

### Task 9: Fixtures dos oráculos e comparação

**Files:**
- Create: `tests/reference/generate_data.py`, `tests/reference/oracle_rugarch.R`, `tests/reference/oracle_arch.py`, `tests/reference/README.md`
- Create (gerados, commitados): `tests/reference/data/sgarch11_norm.csv`, `tests/reference/fixtures/rugarch_sgarch11_norm.json`, `tests/reference/fixtures/arch_sgarch11_norm.json`
- Create: `docs/guide/validation.md`
- Test: `python/tests/test_oracles.py`

**Interfaces:**
- Consumes: pacote Python (Task 8).
- Formato do fixture (fixado): `{"oracle": str, "oracle_version": str, "series": [str], "results": [{"series": str, "loglik": float, "params": {name: float}, "se": {name: float}, "robust_se": {name: float}, "sigma": [float], "aic": float, "bic": float, "hqic": float | null, "sic": float | null}]}`. Os critérios ficam na escala do próprio oráculo.
- Dados: 3 séries sGARCH(1,1)-norm, μ=0.05, ω=0.01, α=0.08, β=0.9, T = 5000, 3000 e 1500, `np.random.default_rng(20261110)`, escritas em CSV com 17 dígitos significativos (série mais curta com células vazias no fim).
- `rugarch`: `ugarchspec(variance.model=list(model="sGARCH", garchOrder=c(1,1)), mean.model=list(armaOrder=c(0,0), include.mean=TRUE), distribution.model="norm")`, `ugarchfit(solver="hybrid")`; `infocriteria()` dá AIC/BIC/SIC/HQIC por observação. Comparação com `var_init="sample"`.
- `arch`: `arch_model(y, mean="Constant", vol="GARCH", p=1, q=1, dist="normal", rescale=False).fit(disp="off")`; SE Hessiana em `fit(cov_type="classic")`, robusta em `cov_type="robust"`; `aic`/`bic` totais. Comparação com `var_init="backcast"` e `per_obs=False`.
- Tolerâncias (fixadas; registradas em `docs/guide/validation.md`): loglik — `ll_ve ≥ ll_oracle − 1e-6·|ll_oracle|` e `|ll_ve − ll_oracle| ≤ 1e-4·|ll_oracle|`; parâmetros — relativo 1e-3 (absoluto 1e-5 se |θ| < 1e-2); SE e SE robusto — relativo 5e-2; σₜ — relativo máximo 1e-3; critérios — absoluto 1e-6 na escala do oráculo após conversão.
- **Pré-requisito:** R ≥ 4.4 com `rugarch` (e `jsonlite`) instalado, e `arch` no venv. Se o R não estiver disponível, o executor para nesta task e pede ao usuário para instalar o R ou rodar `oracle_rugarch.R`; não inventa fixtures.

- [ ] **Step 1: Gerar dados e fixtures**

Run: `python/.venv/Scripts/python tests/reference/generate_data.py && Rscript tests/reference/oracle_rugarch.R && python/.venv/Scripts/python tests/reference/oracle_arch.py`
Expected: os três arquivos gerados; JSON com 3 resultados cada.

- [ ] **Step 2: Escrever o teste**

```python
@pytest.mark.parametrize("oracle,var_init,per_obs", [("rugarch", "sample", True), ("arch", "backcast", False)])
def test_matches_oracle(ctx, oracle, var_init, per_obs):
    fx = load_fixture(oracle); data = load_csv()   # dict nome → np.ndarray sem as células vazias do fim
    for res in fx["results"]:
        f = ctx.fit(ve.Spec(), data[res["series"]], options=ve.FitOptions(var_init=var_init))
        assert f.status == "converged"
        assert ve.loglik(f) >= res["loglik"] - 1e-6 * abs(res["loglik"])
        assert ve.loglik(f) == pytest.approx(res["loglik"], rel=1e-4)
        assert_params(f.coef().estimate, res["params"], rel=1e-3, abs_small=1e-5)
        assert_rel(f.coef("hessian").std_error, res["se"], 5e-2)
        assert_rel(f.coef("robust").std_error, res["robust_se"], 5e-2)
        assert np.max(np.abs(f.sigma / np.array(res["sigma"]) - 1)) <= 1e-3
        assert ve.aic(f, per_obs=per_obs) == pytest.approx(res["aic"], abs=1e-6)
        assert ve.bic(f, per_obs=per_obs) == pytest.approx(res["bic"], abs=1e-6)
```

- [ ] **Step 3: Rodar** — `python/.venv/Scripts/python -m pytest python/tests/test_oracles.py -v`. Expected: PASS. Se falhar por convenção (σ²₀, termo t = 0, sinal do critério), corrigir a convenção no núcleo (Task 3/5) conforme o oráculo, documentar em `docs/guide/validation.md` e rodar a suíte C++ de novo. Não afrouxar tolerância sem registrar o motivo no mesmo arquivo.
- [ ] **Step 4: Commit** — `git commit -m "test: validate sGARCH(1,1)-norm against rugarch and arch"`

---

### Task 10: Documentos do projeto e CI

**Files:**
- Create: `README.md`, `ARCHITECTURE.md`, `STYLEGUIDE.md`, `CHANGELOG.md`, `LICENSE`, `CITATION.cff`, `CONTRIBUTING.md`, `.github/workflows/ci.yml`

**Interfaces:**
- Consumes: tudo acima.

- [ ] **Step 1: Escrever os documentos**

- `LICENSE`: MIT, `Copyright (c) 2026 Gabriel Matte` (igual aos outros projetos do autor).
- `CITATION.cff`: `cff-version: 1.2.0`, título VolEngine, autor Gabriel Matte, versão 0.1.0, licença MIT.
- `ARCHITECTURE.md`: o mapa do código no estilo do FixedWidthParser — camadas (núcleo → interface C → wrappers), onde vive cada responsabilidade (tabela do mapa de arquivos), as regras de estabilidade da seção 5 do spec e o fluxo de um `ve_fit_batch`.
- `STYLEGUIDE.md`: prioridades (corretude, performance, legibilidade) como no FixedWidthParser, para C++23 moderno: `std::expected` para falhas esperadas no núcleo, `std::span`/`std::mdspan` para dados, sem `else` depois de `return`, ternário só com valores simples numa linha, nada de "C++ estilo C" fora de `src/capi/`.
- `CHANGELOG.md`: Keep a Changelog, seção `[Unreleased]` listando o que este plano entrega.
- `README.md`: o que é, exemplo Python mínimo (`ctx.fit(ve.Spec(), returns)`), build com presets.
- `CONTRIBUTING.md`: build, testes C++ e Python, como regenerar fixtures.

- [ ] **Step 2: Escrever o CI**

`ci.yml`: matriz `windows-latest` (preset `msvc-release`), `ubuntu-latest` com GCC ≥ 15, `macos-latest` com Clang/libc++ ≥ 18; em cada uma: configure, build, `ctest`, criar venv, `pip install -e python[test] arch`, `pytest python/tests` com `VOLENGINE_LIB` apontando para a biblioteca construída. Os testes de oráculo usam os fixtures commitados; o CI não roda R.

- [ ] **Step 3: Verificar localmente** — `cmake --build --preset gcc-release && ctest --preset gcc-release && python/.venv/Scripts/python -m pytest python/tests -q` → tudo PASS. `cmake --preset msvc-release && cmake --build --preset msvc-release && ctest --preset msvc-release` → PASS.
- [ ] **Step 4: Commit** — `git commit -m "docs: add project documents and CI workflow"`
