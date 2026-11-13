# VolEngine — Plano 3a: forecast

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A partir de um ajuste, prever σ e a média condicional h = 1..H passos depois do fim da amostra, como o `ugarchforecast` com `n.roll = 0`, em todos os modelos, distribuições e opções do plano 2. Expor isso pela interface C (`ve_forecast_*`) e pelo Python (`Fit.forecast`), validado contra o `rugarch` e, onde a previsão dele é analítica, contra o `arch`.

**Architecture:** O ajuste passa a guardar um estado final pequeno (`ForecastState`). São as últimas L observações de e, σ², q do csGARCH e dos desvios da média, com L = max(m, n, p, q); com ARFIMA, a série inteira dos desvios. Esse estado é preenchido pela mesma avaliação `double` que já produz `residuals` e `sigma2` no fim do ajuste, então nenhum número existente muda. Um módulo novo, `src/forecast/`, aplica a cada modelo a sua recursão h passos à frente. Ela lê o estado nas defasagens dentro da amostra e troca cada choque futuro pela sua esperança. A média segue o ARMA (e o ARFIMA) com resíduos futuros nulos.

**Tech Stack:** C++23, CMake + presets, GoogleTest v1.17.0, Google Benchmark v1.9.1, Python ≥ 3.11 + numpy + pytest, R 4.6.2 + `rugarch` 1.5.6, `arch` 8.0.0.

**Spec:** `docs/superpowers/specs/2026-11-13-volengine-03-forecast-simulacao-diagnosticos-design.md` (seções 2 "3a", 3, 4–7), sobre a spec-mãe `docs/superpowers/specs/2026-11-10-volengine-design.md`.

## Roteiro

- **1, 2a–2f (feitos).**
- **3a (este):** forecast.
- **3b:** simulação (Philox).
- **3c:** diagnósticos.

Depois: 4 DCC; 5 performance; 6 wrapper C++, wheels, 1.0.

## Global Constraints

Valem todas as Global Constraints dos planos 1 e 2a–2f, o `STYLEGUIDE.md` e o `ARCHITECTURE.md`. As que mais pesam aqui:

- A interface C é o único contrato estável:
  - enums `typedef int32_t` + `#define`, só acréscimos;
  - ≤ 5 parâmetros por função; configuração nova é função nova;
  - nenhuma exceção atravessa a fronteira C;
  - handles imutáveis, legíveis de várias threads;
  - visões `const double*` + comprimento, válidas enquanto o handle existir.
- **O plano 3a só lê o ajuste.** Nenhum número de ajuste muda: ℓ, estimativas, erros-padrão e séries continuam iguais. Todos os testes existentes passam sem mudar constantes nem tolerâncias.
- Convenção padrão segue o `rugarch`. O código dele (GPL) pode ser lido para entender convenções, mas nunca copiado ou traduzido. A referência vai para `validation.md`.
- Anti-objetivo: não fazer mais do que o necessário. Ficam fora:
  - quantil, VaR e ES;
  - forecast em lote ou em grade;
  - `n.roll` e rolling forecast;
  - forecast a partir de spec, porque um ajuste com todos os parâmetros fixos já cobre esse caso;
  - forecast do DCC.
- σ previsto é a grandeza do `ugarchforecast` em todos os modelos. Em eGARCH, apARCH e fGARCH com h > 1 ele não é √E[σ²ₜ₊ₕ], e `validation.md` diz isso (decisão aprovada no design).
- Estilo:
  - nunca `else`;
  - funções ≤ 60 linhas, ≤ 5 parâmetros, aninhamento ≤ 3;
  - ≤ 8 módulos por pasta. `src/model/` e `src/capi/` já têm 8, por isso a pasta nova e a subpasta da Task 4;
  - comentários só para restrição externa ou invariante;
  - medições em `docs/`;
  - `-Wnrvo` como erro no GCC.
- Ambiente e commits:
  - commits sem trailer `Co-Authored-By`; LF;
  - Python no venv `python/.venv`, com `VOLENGINE_LIB` apontando para a DLL do preset testado;
  - R em `C:/Program Files/R/R-4.6.2/bin/Rscript.exe`;
  - **testes só em release** (`gcc-release`), por pedido do usuário (2026-11-13), até ele dizer o contrário.

### Novas regras deste plano

**Notação.** n é o número de observações do ajuste, com índices 0..n−1. A previsão de passo h é o índice t = n − 1 + h. Uma defasagem t − i ≤ n − 1 lê o estado; t − i ≥ n é futura. cₜ = Σₖ ζₖ vₜ,ₖ usa as linhas futuras de `vxreg`, com a linha h−1 da matriz para o índice t. Os parâmetros são as estimativas do ajuste em espaço natural. Com VT, o ω é o ω alvo que o ajuste reporta.

**Variância.** Cada família avança uma grandeza x, linear nos choques esperados:

| Modelo | x | termo do choque i, dentro da amostra | termo do choque i, futuro | σ̂ |
| --- | --- | --- | --- | --- |
| sGARCH, iGARCH | σ² | αᵢ e²ₜ₋ᵢ | αᵢ x̂ₜ₋ᵢ | √x |
| gjrGARCH | σ² | (αᵢ + γᵢ 1[eₜ₋ᵢ < 0]) e²ₜ₋ᵢ | (αᵢ + γᵢ π) x̂ₜ₋ᵢ, π = P(z < 0) como em `persistence` | √x |
| eGARCH | log σ² | αᵢ zₜ₋ᵢ + γᵢ (\|zₜ₋ᵢ\| − E\|z\|) | 0 | exp(x/2) |
| apARCH, fGARCH | σ^δ (λ no fGARCH) | o termo de `power_garch_step` | αᵢ κᵢ x̂ₜ₋ᵢ, κᵢ = `shock_moment(i)` | x^(1/δ) |

- Em todas as famílias, xₜ = ωₜ + Σᵢ termoᵢ + Σⱼ βⱼ xₜ₋ⱼ, com ωₜ = ω + cₜ.
- No iGARCH, β_q = 1 − Σα − Σⱼ<q βⱼ, como em `igarch_parameters`.
- **fGARCH:** o termo dentro da amostra é σ^λ (|z − η₂| − η₁(z − η₂))^δ com o \|·\| **exato**. O filtro do ajuste usa a forma suavizada √(u² + 1e-6) (`kFGarchAbsSmoothing`), e o `ugarchforecast` não. Por isso o h = 1 do fGARCH difere do filtro em ~1e-7 relativo, e o forecast segue o `rugarch`. No apARCH os dois são exatos.
- **csGARCH** (a forma do `rugarch`, que não é a recursão com choques esperados):
  - Em h = 1, a recursão do ajuste no índice n:
    - q̂ = ωₜ + ρ q_{n−1} + φ (e²_{n−1} − σ²_{n−1});
    - σ̂² = q̂ + Σᵢ αᵢ (e²ₜ₋ᵢ − qₜ₋ᵢ) + Σⱼ βⱼ (σ²ₜ₋ⱼ − qₜ₋ⱼ).
  - Em h ≥ 2, forma fechada a partir da última observação, sem o choque dela e com o cₜ do próprio passo:
    - q̂ₕ = (1 − ρʰ)/(1 − ρ) · (ω + cₜ) + ρʰ q_{n−1};
    - σ̂²ₕ = q̂ₕ + (Σα + Σβ)ʰ (σ²_{n−1} − q_{n−1}).
- O passo h = 1 é exatamente o passo da recursão do ajuste no índice n, salvo a suavização do fGARCH: só lê defasagens dentro da amostra.

**Média.** dₜ é o desvio sobre o qual o ARMA age: rₜ − μ − Σδₖxₜ,ₖ, menos o termo `archm` quando há ARCH-in-mean.
- Sem ARFIMA:
  - d̂ₜ = Σᵢ φᵢ dₜ₋ᵢ + Σⱼ θⱼ eₜ₋ⱼ, com e = 0 nas defasagens futuras e d̂ nas defasagens futuras de d.
  - A média prevista é μ + Σδₖ xₜ,ₖ (linhas futuras de `mxreg`) + ξ σ̂ₜ^p (com `archm` p) + d̂ₜ.
- **ARFIMA** (a forma do `rugarch`, φ(L)(1 − L)^d d = θ(L) e, sobre os desvios d e não sobre os resíduos do ARMA). πₖ são os pesos de `fractional_difference`.
  - Fₜ = −Σₖ₌₁ᵗ πₖ dₜ₋ₖ, com d̂ nas defasagens futuras.
  - wₛ = Σₖ₌₀ˢ πₖ dₛ₋ₖ é a diferença fracionária truncada de d. Dentro da amostra só as últimas m entram; nas futuras, w = d̂ − F.
  - d̂ₜ = Fₜ + Σᵢ φᵢ wₜ₋ᵢ + Σⱼ θⱼ eₜ₋ⱼ, com e = 0 no futuro.
  - Com d = 0, πₖ = 0 para k ≥ 1, e a regra vira a do ARMA bit a bit.
  - Custo O(n·H), marcado com `ponytail:`.

**Convenção confirmada.** As regras foram lidas antes do plano nas funções impressas do pacote instalado (`rugarch` 1.5.6): `.nsgarchforecast`, `.ngjrgarchforecast`, `.negarchforecast`, `.naparchforecast`, `.nfgarchforecast`, `.ncsgarchforecast`, `.nigarchforecast`, `armaf`, `arfimaf` e os κ que elas recebem (`pneg`, `egarchKappa`, `fgarchKappa`). Confirmado como acima:
- π = P(z < 0) no gjr;
- o termo `archm` dentro de d;
- o \|·\| exato do fGARCH no forecast;
- as formas fechadas do csGARCH e do ARFIMA.

O `rugarch` também tem `archex` (regressores da média multiplicados por σ), que o VolEngine não tem. A Task 5 confirma tudo isso numericamente.

**Erros** (decisão do Claude):
- Ajuste sem ponto finito (séries vazias): `VE_INVALID_ARGUMENT` com a mensagem "fit has no finite point".
- `FAILED` com ponto finito: previsão normal a partir desse ponto, como os outros valores do ajuste.
- `horizon == 0`: `VE_INVALID_ARGUMENT`.
- Regressores:
  - spec com `mxreg` k > 0 exige um painel de exatamente `horizon` linhas × k colunas, sem ranges; com `vxreg`, o mesmo;
  - painel ausente, a mais ou de forma errada é `VE_INVALID_ARGUMENT`, e a mensagem cita `mxreg` ou `vxreg`.

## Review Focus

1. **Ajuste sem ponto finito** (série constante, ou `INSUFFICIENT_DATA`):
   - `ve_forecast_create` dá `VE_INVALID_ARGUMENT` e não cria handle;
   - o Python levanta o erro de sempre;
   - um `FAILED` com ponto finito prevê normalmente.
   → teste na Task 4.
2. **Regressores futuros errados.** Casos:
   - spec com `mxreg` 2 e painel `NULL`;
   - painel com horizon − 1 linhas;
   - painel com 3 colunas;
   - painel com ranges;
   - spec sem regressores e painel não nulo.

   Cada caso dá `VE_INVALID_ARGUMENT` citando `mxreg` ou `vxreg`, e no Python, `ValueError`. → testes na Task 4.
3. **Horizonte longo** (10 000) em modelo estacionário: termina finito e converge para o nível incondicional. No iGARCH, σ̂² cresce ω por passo. → testes nas Tasks 2 e 3.
4. **Forecast vive sem o ajuste**: liberar o `ve_fit` depois de criar o `ve_forecast` não invalida as visões. Duas threads criando forecasts do mesmo ajuste dão resultados iguais. → teste na Task 4.
5. **Mudança de unidades** (r × 100, ajuste com todos os parâmetros fixos nos valores escalados): σ̂ e a média prevista ficam × 100 (rel 1e-12). → teste na Task 2.

---

## Mapa de arquivos

```
src/forecast/state.hpp             ForecastState (novo)
src/forecast/forecast.hpp, .cpp    forecast(FitResult, horizon, FutureRegressors) (novo)
src/forecast/variance_forecast.*   x̂ por família (novo; separado se forecast.cpp passar de ~300 linhas)
src/model/likelihood.hpp, .cpp     LoglikOutputs ganha deviations, permanent
src/fit/fit_result.hpp, fit.cpp    FitResult::state, preenchido na avaliação final
src/capi/results/                  fit_api.cpp (movido) + forecast_api.cpp (novo)
src/capi/handles.hpp               struct ve_forecast
include/volengine/volengine.h      ve_forecast_*
python/src/volengine/_forecast.py  Forecast (novo); _fit.py Fit.forecast; _native.py
tests/unit/forecast/               forecast_test.cpp (novo)
tests/reference/forecast_rugarch.R, fixtures/rugarch_forecast.json, oracle_arch.py, fixtures/arch_forecast.json
python/tests/test_forecast.py, python/tests/test_forecast_oracles.py
tests/benchmarks/fit_benchmarks.cpp  BM_Forecast
```

---

### Task 1: Estado final do ajuste

**Files:**
- Create: `src/forecast/state.hpp`
- Modify: `src/model/likelihood.hpp`, `src/model/likelihood.cpp`, `src/fit/fit_result.hpp`, `src/fit/fit.cpp`
- Test: `tests/unit/forecast/state_test.cpp`, `tests/unit/model/likelihood_test.cpp`

**Interfaces:**
- Produces:
  ```cpp
  namespace ve::forecast {
  // The last L = max(m, n, p, q) observations of the fit at its estimate (all n deviations with ARFIMA);
  // permanent only in csGARCH, else empty.
  struct ForecastState {
      std::vector<double> residuals;   // e
      std::vector<double> sigma2;
      std::vector<double> permanent;   // csGARCH q
      std::vector<double> deviations;  // d, the series the ARMA acts on (archm term included)
  };
  }
  namespace ve::model {
  template <class T> struct LoglikOutputs {
      std::span<T> terms{}, residuals{}, sigma2{};
      std::span<T> deviations{}, permanent{};  // novos: vazios = não calculados
  };
  }
  namespace ve { struct FitResult { /* campos atuais, e: */ forecast::ForecastState state; }; }
  ```
- Os spans novos são escritos só quando não vazios:
  - `deviations` por `arma_mean_residuals` e por `JointMean`, que inclui o termo `archm`; sem `mxreg`, dₜ = rₜ − μ;
  - `permanent` por `csgarch_from_residuals` e `joint_csgarch`.
- Na avaliação final, a mesma chamada que preenche `residuals`/`sigma2` para `fill_series` passa os dois spans novos e guarda as caudas em `FitResult::state`.
- Sem ponto finito, `state` fica vazio.

- [ ] **Step 1: Escrever os testes**

```cpp
TEST(Likelihood, OutputsDeviationsAndPermanent) {
  // csGARCH(1,1)-norm, ARMA(1,0), mxreg 1, r de 8 pontos: deviations[t] == r[t] - mu - delta x[t] (bit a bit);
  // permanent == q da recursão csgarch_variance rodada à parte; loglik igual com e sem os spans novos (bit a bit)
}
TEST(Likelihood, ArchmDeviationsIncludeArchmTerm) { /* gjrGARCH archm 1: deviations[t] == r[t] - mu - xi sigma[t] */ }
TEST(ForecastState, TailsOfTheFit) {
  // fit_one sGARCH(2,1) ARMA(1,1) em série simulada: state.residuals == últimas 2 de result.residuals;
  // state.sigma2[i] ≈ sigma[i]^2 (rel 1e-15); permanent vazio; csGARCH: permanent tem L entradas;
  // ARFIMA: deviations tem n entradas
}
TEST(ForecastState, EmptyWithoutFinitePoint) { /* série constante: status FAILED sem séries, state vazio */ }
```

- [ ] **Step 2: Rodar e ver falhar** — `cmake --build --preset gcc-release && ctest --preset gcc-release -R "Likelihood|ForecastState"` → não compila.
- [ ] **Step 3: Implementar.**
- [ ] **Step 4: Rodar e ver passar.** A suíte release inteira passa, sem mudar constantes. Um `BM_Loglik/s11_norm/dual` antes e depois fica dentro do ruído, porque os spans vazios não custam nada.
- [ ] **Step 5: Commit** — `git commit -m "feat: keep the final state of a fit for forecasting"`

---

### Task 2: Forecast de sGARCH, iGARCH, gjrGARCH e csGARCH, com a média

**Files:**
- Create: `src/forecast/forecast.hpp`, `src/forecast/forecast.cpp`
- Test: `tests/unit/forecast/forecast_test.cpp`

**Interfaces:**
- Consumes: `FitResult::state` (Task 1); `model::layout_of`, `persistence`, `shock_moment` e os `*_parameters` existentes. Se precisar de um deles fora de `likelihood.cpp`, mova a declaração para um header de `src/model/` sem criar módulo novo.
- Produces:
  ```cpp
  namespace ve::forecast {
  // horizon x columns, row-major, like model::RegressorMatrix; empty when the spec has none.
  struct FutureRegressors { model::RegressorMatrix mean{}; model::RegressorMatrix variance{}; };
  struct Forecast { std::vector<double> sigma; std::vector<double> mean; };   // each of size horizon
  // fit has a finite point and future matches the spec (the caller checks; asserted here).
  Forecast forecast(const FitResult& fit, std::size_t horizon, const FutureRegressors& future = {});
  }
  ```
- Regras "Variância" e "Média" para sGARCH, iGARCH, gjrGARCH e csGARCH, com ARMA, `mxreg`, `vxreg` e `archm`. As famílias das Tasks 3 entram no mesmo `switch`, e até lá dão `assert(false)`.

- [ ] **Step 1: Escrever os testes**

```cpp
// Helper do teste: estende a série por um r_n tal que e_n^2 == mean(e^2) da amostra (|e_n|^δ == mean |e|^δ no
// apARCH/fGARCH), com e_n = r_n − média prevista de h = 1, e v_n == média de v. Assim a pré-amostra `sample` e o
// VT ficam iguais. Avalia a loglik da spec com todos os parâmetros fixos nas estimativas sobre n + 1 pontos.
TEST(Forecast, FirstStepMatchesExtendedFilter) {
  // para sGARCH, iGARCH, gjrGARCH (norm, sstd), csGARCH × ARMA {(0,0), (1,1)} × {mxreg 0, 2} × {vxreg 0, 1}
  // × {VT off, on}, e gjrGARCH archm {1, 2}: sigma[0]^2 == sigma2 da extensão no índice n (rel 1e-12) e
  // mean[0] == fitted da extensão no índice n (abs 1e-12 (1 + |mean|))
}
TEST(Forecast, SGarchClosedForm) {
  // sGARCH(1,1), P = alpha + beta: x_h = omega (1 - P^(h-1)) / (1 - P) + P^(h-1) x_1, h = 1..50 (rel 1e-13)
}
TEST(Forecast, LongHorizonReachesUnconditional) {
  // horizon 10000: sGARCH → omega / (1 - P); gjrGARCH (sstd) → omega / (1 - P) com P = persistence;
  // csGARCH → omega / (1 - rho) (rel 1e-10); média ARMA(1,1) estacionária → mu (abs 1e-12); tudo finito
}
TEST(Forecast, CsGarchClosedForm) {
  // csGARCH(1,1) e (2,2) com vxreg 1: h >= 2 pela forma fechada da regra a partir de q e sigma2 da última
  // observação (rel 1e-13); h = 1 pela recursão (coberto por FirstStepMatchesExtendedFilter)
}
TEST(Forecast, IGarchGrowsByOmega) { /* x_h - x_{h-1} == omega para h >= 2 (rel 1e-12), horizon 10000 finito */ }
TEST(Forecast, ArmaMeanByHand) { /* ARMA(2,1) sem regressores, 3 passos calculados à mão a partir de state (1e-14) */ }
TEST(Forecast, UnitEquivariance) {
  // Review Focus 5: sGARCH ARMA(1,1) com todos os parâmetros fixos; série × 100 e mu, omega escalados:
  // sigma e mean × 100 (rel 1e-12)
}
```

- [ ] **Step 2: Rodar e ver falhar** — `ctest --preset gcc-release -R Forecast` → não compila.
- [ ] **Step 3: Implementar.**
- [ ] **Step 4: Rodar e ver passar** — suíte release inteira PASS.
- [ ] **Step 5: Commit** — `git commit -m "feat: forecast sGARCH, iGARCH, gjrGARCH and csGARCH"`

---

### Task 3: Forecast de eGARCH, apARCH, fGARCH e ARFIMA

**Files:**
- Modify: `src/forecast/forecast.cpp` (ou `src/forecast/variance_forecast.*`, se o arquivo passar de ~300 linhas)
- Test: `tests/unit/forecast/forecast_test.cpp`

**Interfaces:**
- Consumes: Task 2. `shock_moment` dá κᵢ, e E|z| é o `abs_moment` que `egarch_parameters` já usa.
- Regra "Variância" para eGARCH, apARCH e os 8 fGARCH. Regra "Média" com ARFIMA, com o comentário `// ponytail: O(n·H) direct sum`.

- [ ] **Step 1: Escrever os testes**

```cpp
// FirstStepMatchesExtendedFilter estendido a eGARCH (norm, std), apARCH (ged) e ARFIMA (sGARCH, ARMA (0,0) e
// (1,1), mxreg 0 e 2), com as mesmas tolerâncias; e aos 8 fGARCH (norm) com rel 1e-5, pela suavização do |·| no
// filtro (regra "fGARCH").
TEST(Forecast, FGarchFirstStepUsesExactAbs) {
  // fGARCH ALLGARCH: sigma[0]^lambda == o passo com |·| exato calculado à mão a partir de state (rel 1e-13), e
  // difere do filtro estendido em mais de 1e-12
}
TEST(Forecast, EGarchClosedForm) { /* eGARCH(1,1): log x_h = omega (1 - b^(h-1)) / (1 - b) + b^(h-1) log x_1 (rel 1e-13) */ }
TEST(Forecast, PowerLongHorizon) {
  // apARCH e fGARCH ALLGARCH, horizon 10000: sigma^delta → omega / (1 - sum alpha kappa - sum beta) (rel 1e-10)
}
TEST(Forecast, ArfimaMeanByHand) { /* d = 0.3, ARMA(1,1), n = 5: 3 passos de F, w e d̂ pela regra, à mão (1e-14) */ }
TEST(Forecast, ArfimaZeroIsPlain) { /* d fixo 0: forecast == forecast da spec sem ARFIMA (bit a bit) */ }
```

- [ ] **Step 2: Rodar e ver falhar** — `ctest --preset gcc-release -R Forecast` → FAIL (assert ou valor).
- [ ] **Step 3: Implementar.**
- [ ] **Step 4: Rodar e ver passar** — suíte release inteira PASS.
- [ ] **Step 5: Commit** — `git commit -m "feat: forecast eGARCH, apARCH, fGARCH and the ARFIMA mean"`

---

### Task 4: Interface C e Python

**Files:**
- Move: `src/capi/fit_api.cpp` → `src/capi/results/fit_api.cpp`. Atualize o `CMakeLists.txt`. Os planos 3b e 3c põem `sim_api` e `diagnostics_api` na mesma subpasta.
- Create: `src/capi/results/forecast_api.cpp`, `python/src/volengine/_forecast.py`
- Modify: `src/capi/handles.hpp`, `include/volengine/volengine.h`, `python/src/volengine/_fit.py`, `_native.py`, `__init__.py`
- Test: `tests/unit/capi/forecast_test.cpp`, `python/tests/test_forecast.py`

**Interfaces:**
- Consumes: `forecast::forecast` (Tasks 2–3).
- Produces:
  ```c
  typedef struct ve_forecast ve_forecast;
  /* Forecast of f for h = 1..horizon after the last observation (rugarch's ugarchforecast, n.roll = 0).
     mxreg and vxreg: panels of horizon rows x the spec's mxreg / vxreg columns, without ranges; NULL when the
     spec has none. VE_INVALID_ARGUMENT for horizon 0, a fit with no finite point, or regressors that do not
     match. The forecast does not depend on f afterwards. */
  VE_API ve_status ve_forecast_create(const ve_fit* f, size_t horizon, const ve_panel* mxreg,
                                      const ve_panel* vxreg, ve_forecast** out);
  VE_API ve_status ve_forecast_horizon(const ve_forecast* fc, size_t* out);
  /* Views valid until fc is freed; *len is the horizon. sigma: forecast conditional standard deviation, the
     quantity of ugarchforecast (not sqrt(E[sigma2]) in eGARCH, apARCH and fGARCH for h > 1); mean: forecast
     conditional mean (rugarch's seriesFor). */
  VE_API ve_status ve_forecast_sigma(const ve_forecast* fc, const double** data, size_t* len);
  VE_API ve_status ve_forecast_mean(const ve_forecast* fc, const double** data, size_t* len);
  VE_API void ve_forecast_free(ve_forecast* fc);
  ```
  ```python
  class Forecast:           # _forecast.py; exportado em volengine
      sigma: NDArray[np.float64]   # read-only view, shape (horizon,)
      mean: NDArray[np.float64]
  class Fit:
      def forecast(self, horizon: int, *, mxreg: ArrayLike | None = None,
                   vxreg: ArrayLike | None = None) -> Forecast: ...
  ```
- Regra "Erros". As mensagens seguem o padrão de `last_error` do lote. O Python reaproveita `open_panel` de `api.py` para os regressores, e um array 1-D vale como uma coluna. `Forecast` guarda o handle, como `Fit`. Se o ajuste for liberado antes, o forecast continua válido.

- [ ] **Step 1: Escrever os testes**

```cpp
TEST(CapiForecast, MatchesCore) { /* ve_fit_batch de 1 job sGARCH ARMA(1,1): views == forecast::forecast (bit a bit) */ }
TEST(CapiForecast, OutlivesFit) { /* Review Focus 4: ve_fit_free antes de ler as views; valores iguais */ }
TEST(CapiForecast, SameFromTwoThreads) { /* duas std::thread criando do mesmo fit: resultados iguais bit a bit */ }
TEST(CapiForecast, InvalidArguments) {
  // Review Focus 1–2: horizon 0; fit sem ponto finito (série constante); mxreg 2 e painel NULL; painel com
  // horizon - 1 linhas; com 3 colunas; com ranges; spec sem regressores e painel não nulo; handle NULL →
  // VE_INVALID_HANDLE. Cada um: *out == NULL e last_error cita "horizon", "finite", "mxreg" ou "vxreg"
}
TEST(CapiForecast, FailedWithFinitePointForecasts) { /* fit FAILED com séries: forecast criado, valores finitos */ }
```

```python
def test_forecast_shapes_and_read_only(): ...       # (10,), (10,), não graváveis; volengine.Forecast exportado
def test_forecast_with_regressors(): ...            # mxreg 2 / vxreg 1: aceita (h, k) e 1-D para k = 1
def test_forecast_bad_regressors_raise(): ...       # ValueError citando "mxreg" / "vxreg"; horizon 0 → ValueError
def test_forecast_outlives_fit(): ...               # del fit; gc.collect(); forecast.sigma ainda legível e igual
```

- [ ] **Step 2: Rodar e ver falhar** — `ctest --preset gcc-release -R CapiForecast` → não compila; `pytest python/tests/test_forecast.py` → `AttributeError`.
- [ ] **Step 3: Implementar.** Mover o arquivo; acrescentar a struct `ve_forecast { ve::forecast::Forecast forecast; }` e as funções; acrescentar o Python e as docstrings.
- [ ] **Step 4: Rodar e ver passar** — suíte release inteira PASS; `pytest` PASS (com o xfail e a falha pré-existente de `sgarch11_vx_std/aparch11_vx_std` desmarcada como antes); `python tests/style/check_style.py` sai 0; `mypy --strict` sem erro novo.
- [ ] **Step 5: Commit** — `git commit -m "feat: expose forecasts through the C API and Python"`

---

### Task 5: Oráculo `rugarch`

**Files:**
- Create: `tests/reference/forecast_rugarch.R`, `tests/reference/fixtures/rugarch_forecast.json`, `python/tests/test_forecast_oracles.py`
- Modify: `tests/reference/README.md`, `docs/guide/validation.md`

**Interfaces:**
- Consumes: `Fit.forecast` (Task 4); `cases.json`, os CSVs de `tests/reference/data/` e os coeficientes do `rugarch` em `fixtures/rugarch.json`. A conversão de nomes é a de `test_oracles.py`; reaproveite as funções, sem duplicar.
- O caso é avaliado nos parâmetros do `rugarch`, para separar erro de fórmula de diferença de otimizador:
  - H = 25 e n = T − 25;
  - o `rugarch` roda `ugarchforecast(spec com fixed.pars = coef do rugarch.json, data = r[1:n], n.ahead = 25, external.forecasts = list(mregfor = X[(n+1):T, ], vregfor = V[(n+1):T, ]))`;
  - o VolEngine ajusta a mesma spec com `fixed` = todas as estimativas (uma avaliação) sobre r[:n] e as linhas [:n] dos regressores, e roda `forecast(25, mxreg=X[n:], vxreg=V[n:])`.
- Todos os casos com o oráculo `rugarch` entram, inclusive os xfail de ajuste: aqui a comparação é no ponto do `rugarch`.
  - Um caso com VT entra se o ω alvo de n pontos bate entre os dois.
  - Se não bater, sai da lista com um parágrafo em `validation.md`.
- Tolerância: σ̂ rel 1e-8; média abs 1e-8 (1 + |média|). Divergência acima disso é investigada, nunca afrouxada sem explicação documentada.
- `validation.md` ganha a seção "`rugarch`, forecast". Ela traz:
  - as regras "Variância" e "Média";
  - a ressalva do σ em eGARCH, apARCH e fGARCH;
  - as fontes da seção "Convenção confirmada" e as duas formas do `rugarch` que não são a recursão com choques
    esperados (csGARCH h ≥ 2, ARFIMA), além do |·| exato do fGARCH;
  - a tabela de diferenças observadas por família.

- [ ] **Step 1: Escrever o script R e gerar a fixture.** `Rscript tests/reference/forecast_rugarch.R` grava `rugarch_forecast.json`, com `{id, n, horizon, sigma[25], mean[25]}` por caso.
- [ ] **Step 2: Escrever o teste** `test_forecast_matches_rugarch[<id>]`, parametrizado pelos ids da fixture, e rodar contra a DLL release. Esperado: PASS em todos. Um caso que falha é investigado com systematic-debugging antes de qualquer xfail, e um xfail strict exige um parágrafo em `validation.md`.
- [ ] **Step 3: Documentar** em `validation.md` e no `README.md` de `tests/reference/` como regenerar a fixture.
- [ ] **Step 4: Rodar** — `pytest python/tests` inteiro PASS (exceto a falha pré-existente desmarcada).
- [ ] **Step 5: Commit** — `git commit -m "test: validate forecasts against rugarch"`

---

### Task 6: Oráculo `arch`

**Files:**
- Modify: `tests/reference/oracle_arch.py` (novo subcomando ou flag `--forecast`), `python/tests/test_forecast_oracles.py`, `docs/guide/validation.md`
- Create: `tests/reference/fixtures/arch_forecast.json`

**Interfaces:**
- Consumes: Task 5 e a convenção `arch` dos testes de oráculo existentes (`var_init="backcast"`, nomes, conversão da sstd).
- Entram os casos com o oráculo `arch` de modelo sGARCH ou gjrGARCH e média constante ou AR, onde a previsão do `arch` é analítica. Para cada um:
  - `arch_model(...).fix(params do arch.json)` sobre r[:n] com `forecast(horizon=25, method="analytic", reindex=False)`;
  - o VolEngine com `var_init="backcast"` e todos os parâmetros fixos nos do `arch`.
- Mesmas tolerâncias da Task 5. As diferenças de convenção (backcast, AR do `arch` sobre r) vão para `validation.md`.

- [ ] **Step 1: Gerar a fixture** — `python/.venv/Scripts/python.exe tests/reference/oracle_arch.py --forecast`.
- [ ] **Step 2: Escrever o teste** `test_forecast_matches_arch[<id>]` e rodar → PASS.
- [ ] **Step 3: Documentar** em `validation.md`.
- [ ] **Step 4: Rodar** — `pytest python/tests` PASS.
- [ ] **Step 5: Commit** — `git commit -m "test: validate forecasts against arch"`

---

### Task 7: Benchmarks e documentação

**Files:**
- Modify: `tests/benchmarks/fit_benchmarks.cpp`, `docs/performance/benchmarks.md`, `README.md`, `CHANGELOG.md`, `ARCHITECTURE.md`

**Interfaces:**
- `BM_Forecast/<series>/h:60`: `forecast::forecast` de um ajuste já feito, sobre as séries dos `BM_FitOne`. Cobre `s11_norm`, `egarch11_std`, `csgarch11_std`, `fgarch_allgarch11_norm` e `arfima_sgarch11_norm`, onde o termo O(n·H) aparece.
- Esta é a máquina dos benchmarks: rode com `--benchmark_repetitions=5 --benchmark_report_aggregates_only=true` e publique a tabela em `benchmarks.md`, no formato das existentes.
- `README.md`: exemplo `f.forecast(10)`, com a ressalva do σ em eGARCH/apARCH/fGARCH e o link para `validation.md`. `CHANGELOG.md`: entrada "Added". `ARCHITECTURE.md`: linhas de `src/forecast/` e `src/capi/results/`, e a frase do `ForecastState`.

- [ ] **Step 1: Escrever os benchmarks** e rodar `build/gcc-bench/volengine_benchmarks --benchmark_filter=BM_Forecast` → termina, todos finitos.
- [ ] **Step 2: Medir** com 5 repetições e publicar.
- [ ] **Step 3: Documentar.**
- [ ] **Step 4: Rodar** — suíte release inteira, `pytest`, `check_style.py` → PASS.
- [ ] **Step 5: Commit** — `git commit -m "bench: add forecast benchmarks and document plan 3a"`
