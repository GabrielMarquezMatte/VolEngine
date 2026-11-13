# VolEngine — Plano 3b: simulação

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Simular P caminhos × H passos de retornos e σ, de duas origens: a partir de um ajuste, continuando do fim da amostra (o `ugarchsim` com `startMethod = "sample"`), ou a partir de uma spec com todos os parâmetros fixos, partindo do estado incondicional (o `ugarchpath`). Vale para todos os modelos, distribuições e opções do plano 2. O RNG é um Philox próprio, e o resultado é bit-idêntico para qualquer número de threads. A simulação fica exposta pela interface C (`ve_simulate_*`) e pelo Python (`ctx.simulate`) e é validada estatisticamente contra o forecast do 3a.

**Architecture:** Um passo de simulação é o passo h = 1 do forecast do 3a, aplicado a um estado que cresce: a variância e a média do passo saem da mesma regra, depois o caminho sorteia z, faz e = σ z e r = média + e, e acrescenta e, σ², q e o desvio ao seu estado. Por isso o 3b não duplica nenhuma recursão. O forecast ganha só os momentos pré-calculados (π, E|z|, κᵢ, quadratura caríssima por passo) e uma função de um passo que devolve também o q do csGARCH e o nível da média. Um módulo novo, `src/sim/`, traz:
- o Philox4x32-10;
- os amostradores das 8 distribuições, construídos sobre as mesmas `dist::Density<double>` da verossimilhança, para que a parametrização seja a do ajuste;
- o estado inicial de cada origem;
- o laço de caminhos, dividido entre as threads do `ve_context`.

Cada caminho tem fluxo próprio, então o resultado não depende da divisão.

**Tech Stack:** C++23, CMake + presets, GoogleTest v1.17.0, Google Benchmark v1.9.1, Boost.Math (já no projeto; `cyl_bessel_k` nos testes do GIG), Python ≥ 3.11 + numpy + pytest.

**Spec:** `docs/superpowers/specs/2026-11-13-volengine-03-forecast-simulacao-diagnosticos-design.md` (seções 2 "3b", 3, 4–9), sobre a spec-mãe `docs/superpowers/specs/2026-11-10-volengine-design.md` (seções 8 e 12: bit-identidade entre números de threads).

## Roteiro

- **1, 2a–2f, 3a (feitos).**
- **3b (este):** simulação.
- **3c:** diagnósticos.

Depois: 4 DCC; 5 performance; 6 wrapper C++, wheels, 1.0.

## Global Constraints

Valem todas as Global Constraints dos planos 1, 2a–2f e 3a, o `STYLEGUIDE.md` e o `ARCHITECTURE.md`. As que mais pesam aqui:

- A interface C é o único contrato estável:
  - enums `typedef int32_t` + `#define`, só acréscimos;
  - ≤ 5 parâmetros por função; configuração nova é função nova;
  - nenhuma exceção atravessa a fronteira C;
  - handles imutáveis, legíveis de várias threads;
  - visões `const double*` + comprimento, válidas enquanto o handle existir.
- **O 3b só lê o ajuste e a spec.** Nenhum número de ajuste nem de forecast muda: todos os testes existentes, inclusive os do 3a e os oráculos de forecast, passam sem mudar constantes nem tolerâncias.
- **Bit-identidade:** mesma semente, mesma origem, mesma versão, mesma máquina e ISA dão os mesmos caminhos com qualquer número de threads. A semente **não** reproduz caminhos entre versões (decisão do autor); o CHANGELOG avisa quando os caminhos de uma semente mudam.
- Com a mesma semente, os caminhos de `paths = 100` são os 100 primeiros de `paths = 1000`, e os de `horizon = 60` são o começo dos de `horizon = 120` (spec, seção 9).
- Sem grade: uma origem por chamada. Ficam fora quantis, percentis, retorno acumulado e qualquer estatística sobre os caminhos (o usuário usa numpy), bootstrap e simulação do DCC.
- Regressores: uma matriz H × k de `mxreg` e uma de `vxreg`, as mesmas para todos os caminhos.
- Layout do resultado: P × H, linha = caminho, para o numpy ver `(paths, horizon)` sem cópia.
- O `rugarch` (GPL) pode ser lido para entender convenções, mas nunca copiado ou traduzido. Os algoritmos dos amostradores vêm dos papers citados.
- Estilo:
  - nunca `else`;
  - funções ≤ 60 linhas, ≤ 5 parâmetros, aninhamento ≤ 3;
  - ≤ 8 módulos por pasta. `src/sim/` nasce com 4, e `src/capi/results/` passa de 2 para 4;
  - comentários só para restrição externa ou invariante;
  - medições em `docs/`;
  - `-Wnrvo` como erro no GCC.
- Ambiente e commits:
  - commits sem trailer `Co-Authored-By`; LF;
  - Python no venv `python/.venv`, com `VOLENGINE_LIB` apontando para a DLL do preset testado;
  - **testes só em release** (`gcc-release`), por pedido do usuário (2026-11-13), até ele dizer o contrário;
  - o `pytest` roda com `-k "not sgarch11_vx_std/aparch11_vx_std"`, a falha pré-existente.

### Novas regras deste plano

**Um passo.** Para um caminho com estado S (um `ForecastState`), o passo h faz:
1. Avalia o forecast de horizonte 1 a partir de S, com a linha h − 1 dos regressores futuros. Isso dá σ²ₕ, a média mₕ, o nível ℓₕ = μ + Σδₖxₕ,ₖ + ξσₕ^p (a média antes do ARMA) e, no csGARCH, qₕ.
2. Sorteia zₕ do fluxo do caminho, faz eₕ = σₕ zₕ e rₕ = mₕ + eₕ.
3. Grava rₕ e σₕ na linha do caminho.
4. Acrescenta a S: eₕ em `residuals`, σ²ₕ em `sigma2`, qₕ em `permanent` (só no csGARCH) e rₕ − ℓₕ em `deviations`.

O estado só cresce, sem descartar o começo. As regras do 3a leem a cauda, então isso basta. Como o forecast de h = 1 só lê defasagens dentro de S, o passo é a recursão do modelo com choques realizados:
- no fGARCH, com o |·| exato, como o forecast;
- no ARFIMA, com a soma sobre todos os desvios de S.

`ponytail:` o ARFIMA custa O((n + h) · m) por passo; fica assim, a não ser que alguém simule ARFIMA em escala.

**Origem "ajuste".** S começa como `FitResult::state`, e θ são as estimativas. Um ajuste sem ponto finito (estado vazio) é erro, e um `FAILED` com ponto finito simula normalmente, como no 3a.

**Origem "spec"** (decisão do Claude, pedida pela spec na seção 8):
- A spec precisa ter todos os parâmetros fixos (os derivados não contam) e VT desligado. θ são os valores fixos.
- Com c₁ = Σζₖ v₀,ₖ (a primeira linha dos `vxreg` futuros, 0 sem eles), o nível incondicional x̄ na escala da família é:

| Modelo | x̄ | σ̄² | condição |
| --- | --- | --- | --- |
| sGARCH, gjrGARCH, apARCH, fGARCH | (ω + c₁)/(1 − P), P = `model::persistence` | x̄ (σ^δ: x̄^(2/δ)) | P < 1 |
| eGARCH | log σ̄² = (ω + c₁)/(1 − Σβ) | exp(·) | \|Σβ\| < 1 |
| csGARCH | q̄ = (ω + c₁)/(1 − ρ) | q̄ | ρ < 1 e P < 1 |
| iGARCH | não existe | — | erro |

- S tem L = max(m, n, p, q, 1) entradas: σ² = σ̄², q = q̄ (só no csGARCH) e desvios 0, ou seja, a média no seu nível incondicional.
- Cada caminho sorteia primeiro L choques zᵢ do seu fluxo e grava os resíduos σ̄ zᵢ, como o `ugarchpath` sorteia o pré-resíduo. Só depois vêm os passos.
- Com essa origem, a média sobre os caminhos de x em cada passo é x̄. O teste da Task 4 confere isso.

**Erros da origem "spec"**, todos `VE_INVALID_ARGUMENT` com mensagem:
- parâmetro livre: cita o nome dele;
- VT ligado;
- iGARCH;
- parâmetros da distribuição fora do domínio (`dist::in_domain`);
- P, ρ ou Σβ fora da condição;
- σ̄² não finito ou ≤ 0.

**Fluxos.** O Philox4x32-10 é o de Salmon et al. (2011), Random123.
- A chave é a semente (2 × 32 bits).
- O contador é (bloco baixo, bloco alto, caminho baixo, caminho alto). O bloco começa em 0 e cada bloco rende dois `uint64`.
- `uniform()` devolve ((u >> 11) + 0,5) · 2⁻⁵³, em (0, 1) aberto.
- O caminho p consome o seu fluxo em ordem: os pré-choques, depois um z por passo. Daí saem os prefixos e a independência da divisão entre threads.

**Amostradores** (decisão do Claude; "algoritmos dos amostradores" estão na seção 8 da spec). Cada um sorteia da `dist::Density<double>` de `make_density`, na parametrização do ajuste:

| Distribuição | Como sortear |
| --- | --- |
| norm | Box–Muller, só o cosseno: z = √(−2 ln u₁) cos(2π u₂). `ponytail:` Ziggurat no plano 5. |
| std | z = N √((ν − 2) / (2 G)), G ~ Gama(ν/2) |
| ged | \|z\| = λ (2 G)^(1/ν), G ~ Gama(1/ν), sinal por u < 1/2; λ = `scale` da `LogDensity` |
| sstd, sged | Fernandez–Steel sobre um x do núcleo: y = ξ\|x\| com probabilidade ξ²/(1 + ξ²), senão −\|x\|/ξ; z = (y − `shift`)/`stretch` (o inverso do mapa de `skewed.hpp`) |
| nig, ghyp | z = `location` + β W + √W N, com β = ρ α, W ~ GIG(λ, χ = δ², ψ = α² − β²) e λ = −1/2 no nig. GIG por Hörmann & Leydold (2014), as três regiões. |
| jsu | z = `center` + `scale` sinh((N + `skew`) / `shape`) |

A Gama é a de Marsaglia & Tsang (2000). Com a < 1, usa-se G(a + 1) · u^(1/a).

O GIG tem densidade ∝ w^(λ−1) exp(−(χ/w + ψ w)/2).

Se a leitura de `skewed.hpp` ou de `hyperbolic.hpp` mostrar que algum mapa acima está invertido, o teste de momentos da Task 1 diz qual. Corrija o mapa e registre o ruling.

**Momentos pré-calculados.** π = P(z < 0), E|z| e κᵢ passam a ser calculados uma vez por forecast e uma vez por simulação, nunca por passo, porque a quadratura das distribuições assimétricas custa milhares de avaliações de densidade.

**Erros da interface C** (decisão do Claude):
- horizon 0 ou paths 0: `VE_INVALID_ARGUMENT`;
- paths · horizon · 2 · 8 bytes acima de `SIZE_MAX`: `VE_INVALID_ARGUMENT` ("too large");
- regressores com linhas ≠ horizon, com ranges, ausentes quando a spec os tem, presentes quando não tem, ou com colunas ≠ as da spec: `VE_INVALID_ARGUMENT` citando `mxreg` ou `vxreg`;
- falta de memória: `VE_OUT_OF_MEMORY`, pelo `guarded` de sempre.

## Review Focus

1. **Tamanho absurdo** (paths = 2⁶², horizon = 60, ou um horizon ≥ 2⁶⁴ vindo do Python): erro de argumento, nunca alocação gigante, wrap ou crash. → testes nas Tasks 5 (C e Python).
2. **Distribuições nas bordas do domínio:** std ν = 2,05, ged ν = 0,25, nig ζ = 1e4, ghyp λ = −3 e ζ = 0,3, jsu shape (τ) = 0,2. 10⁵ sorteios, todos finitos e sem laço infinito de rejeição. → teste na Task 1.
3. **Simulação vive sem a origem:** liberar o `ve_fit`, o `ve_spec`, as opções e o contexto depois de criar o `ve_sim` não invalida as visões. → teste na Task 5.
4. **Semente nas bordas:** 0 e 2⁶⁴ − 1 funcionam e diferem. No Python, uma semente negativa ou ≥ 2⁶⁴ dá `ValueError`, e não um número trocado em silêncio. → testes nas Tasks 3 e 5.
5. **Caminho explosivo:** um iGARCH ajustado, ou um apARCH com P perto de 1, com 60 passos termina finito. Um caminho que estoura para inf fica inf no resultado, e a função não quebra nem trava. → teste na Task 3.

---

## Mapa de arquivos

```
src/forecast/inputs.hpp              Moments; Inputs ganha moments
src/forecast/variance_forecast.*     moments_of; next_permanent
src/forecast/step.hpp                Step first_step(Inputs) (novo; implementado em forecast.cpp)
src/sim/philox.hpp                   philox4x32, Stream (novo)
src/sim/samplers.hpp, .cpp           normal, gamma, gig, Sampler (novo)
src/sim/start.hpp, .cpp              Start, start_of_fit, start_of_spec (novo)
src/sim/simulate.hpp, .cpp           Request, Simulation, simulate (novo)
src/capi/results/regressor_rows.*    cópia e checagem dos regressores futuros (sai de forecast_api.cpp)
src/capi/results/sim_api.cpp         ve_sim_options_*, ve_simulate_*, ve_sim_* (novo)
src/capi/handles.hpp                 ve_sim_options, ve_sim
include/volengine/volengine.h        seção de simulação
python/src/volengine/_simulation.py  Simulation (novo); api.py Context.simulate; _native.py; __init__.py
tests/unit/sim/                      philox_test, samplers_test, simulate_test, start_test (novo)
tests/unit/capi/sim_test.cpp, python/tests/test_simulate.py
tests/benchmarks/fit_benchmarks.cpp  BM_Simulate, BM_Sampler
```

---

### Task 1: Philox e amostradores

**Files:**
- Create: `src/sim/philox.hpp`, `src/sim/samplers.hpp`, `src/sim/samplers.cpp`
- Modify: `CMakeLists.txt`
- Test: `tests/unit/sim/philox_test.cpp`, `tests/unit/sim/samplers_test.cpp`

**Interfaces:**
- Produces:
  ```cpp
  namespace ve::sim {
  std::array<std::uint32_t, 4> philox4x32(std::array<std::uint32_t, 4> counter, std::array<std::uint32_t, 2> key);
  class Stream {                       // the stream of one path (regra "Fluxos")
  public:
      Stream(std::uint64_t seed, std::uint64_t path);
      std::uint64_t next();            // the next 64 bits
      double uniform();                // in (0, 1)
  };
  double normal(Stream& stream);
  double gamma(Stream& stream, double shape);                            // scale 1
  double gig(Stream& stream, double lambda, double chi, double psi);
  class Sampler {                      // z of the standardized density of the fit
  public:
      Sampler(Dist dist, const dist::ShapeParameters<double>& parameters);
      double draw(Stream& stream) const;
  };
  }
  ```

- [ ] **Step 1: Escrever os testes**

```cpp
TEST(Philox, KnownAnswers) {
  // Random123 kat_vectors, philox4x32 R = 10:
  // counter 0, key 0 → {0x6627e8d5, 0xe169c58d, 0xbc57ac4c, 0x9b00dbd8}
  // counter all 0xffffffff, key all 0xffffffff → {0x408f276d, 0x41c83b0e, 0xa20bc7c6, 0x6d5451fd}
  // counter {0x243f6a88, 0x85a308d3, 0x13198a2e, 0x03707344}, key {0xa4093822, 0x299f31d0}
  //   → {0xd16cfe09, 0x94fdcceb, 0x5001e420, 0x24126ea1}
}
TEST(Philox, StreamsArePerPath) {
  // Stream(7, 3) == Stream(7, 3) por 1000 next(); Stream(7, 3) != Stream(7, 4) e != Stream(8, 3) no primeiro next();
  // sementes 0 e 2^64 - 1 funcionam e diferem; uniform() em (0, 1) aberto em 10^6 sorteios
}
TEST(Samplers, GammaMoments) { /* shape 0.3, 1, 7.5: média e variância == shape, 5 erros-padrão, N = 4e5 */ }
TEST(Samplers, GigMoments) {
  // (lambda, chi, psi) = (-0.5, 1, 2), (1.5, 0.3, 4), (-3, 2, 0.5): E W = sqrt(chi/psi) K_{l+1}(w)/K_l(w),
  // E 1/W = sqrt(psi/chi) K_{l+1}(w)/K_l(w) - 2 l / chi, w = sqrt(chi psi) (boost cyl_bessel_k); 5 erros-padrão
}
TEST(Samplers, StandardizedMoments) {
  // norm; std nu 8; ged nu 1.3 e 0.6; sstd (xi 1.4, nu 8); sged (0.7, 1.5); nig (rho -0.4, zeta 1.5);
  // ghyp (0.3, 2, lambda -1.5) e (0.3, 2, lambda 2); jsu (skew 0.8, shape 1.5). N = 4e5, 5 erros-padrão:
  // média 0, variância 1, P(z < 0) == dist::probability_below_zero, E|z| == dist::abs_moment(power 1),
  // E(|z - 0.2| - 0.3 (z - 0.2))^1.5 == dist::shifted_moment(eta1 0.3, eta2 0.2, power 1.5)
}
TEST(Samplers, FiniteAtDomainEdges) {
  // Review Focus 2: std 2.05, ged 0.25, nig zeta 1e4, ghyp (lambda -3, zeta 0.3), jsu shape 0.2: 1e5 sorteios finitos
}
```

- [ ] **Step 2: Rodar e ver falhar** — `cmake --build --preset gcc-release && ctest --preset gcc-release -R "Philox|Samplers"` → não compila.
- [ ] **Step 3: Implementar** pela regra "Amostradores" e "Fluxos". `Sampler` guarda a `Density<double>` de `make_density` e despacha por `std::visit`.
- [ ] **Step 4: Rodar e ver passar** — suíte release inteira PASS.
- [ ] **Step 5: Commit** — `git commit -m "feat: add the Philox stream and the samplers of the eight distributions"`

---

### Task 2: Um passo do forecast, com os momentos pré-calculados

**Files:**
- Create: `src/forecast/step.hpp`
- Modify: `src/forecast/inputs.hpp`, `src/forecast/variance_forecast.hpp`, `src/forecast/variance_forecast.cpp`, `src/forecast/forecast.cpp`
- Test: `tests/unit/forecast/forecast_test.cpp`

**Interfaces:**
- Consumes: o forecast do 3a.
- Produces:
  ```cpp
  namespace ve::forecast::detail {
  // pi = P(z < 0) (gjrGARCH), E|z| (eGARCH), kappa_i (apARCH, fGARCH); 0 or empty where the model reads none.
  struct Moments { double below_zero = 0.0; double abs_moment = 0.0; std::vector<double> shock{}; };
  Moments moments_of(const model::Layout& layout, std::span<const double> theta);
  struct Inputs { /* campos atuais, e: */ const Moments& moments; };
  // csGARCH: q of step 1, omega_1 + rho q_{n-1} + phi (e^2_{n-1} - sigma2_{n-1}).
  double next_permanent(const Inputs& in);
  // step.hpp: the forecast of h = 1 from in.state (in.horizon == 1): sigma2, the mean, its level (the mean before
  // the ARMA part) and the csGARCH q (0 in the other models).
  struct Step { double sigma2; double mean; double level; double permanent; };
  Step first_step(const Inputs& in);
  }
  ```
- `variance_forecast` lê π, E|z| e κᵢ de `in.moments`, e não calcula mais nada disso. `component_variance` usa `next_permanent`. `forecast()` calcula `moments_of` uma vez. Os números do forecast não mudam (Global Constraints).

- [ ] **Step 1: Escrever os testes**

```cpp
TEST(Forecast, FirstStepIsForecastOfOne) {
  // para cada família (sGARCH, iGARCH, gjr sstd, eGARCH, apARCH, fGARCH ALLGARCH, csGARCH) com ARMA(1,1), mxreg 1,
  // vxreg 1, e gjr archm 1, e ARFIMA: first_step == forecast(fit, 1) em sigma2 e mean (bit a bit);
  // level == mean - d̂ do 3a; csGARCH: permanent == q do passo pela regra (rel 1e-15)
}
```

- [ ] **Step 2: Rodar e ver falhar** — `ctest --preset gcc-release -R Forecast` → não compila.
- [ ] **Step 3: Implementar.** `first_step` vive em `forecast.cpp`, onde estão `level` e a média.
- [ ] **Step 4: Rodar e ver passar.** A suíte release inteira passa, com todos os testes do 3a sem mudar tolerância. O `pytest python/tests/test_forecast_oracles.py` também passa.
- [ ] **Step 5: Commit** — `git commit -m "refactor: precompute the forecast moments and expose its first step"`

---

### Task 3: Simulação a partir de um ajuste

**Files:**
- Create: `src/sim/start.hpp`, `src/sim/start.cpp`, `src/sim/simulate.hpp`, `src/sim/simulate.cpp`
- Modify: `CMakeLists.txt`
- Test: `tests/unit/sim/simulate_test.cpp`

**Interfaces:**
- Consumes: Tasks 1 e 2; `Context::with_pool`, dividindo os caminhos em blocos como `run_blocks` de `fit/batch.cpp`.
- Produces:
  ```cpp
  namespace ve::sim {
  struct Start {
      model::Layout layout;
      std::vector<double> theta;
      forecast::ForecastState state;
      forecast::detail::Moments moments;
      std::size_t presample = 0;        // > 0 only from a spec: shocks each path draws before step 1
      double presample_sigma = 0.0;     // their sigma, sqrt of the unconditional sigma2
  };
  Start start_of_fit(const FitResult& fit);   // fit has a finite point (the caller checks)
  struct Request {
      std::size_t horizon = 0;
      std::size_t paths = 0;
      std::uint64_t seed = 0;
      forecast::FutureRegressors future{};    // horizon rows; the caller checks the columns
  };
  // paths x horizon, row = path.
  struct Simulation { std::size_t paths = 0; std::size_t horizon = 0; std::vector<double> returns; std::vector<double> sigma; };
  Simulation simulate(Context& context, const Start& start, const Request& request);
  }
  ```
- Regras "Um passo", "Origem ajuste" e "Fluxos". Cada caminho escreve só a sua linha.

- [ ] **Step 1: Escrever os testes**

```cpp
// Os ajustes vêm das séries de tests/unit/simulated_garch.hpp com fit_one, ou de estados montados à mão como em
// UnitEquivariance (forecast_test.cpp). P = 20000, H = 10, "5 EP" = |média - alvo| <= 5 sd / sqrt(P) + 1e-12.
TEST(Simulate, FirstStepIsTheForecast) {
  // em cada família: sigma[p, 0] == forecast(fit, 1).sigma[0] para todo p (bit a bit)
}
TEST(Simulate, MeanMatchesForecast) {
  // média sobre os caminhos, h = 1..10, contra o forecast do 3a, 5 EP:
  //   sigma^2: sGARCH norm, iGARCH std, gjrGARCH norm e std (só simétricas: o pi do rugarch não é E[z^2 1(z<0)]);
  //   log sigma^2: eGARCH std;  sigma^delta: apARCH ged, fGARCH ALLGARCH norm;
  //   csGARCH norm: E sigma^2 pela recursão com choques esperados, calculada no teste
  //     (E q_h = omega + rho E q_{h-1}; E sigma2_h = E q_h + (sum alpha + sum beta)(E sigma2_{h-1} - E q_{h-1})), não a forma do rugarch;
  //   retornos: ARMA(1,1) com mxreg 1 e ARFIMA (d 0.3), contra forecast.mean
}
TEST(Simulate, SameForAnyThreadCount) { /* contextos de 1, 3 e 16 threads: returns e sigma iguais (memcmp) */ }
TEST(Simulate, Prefixes) {
  // paths 100 == primeiras 100 linhas de paths 1000; horizon 60 == primeiras 60 colunas de horizon 120;
  // sementes 0 e 2^64 - 1 diferem entre si e de 42
}
TEST(Simulate, ExplosivePathsStayDefined) {
  // Review Focus 5: iGARCH ajustado e apARCH com P = 0.999, H = 60: termina; todo valor é finito ou +inf, sem NaN em sigma
}
```

- [ ] **Step 2: Rodar e ver falhar** — `ctest --preset gcc-release -R Simulate` → não compila.
- [ ] **Step 3: Implementar.**
- [ ] **Step 4: Rodar e ver passar** — suíte release inteira PASS.
- [ ] **Step 5: Commit** — `git commit -m "feat: simulate paths that continue a fit"`

---

### Task 4: Simulação a partir de uma spec

**Files:**
- Modify: `src/sim/start.hpp`, `src/sim/start.cpp`, `src/sim/simulate.cpp`
- Test: `tests/unit/sim/start_test.cpp`

**Interfaces:**
- Consumes: Task 3; `model::persistence`, `dist::in_domain`, `garch::raise`.
- Produces:
  ```cpp
  namespace ve::sim {
  // Regra "Origem spec": the unconditional start, or the message of VE_INVALID_ARGUMENT.
  std::expected<Start, std::string> start_of_spec(const Spec& spec, const forecast::FutureRegressors& future);
  }
  ```
- `simulate` sorteia os `presample` choques antes do passo 1, quando `presample > 0`.

- [ ] **Step 1: Escrever os testes**

```cpp
TEST(SimulateSpec, StationaryInExpectation) {
  // todos os parâmetros fixos, P = 20000, H = 5, 5 EP: média de x_h == x̄ em todo h para sGARCH norm,
  // gjrGARCH norm, eGARCH std (log sigma^2), apARCH ged (sigma^delta), csGARCH norm (sigma^2 e q̄); média dos
  // retornos == mu com ARMA(1,1)
}
TEST(SimulateSpec, VxregShiftsTheLevel) { /* sGARCH vxreg 1, v_0 = 2: x̄ == (omega + 2 zeta) / (1 - P) (rel 1e-14) */ }
TEST(SimulateSpec, Rejections) {
  // parâmetro livre (mensagem cita "alpha1"); VT; iGARCH; P = 1.02; shape std 1.5 (fora do domínio):
  // cada um devolve a mensagem, sem Start
}
TEST(SimulateSpec, PresampleComesFirst) {
  // sGARCH(2,1): sigma[p, 0] varia entre caminhos (pré-choques sorteados); horizon 5 == primeiras 5 colunas de
  // horizon 10 e paths 10 == primeiras 10 linhas de paths 50 (bit a bit); 1 e 4 threads iguais
}
```

- [ ] **Step 2: Rodar e ver falhar** — `ctest --preset gcc-release -R SimulateSpec` → não compila.
- [ ] **Step 3: Implementar.**
- [ ] **Step 4: Rodar e ver passar** — suíte release inteira PASS.
- [ ] **Step 5: Commit** — `git commit -m "feat: simulate paths from a spec with every parameter fixed"`

---

### Task 5: Interface C e Python

**Files:**
- Create: `src/capi/results/regressor_rows.hpp`, `src/capi/results/regressor_rows.cpp`, `src/capi/results/sim_api.cpp`, `python/src/volengine/_simulation.py`
- Modify: `src/capi/results/forecast_api.cpp` (usa `regressor_rows`), `src/capi/handles.hpp`, `include/volengine/volengine.h`, `python/src/volengine/api.py`, `_native.py`, `__init__.py`, `CMakeLists.txt`
- Test: `tests/unit/capi/sim_test.cpp`, `python/tests/test_simulate.py`

**Interfaces:**
- Consumes: Tasks 3–4. As checagens de `future_rows` de `forecast_api.cpp` passam para `regressor_rows.*`: copiar linhas (horizon, sem ranges) e conferir colunas contra a spec, com as mensagens atuais. O forecast continua com as mesmas mensagens.
- Produces:
  ```c
  typedef struct ve_sim_options ve_sim_options;
  typedef struct ve_sim ve_sim;
  /* horizon >= 1 steps and paths >= 1; any seed. The same seed, source and library version give the same paths
     on every machine with the same ISA and for every thread count; a new version may change them (CHANGELOG).
     The first k paths of a simulation are those of any simulation with more paths, and likewise for steps. */
  VE_API ve_status ve_sim_options_create(size_t horizon, size_t paths, uint64_t seed, ve_sim_options** out);
  /* Copies the future regressors: horizon rows x the spec's columns, without ranges, the same for every path;
     NULL for none. */
  VE_API ve_status ve_sim_options_set_regressors(ve_sim_options* o, const ve_panel* mxreg, const ve_panel* vxreg);
  VE_API void ve_sim_options_free(ve_sim_options* o);
  /* Paths that continue after the last observation of f (rugarch's ugarchsim, startMethod = "sample"). */
  VE_API ve_status ve_simulate_fit(ve_context* ctx, const ve_fit* f, const ve_sim_options* o, ve_sim** out);
  /* Paths from the unconditional state of s (rugarch's ugarchpath): every parameter fixed, no variance targeting,
     not iGARCH, stationary. */
  VE_API ve_status ve_simulate_spec(ve_context* ctx, const ve_spec* s, const ve_sim_options* o, ve_sim** out);
  VE_API ve_status ve_sim_paths(const ve_sim* sim, size_t* out);
  VE_API ve_status ve_sim_horizon(const ve_sim* sim, size_t* out);
  /* paths x horizon, row-major (row = path); *len = paths * horizon; valid until sim is freed. sigma is the
     conditional standard deviation of each step, returns the simulated series. */
  VE_API ve_status ve_sim_returns(const ve_sim* sim, const double** data, size_t* len);
  VE_API ve_status ve_sim_sigma(const ve_sim* sim, const double** data, size_t* len);
  VE_API void ve_sim_free(ve_sim* sim);
  ```
  ```python
  class Simulation:              # _simulation.py; exportado em volengine
      returns: NDArray[np.float64]   # read-only view, shape (paths, horizon)
      sigma: NDArray[np.float64]
  class Context:
      def simulate(self, source: Fit | Spec, *, horizon: int, paths: int, seed: int,
                   mxreg: ArrayLike | None = None, vxreg: ArrayLike | None = None) -> Simulation: ...
  ```
- Regra "Erros da interface C". No Python:
  - `horizon` e `paths` precisam ser inteiros em [1, 2⁶⁴), e `seed` em [0, 2⁶⁴); fora disso, `ValueError`;
  - regressores como em `Fit.forecast` (1-D vale uma coluna);
  - `Simulation` guarda o handle.

- [ ] **Step 1: Escrever os testes**

```cpp
TEST(CapiSim, FitMatchesCore) { /* ve_fit_batch de 1 job: views == sim::simulate (bit a bit), forma paths x horizon */ }
TEST(CapiSim, SpecMatchesCore) { /* spec com tudo fixo: views == simulate(start_of_spec) */ }
TEST(CapiSim, OutlivesSources) { /* Review Focus 3: libera fit, spec, opções, painéis e contexto antes de ler; iguais */ }
TEST(CapiSim, InvalidArguments) {
  // horizon 0; paths 0; paths 2^62 com horizon 60 (Review Focus 1); fit sem ponto finito; spec com parâmetro livre;
  // mxreg 2 sem regressores; painel com horizon - 1 linhas; com ranges; com 3 colunas; spec sem regressores e painel
  // não nulo; handles NULL → VE_INVALID_HANDLE. Cada um: *out == NULL e last_error cita "horizon", "paths",
  // "too large", "finite", o nome do parâmetro, "mxreg" ou "vxreg"
}
TEST(CapiSim, SeedEdges) { /* Review Focus 4: seeds 0 e UINT64_MAX funcionam e diferem */ }
```

```python
def test_simulate_shapes_and_read_only(): ...     # (paths, horizon), não graváveis; volengine.Simulation exportado
def test_simulate_from_spec(): ...                # Spec com fixed= de todos os parâmetros
def test_simulate_same_for_any_threads(): ...     # Context(threads=1) e Context(threads=4): arrays iguais
def test_simulate_bad_arguments_raise(): ...      # horizon 0, paths -1, seed -1, seed 2**64, horizon 2**64: ValueError;
                                                  # regressores errados: ValueError citando "mxreg" / "vxreg"
def test_simulate_outlives_sources(): ...         # del fit, ctx; gc.collect(); arrays ainda legíveis e iguais
```

- [ ] **Step 2: Rodar e ver falhar** — `ctest --preset gcc-release -R CapiSim` → não compila; `pytest python/tests/test_simulate.py` → `AttributeError`.
- [ ] **Step 3: Implementar.** Os handles são `struct ve_sim_options { std::size_t horizon, paths; std::uint64_t seed; std::vector<double> mean, variance; std::size_t mean_columns, variance_columns; bool has_mean, has_variance; }` e `struct ve_sim { ve::sim::Simulation simulation; }`, com docstrings no Python.
- [ ] **Step 4: Rodar e ver passar.** Precisam passar a suíte release inteira, o `pytest` com o deselect de sempre, o `check_style.py` e o `mypy --strict`, este sem erro novo além dos 6 pré-existentes.
- [ ] **Step 5: Commit** — `git commit -m "feat: expose simulation through the C API and Python"`

---

### Task 6: Benchmarks e documentação

**Files:**
- Modify: `tests/benchmarks/fit_benchmarks.cpp`, `docs/performance/benchmarks.md`, `docs/guide/validation.md`, `README.md`, `CHANGELOG.md`, `ARCHITECTURE.md`, `tests/reference/README.md` (só se citar a simulação)

**Interfaces:**
- `BM_Simulate/<series>/paths:10000/h:60/threads:<1|all>`: `sim::simulate` de um ajuste feito fora da medição, sobre `s11_norm`, `egarch11_std` e `sgarch11_ghyp`, com 1 thread e com todas (spec, seção 7).
- `BM_Sampler/<dist>`: 10⁵ sorteios de cada uma das 8 distribuições, nos parâmetros do `StandardizedMoments`.
- Esta é a máquina dos benchmarks: rode só esses filtros, com `--benchmark_repetitions=5 --benchmark_report_aggregates_only=true`, e publique em `benchmarks.md` no formato das tabelas existentes.
- `validation.md` ganha uma seção "Simulação" com:
  - o Philox e os vetores de teste;
  - os amostradores, com os papers, e o teste de momentos;
  - a tabela de "Média contra o forecast": a grandeza de cada família e por que o gjr só entra com distribuições simétricas (o π do `rugarch` não é E[z² 1(z < 0)] nas assimétricas) e o csGARCH contra a recursão com choques esperados, e não contra a forma fechada do `rugarch`;
  - a origem "spec" e as suas recusas;
  - a garantia de reprodutibilidade.
- Outros documentos:
  - `README.md`: exemplo `ctx.simulate(fit, horizon=60, paths=10_000, seed=42)` e um percentil do retorno acumulado com numpy;
  - `CHANGELOG.md`: entrada "Added", com a frase de que a semente só reproduz caminhos dentro da mesma versão;
  - `ARCHITECTURE.md`: linhas de `src/sim/` e de `src/capi/results/`, e a dependência `forecast` <- `sim` <- `capi`.

- [ ] **Step 1: Escrever os benchmarks** e rodar `build/gcc-bench/volengine_benchmarks --benchmark_filter="BM_Simulate|BM_Sampler"` → termina, todos finitos.
- [ ] **Step 2: Medir** com 5 repetições e publicar.
- [ ] **Step 3: Documentar.**
- [ ] **Step 4: Rodar** — suíte release inteira, `pytest`, `check_style.py` → PASS.
- [ ] **Step 5: Commit** — `git commit -m "bench: add simulation benchmarks and document plan 3b"`
