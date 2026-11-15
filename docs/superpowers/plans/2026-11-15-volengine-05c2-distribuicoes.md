# Plano 5c-2 — Distribuições Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Quatro mudanças nas distribuições:
- a Boost só em `double`, o que restaura a bit-identidade entre threads;
- as derivadas do K de Bessel em ordem livre pelo trapézio, o que deixa o ghyp ~5–10x mais barato;
- o quantil nig/ghyp por tabela com polimento;
- a quadratura de ged/sged correta em ν grande.

**Architecture:**
- Os valores de K continuam vindo da Boost, agora com `promote_double<false>`.
- Uma soma do trapézio sobre K_v(x) = ½ ∫ exp(−x cosh t + v t) dt substitui o Richardson e a razão nos overloads
  `Dual` de ordem livre (`src/dist/bessel.*`).
- O quantil ganha um `dist::QuantileTable`, construído uma vez por margem na simulação de Copula-DCC. O chute vem de
  um Hermite e o acabamento de um passo de Newton com segmento curto.
- A quadratura de momentos ganha pontos de divisão nas bordas da densidade ged quase uniforme.

**Tech Stack:** C++23, Boost.Math, GoogleTest, Google Benchmark.

**Spec:** `docs/superpowers/specs/2026-11-15-volengine-05c-design.md` §11. Design-mãe:
`docs/superpowers/specs/2026-11-14-volengine-05-performance-design.md`.

## Global Constraints

- **Valor bit a bit entre caminhos:**
  - o valor do `Dual` de `log_bessel_k` e `log_bessel_k_scaled` é igual (`EXPECT_EQ`) ao do overload `double`;
  - `objective<double>` e o valor do `Dual` do loglik ghyp seguem bit-idênticos.
- **Bit-identidade entre threads** na mesma máquina e ISA. Nenhuma chamada de função especial da Boost em `src/` fica
  sem `promote_double<false>`.
- **Interface C e Python:** nenhuma assinatura muda.
- **Oráculos** `rugarch`/`rmgarch` verdes nas tolerâncias atuais: nig, ghyp, mvlaplace, cópulas, forecast, simulação
  e GOF.
  - Os univariados estão em `python/tests/oracles/`.
  - Os multivariados estão no gtest (`CopulaOracle`, `DccOracle`, `DccForecastOracle`) e nos `test_*dcc*`/`test_copula*`
    do pytest.
  - Um xfail estrito que passar sai do xfail.
- **`src/dist/` tem 8 módulos,** o teto. Nenhum módulo novo ali.
- **Estilo:** `STYLEGUIDE.md` (≤ 60 linhas por função, ≤ 5 parâmetros, aninhamento ≤ 3, sem `else`).
- **Testes** só no preset `gcc-release`:
  - `C:\msys64\ucrt64\bin` primeiro no PATH;
  - `ctest --preset gcc-release -j 8` roda a suíte;
  - `build/gcc-release/volengine_tests.exe --gtest_filter=X` roda um recorte;
  - pytest com `VOLENGINE_LIB=build/gcc-release/libvolengine.dll`.
- **Medidas no `gcc-bench`,** A/B na mesma sessão contra a base da Task 1 (`build/base-5c2/`).
- **Código e docs do repo** em inglês; `docs/superpowers` em português. Commits sem `Co-Authored-By`.

## Review Focus

1. **Ordem livre que cruza v = 0** (λ = ½ no termo da densidade ghyp): ∂v log K é contínua e vale 0 em v = 0, e o
   sinal segue sign(v). Teste na Task 3.
2. **x ≥ 700 com ordem `Dual`:** o valor vem da Hankel com recorrência, e as derivadas do trapézio têm que bater com
   ele. Teste na Task 3 (x = 900 e 5000).
3. **Entradas fora do domínio no overload `Dual`** (x NaN, negativo, 0, inf; ordem NaN; |v| ≥ 1e5): sem exceção, com
   o valor e as derivadas NaN ou inf como hoje. Teste na Task 3.
4. **u fora da faixa da tabela** (1e-12, 1 − 1e-12), u exatamente num nó, e u = 0, 1 ou NaN: o mesmo resultado que
   `dist::quantile`. Teste na Task 4.
5. **sged com skew extremo e ν grande:** os refinamentos do kink e os das bordas se somam e não podem estourar a
   capacidade de `Breakpoints`. P(z < 0) tem que bater com a forma fechada. Teste na Task 5.

---

### Task 1: Base de medida e o benchmark do quantil

**Files:**
- Modify: `tests/benchmarks/fit_benchmarks.cpp`. O `BM_DistQuantile/{nig,ghyp}` entra ao lado de `BM_LogBesselK`.
- Modify: `docs/performance/profile-5c.md`. O título passa a "Profile and measurements of plans 5c-1 and 5c-2" e
  entra a seção "5c-2 base".

**Interfaces:**
- Produces:
  - `BM_DistQuantile/nig` e `BM_DistQuantile/ghyp`;
  - `build/base-5c2/volengine_benchmarks.exe`, a base das Tasks 2–6;
  - `build/gcc-bench/dist_base.json`.

- [ ] **Step 1: O benchmark.** `run_dist_quantile(state, Dist)` chama `ve::dist::quantile` em 200 u igualmente
  espaçados em (0,001, 0,999), com os parâmetros dos casos de `DistQuantile.InvertsTheCdf`:
  - nig: `{.skew = -0.4, .shape = 1.5}`;
  - ghyp: `{.skew = 0.3, .shape = 2.0, .ghlambda = -1.5}`.

  Tem o contador `calls_per_s`, como em `run_log_bessel_k`.

- [ ] **Step 2: Commit** `test: benchmark the nig and ghyp quantiles`. A base é este commit.

- [ ] **Step 3: Montar a base.**
  - `git worktree add build/base-5c2-src HEAD`;
  - configurar o preset `gcc-bench` desse worktree com saída em `build/base-5c2/` e compilar `volengine_benchmarks`.

- [ ] **Step 4: Rodar a base.**
  - Filtro: `--benchmark_filter='BM_LogBesselK|BM_DistQuantile|BM_(Loglik|FitOne)/sgarch11_(nig|ghyp)|BM_Diagnostics/sgarch11_ghyp/gof|BM_DccLoglik/dcc_mvlaplace'`.
  - `--benchmark_repetitions=3`, saída `build/gcc-bench/dist_base.json`.
  - Esperado: os números de §11.2 do spec (free_order ~25,9 ms, `FitOne` ghyp ~1843 ms) e o quantil ~290 µs por
    chamada.

- [ ] **Step 5: Registrar** a tabela da base em `profile-5c.md` e fazer o commit `docs: plan 5c-2 base measurements`.

### Task 2: A Boost só em `double`

**Files:**
- Modify:
  - `src/dist/bessel.cpp` (`IgnoreErrors` ganha `promote_double<false>`);
  - `src/diagnostics/tails.cpp` (`gamma_q`, `ibeta` com a política `double`);
  - qualquer outra chamada achada no Step 1.
- Test:
  - `tests/unit/dist/bessel_test.cpp`;
  - `tests/unit/dist/cdf_test.cpp`;
  - `tests/unit/diagnostics/portmanteau_test.cpp`, onde já estão os testes de `student_two_sided`.

**Interfaces:**
- Produces: `log_bessel_k` e `log_bessel_k_scaled` calculados em `double`. Assinaturas iguais.

- [ ] **Step 1: Auditoria.** Listar com `grep -rn "boost::math::" src` as chamadas de função especial ou de
  distribuição sem política explícita. Hoje são `bessel.cpp`, `tails.cpp` e os `quantile(normal/student)` de
  `dcc/copula/scores.cpp`; nos de `scores.cpp`, conferir se o tipo de distribuição já carrega a `DoublePolicy`. Anotar
  a lista no commit.

- [ ] **Step 2: Testes de thread que falham.** Seguem o padrão de `DistQuantile.SameBitsOnAnyThread`: o mesmo cálculo
  na thread principal e numa `std::thread`, comparados com `EXPECT_EQ`.
  - `Bessel.SameBitsOnAnyThread`: `log_bessel_k` e `log_bessel_k_scaled` em ordens {−1,5; 0,3; 1,7; 2,5} e
    x {0,2; 3; 40}.
  - `DistQuantile.SameBitsOnAnyThread` ganha nig `{.skew = -0.4, .shape = 1.5}` e ghyp
    `{.skew = 0.3, .shape = 2.0, .ghlambda = -1.5}`.
  - Um teste igual para `gamma_upper` e `student_two_sided`.

- [ ] **Step 3: Rodar e ver falhar.**
  - Rodar `--gtest_filter='*SameBitsOnAnyThread*'`. Esperado: FAIL nos casos novos (x87 diferente na `std::thread`).
  - Se algum passar mesmo assim, registrar no commit que o defeito não se reproduz nesta máquina para aquele caso. O
    item segue pela promessa.

- [ ] **Step 4: Aplicar `promote_double<false>`** em toda chamada da lista do Step 1.

- [ ] **Step 5: Rodar os testes.**
  - `--gtest_filter='*SameBitsOnAnyThread*:Bessel*:Cdf*:DistQuantile*:Moments*'`;
  - os oráculos gtest (`*Oracle*`);
  - `ctest --preset gcc-release -j 8`;
  - `pytest python/tests/oracles`.

  Esperado: PASS. Os casos nig/ghyp/mvlaplace mudam por ulps, dentro das tolerâncias. Se algum oráculo cair, parar e
  trazer o número ao autor antes de mexer em tolerância.

- [ ] **Step 6: A/B** com o filtro da Task 1, `--benchmark_repetitions=3`. Esperado: `BM_LogBesselK/fixed_order`,
  nig, GOF, mvlaplace e `BM_DistQuantile` mais rápidos. Registrar a tabela em `profile-5c.md`.

- [ ] **Step 7: Commit** `fix: Boost special functions in double, whatever the thread's x87 precision`.

### Task 3: Derivadas de ordem livre pelo trapézio

**Files:**
- Modify: `src/dist/bessel.hpp`, `src/dist/bessel.cpp`.
- Test: `tests/unit/dist/bessel_test.cpp`.

**Interfaces:**
- Consumes: `log_bessel_k(double, double)` e `log_bessel_k_scaled(double, double)` da Task 2.
- Produces, em `bessel.hpp`:
  ```cpp
  namespace ve::dist::detail {
  struct LogBesselSlopes { double order; double x; };
  // d log K_order(x) / d order and / d x (unscaled; the scaled x slope is this + 1).
  LogBesselSlopes log_bessel_k_slopes(double order, double x);
  }
  ```
  Os dois overloads com ordem `T` (`log_bessel_k(const T&, const T&)` e `log_bessel_k_scaled(const T&, const T&)`)
  passam a usá-la. Os de ordem `double` não mudam. `order_slope` sai, a menos que reste como fallback (Step 3).

- [ ] **Step 1: Testes que falham.**
  - **`Bessel.OrderSlopeMatchesLongDoubleReference`:** ordens {−3,3; −1,5; −0,5; 0; 0,1; 0,4999; 0,5; 1; 2,5; 3,7;
    10; 50; 200} e x {0,01; 0,1; 1; 1,78; 10; 31,6; 300; 699}. A referência é a diferença central de Richardson em
    long double, com passo 1e-4 sobre `std::log(boost::math::cyl_bessel_k<long double>)`. O mesmo vale para ∂x pela
    razão em long double. Tolerância 1e-12 relativa a max(1, |ref|). Hoje falha em ∂v (o Richardson de passo 1e-3
    erra ~1e-10 em x pequeno).
  - **`Bessel.OrderDerivativeMatchesClosedFormAtHalf`:** a tolerância vai de 1e-8 para 1e-13.
  - **`Bessel.OrderSlopeSmallArgument`:** v = 0,1, x ∈ {1e-8; 1e-6; 1e-4}. A referência é o Richardson em long double
    com passo 1e-6. Tolerância 1e-11.
  - **`Bessel.OrderSlopeZeroOrder`:** em v = 0, ∂v = 0 exato. ∂v(+1e-9) e ∂v(−1e-9) têm sinais opostos e
    |∂v| < 1e-6.
  - **`Bessel.OrderSlopeLargeArgument`:** x ∈ {900; 5000}, ordens {0,3; 2,5; 40}. ∂v e ∂x do scaled batem com
    diferenças de `log_bessel_k_scaled` em `double` a 1e-7 (passo 1e-4).
  - **`Bessel.DerivativesMatchFiniteDifferences`:** `EXPECT_EQ(fixed.grad[1], dual.grad[1])` vira `EXPECT_NEAR` a
    1e-13 relativo. O ∂x da ordem livre vem agora do trapézio.
  - **`Bessel.NeverThrows`** ganha o overload de ordem `Dual` com x ∈ {NaN, −1, 0, inf} e ordem ∈ {NaN, 1e5}. Valor
    igual ao overload `double`, sem exceção.

- [ ] **Step 2: Rodar e ver falhar** `--gtest_filter='Bessel*'`. Esperado: FAIL em `OrderSlopeMatchesLongDouble`,
  `OrderSlopeSmallArgument` e `ClosedFormAtHalf`.

- [ ] **Step 3: Implementar `log_bessel_k_slopes`** em `bessel.cpp`. O algoritmo é o do spike, para v = |order|:
  ```
  peak = asinh(v / x); sigma = (x^2 + v^2)^(-1/4); h = min(0.2, 0.4 sigma)
  nodes t_k = peak ± k h, k = 0, 1, ..., each side until e_k < -40, with
  e_k = -x (cosh t_k - cosh peak) + v (t_k - peak), cosh t_k from e^(t_k) by recurrence (e^peak, e^±h)
  w_k = exp(e_k); S = Σ w_k
  d/dx = -Σ cosh(t_k) w_k / S;  d/dorder = sign(order) Σ t_k w_k / S   (0 at order = 0)
  ```
  - **Domínio validado** em constantes nomeadas: |order| ≤ 200 e 1e-8 ≤ x < ∞, com x finito.
  - **Fora dele** (x < 1e-8, |order| > 200), o caminho atual: Richardson para a ordem e razão para x, guardados como
    fallback com `ponytail:` que diz o teto.
  - **Entrada inválida** (NaN, x ≤ 0): NaN nas duas derivadas.
  - **Overloads `Dual`:** valor do overload `double`, depois `ad::chain(order, x, value, slopes.order, slopes.x)`, e
    `+ 1` em x no scaled.

- [ ] **Step 4: Rodar os testes.**
  - `--gtest_filter='Bessel*:Densities*:Moments*:Likelihood*:*Ghyp*:*Nig*'`;
  - `pytest python/tests/oracles -k "nig or ghyp"`;
  - `ctest -j 8`.

  Esperado: PASS. Os testes de gradiente do ghyp contra diferenças finitas também passam.

- [ ] **Step 5: A/B.** Esperado:
  - `BM_LogBesselK/free_order` de ~25,9 ms para ≤ 5 ms;
  - `BM_Loglik/sgarch11_ghyp/dual` e `BM_FitOne/sgarch11_ghyp` ≥ 3x mais rápidos;
  - nig e o conjunto rápido sem piora.

  Registrar em `profile-5c.md`. Se o ghyp ainda ficar acima de 5x o nig no `FitOne`, deixar um `ponytail:` no
  `log_bessel_k_slopes` com os números (SIMD nos nós é o próximo passo, spec §11.4).

- [ ] **Step 6: Commit** `perf: Bessel K order and argument slopes by the trapezoid rule`.

### Task 4: Quantil nig/ghyp por tabela

**Files:**
- Modify: `src/dist/cdf.hpp`, `src/dist/cdf.cpp`.
- Modify: `src/dcc/future/simulate.cpp`. O `Margin` guarda um `dist::QuantileTable` no lugar de `dist` +
  `parameters`; `margin_quantile` o chama.
- Modify: `tests/benchmarks/fit_benchmarks.cpp`, com `BM_DistQuantile/{nig,ghyp}/table`.
- Test: `tests/unit/dist/cdf_test.cpp`.

**Interfaces:**
- Consumes: `quadrature_cdf`, `integrate_segment`, `solved_quantile` e `needs_quadrature` (internos de `cdf.cpp`);
  `dist::quantile`.
- Produces, em `cdf.hpp`:
  ```cpp
  // The quantile of one distribution, with a table of the cdf built once for nig and ghyp (whose quantile is
  // otherwise a root solve over the quadrature cdf); the other distributions call quantile. Same results as
  // quantile to ~1e-13 relative in z; thread-independent and immutable once built.
  class QuantileTable
  {
  public:
      QuantileTable(Dist dist, const ShapeParameters<double>& parameters);
      double operator()(double u) const;
  };
  ```

- [ ] **Step 1: Testes que falham.**
  - **`QuantileTable.MatchesSolvedQuantile`:** nig `{.skew = -0.4, .shape = 1.5}` e `{.skew = 0.3, .shape = 1.2}`;
    ghyp `{.skew = 0.3, .shape = 2.0, .ghlambda = -1.5}`, `{.skew = 0.3, .shape = 2.0, .ghlambda = 2.0}` e o pico
    estreito `{.skew = -0.2, .shape = 0.3, .ghlambda = -1.5}`. Valores de u: 2001 igualmente espaçados em
    [1e-4, 1 − 1e-4], mais `kProbabilities`. Assertivas:
    - `|table(u) − quantile(u)| ≤ 1e-13 · max(1, |quantile(u)|)`;
    - `|cdf(table(u)) − u| ≤ 1e-13`.
  - **`QuantileTable.OtherDistributionsDelegate`:** para norm, std, ged, sstd, sged e jsu, com os parâmetros de
    `DistQuantile.KnownValues`, `EXPECT_EQ(table(u), quantile(u))`.
  - **`QuantileTable.EdgesLikeQuantile`:** u ∈ {0, 1, NaN, −0,1, 1e-12, 1 − 1e-12}, um u igual ao de um nó, e
    parâmetros fora do domínio. Mesmo resultado que `quantile`: `EXPECT_EQ`, ou ambos NaN.
  - **`QuantileTable.Monotone`:** `table(u)` não decresce nos 2001 u.
  - **`QuantileTable.SameBitsOnAnyThread`:** a tabela é construída numa thread e consultada em outra.

- [ ] **Step 2: Rodar e ver falhar** (`QuantileTable` não existe).

- [ ] **Step 3: Implementar.**
  - **Construção:** uma grade de z com cdf por `quadrature_cdf` no primeiro nó e depois acumulada por
    `integrate_segment`, como no `sorted_cdf`, mais a densidade em cada nó.
  - **Consulta:**
    - busca binária em u;
    - chute por Hermite cúbico de z(u), com inclinações 1/f nos nós;
    - um passo de Newton, com cdf(chute) = cdf do nó vizinho + `integrate_segment(nó, chute)` e o passo
      (cdf − u)/f(chute).
  - **u fora da faixa da tabela:** `quantile(...)`.
  - **Deixado ao implementador, com o teste como juiz:**
    - a grade (número de nós, faixa em u, espaçamento uniforme em z ou adaptado ao `density_scale`);
    - um segundo passo de Newton;
    - PCHIP no lugar do Hermite.
  - **Se 1e-13 contra o `solved_quantile` não fechar** porque a cdf acumulada e a `quadrature_cdf` diferem (o
    comentário de `quadrature_cdf` registra até ~1e-8 entre as duas quadraturas num pico), o polimento passa a usar
    `quadrature_cdf(chute)`. Registrar a escolha e o custo em `profile-5c.md`.

- [ ] **Step 4: Usar na simulação.** `margin_of` constrói `QuantileTable(layout.dist, parameters)` para a margem
  paramétrica. A margem empírica não muda.

- [ ] **Step 5: Rodar os testes.** `--gtest_filter='QuantileTable*:DistQuantile*:Cdf*:DccSimulate*:DccEnd*:DccForecast*'` e
  `ctest -j 8`. Esperado: PASS. As simulações de Copula-DCC com margens std/norm ficam bit a bit iguais, porque
  delegam ao `quantile`.

- [ ] **Step 6: A/B.** `BM_DistQuantile/*/table` contra `BM_DistQuantile/*` da base. Esperado: ≥ 50x. O custo de
  construção da tabela por margem é medido à parte e registrado. Registrar em `profile-5c.md` e trocar o `ponytail:`
  de `solved_quantile` pelo novo teto.

- [ ] **Step 7: Commit** `perf: nig and ghyp quantiles from a cdf table polished by Newton`.

### Task 5: ged/sged em ν grande

**Files:**
- Modify: `src/dist/moments.hpp`: `Breakpoints` (capacidade), `add_side_refinements` e uma função nova
  `add_edge_refinements`.
- Test:
  - `tests/unit/dist/moments_test.cpp`;
  - `tests/unit/sim/samplers_test.cpp` (`GedWithLargeShape`).

**Interfaces:**
- Consumes: `LogDensity<T>` (ged: `power` = ν, densidade ∝ exp(−½ |z/scale|^ν)), `SkewedLogDensity<T>`,
  `side_scales`, `add_refinements`.
- Produces: nada público. `shifted_moment` e `probability_below_zero` mantêm as assinaturas.

- [ ] **Step 1: Testes que falham.**
  - **`Moments.GedLargeShapeMatchesClosedForm`:** ν ∈ {2; 5; 20; 50; 200; 500; 1000}, η₁ ∈ {0; 0,3}, p ∈ {1; 1,5; 2}.
    Compara `shifted_moment(Ged, ν, η₁, 0, p)` com `asymmetric_moment(Ged, ν, η₁, p)` (a forma fechada) a 1e-12
    relativo.
  - **`Moments.SgedLargeShapeProbabilityBelowZero`:** ν no mesmo conjunto, skew ∈ {0,05; 0,8; 1,2; 20}. Compara
    `probability_below_zero(Sged, ...)` com `dist::cdf(Sged, ..., 0.0)` a 1e-12 absoluto.
  - **`Moments.SgedLargeShapeDerivativesMatchFiniteDifferences`:** ν = 300, skew 1,2, η₁ 0,3, η₂ 0,2, p 1,5. O padrão
    é o de `SkewedDerivativesMatchFiniteDifferences`.
  - **`Samplers.GedWithLargeShape`:** volta a conferir `P(z < 0)` e o momento deslocado `(0,3; 0,2; 1,5)`, como o teste
    de cima do mesmo arquivo. O comentário que explica a omissão sai.

- [ ] **Step 2: Rodar e ver falhar.** Esperado: FAIL de ν ~ 50 para cima. Se falhar por outra causa que não as bordas
  (por exemplo, a cauda de `tail_exponent`), registrar e corrigir a causa.

- [ ] **Step 3: Implementar `add_edge_refinements`.**
  - **Quando entra:** só para ged e sged com ν acima de um limiar nomeado. O limiar é o menor ν em que a quadratura
    atual erra mais de 1e-13 nos testes do Step 1. Abaixo dele, os pontos e os valores ficam bit a bit iguais (o
    `Moments.SymmetricUnchanged` e os oráculos garantem isso).
  - **Pontos:** um em cada borda, |y| = scale · 2^(1/ν) no kernel, levado a z por lado como em `side_scales`. Cada
    borda ganha refinamentos dos dois lados, na escala scale/ν, pela máquina de `add_refinements`.
  - **Só valores,** como os de `add_location_refinements`, porque a densidade é suave ali.
  - **Capacidade:** aumentar `Breakpoints::points` para caber kink, η₂, os refinamentos do kink, os da localização e
    os das bordas. Um `assert`, ou um teste com skew 0,05 e 20, cobre o pior caso.

- [ ] **Step 4: Rodar os testes.**
  - `--gtest_filter='Moments*:Samplers*:Forecast*:*Aparch*:*Fgarch*'`;
  - `pytest python/tests/oracles`;
  - `ctest -j 8`.

  Esperado: PASS, sem nenhum valor de oráculo alterado.

- [ ] **Step 5: Commit** `fix: ged and sged moments at large shapes split at the density's edges`.

### Task 6: Tabelas finais e documentação

**Files:**
- Modify:
  - `docs/performance/profile-5c.md` e `docs/performance/benchmarks.md`, a seção de `BM_LogBesselK` e a linha
    `sgarch11_ghyp` das tabelas de ajuste (a nota "the gradient costs 38 `double` evaluations" ganha o número novo);
  - `docs/superpowers/specs/2026-11-15-volengine-05c-design.md`, com a seção nova "12. 5c-2 feito";
  - `CHANGELOG.md` (`[Unreleased]`), no padrão das entradas do 5b:
    - em "Fixed": a bit-identidade de nig/ghyp/mvlaplace e dos p-valores entre threads, e os momentos ged/sged em ν
      grande;
    - em "Changed": nig/ghyp movem por ulps, e o gradiente ghyp vem de uma quadratura.

- [ ] **Step 1: Tabela final** do filtro da Task 1, mais `BM_DistQuantile/*/table`, com 5 repetições, base e nova na
  mesma sessão. Rodar também o conjunto rápido (`quick.py run`/`compare`). Esperado: nenhum secundário pior acima do
  ruído.

- [ ] **Step 2: Suíte inteira.** `ctest --preset gcc-release -j 8` e `pytest python/tests` com a DLL do `gcc-release`.
  Esperado: tudo verde.

- [ ] **Step 3: Documentação.** As tabelas, a seção 12 do spec (números e o que ficou para o autor) e o `CHANGELOG`.
  Depois, `git worktree remove build/base-5c2-src`.

- [ ] **Step 4: Commit** `docs: record the plan 5c-2 benchmarks`.
