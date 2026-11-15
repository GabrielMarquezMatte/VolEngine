# Plano 5b — SIMD Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Derrubar o pipeline do TCC com todas as threads (grade univariada + grade do DCC) com SIMD dentro de uma
avaliação: kernels Highway para a recursão e o Cholesky do DCC, gradiente do DCC por tangentes na recursão com reverse
dentro de t, e um laço vetorial único da densidade univariada (norm/std/ged), fundido com o adjunto.

**Architecture:**
- Os kernels ficam em `src/simd/`, com despacho em tempo de execução do Highway, atrás de funções sobre
  `std::span<double>`.
- A instanciação `double` da verossimilhança do DCC e da univariada chama os kernels; o `Dual` continua como está,
  servindo de referência e de fallback.
- O gradiente do DCC é um arquivo novo `src/dcc/gradient.{hpp,cpp}`: k matrizes tangentes de Q_t, mais a densidade de t
  puxada até Q_t.

**Tech Stack:** C++23, Google Highway 1.2.0 (FetchContent), GoogleTest, Google Benchmark, AMD uProf.

**Spec:** `docs/superpowers/specs/2026-11-14-volengine-05b-simd-design.md` (design-mãe:
`docs/superpowers/specs/2026-11-14-volengine-05-performance-design.md`).

## Global Constraints

- **Highway 1.2.0**, via FetchContent. Despacho em tempo de execução (`HWY_DYNAMIC_DISPATCH`) num build genérico
  x86-64; nada de `-march=native`.
- **Intrínsecos e tipos do Highway só dentro de `src/simd/`.** O resto do código vê funções sobre `std::span<double>`.
- **Bit-identidade** (opção a do autor):
  - na mesma máquina, os resultados não dependem do número de threads;
  - o ℓ do adjunto é bit-idêntico ao de `loglik<double>`;
  - `double` e `Dual` passam a diferir em ulps, e os testes desse par ganham uma tolerância registrada;
  - entre ISAs, os resultados só batem dentro de tolerância.
- **Interface C:** nenhuma assinatura muda.
- **Oráculos** (rugarch, rmgarch, arch) verdes nas tolerâncias atuais.
- **Estilo:** `STYLEGUIDE.md` inteiro (≤ 60 linhas, ≤ 5 parâmetros, aninhamento ≤ 3, sem `else`, ≤ 8 módulos por
  pasta).
- **Testes** só no preset `gcc-release`:
  - rodar com `C:\msys64\ucrt64\bin` primeiro no PATH;
  - `ctest --preset gcc-release -j 8` roda a suíte inteira;
  - `build/gcc-release/volengine_tests.exe --gtest_filter=X` roda um recorte;
  - pytest com `VOLENGINE_LIB=build/gcc-release/libvolengine.dll`.
- **Medidas no `gcc-bench`,** com o A/B na mesma sessão. O "antes" sai de uma build do commit base num worktree,
  porque o benchmark carrega `libvolengine.dll`.
- **Código e docs do repo** em inglês; `docs/superpowers` em português. Commits sem `Co-Authored-By`.

## Review Focus

1. **m ou T que não são múltiplos do número de lanes** (m = 2, 3, 5, 21; T ímpar): o resto do laço vetorial dá o mesmo
   que a referência escalar dentro da tolerância do kernel. Testes nas Tasks 3, 4 e 5.
2. **R_t que não é positiva definida no meio do caminho:** o kernel de Cholesky devolve falha, o termo vira NaN como no
   escalar, e o objetivo devolve +∞. Testes nas Tasks 3 e 4.
3. **Máquina sem AVX2 (SSE4 ou escalar):** todos os kernels passam no target mais baixo compilado, forçado no teste.
   Tasks 1, 3 e 5.
4. **FDCC com grupos de tamanhos diferentes** e **aDCC** (indicador de choque negativo): o gradiente tangente bate com o
   `Dual`. Task 4.
5. **σ²_t inválida (≤ 0 ou não finita) dentro do laço vetorial univariado:** o resultado é `kInvalidLoglik` exato, como
   antes. Task 5.

---

### Task 1: Highway, `src/simd/` e verificação da toolchain

**Files:**
- Modify: `CMakeLists.txt` (FetchContent de Highway 1.2.0; `src/simd/*.cpp` no `volengine_core`)
- Create: `src/simd/reduce.hpp`, `src/simd/reduce.cpp` (um kernel mínimo despachado: `double simd::sum(std::span<const double>)`)
- Test: `tests/unit/simd/reduce_test.cpp`
- Modify: `docs/performance/measuring.md` (seção "SIMD targets")

**Interfaces:**
- Produces: `double simd::sum(std::span<const double> x)`, despachado. Produz também o padrão de despacho e o padrão
  de teste em todos os targets, que as Tasks 3–5 copiam.

- [ ] **Step 1: Write the failing test** `SimdSum.MatchesScalarOnEveryTarget`. Para cada target suportado e compilado
  (helpers de teste do Highway, `hwy::SupportedAndGeneratedTargets()` com `hwy::SetSupportedTargetsForTest`):
  - entradas de tamanho 0, 1, 3, 7, 21, 5000;
  - `simd::sum` dentro de 4 ulps da soma escalar em `long double`;
  - imprime o nome de cada target testado.
- [ ] **Step 2: Run** `volengine_tests.exe --gtest_filter=SimdSum*`. Expected: FAIL (não compila).
- [ ] **Step 3: Implement.** Highway por FetchContent, com `HWY_ENABLE_TESTS`, `HWY_ENABLE_EXAMPLES` e
  `HWY_ENABLE_CONTRIB` como o kernel precisar. Depois o kernel com `foreach_target.h` e `HWY_EXPORT`.
- [ ] **Step 4: Run** o teste. Expected: PASS em todos os targets impressos.
- [ ] **Step 5: Verificar a toolchain** (design §5):
  - registrar quais targets o MinGW-GCC compila e roda: SSE2, SSSE3, SSE4, AVX2, AVX3;
  - forçar o AVX2 num laço de 10⁶ chamadas do kernel com acumuladores na pilha, sem crash;
  - compilar e rodar o mesmo teste no preset `msvc-release`;
  - escrever o resultado em `docs/performance/measuring.md`.

  **Stop condition:** se AVX2 não rodar no `gcc-bench` (desligado pelo Highway ou crash por alinhamento, GCC bug
  54412), parar e levar ao autor a escolha do build oficial.
- [ ] **Step 6: Commit** `feat: add Highway and the dispatched SIMD kernel module`

### Task 2: Profile do DCC

**Files:**
- Create: `docs/performance/profile-5b.md`

- [ ] **Step 1: Profile.** uProf, amostragem por tempo, sobre `volengine_benchmarks --benchmark_filter=BM_DccFit/fdcc_mvnorm`
  e `/dcc_mvnorm`, com o preset `msvc-profile` (mesmo procedimento de `docs/performance/measuring.md`).
- [ ] **Step 2: Record.** As funções mais quentes, e as fatias da recursão de Q_t, da normalização, do Cholesky e solve
  e da densidade, no objetivo em `Dual` e no `reported_loglik` em `double`.

  Expected: recursão e Cholesky cada um entre ~30% e ~60%. Se for outra divisão, registrar um `Ruling:` com a ordem dos
  kernels das Tasks 3 e 4.
- [ ] **Step 3: Commit** `docs: profile the DCC fit before the SIMD kernels`

### Task 3: Kernels de valor do DCC

**Files:**
- Create: `src/simd/dcc_kernels.{hpp,cpp}`, `src/simd/cholesky_kernels.{hpp,cpp}`
- Modify: `src/dcc/recursion.hpp` e `src/dcc/likelihood.cpp`: a instanciação `double` de `CorrelationRecursion` e de
  `observation_term` chama os kernels; o template genérico do `Dual` não muda.
- Modify: `tests/benchmarks/fit_benchmarks.cpp` (`BM_DccLoglik/<spec>/{double,dual}`, m = 21, T = 5000)
- Test: `tests/unit/simd/dcc_kernels_test.cpp`, `tests/unit/simd/cholesky_kernels_test.cpp`, `tests/unit/dcc/likelihood_test.cpp`

**Interfaces:**
- Produces (os nomes ficam a cargo do implementador; registrar no ledger):
  - um passo da recursão sobre Q empacotado, Q_t = C + Σ A∘(zz')_{t-j} + Σ G∘(nn')_{t-j} + Σ B∘Q_{t-k}, com pesos
    escalares (DCC/aDCC) ou matriciais (FDCC);
  - a normalização R = D Q D;
  - `bool simd::cholesky_factor(std::span<double> a, std::size_t n)` e `void simd::cholesky_solve(...)`, com as mesmas
    pré-condições e o mesmo layout de `linalg::cholesky_factor` e `linalg::cholesky_solve`.

- [ ] **Step 1: Write the failing tests**:
  - cada kernel contra a referência escalar existente (`linalg::cholesky_*`, o passo genérico de `recursion.hpp`) em
    todo target, com m ∈ {2, 3, 5, 21}. Tolerância relativa documentada por kernel, ≤ 1e-14 nos elementos;
  - `cholesky_factor` devolve `false` numa matriz não positiva definida;
  - `DccLikelihood.DoubleMatchesDualValue`: DCC, aDCC e FDCC × mvnorm, mvt e mvlaplace; `correlation_terms<double>`
    contra o valor de `correlation_terms<Dual<8>>`, tolerância relativa 1e-12 por termo;
  - um caso com R_t não positiva definida no meio do caminho, com termo NaN nos dois caminhos.
- [ ] **Step 2: Run** `--gtest_filter=*Kernel*:DccLikelihood*`. Expected: FAIL (não compila).
- [ ] **Step 3: Implement** os kernels e a instanciação `double`. A densidade por t continua escalar. Os laços
  internos do Cholesky (produtos internos das linhas) e as atualizações elemento a elemento são vetoriais.
- [ ] **Step 4: Run** o filtro, a suíte inteira e o pytest. Expected: tudo verde, oráculos rmgarch inclusive.
  - Os testes bit-exatos dos erros-padrão do DCC (diferenças centrais em `double`) passam a ter tolerância: medir o
    ruído novo e registrar o valor no teste.
- [ ] **Step 5: Medir A/B** `BM_DccLoglik/*/double`, `BM_DccStandardErrors` e `BM_DccFit` (este muda pouco, porque o
  gradiente ainda é `Dual`). Registrar em `docs/performance/profile-5b.md`.
- [ ] **Step 6: Commit** `perf: SIMD kernels for the DCC recursion and Cholesky`

### Task 4: Gradiente do DCC — tangentes na recursão, reverse dentro de t (com portão)

**Files:**
- Create: `src/dcc/gradient.{hpp,cpp}`
- Modify: `src/dcc/fit.cpp`: `MeanObjective` usa o gradiente novo quando ele cobre a spec; senão, o `Dual` atual.
- Test: `tests/unit/dcc/gradient_test.cpp`

**Interfaces:**
- Consumes: os kernels da Task 3.
- Produces:
  ```cpp
  namespace ve::dcc {
  // True for the specs objective_gradient computes: DCC, aDCC and FDCC x mvnorm, mvt and mvlaplace (listed
  // explicitly; anything else falls back to the Dual).
  bool has_fast_gradient(const DccSpec& spec);
  // objective (as dcc::objective<double>) and its gradient over the natural-space theta. Buffers kept between calls.
  class ObjectiveGradient {
  public:
      double evaluate(const StandardizedPanel& panel, const DccSpec& spec, std::span<const double> theta,
                      std::span<double> grad);
  };
  }
  ```

- [ ] **Step 1: Write the failing test** `DccGradient.MatchesDual`, um teste de propriedade:
  - DCC, aDCC e FDCC × mvnorm, mvt e mvlaplace;
  - m ∈ {2, 3, 21};
  - FDCC com grupos de tamanhos 7/7/7 e 10/11;
  - pontos interior, persistência 0,995 e mvt com ν = 4,5 e 30.

  Asserções:
  - o valor de `evaluate` é bit-idêntico a `objective<double>`;
  - o gradiente bate com o de `objective<Dual<kMaxParameterCount>>` com |Δ| ≤ 1e-9 · max(1, ‖∇‖∞);
  - num ponto fora do domínio, o valor é +∞ como em `objective<double>`.

  `DccGradient.CoversListedSpecs` verifica quais specs `has_fast_gradient` aceita.
- [ ] **Step 2: Run** `--gtest_filter=DccGradient*`. Expected: FAIL (não compila).
- [ ] **Step 3: Implement**, por t, com os kernels da Task 3:
  - **Valor:** o mesmo caminho de `objective<double>` (recursão, normalização, Cholesky, solve).
  - **Tangentes:** dQ_t^(i), uma por parâmetro livre, com a mesma recursão linear de Q_t, mais o termo direto de
    ∂C/∂θ_i, ∂A/∂θ_i e ∂B/∂θ_i.
  - **Parciais da densidade de t:** a = ∂ℓ_t/∂log|R| e b = ∂ℓ_t/∂q, mais o direto na forma, por
    `log_density<ad::Dual<3>>` (lanes log|R|, q, forma) sobre a função existente.
  - **Para R_t:** G_R = a·R⁻¹ − b·w wᵀ, com w = R⁻¹z e R⁻¹ obtida do fator.
  - **Para Q_t:** pela normalização R = D Q D, com D = diag(Q)^-½. Fora da diagonal, G_Q = D G_R D. A diagonal recebe
    −½ Σ_j (G_R,ij + G_R,ji) R_ij / Q_ii (j ≠ i).
  - **Contração:** ∂ℓ_t/∂θ_i = ⟨G_Q, dQ_t^(i)⟩.

  O teste é o árbitro de cada fórmula. Depois ligar `MeanObjective`: ∇_x = Jᵀ ∂ℓ/∂θ pelo `Dual` de `to_natural`, como
  `NegativeMeanLoglik` no univariado.
- [ ] **Step 4: Run** o filtro e a suíte inteira. Expected: PASS; oráculos rmgarch verdes.
- [ ] **Step 5: Portão (design §5, decisão do autor).** A/B na mesma sessão de `BM_DccFit` (9 specs) e
  `BM_DccGrid/threads:1`, caminho novo contra o `Dual` (commit da Task 3).
  - **Go:** `BM_DccGrid/threads:1` < 0,9 do tempo. Registrar e commitar.
  - **No-go:** registrar os números em `profile-5b.md`, reverter o `src/fit.cpp` do DCC ao `Dual` (o arquivo
    `gradient.*` sai do build), e anotar `Ruling: no-go` no ledger. O 5b segue para a Task 5.
- [ ] **Step 6: Commit** `perf: DCC gradient by tangents of the recursion and a per-observation pullback` (ou, no
  no-go, `docs: record the DCC tangent gradient measurement`).

### Task 5: Laço vetorial único da densidade univariada, fundido com o adjunto

**Files:**
- Create: `src/simd/density_kernels.{hpp,cpp}`
- Modify:
  - `src/model/likelihood.cpp`: em `T = double`, norm, std e ged chamam o kernel de valor em todo modelo;
  - `src/model/likelihood/adjoint.cpp`: `density_backward` dá lugar ao kernel fundido;
  - `src/dist/symmetric.hpp`: `term_adjoint` fica como referência escalar dos testes, ou sai se nada mais o usar.
- Test: `tests/unit/simd/density_kernels_test.cpp`, `tests/unit/model/likelihood_test.cpp`,
  `tests/unit/model/loglik_adjoint_test.cpp`

**Interfaces:**
- Produces:
  ```cpp
  namespace ve::simd {
  // sum_t log f(e_t / sigma_t) - log(sigma2_t) / 2 under density (norm, std or ged), writing terms when not empty;
  // kInvalidLoglik when some sigma2_t is not positive and finite or the sum is not finite (as model::sum_of_terms).
  double symmetric_terms(const dist::LogDensity<double>& density, std::span<const double> e,
                         std::span<const double> sigma2, std::span<double> terms);
  // The same sum, bit for bit, plus d/de_t and d/dsigma2_t into the adjoint spans and the sums of d/dscale and
  // d/dpower into partials (TermAdjoint semantics of dist/symmetric.hpp).
  double symmetric_terms_with_partials(const dist::LogDensity<double>& density, SymmetricSeries series,
                                       SymmetricPartials& partials);
  }
  ```
  O agrupamento dos spans em `SymmetricSeries` e `SymmetricPartials` existe para respeitar o limite de 5 parâmetros.
  Os campos ficam a cargo do implementador.

- [ ] **Step 1: Write the failing tests**:
  - os dois kernels contra a referência escalar (`sum_of_terms` com `log_density`, e `term_adjoint`) em todo target,
    com T ∈ {1, 3, 7, 600, 5001}, norm, std (ν 2,5, 8, 30) e ged (ν 0,8, 1,5, 3): termo a termo ≤ 4 ulps nos valores,
    1e-13 relativo nas parciais;
  - o valor dos dois kernels é bit-idêntico entre si;
  - σ²_t ≤ 0 ou NaN no meio do vetor dá `kInvalidLoglik` exato;
  - resíduo exatamente zero com ged dá a mesma convenção do `Dual`.
- [ ] **Step 2: Run** `--gtest_filter=*DensityKernel*`. Expected: FAIL (não compila).
- [ ] **Step 3: Implement** os kernels. Ligar `loglik<double>` (só `T = double`) e o adjunto.
- [ ] **Step 4: Update the pair tests.**
  - `LikelihoodGrid.DoubleAndDualAgreeOnValue` e os testes que comparam `double` com `Dual` bit a bit passam a usar a
    tolerância medida, registrada no teste.
  - O teste de propriedade do adjunto continua exigindo o ℓ bit-idêntico a `loglik<double>`, e o gradiente contra o
    `Dual` em 1e-10.
- [ ] **Step 5: Run** a suíte inteira e o pytest. Expected: tudo verde, oráculos nas tolerâncias atuais. Um oráculo que
  mude vai ao autor com os números.
- [ ] **Step 6: Commit** `perf: one vector loop for the symmetric densities, fused with the adjoint`

### Task 6: Medida, plataformas e documentação

**Files:**
- Modify: `docs/performance/profile-5b.md`, `docs/performance/benchmarks.md`, `docs/performance/python-vs-arch.md`,
  `CHANGELOG.md`, `docs/superpowers/specs/2026-11-14-volengine-05b-simd-design.md` (§11 "feito")

- [ ] **Step 1: Platforms.** O preset `msvc-release` compila e passa a suíte. Clang, se estiver instalado: compila. Se
  não estiver, registrar a ausência.
- [ ] **Step 2: A/B na mesma sessão**, com o base no worktree do commit antes da Task 3:
  - `BM_DccFit` (9 specs), `BM_DccGrid` com 1 thread e com todas;
  - `BM_TccGrid` com 1 thread e com todas, com e sem erros-padrão;
  - `BM_Loglik`, `BM_FitOne`, `BM_DccStandardErrors`, `BM_CopulaFit`;
  - `grid_profile` com `compare-profile` (status e ℓ);
  - Python contra `arch`.

  Expected: o pipeline (`BM_TccGrid` + `BM_DccGrid`, ambos com todas as threads) cai em relação a 2,72 + 2,35 s.
- [ ] **Step 3: Write** os números nos docs. No CHANGELOG, em "Performance" e em "Changed": `double` e `Dual` diferem
  em ulps, e os resultados entre ISAs batem só dentro de tolerância.
- [ ] **Step 4: Run** `ctest --preset gcc-release -R "^style$"`. Expected: PASS.
- [ ] **Step 5: Commit** `docs: record the plan 5b benchmarks`
