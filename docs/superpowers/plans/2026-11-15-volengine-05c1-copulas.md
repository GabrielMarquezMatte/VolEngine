# Plano 5c-1 — Cópulas Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Tirar a cópula DCC-t dos ~36 s (e a DCC-mvnorm de ~0,9 s) com quatro mudanças: o gradiente por tangentes do
5b no ajuste interno, partida a quente no perfil de ν, Kendall pelo algoritmo de Knight e a correção da cópula
estática t que hoje falha em m = 21.

**Architecture:** O ajuste interno de cada ponto do perfil já é o objetivo DCC comum sobre o painel de scores. Basta
deixar `has_fast_gradient` aceitar specs de cópula. O perfil de Brent (`profile_shape` em `src/dcc/copula/fit.cpp`)
passa a guardar os pontos avaliados. Cada ajuste novo parte do ponto de ν mais próximo, e o ponto final reaproveita o
que o Brent já avaliou. O τ_b de Kendall troca o laço O(T²) por ordenação com contagem de inversões, com as mesmas
contagens inteiras.

**Tech Stack:** C++23, Boost.Math (Brent, quantis), GoogleTest, Google Benchmark.

**Spec:** `docs/superpowers/specs/2026-11-15-volengine-05c-design.md` (§3–§9). Design-mãe:
`docs/superpowers/specs/2026-11-14-volengine-05-performance-design.md`.

## Global Constraints

- **Perfil:** intervalo [4, 50] (`kShapeFloor`, `kShapeCeiling`), 30 bits, 100 iterações. Nada disso muda.
- **Gradiente:** o valor de `ObjectiveGradient::evaluate` é bit-idêntico a `objective<double>`, e o gradiente bate com
  o `Dual` a 1e-9 da maior componente (`kGradientTolerance`).
- **Kendall:** τ_b bit-idêntico ao laço atual.
- **Interface C e Python:** nenhuma assinatura muda.
- **Oráculos** `rmgarch` das cópulas verdes nas tolerâncias atuais:
  - `CopulaOracle.*`;
  - `CopulaStatic.MatchesRmgarch`;
  - `CopulaFit.MatchesRmgarchAtItsPoint`;
  - `python/tests/oracles/test_copula_oracles.py`.

  Um xfail estrito que passar sai do xfail.
- **Bit-identidade entre números de threads** e entre um ajuste sozinho e o mesmo ajuste num lote misto
  (`DccBatch.MixedBatchSameAsAlone`).
- **Estilo:** `STYLEGUIDE.md` (≤ 60 linhas por função, ≤ 5 parâmetros, aninhamento ≤ 3, sem `else`, ≤ 8 módulos por
  pasta).
- **Testes** só no preset `gcc-release`:
  - `C:\msys64\ucrt64\bin` primeiro no PATH;
  - `ctest --preset gcc-release -j 8` roda a suíte inteira;
  - `build/gcc-release/volengine_tests.exe --gtest_filter=X` roda um recorte;
  - pytest com `VOLENGINE_LIB=build/gcc-release/libvolengine.dll`.
- **Medidas no `gcc-bench`,** A/B na mesma sessão.
  - O "antes" é a build de `6f7053d` (igual a `4ed19d8` no código), feita num worktree em `build/base-5c1/`.
  - Os benchmarks de ajuste de cópula têm `Iterations(1)`: use `--benchmark_repetitions=3` no A/B e 5 na tabela
    final.
- **Código e docs do repo** em inglês; `docs/superpowers` em português. Commits sem `Co-Authored-By`.

## Review Focus

1. **Ponto do perfil que falha** (ℓ não finito, ou `fit_dcc` `Failed`) virando a partida a quente do vizinho. A
   partida só vem de pontos com ℓ finito; sem nenhum, usa o início da spec. Teste na Task 3.
2. **Estimativa vizinha na fronteira** (aDCC com g ≈ 4,8e-8, a + b perto de 1) usada como partida. O ajuste interno
   tem que aceitar uma partida assim sem falhar nem ficar no ponto inicial. Teste na Task 3, com a aDCC-t do copula3.
3. **`spec.start` dado pelo usuário com ν livre:** o primeiro ponto do perfil respeita o início do usuário; os
   seguintes partem do vizinho. Teste na Task 3.
4. **Empates nas uniformes** (cópula empírica: postos repetidos; colunas com valores iguais) no Kendall de Knight:
   contagens iguais às do laço. Teste na Task 4.
5. **Cópula estática t em m grande** (o caso que hoje falha): ajuste finito e `CONVERGED`. Teste na Task 1.

---

### Task 1: Base de medida e a falha da cópula estática t em m = 21

**Files:**
- Create: `docs/performance/profile-5c.md`
- Modify: o arquivo da causa, a decidir pelo diagnóstico (provavelmente `src/dcc/copula/fit.cpp` ou
  `src/dcc/copula/static_fit.cpp`)
- Test: `tests/unit/dcc/copula/static_fit_test.cpp`

**Interfaces:**
- Produces: `build/base-5c1/volengine_benchmarks.exe` (a base do A/B das Tasks 2–4) e
  `build/gcc-bench/copula_base.json`.

- [ ] **Step 1: Montar a base.** `git worktree add build/base-5c1-src 6f7053d`, depois configurar e compilar o preset
  `gcc-bench` desse worktree com saída em `build/base-5c1/`.

- [ ] **Step 2: Rodar a base** com `--benchmark_filter='BM_CopulaFit|BM_CopulaGrid|BM_CopulaStandardErrors|BM_DccFit/(dcc_mvnorm|dcc_mvt|adcc_mvt)|BM_DccGrid/m:21/T:5000/threads:1'`,
  `--benchmark_repetitions=3`, saída `build/gcc-bench/copula_base.json`. Esperado: os mesmos erros de
  `static_mvt_{kendall,ml}_param` ("the DCC fit has no finite point") e `dcc_mvt_param` em ~35 s.

- [ ] **Step 3: Diagnosticar a falha** com superpowers:systematic-debugging.
  - Em `4ed19d8`, a falha é a mesma da base.
  - O plano 4b registrou esses dois casos funcionando em `c5cb446` (5,27 s e 2,11 s). Se a causa não aparecer
    lendo o código, `git bisect` entre `c5cb446` e `4ed19d8` com `BM_CopulaFit/static_mvt_ml_param`.
  - Registrar a causa em `profile-5c.md` em duas linhas.

- [ ] **Step 4: Escrever o teste que falha** `CopulaStatic.StudentFitsAtManySeries` em `static_fit_test.cpp`. Usar o
  menor painel que reproduz a falha: m = 21, como no benchmark, ou menos se o diagnóstico mostrar que basta.
  - Para Kendall e ML: `fit.status == FitStatus::Converged`, `std::isfinite(fit.loglik)` e ν em (4, 50).
  - Em ML, também ℓ ≥ ℓ do Kendall no mesmo painel menos 1e-6·|ℓ|.

  Rodar e ver falhar.

- [ ] **Step 5: Corrigir na causa,** rodar o teste e o recorte `--gtest_filter='Copula*:DccBatch*'`. Esperado: PASS.

- [ ] **Step 6: Commit.** `git commit -m "fix: <causa> in the static Student copula"` com o teste e `profile-5c.md`
  (a base, a causa).

### Task 2: Cópulas DCC/aDCC/FDCC no gradiente por tangentes

**Files:**
- Modify: `src/dcc/likelihood/gradient.cpp:375-380` (`has_fast_gradient`), `src/dcc/likelihood/gradient.hpp`
  (comentário)
- Test: `tests/unit/dcc/likelihood/gradient_test.cpp`

**Interfaces:**
- Produces: `has_fast_gradient(spec)` verdadeiro para DCC/aDCC/FDCC × mvnorm/mvt/mvlaplace com qualquer `copula`; falso
  para `DccModel::Static`.

- [ ] **Step 1: Testes que falham.**
  - Em `CoversListedSpecs`, a cópula DCC-t paramétrica passa a dar `EXPECT_TRUE`; a estática continua `EXPECT_FALSE`.
  - Novo `DccGradient.MatchesDualOnCopulaScores`:
    - os painéis são `copula::copula_panel(inputs.uniforms_of(spec), *inputs.univariate, dist, nu)` do copula3 (os
      helpers de `tests/unit/dcc/copula/copula3_panel.hpp`);
    - casos: `DccCopula::Parametric` e `Empirical` × `Convention::Standard` e `Rmgarch` × DCC/aDCC × mvnorm/mvt, com
      ν = 6 e persistência 0,93;
    - as mesmas asserções de `MatchesDual`: `EXPECT_EQ` do valor contra `objective<double>` e `EXPECT_NEAR` a
      `kGradientTolerance * scale`.

  Rodar e ver falhar só o `CoversListedSpecs`: o `MatchesDualOnCopulaScores` já passa, porque o avaliador não lê
  `copula`. Se não passar, parar: o objetivo não é o mesmo.

- [ ] **Step 2: Implementar.** Em `has_fast_gradient`, remover `spec.copula == DccCopula::None`. O comentário diz que
  o painel de scores de uma cópula é um painel DCC comum.

- [ ] **Step 3: Rodar** `--gtest_filter='DccGradient*:Copula*:DccBatch*'` e o pytest de
  `test_copula_oracles.py`. Esperado: PASS. `CopulaFit.MvnormIsDccOnScores` continua com `EXPECT_EQ`, porque os dois
  lados passam pelo mesmo caminho.

- [ ] **Step 4: A/B.** Rodar o mesmo filtro da Task 1, Step 2, e depois `python tests/benchmarks/quick.py compare`
  contra `copula_base.json`.
  - Esperado: `BM_CopulaFit/dcc_mvnorm_param` perto de 0,15 s e `dcc_mvt_param` perto de 35 × 0,15 s.
  - `BM_DccFit` e `BM_DccGrid` iguais dentro do ruído.
  - Registrar a tabela em `profile-5c.md`.

- [ ] **Step 5: Commit.** `git commit -m "perf: copula DCC fits on the tangent gradient"`.

### Task 3: Perfil de ν com partida a quente

**Files:**
- Modify: `src/dcc/copula/fit.cpp` (`ProfilePoint`, `dynamic_point`, `static_point`, `point_at`, `profile_shape`,
  `fit_copula`), `src/dcc/copula/fit.hpp:63-65` (comentário), `src/dcc/copula/static_fit.hpp` e `.cpp`
  (`static_ml` recebe a partida)
- Test: `tests/unit/dcc/copula/fit_test.cpp`

**Interfaces:**
- Consumes: Task 2 (o ajuste interno já é rápido; esta task não depende disso para estar correta).
- Produces:
  - `struct ProfilePoint { FitStatus status; std::vector<double> theta; double loglik; double nu; }`. Sai do
    namespace anônimo de `fit.cpp` para `ve::dcc::copula::detail` em `fit.hpp`, junto de `nearest_finite`. `theta` é
    completo, na ordem do layout; `nu` é o ponto avaliado.
  - `point_at(inputs, spec, nu, config, const ProfilePoint* warm)`. Com `warm == nullptr`, o comportamento é o de
    hoje.
  - `static_ml(scores, layout, shape, config, std::span<const double> start)`. `start` vazio lê `layout.given`, como
    hoje.

- [ ] **Step 1: Testes que falham** em `fit_test.cpp`:
  - `CopulaFit.ProfileWarmStartKeepsTheMaximum`: DCC-t e aDCC-t paramétricas do copula3.
    - ν dentro de 1e-4·ν do ν de referência medido na base. Antes de mudar o código, calcular em `6f7053d` e gravar
      no teste: ν, ℓ e a estimativa, com 17 dígitos. A tolerância de ν é larga porque o perfil é plano perto do
      máximo; o critério que vale é o de ℓ.
    - ℓ ≥ ℓ_ref − 1e-8·|ℓ_ref|.
    - Para a aDCC (Review Focus 2): status `Converged` e g ≥ 0.
  - `CopulaFit.ProfileIsDeterministic`: duas chamadas de `fit_copula` na mesma spec dão `estimate` e `loglik` iguais
    bit a bit.
  - `CopulaFit.ProfileFirstPointHonoursUserStart` (Review Focus 3): uma spec com `start = {{"dcca1", 0.03},
    {"dccb1", 0.95}}` converge para o mesmo ν (1e-4 relativo) que a spec sem `start`.
  - `CopulaFit.ProfileSkipsFailedNeighbours` (Review Focus 1): a escolha do vizinho exposta como função livre
    testável, `const ProfilePoint* nearest_finite(std::span<const ProfilePoint>, double nu)`, declarada em
    `fit.hpp` no namespace `ve::dcc::copula::detail`. Casos:
    - devolve `nullptr` com só pontos de ℓ NaN;
    - escolhe o de |Δν| menor;
    - num empate de |Δν|, escolhe o de ν menor.

  Rodar e ver falhar (não compila: `nearest_finite`).

- [ ] **Step 2: Implementar.**
  - `profile_shape` guarda os `ProfilePoint` avaliados num vetor local, em ordem de avaliação.
  - Cada chamada do Brent pede `nearest_finite(pontos, nu)` e passa o resultado a `point_at`.
  - `dynamic_point` com `warm`: `inner_spec.start` passa a ser os nomes estimados de `layout_of(inner_spec, m)` com
    os valores de `warm->theta`. O `start` do usuário fica só no primeiro ponto.
  - `static_point` com `warm`: passa `warm->theta[c_begin .. c_begin + c_count)` como `start` do `static_ml`.
  - `profile_shape` devolve também o ponto do ν vencedor, procurado pelos bits de `best.first`. `fit_copula` usa esse
    ponto em vez de chamar `point_at` de novo. Sem ν livre, a chamada única segue com `warm == nullptr`.

  Atualizar o comentário de `fit_copula` em `fit.hpp` (a partida a quente e o ponto reaproveitado). Em
  `profile_shape`, um `ponytail:` aponta o próximo passo: Newton no perfil pelo teorema do envelope (spec §5), se a
  cópula t ainda ficar muito acima do DCC-t comum.

- [ ] **Step 3: Rodar** `--gtest_filter='Copula*:DccBatch*:CopulaOracle*'` e o pytest de cópulas. Esperado: PASS,
  incluindo `ProfileFindsTheShape`, `ShapeAtTheCeiling` e `MixedBatchSameAsAlone`.

- [ ] **Step 4: A/B** como na Task 2, Step 4. Esperado: `dcc_mvt_param` e `adcc_mvt_param` abaixo do resultado da
  Task 2. A estática t também ganha, pela partida do ML. Registrar em `profile-5c.md`, com o número de pontos do
  perfil e as iterações médias do ajuste interno, antes e depois.

- [ ] **Step 5: Commit.** `git commit -m "perf: warm-start the copula shape profile from the nearest point"`.

### Task 4: Kendall pelo algoritmo de Knight

**Files:**
- Modify: `src/dcc/copula/static_fit.cpp:28-46` (`tau_b`), `src/dcc/copula/static_fit.hpp:18` (sai o `ponytail:`)
- Test: `tests/unit/dcc/copula/static_fit_test.cpp`

**Interfaces:**
- Produces: `kendall_correlation(u, t, m)` com a mesma assinatura e os mesmos bits.

- [ ] **Step 1: Teste que falha (ou que fixa os bits)** `CopulaStatic.KendallMatchesPairCount`.
  - Copiar para o teste o `tau_b` O(T²) atual como referência local.
  - Comparar com `EXPECT_EQ`, entrada por entrada, a matriz de `kendall_correlation` com sin(π τ_b / 2) da
    referência, nestes painéis:
    - aleatório m = 4, T = 257;
    - com empates numa coluna (valores arredondados a 1/16);
    - com empates conjuntos nas duas colunas;
    - T = 2;
    - colunas iguais (τ = 1).

  Com o código atual passa. O teste fica como guarda da troca.

- [ ] **Step 2: Implementar `tau_b` por Knight** (1966), com contagens em `std::int64_t`:
  1. Ordenar os índices por (x, y). Contar n1, os pares empatados em x, e n3, os pares empatados em (x, y), pelos
     comprimentos dos blocos iguais.
  2. Com merge sort estável em y sobre essa ordem, contar as trocas `swaps`. Depois contar n2, os pares empatados em
     y, pelos blocos de y já ordenado.
  3. `score = n0 − n1 − n2 + n3 − 2·swaps`, com n0 = T(T−1)/2.
  4. Devolver `static_cast<double>(score) / std::sqrt((pairs - ties_a) * (pairs - ties_b))`, com `pairs`, `ties_a` =
     n1 e `ties_b` = n2 em `double`. É a mesma expressão de hoje, para os mesmos bits.

  As colunas a e b saem de u com passo m, para um buffer local por par (ou um por coluna, ordenado uma vez se medir
  melhor).

- [ ] **Step 3: Rodar** `--gtest_filter='CopulaStatic*:CopulaOracle*:CopulaFit*'`. Esperado: PASS sem tolerância
  nova.

- [ ] **Step 4: A/B** de `BM_CopulaFit/static_mvnorm_kendall_param` e `static_mvt_kendall_param`. Esperado: de ~3,8 s
  para a ordem de ms no Kendall. Registrar em `profile-5c.md`.

- [ ] **Step 5: Commit.** `git commit -m "perf: Kendall's tau-b by Knight's algorithm"`.

### Task 5: Medir o resto, tabelas finais e documentação

**Files:**
- Modify: `docs/performance/benchmarks.md` (seção "Copula-DCC workload (plan 4b)": sai o "Pending"; entram as 5
  repetições), `docs/performance/profile-5c.md`, `docs/superpowers/specs/2026-11-15-volengine-05c-design.md` (seção
  nova "10. 5c-1 feito"), `CHANGELOG.md` (`[Unreleased]`: em "Fixed", a cópula estática t; em "Changed", o caminho
  das estimativas de cópula com ν livre, que se movem dentro da tolerância; no padrão das entradas do 5b).

- [ ] **Step 1: Tabela final.**
  - Rodar `BM_CopulaFit`, `BM_CopulaGrid` e `BM_CopulaStandardErrors` com 5 repetições, base e nova, na mesma sessão.
  - Rodar também os secundários `BM_DccFit` e `BM_DccGrid/threads:1` e o conjunto rápido (`quick.py run`/`compare`).
  - Esperado: nenhum secundário pior acima do ruído.

- [ ] **Step 2: Medir o ML estático e os erros-padrão da cópula.**
  - Pegar o tempo de `static_mvt_ml_param` e de `BM_CopulaStandardErrors`.
  - Se um deles passar de 1 s, fazer um profile (`msvc-profile`, como em `measuring.md`) e registrar onde está o
    tempo, sem otimizar.
  - O que aparecer volta ao autor como candidato.

- [ ] **Step 3: Suíte inteira.** `ctest --preset gcc-release -j 8` e `pytest python/tests` com a DLL do
  `gcc-release`. Esperado: tudo verde; xfails estritos que passarem saem do xfail.

- [ ] **Step 4: Documentação.**
  - `benchmarks.md` e `profile-5c.md` com as tabelas.
  - Na spec do 5c, a seção 10 com:
    - os números;
    - a causa da falha estática;
    - o que ficou para o autor.
  - Remover `git worktree` da base (`git worktree remove build/base-5c1-src`).

- [ ] **Step 5: Commit.** `git commit -m "docs: record the plan 5c-1 benchmarks"`.
