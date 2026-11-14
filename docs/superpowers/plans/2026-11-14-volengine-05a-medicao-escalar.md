# VolEngine — Plano 5a: medição e caminho escalar

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** saber em segundos onde está o tempo da grade de 1080 ajustes e deixá-la o mais rápida possível no caminho
escalar: ferramental de medição, profile publicado, opção de erros-padrão, `lgamma` sem corrida, L-BFGS que não
para em ridge e as otimizações que o profile apontar.

**Architecture:**
- **Contadores internos:** o núcleo conta o trabalho de cada ajuste em `FitResult::work` (iterações, avaliações,
  segundos totais e do otimizador). Nada disso passa pela interface C, e o tempo nunca entra numa conta.
- **Ferramentas de medição** em `tests/benchmarks/`:
  - um executável `volengine_grid_profile`, que roda a grade (ou um recorte) por `ve::run_batch` e agrega o trabalho
    por spec, distribuição, ordem e ARMA;
  - benchmarks pequenos no Google Benchmark;
  - um script `quick.py` que roda o conjunto rápido e compara duas rodadas.
- **Única adição à interface C:** `ve_fit_options_set_standard_errors`. Desligada, o ajuste não calcula a inferência
  no ponto estimado: Hessiana numérica, robustos, scores e Nyblom.
- **Otimizador:** a correção do ridge e as otimizações escalares ficam em `src/optim/` e `src/fit/`. Cada uma entra com
  A/B medido.

**Tech Stack:** C++23, CMake + presets, GoogleTest v1.17.0, Google Benchmark v1.9.1, Boost.Math 1.87 (standalone),
BS::thread_pool v5.0.0, AMD uProf (`C:/Program Files/AMD/AMDuProf/bin/AMDuProfCLI.exe`), Python ≥ 3.11 + numpy +
pytest + mypy.

**Spec:** `docs/superpowers/specs/2026-11-14-volengine-05-performance-design.md`, sobre a spec-mãe
`docs/superpowers/specs/2026-11-10-volengine-design.md`, seção 8. Leia também `docs/performance/benchmarks.md` e
`STYLEGUIDE.md` (Priorities e Performance Notes).

## Roteiro

- **1, 2a–2f, 3a–3c, 4a–4c (feitos).**
- **5a (este):** medição, profile e caminho escalar.
- **Depois:** 5b, SIMD, redesenhado com os números de 5a; 5c, os outros pontos quentes; 6, wrapper C++, wheels e 1.0.

## Global Constraints

- O número oficial é o do preset `gcc-bench` (genérico, sem `-march=native`). Um número `native-release` só aparece
  marcado como informação extra.
- **Carga de referência:** `BM_TccGrid`, 1080 ajustes:
  - 20 séries × T = 5000;
  - {p, q ∈ 1..3} × {ARMA(0,0), ARMA(1,1)} × {norm, std, ged}.

  Piso: abaixo de 10 s com todas as threads no Ryzen 7 5700X. O piso não encerra o plano.
- **Oráculos:** `rugarch`, `rmgarch` e `arch` continuam verdes nas mesmas tolerâncias. ℓ do VolEngine ≥ ℓ do oráculo
  menos a tolerância. Um xfail estrito que passar a passar sai da lista.
- **Bit-identidade** entre números de threads na mesma máquina e ISA. O resultado de um job não depende de quantas
  threads rodam nem dos outros jobs do lote.
- **Interface C:** nenhuma assinatura existente muda. A única adição é `ve_fit_options_set_standard_errors`.
- **`tol`** mantém o significado documentado: max|g| ≤ tol·max(1, |f|), padrão 1e-6, `max_iter` 500.
- **Mudanças no otimizador** são validadas além da grade: `BM_FitOne` de `named_specs`, `BM_DccGrid` e oráculos. Uma
  piora acima do ruído bloqueia a mudança até o autor aceitar.
- **Toda otimização** registra antes/depois medidos em sequência, na mesma sessão, em `docs/performance/`.
- **Estilo:** o `STYLEGUIDE.md` inteiro. Código, identificadores e `docs/` (fora de `docs/superpowers/`) em inglês.
- **Testes** só no preset `gcc-release` (pedido do autor): `cmake --build --preset gcc-release` e
  `ctest --preset gcc-release`. Python com
  `VOLENGINE_LIB=$PWD/build/gcc-release/libvolengine.dll python/.venv/Scripts/python -m pytest python/tests -q`.
- **Executáveis** rodam com `/c/msys64/ucrt64/bin` primeiro no `PATH`. A libstdc++ do mingw64 do Git não tem os
  símbolos.
- **Commits** sem trailer `Co-Authored-By`.

## Review Focus

1. **Opção desligada × consumidores dos dados que ela pula.** O DCC (`ve_dcc_fit_batch`, `ve_dcc_fit_standard_errors`)
   recebe `ve_fit` ajustados com a opção desligada. Esperado: funciona igual, porque o DCC refiltra cada série pela
   própria spec e não lê `se_hessian` nem `nyblom`. Teste na Task 5.
2. **Opção desligada × forecast e simulação.** `ve_forecast_create` e `ve_simulate_fit` sobre um ajuste sem erros-padrão
   dão os mesmos números que com eles. Teste na Task 5.
3. **Ajuste `FAILED` ou `INSUFFICIENT_DATA` com a opção desligada.** Os acessores de erro-padrão continuam devolvendo
   o erro da opção, não NaN silencioso. O status e o resto do ajuste ficam iguais aos de antes. Teste na Task 5.
4. **Correção do ridge × resolução de f.** Uma função que não pode descer mais, por arredondamento, continua
   `Converged` (o teste existente `LineSearchFailureWithNearlySmallGradientConverges`). Não pode virar `Failed` em
   massa nos ajustes que hoje convergem. Coberto pela Task 7 (contagem de status na grade antes/depois).
5. **Profile com contadores × bit-identidade.** Medir o tempo dentro de `fit_one` não muda nenhum bit da estimativa.
   Coberto pelos testes `Deterministic`/`DeterministicAcrossBuckets` existentes, que continuam passando, e pela
   Task 1.

---

### Task 1: Contadores de trabalho do ajuste

**Files:**
- Modify: `src/optim/lbfgs.hpp`, `src/optim/line_search.hpp`
- Modify: `src/fit/fit_result.hpp`, `src/fit/fit.cpp`
- Test: `tests/unit/optim/lbfgs_test.cpp`, `tests/unit/fit/fit_test.cpp`

**Interfaces:**
- Produces:
  - `optim::LbfgsResult` ganha `int evaluations`: chamadas da função objetivo, incluída a do ponto inicial.
  - `struct ve::FitWork { int iterations = 0; int evaluations = 0; double seconds = 0.0; double optimizer_seconds = 0.0; };`
    em `fit_result.hpp`;
  - `FitResult::work`, comentado como "work of the fit, for profiling; not in the C API; timing never feeds the
    computation".

- [ ] **Step 1: Escrever os testes que falham**

```cpp
// lbfgs_test.cpp
TEST(Lbfgs, CountsEvaluations)
{
    int calls = 0;
    auto counted = [&calls](std::span<const double> x, std::span<double> g)
    {
        ++calls;
        return weighted_quadratic(x, g);
    };
    std::vector<double> x(10, 1.0);
    LbfgsResult result = ve::optim::lbfgs_minimize(counted, x, {});
    EXPECT_EQ(result.evaluations, calls);
    EXPECT_GE(result.evaluations, result.iterations + 1);
}

// fit_test.cpp
TEST(Fit, ReportsItsWork)
{
    FitResult result = fit_simulated(Spec{});
    ASSERT_EQ(result.status, FitStatus::Converged);
    EXPECT_GT(result.work.iterations, 0);
    EXPECT_GE(result.work.evaluations, result.work.iterations + 1);
    EXPECT_GT(result.work.optimizer_seconds, 0.0);
    EXPECT_GE(result.work.seconds, result.work.optimizer_seconds);
}

TEST(Fit, InsufficientDataDoesNoOptimizerWork)
{
    FitResult result = ve::fit_one(std::span<const double>(simulated()).first(3), Spec{}, ve::FitConfig{});
    EXPECT_EQ(result.work.iterations, 0);
    EXPECT_EQ(result.work.evaluations, 0);
}
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `cmake --build --preset gcc-release` (falha de compilação: `evaluations` e `work` não existem).

- [ ] **Step 3: Implementar**
  - `LineSearch` acumula as avaliações (o `evaluate` já conta `trials_used_` por busca). `Minimizer::finish` devolve o
    total mais 1, a avaliação do ponto inicial.
  - `fit_one(const model::Data&, ...)` mede `seconds` com `std::chrono::steady_clock` em volta de `estimate`.
  - `estimate_parameters` mede `optimizer_seconds` em volta do `lbfgs_minimize` e copia `iterations` e `evaluations`.

- [ ] **Step 4: Rodar a suíte**

Run: `cmake --build --preset gcc-release && ctest --preset gcc-release`
Expected: tudo verde, inclusive `Fit.Deterministic` e `FitGeneric.DeterministicAcrossBuckets`.

- [ ] **Step 5: Commit**

```bash
git add src/optim src/fit tests/unit/optim/lbfgs_test.cpp tests/unit/fit/fit_test.cpp
git commit -m "feat: count optimizer work and time per fit"
```

---

### Task 2: Executável de profile da grade

**Files:**
- Create: `tests/benchmarks/grids.hpp`: `make_spec`, `kSymmetricDists`, `kAllDists`, `tcc_grid` e `spec_label`. Os quatro
  primeiros saem de `fit_benchmarks.cpp`.
- Create: `tests/benchmarks/grid_profile.cpp`
- Modify: `tests/benchmarks/fit_benchmarks.cpp` (usa `grids.hpp`), `CMakeLists.txt` (alvo `volengine_grid_profile`,
  dentro de `if(VE_BUILD_BENCHMARKS)`, mesmas opções e link de `volengine_benchmarks` sem `benchmark::benchmark`)

**Interfaces:**
- Consumes: `FitResult::work` (Task 1); `ve::run_batch(Context&, const Batch&)` e `ve::Batch` (`src/fit/batch.hpp`);
  `ve::bench::simulate_panel` e `simulated_model_for` (`simulated_panel.hpp`).
- Produces:
  - `std::string spec_label(const ve::Spec&)`, no formato `sgarch_p1q1_arma00_norm` (`arma11` com ARMA(1,1));
  - o executável `volengine_grid_profile [--series N] [--threads N] [--filter TEXT] [--standard-errors on|off] [--json PATH]`.
    - `--series`: 1..20, padrão 20.
    - `--threads`: 0 = hardware, o padrão.
    - `--filter`: substring de `spec_label`.
    - `--standard-errors`: padrão `on`. A opção só existe a partir da Task 5; até lá, `off` é erro, com a mensagem
      "needs ve_fit_options_set_standard_errors".
    - `--json`: grava um registro por ajuste:
      `{"spec", "series", "status", "seconds", "optimizer_seconds", "iterations", "evaluations", "loglik"}`.

- [ ] **Step 1: Mover a grade para `grids.hpp`**, sem mudar `BM_TccGrid`. O painel continua sendo o de sGARCH(1,1)-std
  (ν = 8) de `run_batch`.

- [ ] **Step 2: Escrever `grid_profile.cpp`**
  - Jobs em ordem spec-major, como `BatchWorkload`.
  - Mede o tempo real de `ve::run_batch` e imprime:
    - **cabeçalho:** `fits`, `threads`, `wall_s`, `fit_s_sum` e `pool_use` = `fit_s_sum / (threads × wall_s)`;
    - **quatro tabelas,** por spec (as 20 de maior tempo total), por distribuição, por ordem `pXqY` e por ARMA;
    - **colunas:** `group fits total_s share% mean_ms iter_mean eval_mean eval_per_iter optimizer% converged max_iter other`;
    - cada tabela ordenada por `total_s`, decrescente.
  - Funções ≤ 60 linhas, ≤ 5 parâmetros, sem `else`.

- [ ] **Step 3: Compilar e rodar um recorte**

Run: `cmake --preset gcc-bench && cmake --build --preset gcc-bench && build/gcc-bench/volengine_grid_profile --series 1 --threads 1 --filter p1q1 --json build/gcc-bench/profile_smoke.json`
Expected:
- 6 ajustes (p1q1 × 2 ARMA × 3 distribuições);
- `pool_use` perto de 1,0 com 1 thread;
- JSON com 6 registros, todos com `iterations > 0`.

- [ ] **Step 4: Rodar a grade inteira e conferir contra o benchmark**

Run: `build/gcc-bench/volengine_grid_profile --json build/gcc-bench/profile_full.json`
Expected:
- 1080 ajustes;
- `wall_s` na faixa de `BM_TccGrid/threads:hw` (~10–11 s);
- `converged + max_iter = 1080`.

- [ ] **Step 5: Commit**

```bash
git add tests/benchmarks CMakeLists.txt
git commit -m "bench: add grid profile executable"
```

---

### Task 3: Conjunto rápido, A/B e preset de profile

**Files:**
- Modify: `tests/benchmarks/fit_benchmarks.cpp`
- Create: `tests/benchmarks/quick.py` (só biblioteca padrão)
- Modify: `CMakePresets.json` (preset `gcc-profile`)
- Create: `docs/performance/measuring.md`

**Interfaces:**
- Consumes: `FitResult::work` (Task 1); `tcc_grid`/`make_spec`/`kSymmetricDists` (Task 2); o JSON do profile (Task 2).
- Produces:
  - `BM_TccGridSample/threads:1`: 18 ajustes, 1 thread, um por (p, q, ARMA).
    - p, q ∈ 1..3 e ARMA ∈ {(0,0), (1,1)}.
    - Distribuição `kSymmetricDists[(p + q + arma) % 3]`, com `arma` 0 ou 1.
    - Série: `simulate_panel(simulated_model_for(heavy_tails), kObservations, 1)`.
  - Contadores `iterations` e `evaluations` em todo `BM_FitOne`.
  - `python tests/benchmarks/quick.py run --out FILE [--exe PATH]`:
    - roda o filtro fixo `QUICK_FILTER` com `--benchmark_repetitions=3 --benchmark_report_aggregates_only=true`;
    - o filtro cobre `BM_Loglik/{s11_norm,s21_std,arma11_s11_ged,arma22_s33_std}/{double,dual}`,
      `BM_FitOne/{s11_norm,s21_std,arma11_s11_ged,arma22_s33_std,egarch11_std}` e `BM_TccGridSample`.
  - `python tests/benchmarks/quick.py compare A.json B.json`:
    - uma linha por benchmark: mediana A, mediana B, razão B/A;
    - `noise` quando |B − A| ≤ max(sd A, sd B).
  - `python tests/benchmarks/quick.py compare-profile A.json B.json`, sobre dois JSON do profile:
    - razão do tempo total;
    - razão das avaliações;
    - o pior Δℓ por ajuste, (ℓB − ℓA)/|ℓA|;
    - a contagem de status de cada lado.
  - `python tests/benchmarks/quick.py selftest`: asserts sobre JSONs sintéticos em memória, um para cada comando
    `compare`.

- [ ] **Step 1: Escrever `quick.py selftest` e ver falhar** (os comandos ainda não existem)

Run: `python tests/benchmarks/quick.py selftest`
Expected: falha (`compare` não implementado).

- [ ] **Step 2: Implementar `quick.py`** com `argparse` e `json`. Lê `aggregate_name` `median`/`stddev` e `real_time`
  do JSON do Google Benchmark.

- [ ] **Step 3: Rodar o selftest e o style**

Run: `python tests/benchmarks/quick.py selftest && python tests/style/check_style.py`
Expected: os dois saem 0.

- [ ] **Step 4: Adicionar `BM_TccGridSample` e os contadores**, depois rodar o conjunto rápido

Run: `cmake --build --preset gcc-bench && python tests/benchmarks/quick.py run --out build/gcc-bench/quick_base.json`
Expected:
- termina em cerca de 1 minuto;
- `BM_TccGridSample` abaixo de ~1,5 s por repetição;
- `BM_FitOne/*` com `iterations` e `evaluations` no JSON.

- [ ] **Step 5: Preset `gcc-profile`**
  - Herda `gcc-bench` e acrescenta `CMAKE_CXX_FLAGS = "-g -fno-omit-frame-pointer"`.
  - Tem build preset de mesmo nome.

  Run: `cmake --preset gcc-profile && cmake --build --preset gcc-profile`, e depois
  `"C:/Program Files/AMD/AMDuProf/bin/AMDuProfCLI.exe" collect --config tbp -o build/uprof build/gcc-profile/volengine_grid_profile.exe --series 2 --threads 1`
  e `AMDuProfCLI report` sobre a sessão.

  Expected: as funções quentes aparecem com nome.
  - Se o uProf não ler os símbolos DWARF do MinGW, use um preset MSVC com `/Zi` (PDB) só para localizar (spec §4).
  - Anote em `measuring.md` qual caminho funcionou.

- [ ] **Step 6: Escrever `docs/performance/measuring.md`** (inglês). Um comando curto para cada:
  - conjunto rápido;
  - A/B (`quick.py run` antes e depois, seguidos de `compare`);
  - profile da grade, com `--json` e `compare-profile`;
  - uProf.

  Mais a regra de A/B na mesma sessão. Linkar de `docs/performance/benchmarks.md`.

- [ ] **Step 7: Commit**

```bash
git add tests/benchmarks CMakePresets.json docs/performance
git commit -m "bench: add quick suite, A/B script and profile preset"
```

---

### Task 4: Profile de linha de base e checkpoint com o autor

**Files:**
- Create: `docs/performance/profile-5a.md` (inglês)

**Interfaces:**
- Consumes: as ferramentas das Tasks 2 e 3.
- Produces: `build/gcc-bench/profile_base.json` e `build/gcc-bench/quick_base.json` (locais, não commitados), a base dos
  A/B das Tasks 5–8.

- [ ] **Step 1: Medir**, com a máquina sem outra carga pesada:
  - `volengine_grid_profile --json build/gcc-bench/profile_base.json` (todas as threads);
  - `volengine_grid_profile --threads 1 --json build/gcc-bench/profile_base_1t.json`;
  - `quick.py run --out build/gcc-bench/quick_base.json`;
  - uProf sobre `volengine_grid_profile --series 2 --threads 1`.

- [ ] **Step 2: Escrever `profile-5a.md`**
  - **Máquina e commit.**
  - **As tabelas do profile:** por spec (top 20), distribuição, ordem e ARMA.
  - **Divisão do tempo de um ajuste:** otimizador contra inferência no ponto estimado (`1 − optimizer%`).
  - **Avaliações por iteração:** a média e as specs com maior razão.
  - **`pool_use`** com 16 threads e o que ele implica para a cauda.
  - **As 15 funções mais quentes do uProf,** com a parte de cada uma.
  - **Fits `MAX_ITER`:** quais são e quanto do tempo total levam.
  - **Lista de candidatos** para a Task 8, ordenada pelo ganho estimado a partir dos números. Os candidatos já vistos
    na leitura do código, a confirmar ou descartar pelos números:
    - **(a) zoom por interpolação:** `LineSearch::zoom` bissecta; a troca é por interpolação cúbica salvaguardada;
    - **(b) valor antes do gradiente:** avaliar f em `double` antes do gradiente, nas tentativas da busca linear que
      podem falhar o decréscimo suficiente. Gradiente só se passar;
    - **(c) ordem dos jobs:** os mais caros primeiro no lote, por um custo estático da spec (k, distribuição);
    - **(d) blocos menores:** `kBlocksPerThread` em `src/fit/batch.cpp` menor, ou blocos dinâmicos, se `pool_use`
      for baixo;
    - **(e) larguras de `Dual`:** um k = 4 hoje usa `Dual<8>`, e um k = 9 ou 10 usa `Dual<16>`. Larguras 4 e 12;
    - **(f) alocações:** as de `loglik_with_scores_with` e de `standard_errors`.

- [ ] **Step 3: Commit**

```bash
git add docs/performance/profile-5a.md
git commit -m "docs: publish the plan 5a baseline profile"
```

- [ ] **Step 4: CHECKPOINT. Parar e mostrar ao autor:**
  - o resumo do profile;
  - a lista de candidatos ordenada.

  O autor aprova a lista da Task 8. Se o gradiente (`Dual`) dominar, apresentar as opções da spec §5 com os números,
  antes de qualquer código no caminho do gradiente:
  - `Dual` mais barato;
  - AD reverso genérico;
  - adjunto à mão.

  O candidato (e) é caminho do gradiente e só entra com essa aprovação.

---

### Task 5: Opção de erros-padrão

**Files:**
- Modify: `src/core/spec.hpp` (`FitConfig::standard_errors`), `src/fit/fit.cpp`, `src/fit/fit.hpp`
- Modify: `include/volengine/volengine.h`, `src/capi/spec_api.cpp`, `src/capi/results/fit_api.cpp`,
  `src/capi/results/diagnostics_api.cpp`
- Modify: `python/src/volengine/_native.py`, `_spec.py`, `_fit.py`; `python/tests/typing_cases.py` se ele lista
  `FitOptions`
- Modify: `CHANGELOG.md`, `README.md` (onde fala de `FitOptions`), `docs/performance/profile-5a.md` (o A/B)
- Test: `tests/unit/fit/fit_test.cpp`, `tests/unit/capi/results/fit_accessors_test.cpp`, `tests/unit/capi/dcc_test.cpp`,
  `python/tests/test_fit.py`

**Interfaces:**
- Produces:
  - `FitConfig::standard_errors` (`bool`, padrão `true`);
  - `VE_API ve_status ve_fit_options_set_standard_errors(ve_fit_options* o, int32_t on);`
    - `on` 0 ou 1; outro valor é `VE_INVALID_ARGUMENT` com "standard_errors must be 0 or 1";
    - `o` nulo é `VE_INVALID_HANDLE`;
    - comentário no header: "Default 1. With 0 the fit skips the inference at the estimate (Hessian and robust standard
      errors, scores, Nyblom): ve_fit_std_errors, ve_fit_t_values, ve_fit_p_values and ve_fit_nyblom then return
      VE_INVALID_ARGUMENT. Everything else is computed as before.";
  - com a opção desligada, `ve_fit_std_errors`, `ve_fit_t_values`, `ve_fit_p_values` e `ve_fit_nyblom` devolvem
    `VE_INVALID_ARGUMENT`. A mensagem é exatamente: "standard errors were not computed: fit with
    ve_fit_options_set_standard_errors(o, 1)". O teste vale para qualquer status do ajuste;
  - `fit::loglik_with_scores` com `out.scores` vazio avalia em `double`, sem scores. É o caminho da opção desligada;
  - Python:
    - `FitOptions(standard_errors: bool = True)`;
    - `Fit.estimate` (property, `NDArray[np.float64]`, por `ve_fit_estimates`);
    - `Fit.coef()` e `Fit.nyblom()` levantam `VolEngineError` com a mensagem nativa quando a opção estava desligada.

- [ ] **Step 1: Escrever os testes que falham**

```cpp
// fit_test.cpp
TEST(Fit, StandardErrorsOffSkipsInferenceOnly)
{
    ve::FitConfig off;
    off.standard_errors = false;
    FitResult with = fit_simulated(Spec{});
    FitResult without = ve::fit_one(simulated(), Spec{}, off);
    ASSERT_EQ(without.status, with.status);
    for (std::size_t i = 0; i < with.estimate.size(); ++i)
    {
        EXPECT_EQ(bits(without.estimate[i]), bits(with.estimate[i]));
    }
    EXPECT_NEAR(without.loglik, with.loglik, 1e-12 * std::abs(with.loglik));
    for (std::size_t t = 0; t < with.sigma.size(); ++t)
    {
        EXPECT_NEAR(without.sigma[t], with.sigma[t], 1e-14 * with.sigma[t]);
    }
    EXPECT_TRUE(all_nan(without.se_hessian));
    EXPECT_TRUE(all_nan(without.se_robust));
    EXPECT_TRUE(without.nyblom.individual.empty());
    EXPECT_LT(without.work.seconds - without.work.optimizer_seconds,
              with.work.seconds - with.work.optimizer_seconds);
}
```

  Os outros testes, cada um com sua asserção:
  - `fit_accessors_test.cpp`:
    - `CapiFit.StandardErrorsOffIsInvalidArgument`: os quatro acessores devolvem `VE_INVALID_ARGUMENT`, com
      `ve_last_error()` igual à mensagem acima, e `ve_fit_estimates` devolve `VE_OK`;
    - `CapiFit.StandardErrorsOffOnFailedFit`: uma série constante dá `VE_FIT_FAILED`, e o acessor de erro-padrão
      ainda devolve a mensagem da opção;
    - `CapiFit.StandardErrorsOptionRejectsOtherValues`: `on = 2` dá `VE_INVALID_ARGUMENT`; `o = NULL` dá
      `VE_INVALID_HANDLE`.
  - `dcc_test.cpp`: `CapiDcc.UnivariatesWithoutStandardErrorsGiveSameDcc`. O DCC-mvnorm sobre ajustes com a opção desligada
    tem estimativas e erros-padrão de dois estágios bit-idênticos aos do DCC sobre ajustes com ela ligada.
  - `test_fit.py`:
    - `test_standard_errors_off`: `fit.coef()` levanta `VolEngineError`; `fit.estimate` é igual a
      `fit_on.coef().estimate`; `fit.forecast(10).sigma` é igual ao de `fit_on`; `ctx.simulate(fit, horizon=5,
      paths=3, seed=1).paths` é igual ao de `fit_on`.

- [ ] **Step 2: Rodar e ver falhar**

Run: `cmake --build --preset gcc-release` (não compila: `standard_errors` não existe). Depois do C++, `pytest
python/tests/test_fit.py -k standard_errors_off` falha.

- [ ] **Step 3: Implementar.** `fill_at_estimate` com a opção desligada:
  - chama `loglik_with_scores` com `scores` vazio;
  - preenche ℓ, a série e o estado;
  - não chama `standard_errors` nem `nyblom_at_estimate`.

  Os acessores olham `result.config.standard_errors`. No Python, `_native.py` registra a função nova (mesma assinatura
  de `ve_fit_options_set_var_init`).

- [ ] **Step 4: Rodar tudo**

Run: `cmake --build --preset gcc-release && ctest --preset gcc-release`, depois o pytest completo,
`python/.venv/Scripts/python -m mypy --strict python/src` e `python tests/style/check_style.py`.
Expected: tudo verde.

- [ ] **Step 5: Medir o A/B e registrar**

Run: `volengine_grid_profile --standard-errors off --json build/gcc-bench/profile_se_off.json` e
`quick.py compare-profile build/gcc-bench/profile_base.json build/gcc-bench/profile_se_off.json`.
Registrar em `profile-5a.md` o tempo da grade com a opção ligada e desligada.

- [ ] **Step 6: Commit**

```bash
git add src include python tests CHANGELOG.md README.md docs/performance
git commit -m "feat: let fits skip standard errors and Nyblom"
```

---

### Task 6: `lgamma` sem corrida entre threads

**Files:**
- Modify: `src/ad/special.hpp` / `src/ad/special.cpp` (`ve::ad::log_gamma(double)`), `src/ad/dual.hpp:315-318`,
  `src/dist/symmetric.hpp`, `src/dist/cdf.cpp:196`, `src/dcc/likelihood.cpp:37-42`, `src/dcc/copula/scores.cpp:42`
- Modify: `docs/superpowers/specs/2026-11-10-volengine-design.md` §10 (Boost.Math: Bessel K, quantis e `lgamma`)
- Test: `tests/unit/ad/special_test.cpp`

**Interfaces:**
- Produces:
  - `double ve::ad::log_gamma(double x)` para x > 0, por `boost::math::lgamma`, que não escreve em `signgam`;
  - `Dual<N> lgamma(const Dual<N>&)` passa a usá-la.

  Nenhum `std::lgamma` sobra em `src/`.

- [ ] **Step 1: Escrever os testes que falham**
  - `Special.LogGammaMatchesKnownValues`: os valores abaixo, com `EXPECT_NEAR` relativo 1e-15.
    - log Γ(0.5) = 0.5723649429247001;
    - log Γ(1) = 0;
    - log Γ(4) = log 6;
    - log Γ(100) = 359.1342053695754.
  - `Special.LogGammaIsSameOnEveryThread`: 8 `std::jthread` calculam `log_gamma` em 1000 pontos de (0.1, 200), e os
    bits são iguais aos da thread principal.

- [ ] **Step 2: Rodar e ver falhar** (`log_gamma` não existe)

- [ ] **Step 3: Implementar e trocar todo `std::lgamma`/`using std::lgamma` em `src/`**

Run: `grep -rn "lgamma" src` deve mostrar só `log_gamma` e o `lgamma` de `Dual`.

- [ ] **Step 4: Suíte, oráculos e A/B**

Run: `ctest --preset gcc-release`, pytest completo, `quick.py run` + `compare` contra a base.
Expected: verde, sem piora acima do ruído (`lgamma` roda uma vez por avaliação, não por t).

- [ ] **Step 5: Commit**

```bash
git add src tests/unit/ad docs/superpowers/specs/2026-11-10-volengine-design.md
git commit -m "fix: compute log-gamma without the signgam race"
```

---

### Task 7: L-BFGS que não para em ridge

Contexto: `Minimizer::run` (`src/optim/lbfgs.hpp:63-90`) devolve `Converged` quando um passo é `Flat` ou `Failed` e
max|g| ≤ √gtol·escala. Em ridge isso para longe do ótimo. Antes de mexer, use a skill `diagnosing-bugs` para achar o
mecanismo em pelo menos um caso real (FDCC `grid5`, abaixo).

**Files:**
- Modify: `src/optim/lbfgs.hpp`, `src/optim/line_search.hpp`, `src/optim/curvature_memory.hpp` (o que o diagnóstico
  pedir)
- Test: `tests/unit/optim/lbfgs_test.cpp`, `tests/unit/dcc/oracles/oracle_test.cpp`,
  `python/tests/oracles/test_oracles.py`
- Modify: `docs/guide/validation.md` (tira os casos de "known divergence"; reescreve
  "VolEngine: gradient tolerance"), `docs/performance/profile-5a.md` (A/B)

**Interfaces:**
- Consumes: `LbfgsResult::evaluations` (Task 1), `quick.py compare-profile` (Task 3).
- Produces: nenhuma assinatura nova. `Stop::Converged` passa a significar:
  - max|g| ≤ gtol·escala; ou
  - nenhuma descida possível na resolução de f, depois de reiniciar a memória.

- [ ] **Step 1: Escrever os testes que falham**
  - `test_oracles.py`:
    - tirar de `KNOWN_DIVERGENCES` as quatro entradas `OPTIMIZER`. `QUADRATURE` fica;
    - tirar o `tol = 1e-10` de `fit_options` para `sgarch11_vx_std/aparch11_vx_std`.
  - `oracle_test.cpp`: `DccOracle.Grid5FdccMvnormDefaultConfigLeavesStart`. `grid5_FDCC_mvnorm` com `FitConfig{}`
    padrão:
    - `status == Converged`;
    - ℓ ≥ ℓ do oráculo − 1e-7·|ℓ do oráculo| (a tolerância de ℓ do teste de cópula, `oracle_test.cpp:267`; o teste DCC existente usa 1e-10 com `tol = 1e-10`);
    - a estimativa difere do ponto inicial em pelo menos um parâmetro.
  - `lbfgs_test.cpp`: `Lbfgs.RidgeIsNotReportedConverged`, a menor função objetivo que reproduz o mecanismo achado no
    diagnóstico. Asserção: `stop == Converged` implica max|g| ≤ gtol·max(1, |f|) no ponto devolvido, ou o ponto está a
    menos de 1e-6 do mínimo conhecido.

- [ ] **Step 2: Rodar e ver falhar**

Run: `ctest --preset gcc-release -R "Lbfgs|DccOracle"` e
`pytest python/tests/oracles/test_oracles.py -q`.
Expected: os testes novos falham, e os 5 casos Python falham.

- [ ] **Step 3: Implementar a correção que o diagnóstico indicar.** Restrições:
  - `tol` mantém o significado;
  - `LineSearchFailureWithNearlySmallGradientConverges` e os outros testes de `lbfgs_test.cpp` continuam passando.

- [ ] **Step 4: Rodar tudo**

Run: `ctest --preset gcc-release`, pytest completo.
Expected: verde, com os 5 casos Python passando sem xfail.

- [ ] **Step 5: A/B na grade e fora dela**

Run:
- `volengine_grid_profile --json build/gcc-bench/profile_ridge.json`;
- `quick.py compare-profile` contra a base;
- `quick.py run` + `compare`;
- `volengine_benchmarks --benchmark_filter="BM_DccGrid/threads:1" --benchmark_repetitions=3`, antes e depois.

Expected:
- `converged` da grade ≥ o da base;
- nenhum Δℓ < −1e-9 relativo;
- `BM_FitOne/arma22_s33_std` com `CONVERGED`.

Uma piora de tempo acima do ruído em qualquer caso: parar e mostrar os números ao autor (Global Constraints).

- [ ] **Step 6: Atualizar `validation.md`** (casos e texto do critério de parada) e registrar o A/B em
  `profile-5a.md`.

- [ ] **Step 7: Commit**

```bash
git add src/optim tests python/tests docs
git commit -m "fix: keep L-BFGS from stopping on ridges"
```

---

### Task 8: Otimizações escalares guiadas pelo profile

Uma iteração deste ciclo por candidato da lista aprovada na Task 4, na ordem aprovada. Cada candidato é um commit
próprio.

**Files:** os que o candidato tocar (`src/optim/*`, `src/fit/*`, `src/model/*`); `docs/performance/profile-5a.md`.

**Interfaces:**
- Consumes: as ferramentas das Tasks 2–3 e a base mais recente (a última rodada aceita vira a nova base).
- Produces: nenhuma assinatura pública. Interna só se o candidato pedir, e mantendo os chamadores.

- [ ] **Step 1: Teste que fixa o comportamento antes de mexer.** A forma depende do candidato:
  - **busca linear (a, b):** um teste em `lbfgs_test.cpp` que conta avaliações num problema conhecido (Rosenbrock a
    partir de (−1.2, 1)) e exige menos que hoje. Hoje = o número medido no Step 1, anotado no teste;
  - **ordem dos jobs ou blocos (c, d):** um teste em `tests/unit/capi/batch_test.cpp` de que os resultados do lote são
    bit-idênticos com 1, 3 e 16 threads, se ainda não existir com essa cobertura;
  - **larguras de `Dual` (e):** os testes de gradiente contra diferenças finitas das larguras novas;
  - **alocações (f):** sem teste novo; a suíte existente cobre.

- [ ] **Step 2: Implementar o candidato.**

- [ ] **Step 3: Verificar**

Run: `ctest --preset gcc-release`, pytest completo, `quick.py run` + `compare` e `volengine_grid_profile --json` +
`compare-profile` contra a base.
Expected:
- verde;
- ganho acima do ruído na grade;
- nenhuma piora acima do ruído nos casos fora dela;
- nenhum Δℓ < −1e-9 relativo;
- status da grade sem piora.

- [ ] **Step 4: Decidir.**
  - **Ganho e nenhuma piora:** registrar o A/B em `profile-5a.md` e fazer commit (`perf: <candidato>`).
  - **Sem ganho:** reverter e anotar em `profile-5a.md` que foi medido e descartado.
  - **Piora fora da grade:** parar e mostrar ao autor.

- [ ] **Step 5: Depois do último candidato, perfilar de novo** com o uProf. Se surgir um ponto quente novo e grande,
  mostrar ao autor antes de seguir; não entra sem aprovação.

---

### Task 9: Fechamento

**Files:**
- Modify: `docs/performance/benchmarks.md`, `docs/performance/python-vs-arch.md`, `CHANGELOG.md`,
  `docs/superpowers/specs/2026-11-14-volengine-05-performance-design.md` (seção "5a feito", com os números)

- [ ] **Step 1: Suítes completas**

Run: `ctest --preset gcc-release`, pytest completo, `mypy --strict python/src`, `python tests/style/check_style.py`.
Expected: tudo verde.

- [ ] **Step 2: Benchmarks oficiais**, 5 repetições, máquina sem outra carga:
  - `BM_TccGrid/threads:{1,hw}`, uma vez com erros-padrão e uma sem. O benchmark ganha o parâmetro `se:{on,off}`, que
    passa a opção da Task 5;
  - todos os `BM_FitOne` e `BM_Loglik`;
  - `BM_Batch`;
  - `BM_DccGrid`;
  - `python/benchmarks/bench_vs_arch.py`.

- [ ] **Step 3: Atualizar `benchmarks.md`**
  - Tabelas novas com o commit medido.
  - Uma seção "Plan 5a" com antes/depois da grade e o que mudou.
  - Atualizar `python-vs-arch.md` com a rodada nova.

- [ ] **Step 4: CHANGELOG e spec.** CHANGELOG:
  - **Added:** `ve_fit_options_set_standard_errors`, `FitOptions.standard_errors`, `Fit.estimate`;
  - **Changed:** critério de parada, os casos de oráculo que saíram de xfail, `log_gamma`;
  - **Performance:** a grade antes e depois.

  Na spec do plano 5: o resultado de 5a e o que fica para o design de 5b.

- [ ] **Step 5: Commit**

```bash
git add docs CHANGELOG.md tests/benchmarks python/benchmarks
git commit -m "docs: record plan 5a benchmarks"
```
