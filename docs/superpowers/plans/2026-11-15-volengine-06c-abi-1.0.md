# VolEngine 6c: auditoria da ABI e 1.0 — plano de implementação

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Congelar a interface C na 1.0: corrigir o que a auditoria achou, versionar o contrato da ABI e conferi-lo no CI
(contrato + abidiff), subir `VE_ABI_VERSION` para 1, documentar e publicar `v1.0.0-rc.1` e `v1.0.0`.

**Architecture:** As mudanças de ABI (renomes, semântica de SE do DCC) entram primeiro, com C, Python e wrapper juntos
em cada task. Depois o contrato (`include/volengine/volengine.abi`, gerado de `volengine.h` por
`scripts/abi_contract.py`) vira a fonte do `.def` do Windows, e um job abidiff compara a `.so` atual com o XML da última
release. Por último, versão 1, documentação e as duas tags.

**Tech Stack:** C++23 (GCC 15 / MinGW ucrt64, MSVC, Apple Clang), CMake + ctest, Python 3.11+ com uv, GoogleTest,
libabigail (`abidw`, `abidiff`), GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-11-15-volengine-06-release-design.md` §6 (e §5 da spec-mãe
`docs/superpowers/specs/2026-11-10-volengine-design.md`). Achados do 6b: `docs/superpowers/notes/2026-11-15-abi-findings-6b.md`.

## Global Constraints

- Depois da 1.0, nenhuma assinatura muda e nenhum valor de enum some; acréscimo não quebra (spec-mãe §5).
- Nenhuma struct pública; configuração nova = função nova; nunca `_ex` (spec-mãe §5).
- Nomes pelo conceito, iguais em todas as famílias (spec-mãe §5).
- `VE_ABI_VERSION` = 1 na 1.0; o loader Python e o wrapper conferem igualdade; major do semver = `VE_ABI_VERSION` (spec 6 §6).
- abidiff só informa de `v0.9.0` até a 1.0; a partir de `v1.0.0`, mudança incompatível falha o CI (spec 6 §6).
- O `.def` do Windows sai do contrato, fonte única dos símbolos exportados (spec 6 §6).
- Asset da release: `volengine-native-<os>-<arch>.<ext>` (spec 6 §5.1); os nomes não mudam.
- Código, mensagens e docs do repositório em inglês; `docs/superpowers/` em português (STYLEGUIDE).
- STYLEGUIDE: sem `else`/`elif`; função ≤ 60 linhas, ≤ 5 parâmetros, aninhamento ≤ 3; pasta ≤ 8 módulos.
- Testes só no preset release; no Windows, rodar executáveis MinGW com `/c/msys64/ucrt64/bin` primeiro no PATH.
- O autor trabalha direto em master. Commits sem trailer `Co-Authored-By`.

## Decisões do autor (2026-11-15)

1. Critérios unificados: `ve_fit_loglik` e `ve_fit_criterion(f, which, scale, out)`; `VE_DCC_CRITERION_*` vira
   `VE_CRITERION_*` para as duas famílias; `ve_loglik`, `ve_aic`, `ve_bic`, `ve_hqic`, `ve_sic` saem.
2. Erros-padrão desligados: `ve_dcc_fit_std_errors`, `_t_values`, `_p_values` devolvem `VE_INVALID_ARGUMENT`, como o
   univariado.
3. `ve_dcc_fit_standard_errors` vira `ve_dcc_standard_errors` (mesma assinatura).

## Achados da auditoria e destino

Passada nas 133 funções contra a spec-mãe §5. O registro final (Task 4) lista cada um como corrigido ou aceito.

| # | Achado | Destino |
|---|---|---|
| A1 | Critérios em duas formas (univariado sem prefixo `ve_fit_`, uma função por critério) | Task 3 (decisão 1) |
| A2 | SE desligado: DCC dá NaN, univariado dá erro (6b #4) | Task 2 (decisão 2) |
| A3 | `ve_dcc_fit_standard_errors` contra `ve_dcc_fit_std_errors` | Task 3 (decisão 3) |
| A4 | `loglik` com SE desligado difere no 13º dígito (6b #3): com SE o valor e as séries vêm do passo `Dual` (`loglik_with_scores_with`, `src/fit/fit.cpp:203`), sem SE do passo `double` (`:230`) | Task 1 |
| A5 | DCC: significado de `p`/`q` de `ve_dcc_spec_set_order`, semântica de `groups`, nomes (`dcca1.., dccb1.., dccg1.., C1.., mshape`) e o `k` do critério (inclui primeira etapa e Qbar, 6b #5) não documentados | Task 4 |
| A6 | Mensagem de `ve_dcc_forecast_create` não nomeia as contagens (6b #6) | Task 4 |
| A7 | Formatos de erro do setter e do lote (6b #1), passo negativo (6b #2), `joint` do Nyblom é um `double` | Task 4 (frase no header) |
| A8 | `ve_abi_version`, `ve_last_error` e os `*_free` não devolvem `ve_status` | Aceito: precisam funcionar sem status (versão antes de tudo, mensagem do erro, liberar nunca falha) |
| A9 | Saídas `double* out` sem capacidade | Aceito: o tamanho é consultável (`*_param_count`, `*_spec_count`) e documentado; mudar seria a API inteira |
| A10 | `.lib` do MSVC grava `volengine-native-win-x64.dll`, `.dll.a` do MinGW grava `libvolengine.dll` (6b #8) | Task 6 |
| A11 | `ve_dcc_fit_batch` sem tamanho de `out` (6b #7), um nome por chamada (6b #9) | Aceito (6b) |
| A12 | "every machine with the same ISA" na simulação: promete demais entre CPUs com despacho SIMD diferente | Task 4: "the same ISA and SIMD target chosen at run time" |

## Review Focus

1. Declaração `VE_API` em várias linhas (`ve_batch_set_jobs`, `ve_dcc_fit_standard_errors`) ou com `/* ... */` no fim
   (`VE_MODEL_EGARCH 1 /* gamma1..gammap */`): o contrato normaliza para uma linha sem comentário (teste na Task 5).
2. Ler os SEs de um DCC ajustado com SE desligado **depois** de `ve_dcc_standard_errors`: o fit novo tem SEs (teste na Task 2).
3. Primeiro job abidiff sem XML na release (`v0.9.0` não tem): informa e passa, não falha (teste na Task 7).
4. Ajuste com SE ligado depois da Task 1: `sigma`, `residuals` e `loglik` iguais bit a bit ao desligado, também no
   caminho `Dual` (csGARCH, ghyp) e no adjunto (sGARCH norm) (teste na Task 1).
5. Consumidor MSVC sobre a release: a DLL baixada e o `.lib` concordam em `libvolengine.dll` (conferido na Task 9, rc.1).

---

### Task 1: `loglik` e séries iguais com SE ligado ou desligado (A4)

**Files:**
- Modify: `src/fit/fit.cpp:330-355` (o trecho que chama `fit::loglik_with_scores`)
- Test: `tests/unit/fit/standard_errors_option_test.cpp` (criar; registrar em `CMakeLists.txt` em `volengine_tests`)

**Interfaces:**
- Produces: `FitResult::loglik`, `residuals`, `sigma` independentes de `config.standard_errors`, bit a bit.

- [ ] **Step 1: Teste falhando** `TEST(StandardErrorsOption, SameNumbersOnAndOff)`: para cada (modelo, dist) em
  {(sGARCH, norm), (eGARCH, std), (csGARCH, ged), (apARCH, sstd), (sGARCH, ghyp)}, ajustar a mesma série
  (`PortableNormal`, seed 7, T = 1500) com SE ligado e desligado e conferir `EXPECT_EQ` (bit a bit, sem tolerância) em
  `loglik`, `estimate`, cada elemento de `sigma` e `residuals`.
- [ ] **Step 2: Rodar** `ctest --preset native-release -R StandardErrorsOption` — esperado FAIL em `loglik` (13º dígito).
- [ ] **Step 3: Implementar.** Com SE ligado, `loglik`, resíduos e sigma vêm sempre de `loglik_series_with_double`; o passo
  `Dual` (`loglik_with_scores_with`) preenche só `scores`. Ou seja: chamar o passo `double` para a série e o `loglik`, e
  o passo com scores numa `LoglikSeries` cujos vetores de série são descartados. Custo: um passo `double` a mais por
  ajuste com SE.
- [ ] **Step 4: Rodar** o teste novo e a suíte inteira (`ctest --preset native-release -j 8`) — PASS; oráculos dentro da tolerância.
- [ ] **Step 5: Benchmark.** `python tests/benchmarks/quick.py run` antes e depois (SE ligado, ajuste único e grid
  `se:on`); diferença esperada ≤ 1 passo `double`. Registrar os números na mensagem de commit.
- [ ] **Step 6: Commit** `fix(fit): loglik and series do not depend on the standard errors option`

### Task 2: SE desligado é erro também no DCC (A2)

**Files:**
- Modify: `src/dcc/fit.hpp:24-40` (`DccFitResult`), `src/dcc/fit.cpp` (onde lê `options.standard_errors`),
  `src/capi/dcc/dcc_fit_api.cpp:35-57` (`fill_statistic`), `src/capi/dcc/dcc_standard_errors_api.cpp`,
  `include/volengine/volengine.h` (comentário de `ve_dcc_options_set_standard_errors`)
- Modify: `python/src/volengine/_dcc.py` (docstring de `coef`), `cpp/include/volengine/dcc.hpp`
- Test: `tests/unit/capi/dcc_test.cpp`, `python/tests/dcc/test_dcc.py`, `cpp/tests/dcc_test.cpp:114-130`

**Interfaces:**
- Produces: `bool DccFitResult::standard_errors` (true quando o fit tem tabelas de SE; `ve_dcc_fit_batch` copia a opção,
  o resultado de `ve_dcc_fit_standard_errors`/`ve_dcc_standard_errors` é sempre true). `fill_statistic` devolve
  `ve::capi::no_standard_errors()` (`src/capi/error.hpp:17`) quando false.

- [ ] **Step 1: Testes falhando.**
  - C: `TEST(DccCapi, StandardErrorsOffAreInvalidArgument)`: ajuste com `ve_dcc_options_set_standard_errors(o, 0)`;
    `ve_dcc_fit_std_errors`, `_t_values`, `_p_values` → `VE_INVALID_ARGUMENT` com a mesma mensagem de `ve_fit_std_errors`;
    `ve_dcc_fit_estimates` e `ve_dcc_fit_loglik` → `VE_OK`. Depois `ve_dcc_fit_standard_errors` sobre esse fit →
    o fit novo dá `VE_OK` e valores finitos (Review Focus 2). `loglik` igual bit a bit ao do ajuste com SE ligado.
  - Python: `DccFit.coef()` de `fit_dcc(..., standard_errors=False)` levanta `VolEngineError`; `estimates` funciona.
  - C++: `Dcc.StandardErrorsOffGiveNaN` vira `Dcc.StandardErrorsOffAreAnError`: `std_errors()` sem valor,
    `error().status == invalid_argument`.
- [ ] **Step 2: Rodar** os três — FAIL (hoje `VE_OK` com NaN).
- [ ] **Step 3: Implementar** o campo e o teste em `fill_statistic`. Header: "With 0, ve_dcc_fit_std_errors, _t_values
  and _p_values return VE_INVALID_ARGUMENT (as ve_fit_std_errors); ve_dcc_fit_standard_errors gives a fit that has them."
  Docstrings Python/wrapper que falam em NaN passam a falar em erro.
- [ ] **Step 4: Rodar** `ctest --preset native-release -j 8`, `uv run --project python pytest python/tests -q`,
  testes do `cpp/` (`cpp/tests/ci_check.sh` ou o preset local usado no 6b) — PASS.
- [ ] **Step 5: Commit** `fix(capi)!: DCC standard errors that were not computed are an error, as in the univariate fit`

### Task 3: renomes da ABI (A1, A3)

**Files:**
- Modify: `include/volengine/volengine.h:277-283, 378-381, 423-431`
- Modify: `src/capi/results/fit_api.cpp` (critérios), `src/capi/dcc/dcc_fit_api.cpp:216-235`,
  `src/capi/dcc/dcc_standard_errors_api.cpp`
- Modify: `python/src/volengine/_native.py:56,63,67,127-131`, `_fit.py:88-102`, `_dcc.py:210-265`
- Modify: `cpp/include/volengine/fit.hpp:165-210`, `dcc.hpp:49,336-345`, `context.hpp:186`
- Test: `tests/unit/capi/results/fit_accessors_test.cpp`, `tests/unit/capi/batch_test.cpp`, `tests/unit/capi_scope.hpp`,
  `tests/unit/capi/dcc_test.cpp`, `tests/unit/capi/c_header_check.c`, `cpp/tests/fit_test.cpp`, `cpp/tests/dcc_test.cpp`

**Interfaces:**
- Produces (C):
  ```c
  #define VE_CRITERION_AIC 0
  #define VE_CRITERION_BIC 1
  #define VE_CRITERION_HQIC 2
  #define VE_CRITERION_SIC 3
  VE_API ve_status ve_fit_loglik(const ve_fit* f, double* out);
  VE_API ve_status ve_fit_criterion(const ve_fit* f, int32_t which, int32_t scale, double* out);
  VE_API ve_status ve_dcc_fit_criterion(const ve_dcc_fit* f, int32_t which, int32_t scale, double* out); /* VE_CRITERION_* */
  VE_API ve_status ve_dcc_standard_errors(ve_context* ctx, const ve_dcc_fit* f, const ve_fit* const* fits, size_t m,
                                          ve_dcc_fit** out);
  ```
  Saem: `ve_loglik`, `ve_aic`, `ve_bic`, `ve_hqic`, `ve_sic`, `VE_DCC_CRITERION_*`, `ve_dcc_fit_standard_errors`.
- Produces (C++): `enum class ve::Criterion { aic, bic, hqic, sic }` substitui `DccCriterion`;
  `Fit::criterion(Criterion, IcScale = IcScale::per_obs)` novo; `Fit::aic()`..`sic()` ficam como atalhos sobre ele.
- Python: API pública igual (`Fit.aic()` etc., `DccFit.criterion`, `DccFit.standard_errors`); só os nomes nativos mudam.

- [ ] **Step 1: Testes falhando.** Trocar as chamadas nos testes C pelos nomes novos e acrescentar
  `TEST(FitAccessors, CriterionMatchesEachKind)`: para cada `which` e cada `scale`, `ve_fit_criterion` igual ao valor
  que o teste antigo de `ve_aic`/... esperava; `which = 4` e `scale = 2` → `VE_INVALID_ARGUMENT`. No `c_header_check.c`,
  usar `VE_CRITERION_BIC` e `ve_fit_criterion`.
- [ ] **Step 2: Rodar** — não compila (nomes não existem).
- [ ] **Step 3: Implementar.** `ve_fit_criterion` usa a mesma validação de `which`/`scale` do DCC (mensagens
  `"unknown criterion"`, `"unknown criterion scale"`). O comentário do header diz que `which` vale para os dois.
  Atualizar Python e wrapper; no wrapper, `Context::dcc_standard_errors` chama `ve_dcc_standard_errors`.
- [ ] **Step 4: Rodar** C (`ctest`), Python (`pytest` + `mypy --strict python/src`), `cpp/` (testes +
  `python cpp/scripts/check_abi_coverage.py`) — PASS.
- [ ] **Step 5: Sem resto.** `git grep -nE "ve_(loglik|aic|bic|hqic|sic)\b|VE_DCC_CRITERION|ve_dcc_fit_standard_errors"`
  — vazio fora de `CHANGELOG.md` e `docs/superpowers/`.
- [ ] **Step 6: Commit** `refactor(capi)!: one criterion function per family, ve_dcc_standard_errors`

### Task 4: documentação do header e registro da auditoria (A5–A8, A12)

**Files:**
- Modify: `include/volengine/volengine.h` (comentários; nenhuma assinatura)
- Modify: `src/capi/dcc/dcc_forecast_api.cpp` (mensagem A6)
- Create: `docs/superpowers/notes/2026-11-15-abi-audit-6c.md` (a tabela de achados acima com o destino final de cada um
  e o commit que o fechou)
- Test: `tests/unit/capi/dcc_test.cpp` (ou o teste de forecast DCC existente)

- [ ] **Step 1: Teste falhando** `TEST(DccForecastCapi, SeriesMismatchNamesBothCounts)`: `ve_dcc_future_create(2, h, ...)`
  para um fit de 3 séries → `VE_INVALID_ARGUMENT` e `ve_last_error()` contém `"2"` e `"3"`.
- [ ] **Step 2: Rodar** — FAIL. **Step 3:** mensagem `"x has <a> series, the fit has <b>"`. **Step 4:** PASS.
- [ ] **Step 5: Header.** Conferindo cada frase no código antes de escrevê-la:
  - `ve_dcc_spec_set_order`: qual de `p`/`q` é o termo `a` (choques) e qual o `b`, faixa 1..`VE_MAX_ORDER`, FDCC só (1,1);
  - `ve_dcc_spec_set_groups`: `count` = número de séries, valores e numeração dos grupos, só FDCC;
  - nomes de `ve_dcc_fit_param_name`: `dcca1..`, `dccb1..`, `dccg1..` (aDCC), `fdcca*`/`fdccb*`, `C1..` (estático ML), `mshape` (mvt);
  - `ve_dcc_fit_criterion`: `k` conta as duas etapas (parâmetros estimados de cada série, os do DCC e os m(m-1)/2 de
    Qbar), como o rmgarch; `ve_dcc_fit_param_count` conta só os do DCC;
  - `ve_spec_set_order` e `ve_fit_batch`: o setter recusa sem prefixo, o lote prefixa `"spec <i>: "`;
  - `ve_panel_set_strides`: passo negativo exige o ponteiro bruto do elemento (0, 0);
  - `ve_fit_nyblom`: `joint` é um `double`;
  - antes de `ve_abi_version`: as únicas funções que não devolvem `ve_status` são `ve_abi_version`, `ve_last_error` e
    os `*_free` (que aceitam NULL);
  - simulação: "on every machine with the same ISA and the same SIMD target chosen at run time".
- [ ] **Step 6: Registro** em `docs/superpowers/notes/2026-11-15-abi-audit-6c.md`.
- [ ] **Step 7: Commit** `docs(capi): DCC orders, groups, names and criterion count; audit record for 1.0`

### Task 5: contrato da ABI

**Files:**
- Create: `scripts/abi_contract.py` (só biblioteca padrão), `include/volengine/volengine.abi` (gerado)
- Create: `tests/abi/test_abi_contract.py` (`unittest`)
- Modify: `CMakeLists.txt` (dois testes ctest: `abi_contract` e `abi_contract_unit`), `.github/workflows/ci.yml`
  (`fetch-depth: 0` no checkout do job Linux e um passo de base), `CONTRIBUTING.md` (como regenerar)

**Interfaces:**
- Produces:
  - `python scripts/abi_contract.py write` — regrava `volengine.abi` a partir de `volengine.h`.
  - `python scripts/abi_contract.py check [--base FILE]` — exit 1 com diff unificado se o arquivo difere do header;
    com `--base`, exit 1 listando cada linha da base que sumiu (só acréscimo passa).
  - `python scripts/abi_contract.py base-ref` — imprime a última tag estável `vN.M.P` com N ≥ 1 (sem `-`), ou nada.
  - `python scripts/abi_contract.py def --library NAME OUT` — escreve `LIBRARY NAME`, `EXPORTS` e uma função por linha.
  - `python scripts/abi_contract.py exports DEF` — exit 1 se os nomes de um `.def` do linker diferem das funções do contrato.
  - `functions(contract_lines) -> list[str]` (nomes, na ordem do header), usado por `def` e `exports`.
- Formato de `volengine.abi`, uma linha por item, na ordem do header, comentários e espaços normalizados:
  ```
  define VE_ABI_VERSION 0
  typedef int32_t ve_status
  typedef struct ve_context ve_context
  function int32_t ve_abi_version(void)
  function ve_status ve_spec_set_order(ve_spec*, int32_t, int32_t)
  ```
  Nomes de parâmetro saem (renomear parâmetro não quebra a ABI). Ficam de fora o guard, `VE_API`, `VE_BUILDING` e o
  `extern "C"`.

- [ ] **Step 1: Testes falhando** em `tests/abi/test_abi_contract.py`:
  - `test_multiline_declaration_is_one_line`: `"VE_API ve_status ve_f(const size_t* a,\n    size_t n);"` →
    `["function ve_status ve_f(const size_t*, size_t)"]` (Review Focus 1);
  - `test_define_drops_trailing_comment`: `"#define VE_X 1 /* note */"` → `["define VE_X 1"]`;
  - `test_pointer_to_const_pointer`: `const ve_spec* const* specs` → `const ve_spec* const*`;
  - `test_removed_line_fails_base` / `test_added_line_passes_base`;
  - `test_def_lists_functions_only`: `def` sobre um contrato com define, typedef e duas funções → `LIBRARY x.dll`,
    `EXPORTS`, as duas funções;
  - `test_header_matches_committed_contract`: o `volengine.h` real gera exatamente `volengine.abi`.
- [ ] **Step 2: Rodar** `python -m unittest tests/abi/test_abi_contract.py` — FAIL (script não existe).
- [ ] **Step 3: Implementar** o script; rodar `write`; conferir que `volengine.abi` tem 133 − 6 + 3 = 130 linhas
  `function` (Task 3: saem 5 critérios e o nome antigo de SE; entram `ve_fit_loglik`, `ve_fit_criterion`,
  `ve_dcc_standard_errors`) — ajustar a conta ao que o header tiver de fato.
- [ ] **Step 4: ctest e CI.** `abi_contract` roda `check`; `abi_contract_unit` roda o `unittest`. No `ci.yml`, passo
  Linux: `base=$(python scripts/abi_contract.py base-ref)`; se não vazio, `git show "$base:include/volengine/volengine.abi" > /tmp/base.abi`
  e `check --base /tmp/base.abi`.
- [ ] **Step 5: Rodar** `ctest --preset native-release -R abi_contract` — PASS. Mudar um `#define` no header à mão →
  FAIL com diff; desfazer.
- [ ] **Step 6: Commit** `feat(abi): versioned ABI contract checked against volengine.h`

### Task 6: `.def` e import library do Windows a partir do contrato (A10)

**Files:**
- Modify: `python/scripts/build_wheel.sh:36-42,60-64`, `.github/workflows/release.yml:41-43`,
  `cpp/cmake/FetchNativeLib.cmake:21-50`, `cpp/README.md` (nome da DLL no Windows)

**Interfaces:**
- Consumes: `abi_contract.py def` e `exports` (Task 5).
- Produces: asset `volengine-native-win-x64.def` gerado do contrato com `LIBRARY libvolengine.dll`; `.lib` e `.dll.a`
  gravam `libvolengine.dll`; o `cpp/` sempre salva a DLL baixada como `libvolengine.dll`.

- [ ] **Step 1:** `build_wheel.sh` no Windows: troca a contagem `grep -c` por `abi_contract.py exports "$lib_dir/volengine.def"`
  (falha se o linker exporta algo fora do contrato ou falta algo) e copia para `out/` o `.def` gerado por
  `abi_contract.py def --library libvolengine.dll`.
- [ ] **Step 2:** `release.yml`: `lib /def:volengine-native-win-x64.def /name:libvolengine.dll /machine:x64 /out:volengine-native-win-x64.lib`.
- [ ] **Step 3:** `FetchNativeLib.cmake`: o ramo MSVC salva a DLL como `${_cache}/libvolengine.dll` (igual ao MinGW);
  atualizar o comentário das linhas 21-23.
- [ ] **Step 4: Verificar** localmente no PowerShell (não no Bash, cujo `ld` grava `.def` vazio): build MinGW,
  `abi_contract.py exports build/<preset>/volengine.def` → ok; `lib /def:<gerado> /name:libvolengine.dll` e
  `dumpbin /headers <lib> | findstr libvolengine.dll` → presente; consumidor `cpp/tests/package` com MSVC sobre a DLL
  local → roda.
- [ ] **Step 5: Commit** `fix(release): one DLL name for the MinGW and MSVC import libraries, .def from the ABI contract`

### Task 7: abidiff

**Files:**
- Create: `scripts/abi_dump.sh`
- Modify: `.github/workflows/ci.yml` (job `abi`), `.github/workflows/release.yml` (job `abi-dump` e o asset)

**Interfaces:**
- Produces:
  - `scripts/abi_dump.sh OUT.xml` — configura `-DCMAKE_BUILD_TYPE=Release -DCMAKE_C_FLAGS=-g -DCMAKE_CXX_FLAGS=-g -DVE_BUILD_TESTS=OFF`
    em `build/abi`, compila `volengine` e roda `abidw --headers-dir include/volengine --drop-private-types --out-file OUT.xml build/abi/libvolengine.so`.
  - Asset de release `volengine-abi-linux-x64.xml`.
  - Job `abi` (ubuntu-24.04, GCC 15 como o `build-test`, `sudo apt-get install -y abigail-tools`, `GH_TOKEN`):
    1. `scripts/abi_dump.sh current.xml`;
    2. `tag=$(gh release list --limit 1 --json tagName -q '.[0].tagName')` (a mais recente, prerelease incluída);
       `gh release download "$tag" --pattern volengine-abi-linux-x64.xml` — se não houver o asset, `::notice::` e sai 0
       (Review Focus 3);
    3. `abidiff baseline.xml current.xml`; código com bit 8 (`ABIDIFF_ABI_INCOMPATIBLE_CHANGE`) falha só se a major da
       `tag` (sem `v`) for ≥ 1; senão imprime o relatório e passa. Qualquer outro bit de erro (1, 2) falha sempre.

- [ ] **Step 1: Rodar** `scripts/abi_dump.sh` em Docker `ubuntu:24.04` com GCC 15 e `abigail-tools`; o XML lista
  `ve_fit_criterion`. Rodar `abidiff` do XML contra ele mesmo → exit 0.
- [ ] **Step 2: Provar a falha:** remover um `VE_API` numa cópia, gerar outro XML, `abidiff` → bit 8 ligado.
- [ ] **Step 3:** jobs no `ci.yml` e no `release.yml` (o `publish-github` baixa o XML e o inclui em `files:`).
- [ ] **Step 4: Verificar** num push: job `abi` verde com o aviso "no ABI baseline in v0.9.0".
- [ ] **Step 5: Commit** `ci(abi): abidiff against the last release`

### Task 8: versão 1 e documentação

**Files:**
- Modify: `include/volengine/volengine.h:22` (`VE_ABI_VERSION 1`), `include/volengine/volengine.abi` (regenerado),
  `python/src/volengine/_native.py:16`, `python/pyproject.toml:7` (`1.0.0`), `CMakeLists.txt:2` (`1.0.0`),
  `CITATION.cff` (`version: 1.0.0`, `date-released: 2026-11-15` ou o dia da tag), `CHANGELOG.md`, `README.md`
- Create: `docs/guide/python.md`, `docs/guide/cpp.md`, `python/tests/test_docs.py`
- Test: `tests/unit/capi/abi_test.cpp`, `cpp/tests/error_test.cpp` (já comparam com a macro), `python/tests/test_loader.py`

- [ ] **Step 1: Teste falhando** `python/tests/test_docs.py::test_python_blocks_run`: extrai cada bloco ` ```python `
  de `README.md` e `docs/guide/python.md` e roda com `exec` num namespace novo (dados sintéticos dos próprios blocos,
  como o exemplo atual do README). FAIL enquanto o guia não existe.
- [ ] **Step 2: Versão.** `VE_ABI_VERSION 1`, `abi_contract.py write`, loader `VE_ABI_VERSION = 1`, versões 1.0.0.
  `test_loader.py:45` tem `"ABI version 0, this package needs 1"` literal: passa a montar a mensagem de
  `_native.VE_ABI_VERSION` (`f"ABI version {v}, this package needs {v + 1}"`).
- [ ] **Step 3: Docs.**
  - `README.md`: status "1.0: the C ABI is stable" (tira "Until 1.0 the C ABI may change"); exemplo Python acrescenta
    `fit.forecast(10).sigma` e um DCC de 3 séries (`ctx.fit_dcc` + `coef()` + `correlation[-1]`); exemplo C++ com o
    `FetchContent_Declare(volengine GIT_REPOSITORY ... GIT_TAG v1.0.0 SOURCE_SUBDIR cpp)` e o `main.cpp` de
    `cpp/tests/package/` resumido.
  - `docs/guide/python.md`: instalar, ajustar (spec, lote), coeficientes, critérios, diagnósticos, forecast,
    simulação, DCC e Copula-DCC — um bloco curto por item.
  - `docs/guide/cpp.md`: `FetchContent`, `find_package(volengine 1.0)`, `volengine_copy_native_library`,
    `VOLENGINE_NATIVE_LIB`, erros por `std::expected` e um ajuste com forecast.
  - `CHANGELOG.md`: `## [1.0.0]` com "Changed (breaking)": decisões 1–3 e o SE do DCC; "Fixed": Task 1 e o nome da DLL;
    "Added": contrato, abidiff, `ve_fit_criterion`; a frase "Until 1.0 the C ABI may break" vira a regra da 1.0.
- [ ] **Step 4: Rodar** tudo: `ctest --preset native-release -j 8`, `pytest python/tests -q` (inclui `test_docs`),
  `mypy --strict python/src`, testes do `cpp/`, `check_abi_coverage.py`.
- [ ] **Step 5: Commit** `release: VE_ABI_VERSION 1 and 1.0.0, README and guides`

### Task 9: `v1.0.0-rc.1` e `v1.0.0` (cada push de tag só com o OK do autor)

- [ ] **Step 1:** push de master; `ci.yml`, `wheels.yml`, `cpp.yml` e `abi` verdes.
- [ ] **Step 2: Com OK do autor**, `git tag v1.0.0-rc.1 && git push origin v1.0.0-rc.1`. Release prerelease com as 9
  assets + `.def` fora + XML do abidw + SBOM; PyPI `1.0.0rc1`.
- [ ] **Step 3: Conferir o rc:** `pip install --pre volengine==1.0.0rc1` num venv limpo e rodar o exemplo do README;
  `cpp/tests/package` com `FetchContent` no `GIT_TAG v1.0.0-rc.1`, MinGW e MSVC no Windows (Review Focus 5) e Linux
  (Docker); job `abi` seguinte passa a ter baseline (rc.1) e compara.
- [ ] **Step 4: Com OK do autor**, `git tag v1.0.0 && git push origin v1.0.0`; repetir a conferência do Step 3 com
  `volengine==1.0.0` e `GIT_TAG v1.0.0`.
- [ ] **Step 5:** atualizar a memória do roadmap (plano 6 fechado, 1.0 publicada).
