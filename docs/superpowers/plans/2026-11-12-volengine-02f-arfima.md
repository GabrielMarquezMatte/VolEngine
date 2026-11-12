# VolEngine — Plano 2f: ARFIMA

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Sobre o 2e, aceitar a média ARFIMA do `rugarch` (`arfima = TRUE`): um parâmetro d de diferença fracionária aplicado aos resíduos da média, em todos os modelos e distribuições, com o mínimo para a paridade. A validação é contra o `rugarch`.

**Architecture:** A média calcula os resíduos do 2e (ARMA sobre rₜ − μ − Σδₖxₜ,ₖ) e, com ARFIMA, aplica sobre eles a diferença fracionária truncada (1 − L)^d, no próprio vetor. Esses resíduos alimentam tudo o que já os lê: pré-amostra, VT, densidade e saídas. Nada mais muda de forma: o d é um parâmetro `Real` com mapa logístico em (1e-8, 0.5), e o caminho sem ARFIMA não é tocado.

**Tech Stack:** C++23, CMake + presets, GoogleTest v1.17.0, Google Benchmark v1.9.1, BS::thread_pool v5.0.0, Boost.Math `boost-1.87.0`, Python ≥ 3.11 + numpy + pytest, R 4.6.2 + `rugarch` 1.5.6.

**Spec:** `docs/superpowers/specs/2026-11-10-volengine-design.md`, seção 3 ("Opções de especificação do `rugarch` que mudam o modelo entram: … ARFIMA"). Os planos 2a–2e registram as convenções que este plano estende.

## Roteiro

- **2a–2e (feitos):** motor genérico, variantes, distribuições assimétricas, parâmetros fixos, valores iniciais, VT, regressores e ARCH-in-mean.
- **2f (este):** ARFIMA, como opção de paridade. O usuário não o usa nas próprias análises (decisão do usuário, 2026-11-12): o plano faz o mínimo que passa como modelo.

Depois: 3 diagnósticos/forecast/simulação; 4 DCC; 5 performance; 6 wrapper C++, wheels, 1.0.

## Global Constraints

Valem todas as Global Constraints dos planos 1, 2a–2e, o `STYLEGUIDE.md` e o `ARCHITECTURE.md`. As que mais pesam aqui:

- A interface C é o único contrato estável. Enums `typedef int32_t` + `#define`, só acréscimos; ≤ 5 parâmetros por função; configuração nova é função nova; nenhuma exceção atravessa a fronteira C.
- Ajustes bit-idênticos para qualquer número de threads na mesma máquina e ISA. **Uma spec sem ARFIMA dá os mesmos números do 2e:** todos os testes existentes passam sem mudar constantes nem tolerâncias.
- Convenção padrão segue o `rugarch`. O código dele (GPL) pode ser lido para entender convenções, nunca copiado ou traduzido. Uma dúvida de convenção se resolve lendo a fonte, e a referência vai para `validation.md`.
- Anti-objetivo: não fazer mais do que o necessário. Ficam fora o ARFIMA junto com `archm`, forecast e simulação ARFIMA (o plano 3 decide), a convolução por FFT e d negativo.
- Estilo: nunca `else`; funções ≤ 60 linhas; ≤ 5 parâmetros; aninhamento ≤ 3; ≤ 8 módulos por pasta; comentários só para restrição externa ou invariante; medições em `docs/`. `-Wnrvo` como erro no GCC.
- Commits sem trailer `Co-Authored-By`; LF. Python no venv `python/.venv`; R em `C:/Program Files/R/R-4.6.2/bin/Rscript.exe`.

### Novas regras deste plano

**Convenção do `rugarch`.** Foi confirmada antes do plano com `ugarchfilter` (parâmetros fixos, d = 0.3, sGARCH(1,1)-norm, T = 400) contra uma implementação independente em R. Os casos foram ARMA (0,0), (1,0), (0,1) e (1,1), cada um com e sem `mxreg` 2. As diferenças ficaram em e ≤ 4.4e-16, σ ≤ 4.4e-16 e ℓ ≤ 5e-13. Lido: `.arfimaxfilter`, `.fracdiff`, `.sgarchLLH` e `.meqstart`, impressos do pacote instalado. A Task 6 refaz a confirmação com ajustes.

**Diferença fracionária.** Seja eₜ o resíduo da regra "Média" do 2e (sem `archm`) e T o tamanho da série. Com ARFIMA, o resíduo passa a ser

  yₜ = Σₖ₌₀ᵗ πₖ eₜ₋ₖ, com π₀ = 1 e πₖ = πₖ₋₁ (k − 1 − d)/k, para t = 0..T−1.

A expansão é truncada no início da amostra, sem pré-amostra. yₜ substitui eₜ em tudo:

- a pré-amostra `VE_VARINIT_SAMPLE`, que é a média de y², ou de |y|^δ no apARCH e no fGARCH;
- o número de observações retidas, que continua max(m, n, p, q) (o d não conta);
- o alvo do VT, a densidade e os resíduos de saída.

Com d = 0, πₖ = 0 para k ≥ 1, e yₜ = eₜ bit a bit.

**Backcast** (`VE_VARINIT_BACKCAST`, decisão do Claude). Não muda: continua o OLS de r em [1, X] do 2e, que não depende de parâmetro. O `arch` não tem ARFIMA, então não há convenção a seguir. O backcast também já ignora o ARMA.

**Parâmetro, nome e ordem.** `arfima` vem logo depois de `ma..`: `mu`, `ar1..`, `ma1..`, `arfima`, `archm`, `mxreg1..`. Uma spec nunca tem `arfima` e `archm` ao mesmo tempo, e `spec_problem` devolve "arfima cannot be combined with archm". Por isso o layout mais largo continua com 59 entradas e `kMaxParameterCount` não muda.

**Domínio, mapa e fronteira.** O `rugarch` usa o intervalo fechado [1e-8, 0.5], com LB 1e-8 e UB 0.5 no `.meqstart`. Aqui o intervalo é aberto: d = 1e-8 + (0.5 − 1e-8)·logistic(u), e `is_valid` exige 1e-8 < d < 0.5. O `ParameterKind` é `Real`, sem escala da série, e o passo da Hessiana é o dos `Real`. `on_boundary` marca o d quando ele fica a menos de 1e-8 de qualquer uma das pontas. O `rugarch` costuma parar em d = 1e-8 em séries sem memória.

**Valor inicial** (decisão do Claude). d = 0.25. O `rugarch` usa uma estimativa de Hurst menos 0.5 (`.absHmom`), que só se adota se a Task 6 mostrar ótimos diferentes. Fixos e iniciais do usuário funcionam como nos outros parâmetros. Um início ou fixo fora de (1e-8, 0.5) termina em `VE_FIT_FAILED` com NaN, como os outros parâmetros limitados.

**Custo.** A soma custa O(T²) por avaliação, em `double` e em `Dual`. Fica marcado como `ponytail:` no código, com a FFT como saída se um dia for preciso.

## Review Focus

1. **Série sem memória longa** (a do plano 1, sGARCH(1,1)-norm): o d vai para perto de 1e-8. O ajuste termina finito, o ℓ̂ com ARFIMA fica ≥ o ℓ̂ sem ARFIMA − 1e-6|ℓ|, e o erro-padrão de d é NaN se `on_boundary`. Nenhuma exceção. → teste na Task 4.
2. **Início ou fixo de d fora do domínio** (0.6, 0.5, 0, −0.1): `VE_FIT_FAILED` com NaN nesse job, e os outros jobs do lote não são afetados. → testes nas Tasks 4 e 5.
3. **ARFIMA com `archm`**: `ve_spec_validate` e o lote dão `VE_INVALID_ARGUMENT` com uma mensagem que cita "arfima" e "archm", e nenhum handle é criado. O Python levanta o erro de spec de sempre. → teste na Task 5.
4. **Painel irregular com 4 threads**: a truncagem de cada job começa na primeira linha da sua fatia, não na linha 0 do painel. O resultado é bit-idêntico ao `fit_one` nas fatias. → teste na Task 5.
5. **Mudança de unidades** (r × 100): d̂ igual e ℓ̂ deslocado por T·log 100, porque o d não tem unidade. → teste na Task 4.

---

## Mapa de arquivos

```
src/core/spec.hpp                  Spec::arfima
src/model/layout.*                 Layout::arfima, arfima_index(); nome; kind Real; spec_problem com archm
src/mean/fractional.hpp            fractional_difference (novo)
src/model/likelihood.cpp           mean_residuals aplica a diferença fracionária
src/model/parameter_map.*          mapa logístico de d; is_valid; on_boundary
src/model/start_values.*           d = 0.25
src/capi/spec_api.cpp, include/volengine/volengine.h   ve_spec_set_arfima
python/src/volengine/_spec.py, _native.py              Spec(arfima=)
tests/...                          espelhando src/; série e casos novos em tests/reference/
```

---

### Task 1: Spec e Layout

**Files:**
- Modify: `src/core/spec.hpp`, `src/model/layout.hpp`, `src/model/layout.cpp`
- Test: `tests/unit/model/layout_test.cpp`

**Interfaces:**
- Produces:
  ```cpp
  namespace ve { struct Spec { /* campos do 2e, e: */ bool arfima = false; }; }
  namespace ve::model {
  struct Layout { /* campos do 2e, e: */ bool arfima = false;
      bool has_arfima() const { return arfima; }
      std::size_t arfima_index() const; };   // ma_begin() + ma; archm_index() passa a somar has_arfima()
  }
  ```
- Regra "Parâmetro, nome e ordem". O comentário de `kMaxParameterCount` passa a dizer que o layout mais largo tem `archm` ou `arfima`, nunca os dois.

- [ ] **Step 1: Escrever os testes**

```cpp
TEST(Layout, ArfimaFollowsMa) {
  // sGARCH(1,1), ARMA(1,1), arfima, mxreg 1: mu ar1 ma1 arfima mxreg1 omega alpha1 beta1; kind(arfima_index()) == Real
}
TEST(Layout, WidestArfimaFitsWidestDual) { /* ALLGARCH, ARMA(6,6), (6,6), arfima, mxreg 8, vxreg 8, ghyp: count() == 59 */ }
TEST(Layout, ArfimaWithArchmIsAProblem) { /* arfima + archm 1: spec_problem contém "arfima" e "archm"; arfima só: nullopt */ }
```

- [ ] **Step 2: Rodar e ver falhar** — `cmake --build --preset gcc-debug && ctest --preset gcc-debug -R Layout` → não compila.
- [ ] **Step 3: Implementar.**
- [ ] **Step 4: Rodar e ver passar** — suíte inteira PASS, sem mudar constantes.
- [ ] **Step 5: Commit** — `git commit -m "feat: add the arfima parameter to the layout"`

---

### Task 2: Diferença fracionária na verossimilhança

**Files:**
- Create: `src/mean/fractional.hpp`
- Modify: `src/model/likelihood.cpp` (`mean_residuals`)
- Test: `tests/unit/mean/fractional_test.cpp`, `tests/unit/model/likelihood_test.cpp`

**Interfaces:**
- Consumes: `Layout::arfima_index` (Task 1).
- Produces:
  ```cpp
  namespace ve::mean {
  // The rule "Diferença fracionária", in place: residuals[t] becomes sum_{k=0}^{t} pi_k residuals[t - k].
  template <class T> void fractional_difference(std::span<T> residuals, const T& d);
  }
  ```
- O cálculo é feito no próprio vetor, de t = T−1 até 0, já que yₜ só lê e com índice ≤ t. Os π são calculados uma vez por chamada. Leva o comentário `// ponytail: O(T^2) direct sum; FFT convolution if long series ever need it`.
- `mean_residuals` aplica a diferença fracionária depois do ARMA, com `theta[layout.arfima_index()]`, quando `layout.has_arfima()`. Assim pré-amostra, VT, `targeted_omega_at`, densidade e saídas leem y sem mudança. `presample` e `backcast` não mudam.

- [ ] **Step 1: Escrever os testes**

```cpp
TEST(Fractional, HandCase) {
  // e = (1, 2, -1, 0.5), d = 0.3: pi = (1, -0.3, -0.105, -0.0595); y = (1, 1.7, -1.705, 0.5305) (1e-15)
}
TEST(Fractional, ZeroIsIdentity) { /* d = 0: y == e bit a bit */ }
TEST(Likelihood, ArfimaHandCase) {
  // r de 6 pontos, ARMA(1,0), sGARCH(1,1)-norm, sample init, d = 0.3: e pela regra do 2e, y pela regra,
  // sigma2 com 1 observação retida na média de y^2, l à mão (rel 1e-14)
}
TEST(Likelihood, ArfimaZeroIsPlain) { /* d = 0: loglik == loglik da spec sem arfima, bit a bit, ARMA(1,1), mxreg 2 */ }
TEST(Likelihood, ArfimaTargetsDifferencedResiduals) { /* VT sGARCH: targeted_omega_at == mean(y^2)(1 - P) (1e-15) */ }
// GradientMatchesFiniteDifferences, DoubleAndDualAgreeOnValue e TermsSumToTotalAndScoresSumToGradient ganham
// arfima ∈ {off, d = 0.3} em SGarch, EGarch, ApArch, CsGarch × ARMA {(0,0),(1,1)} × {sample, backcast} × mxreg {0, 2}.
```

- [ ] **Step 2: Rodar e ver falhar** — `ctest --preset gcc-debug -R "Fractional|Likelihood"` → não compila.
- [ ] **Step 3: Implementar.**
- [ ] **Step 4: Rodar e ver passar** — suíte inteira PASS; `Likelihood.MatchesPlanOneSimulatedValues` sem mudar constantes.
- [ ] **Step 5: Commit** — `git commit -m "feat: apply the fractional difference to the mean residuals"`

---

### Task 3: Mapa, validade e início

**Files:**
- Modify: `src/model/parameter_map.hpp`, `src/model/parameter_map.cpp`, `src/model/start_values.hpp`, `src/model/start_values.cpp`
- Test: `tests/unit/model/parameter_map_test.cpp`, `tests/unit/model/start_values_test.cpp`

**Interfaces:**
- Consumes: Task 1.
- Regras "Domínio, mapa e fronteira" e "Valor inicial". `to_natural`, `to_unconstrained`, `is_valid` e `on_boundary` ganham o d. Os comentários desses quatro e de `start_values` citam a regra. `natural_scale` não muda: o d é `Real`.

- [ ] **Step 1: Escrever os testes**

```cpp
TEST(ParameterMap, ArfimaRoundTrip) { /* d ∈ {1e-6, 0.25, 0.49} em sGARCH, eGARCH, apARCH, csGARCH: θ → u → θ' (rel 1e-12) */ }
TEST(ParameterMap, ArfimaValidity) { /* is_valid: 0.25 true; 1e-8, 0, -0.1, 0.5, 0.6 false */ }
TEST(ParameterMap, ArfimaBoundary) { /* on_boundary: 1.5e-8 e 0.5 - 5e-9 true; 0.25 false */ }
TEST(StartValues, ArfimaStart) { /* 0.25; start {arfima: 0.1} → 0.1; fixed {arfima: 0.2} → 0.2 e o resto como hoje */ }
```

- [ ] **Step 2: Rodar e ver falhar** — `ctest --preset gcc-debug -R "ParameterMap|StartValues"` → FAIL.
- [ ] **Step 3: Implementar.**
- [ ] **Step 4: Rodar e ver passar** — suíte inteira PASS.
- [ ] **Step 5: Commit** — `git commit -m "feat: map, validate and start the arfima parameter"`

---

### Task 4: Ajuste

**Files:**
- Modify: `tests/unit/simulated_garch.hpp`. Acrescente as duas séries da Task 6, simuladas de forma independente da biblioteca. Primeiro os choques GARCH εₜ. Depois a integração fracionária truncada eₜ = Σₖ₌₀ᵗ ψₖ εₜ₋ₖ, com ψ₀ = 1 e ψₖ = ψₖ₋₁ (k − 1 + d)/k. Por fim a inversa do ARMA e rₜ = μ + dₜ. São 1000 observações de aquecimento, descartadas, com a truncagem começando no aquecimento.
- Test: `tests/unit/fit/fit_test.cpp`

**Interfaces:**
- Consumes: Tasks 1–3. `fit_one`, `standard_errors` e o lote não mudam. Se algum deles precisar mudar, pare e reporte.

- [ ] **Step 1: Escrever os testes**

```cpp
// T = 3000, séries da Task 6 pelo helper
TEST(FitArfima, BeatsTrueParameters) {
  // as duas séries com o próprio modelo: Converged, l(θ̂) ≥ l(θ verdadeiro) − 1e-6|l|, |d̂ − d| ≤ 0.05
}
TEST(FitArfima, NoMemoryNestsPlain) {
  // Review Focus 1: série sGARCH(1,1)-norm do plano 1 com arfima: sem exceção, Converged ou MaxIter,
  // l̂ ≥ l̂ sem arfima − 1e-6|l|; SE de d NaN se on_boundary, finito se não
}
TEST(FitArfima, UnitEquivariance) { /* Review Focus 5: arfima_sgarch11_norm com r·100: |d̂' − d̂| ≤ 1e-5, l̂' == l̂ − T log 100 (rel 1e-8) */ }
TEST(FitArfima, StartOutsideDomainFails) { /* Review Focus 2: start {arfima: 0.6} e fixed {arfima: 0.5}: Failed, NaN, sem exceção */ }
```

- [ ] **Step 2: Rodar e ver falhar** — `ctest --preset gcc-debug -R FitArfima` → FAIL (o helper ainda não existe).
- [ ] **Step 3: Implementar o helper.** Se uma tolerância de recuperação não fechar, registre o valor observado no relatório antes de mexer.
- [ ] **Step 4: Rodar e ver passar** — suíte C++ inteira PASS, `-Wnrvo` sem aviso.
- [ ] **Step 5: Commit** — `git commit -m "test: fit ARFIMA series"`

---

### Task 5: Interface C, lote e Python

**Files:**
- Modify: `include/volengine/volengine.h`, `src/capi/spec_api.cpp`, `python/src/volengine/_spec.py`, `python/src/volengine/_native.py`
- Test: `tests/unit/capi/spec_test.cpp`, `tests/unit/capi/batch_test.cpp`, `python/tests/test_fit.py`

**Interfaces:**
- Produces (C, só acréscimo):
  ```c
  /* ARFIMA mean (rugarch's arfima = TRUE), 0 or 1 (default 0); other values are VE_INVALID_ARGUMENT and leave the
     spec unchanged. Cannot be combined with ve_spec_set_archm: ve_spec_validate and the batch reject the spec. */
  VE_API ve_status ve_spec_set_arfima(ve_spec* s, int32_t on);
  ```
  O header documenta a regra "Diferença fracionária", o domínio (1e-8, 0.5) e o nome `arfima` depois de `ma`, no bloco da ordem dos parâmetros (linha ~59) e no bloco da média (linha ~225).
- Python: `Spec(..., arfima: bool = False)`, com a docstring da regra e da ordem dos nomes; `ve_spec_set_arfima` em `_native.py`.

- [ ] **Step 1: Escrever os testes**

```cpp
TEST(CapiSpec, ArfimaSetter) { /* 0 e 1 OK; 2 e -1 → INVALID_ARGUMENT, spec inalterada */ }
TEST(CapiSpec, ArfimaWithArchmInvalid) {
  // Review Focus 3: set_arfima 1 + set_archm 1: ve_spec_validate → INVALID_ARGUMENT com "arfima" e "archm";
  // no lote → INVALID_ARGUMENT com "job <i>", fits_out todo NULL
}
TEST(CapiFit, RaggedArfimaMatchesSlices) {
  // Review Focus 4: 3 séries irregulares no painel, arfima e mxreg 1, threads 1 e 4: bit-idêntico a ve::fit_one(Data das fatias)
}
TEST(CapiFit, ArfimaStartOutsideDomainFailsAlone) { /* Review Focus 2: job 1 com start arfima 0.6 → VE_FIT_FAILED; jobs 0 e 2 Converged */ }
TEST(CapiFit, ArfimaParamName) { /* ARMA(1,1): ... ma1 arfima omega ... via ve_fit_param_name */ }
```
```python
def test_arfima_names(ctx, series): ...              # Spec(arma=(1, 1), arfima=True): "arfima" entre "ma1" e "omega"
def test_arfima_with_archm_raises(ctx, series): ...  # Spec(arfima=True, archm=1): a exceção de check, com "archm" na mensagem
```

- [ ] **Step 2: Rodar e ver falhar** — `ctest --preset gcc-debug -R "CapiSpec|CapiFit"` e pytest → FAIL.
- [ ] **Step 3: Implementar.**
- [ ] **Step 4: Rodar e ver passar** — suítes C++ e Python inteiras PASS; header C11-limpo; `mypy --strict` PASS.
- [ ] **Step 5: Commit** — `git commit -m "feat: expose ARFIMA through the C API and Python"`

---

### Task 6: Oráculo

**Files:**
- Modify: `tests/reference/generate_data.py`, `tests/reference/cases.json`, `tests/reference/oracle_rugarch.R`, `tests/reference/README.md`, `python/tests/test_oracles.py`, `docs/guide/validation.md`
- Create (gerado, commitado): `tests/reference/data/grid_2f.csv`; fixtures novas em `tests/reference/fixtures/rugarch.json`

**Interfaces:**
- Consumes: pacote Python (Task 5). Pré-requisito: R 4.6.2 + `rugarch` 1.5.6. Se faltar, pare e peça ao usuário.
- Séries (T = 3000, `numpy.random.default_rng(20261115)`, σ² inicial 1, 1000 de aquecimento descartadas, geradas como no helper da Task 4):

| Série | Processo |
|---|---|
| `arfima_sgarch11_norm` | μ = 0.05, d = 0.3; sGARCH(1,1) ω = 0.02, α = 0.08, β = 0.9; norm |
| `arfima1_gjrgarch11_std` | μ = 0.02, AR(1) φ = 0.2, d = 0.25; gjrGARCH(1,1) ω = 0.02, α = 0.03, γ = 0.09, β = 0.9; std ν = 6 |

- Casos (só `rugarch`; o `arch` não tem ARFIMA):
  - cada série com o próprio modelo;
  - `arfima_sgarch11_norm` em eGARCH(1,1)-std, apARCH(1,1)-norm e csGARCH(1,1)-norm;
  - `arfima_sgarch11_norm` com ARMA(1,1);
  - `arfima_sgarch11_norm` com `mxreg` [x1] de `regressors_2e.csv`;
  - `arfima_sgarch11_norm` com VT.
- O caso ganha o campo `"arfima": true` (opcional), documentado no README; `case_spec` o repassa.
- Vale a regra de interior do 2b–2e: se o `rugarch` parar em d = 1e-8 ou 0.5, o caso é trocado, com o motivo registrado em "Case selection". As tolerâncias são as do 2a, por grandeza, sem afrouxar.
- `validation.md` ganha a seção ARFIMA com:
  - a regra "Diferença fracionária" e as fontes lidas;
  - os números da confirmação com `ugarchfilter`;
  - o domínio aberto, contra o intervalo fechado do `rugarch`;
  - o início 0.25, contra o `.absHmom`;
  - o backcast sem ARFIMA;
  - a exclusão do `archm`;
  - o custo O(T²).

  Se o oráculo divergir, a correção vai no núcleo, com os testes de gradiente mantidos, e a mudança é registrada.

- [ ] **Step 1: Gerar dados e fixtures** — `python/.venv/Scripts/python tests/reference/generate_data.py && Rscript tests/reference/oracle_rugarch.R` → um resultado por caso; os dados e fixtures antigos não mudam (só acréscimos).
- [ ] **Step 2: Rodar** — `VOLENGINE_LIB=<abs> python/.venv/Scripts/python -m pytest python/tests/test_oracles.py -v` → PASS; depois de cada correção no núcleo, as suítes C++ e Python inteiras.
- [ ] **Step 3: Commit** — `git commit -m "test: validate ARFIMA against rugarch"`

---

### Task 7: Benchmarks e documentação

**Files:**
- Modify: `tests/benchmarks/fit_benchmarks.cpp`, `docs/performance/benchmarks.md`, `README.md`, `CHANGELOG.md`, `ARCHITECTURE.md`

**Interfaces:**
- Benchmarks novos (os existentes não mudam): `BM_Loglik/arfima_sgarch11_norm/{double,dual}` e `BM_FitOne/arfima_sgarch11_norm`, T = 5000. Nesta máquina só compilar e rodar curto, com `--benchmark_min_time=0.01s --benchmark_filter=arfima`. `benchmarks.md` anota o custo O(T²) e que a medição fica para a máquina de benchmark.
- README: um exemplo curto com `Spec(arfima=True)`, a regra numa linha e o link para `validation.md`.
- CHANGELOG: entrada em `[Unreleased]`.
- ARCHITECTURE: `mean/fractional.hpp` e a diferença fracionária depois do ARMA.

- [ ] **Step 1: Escrever** — `cmake --preset gcc-bench && cmake --build --preset gcc-bench` sem aviso; a rodada curta termina.
- [ ] **Step 2: Verificar** — `ctest --preset gcc-release` e pytest inteiros PASS; `check_style.py .` sai 0.
- [ ] **Step 3: Commit** — `git commit -m "bench: add ARFIMA benchmarks and document plan 2f"`
