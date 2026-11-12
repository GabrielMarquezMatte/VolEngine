# VolEngine — Plano 2d: parâmetros fixos, valores iniciais e variance targeting

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Sobre os 14 modelos e as 8 distribuições do 2c, deixar o usuário fixar qualquer parâmetro, dar valores iniciais e pedir variance targeting, como no `rugarch` (`fixed.pars`, `start.pars`, `variance.targeting`), validado contra `rugarch` (e contra o `arch` no caso todo fixo) e exposto pela interface C e pelo Python.

**Architecture:** A `Spec` guarda os valores fixos e iniciais por nome; o `Layout` os resolve em índices e passa a saber quais parâmetros são estimados (nem fixos, nem derivados). O mapa irrestrito ↔ natural só escreve entradas livres. No bloco de pesos, os parâmetros fixos viram termos constantes da persistência e os livres dividem o orçamento que sobra, pela mesma regra logistic × softmax de hoje. Sob variance targeting, ω deixa de ser estimado e é calculado dentro do `loglik`, a partir dos resíduos correntes, de modo que a AD já carrega a dependência. Sem parâmetro fixo, sem valor inicial e sem VT, tudo segue pelos caminhos de hoje, bit a bit.

**Tech Stack:** C++23, CMake + presets, GoogleTest v1.17.0, Google Benchmark v1.9.1, BS::thread_pool v5.0.0, Boost.Math `boost-1.87.0`, Python ≥ 3.11 + numpy + pytest, `arch` 8.0.0, R 4.6.2 + `rugarch` 1.5.6.

**Spec:** `docs/superpowers/specs/2026-11-10-volengine-design.md` (seção 3, "Opções de especificação do `rugarch` que mudam o modelo"). Os planos 2a, 2b e 2c (`docs/superpowers/plans/2026-11-11-volengine-02{a,b,c}-*.md`) registram as convenções que este plano estende.

## Roteiro

O 2d original foi dividido em três (decisão do usuário, 2026-11-12):

- **2a, 2b, 2c (feitos):** motor genérico, variantes GARCH, distribuições assimétricas.
- **2d (este):** parâmetros fixos, valores iniciais, variance targeting. Mudam o papel dos parâmetros, não as recursões.
- **2e:** `mxreg`/`vxreg` (matrizes T×k por job do lote, parte do contrato de entrada) e `archm`/`archpow` (média e variância intercaladas a cada t).
- **2f:** ARFIMA (diferença fracionária, O(T²) por avaliação).

Depois, como antes: 3 diagnósticos/forecast/simulação; 4 DCC; 5 performance; 6 wrapper C++, wheels, 1.0.

## Global Constraints

Valem todas as Global Constraints dos planos 1, 2a, 2b e 2c, o `STYLEGUIDE.md` e o `ARCHITECTURE.md`. As que mais pesam aqui:

- A interface C é o único contrato estável. Enums `typedef int32_t` + `#define`, só acréscimos; ≤ 5 parâmetros por função; configuração nova é função nova; nenhuma exceção atravessa a fronteira C.
- Ajustes bit-idênticos para qualquer número de threads na mesma máquina e ISA. **Uma spec sem parâmetro fixo, sem valor inicial e sem VT dá os mesmos números do 2c:** todos os testes existentes passam sem mudar constantes nem tolerâncias. O caminho sem fixos mantém a aritmética de hoje: os ramos novos só entram quando o `Layout` tem fixos, iniciais ou VT.
- Convenção padrão segue o `rugarch`. O código dele (GPL) pode ser lido para entender convenções, nunca copiado ou traduzido.
- **Dúvida de convenção ou fórmula se resolve lendo a fonte** (decisão do usuário, 2026-11-12): papers, a vinheta e os manuais do `rugarch`, a documentação do `arch`. Pesquise antes de deixar algo "a confirmar". Só fica para a Task 7 o que depende dos números do oráculo, e a referência usada vai para `validation.md`.
- Anti-objetivo: não fazer mais do que o necessário. Ficam fora: VT com variância numérica dada pelo usuário (o `variance.targeting = <número>` do `rugarch`), o `fixed.se` do `rugarch` (erro-padrão dos fixos) e uma função para remover um fixo ou um inicial. Cada um entra depois como função nova, sem quebra.
- Estilo: nunca `else`; funções ≤ 60 linhas; ≤ 5 parâmetros; aninhamento ≤ 3; ≤ 8 módulos por pasta; comentários só para restrição externa ou invariante; medições em `docs/`. `-Wnrvo` como erro no GCC.
- Commits sem trailer `Co-Authored-By`; LF. Python no venv `python/.venv`.

### Novas regras deste plano

**Papéis dos parâmetros.** Cada parâmetro do `Layout` é *estimado*, *fixo* ou *derivado*:

- **fixo:** o usuário deu o valor (`ve_spec_set_fixed`). Não entra no otimizador, sai no ajuste com esse valor, tem erro-padrão NaN (Hessiana e robusto) e não conta nos critérios.
- **derivado:** a função dos outros parâmetros que o modelo define. O β_q do iGARCH (como hoje) e o ω sob VT. Também tem nome, valor, SE NaN e não conta nos critérios.
- **estimado:** todo o resto. `estimated_count()` = `count()` − fixos − derivados. É o k dos critérios, como o `np = sum(ipars[, "Estimate"])` do `rugarch`; a Task 7 confirma.

**Valor inicial** (`ve_spec_set_start`): troca o valor inicial padrão do parâmetro (a tabela de `start_values.hpp`). Se o parâmetro for derivado, o valor é ignorado: assim um warm-start com todas as estimativas do ajuste anterior funciona. Se o ponto inicial montado sair do domínio do modelo, o ajuste termina com `VE_FIT_FAILED` e NaN, como um início inválido hoje.

**Nomes.** Fixos e iniciais são guardados por nome, os do `rugarch`, os mesmos de `ve_fit_param_name`. Um nome fica em no máximo uma das duas listas: fixar um nome remove o inicial dele, e o contrário também. Repetir um nome troca o valor. Os nomes são resolvidos na hora do ajuste (`ve_fit_batch`) e em `ve_spec_validate`, nunca nos setters, para que a ordem dos setters não importe.

**Specs rejeitadas** (`model::spec_problem`, que vira `VE_INVALID_ARGUMENT` com a mensagem em `ve_last_error()`; no lote, prefixada por `spec <i>: `):

| Condição | Mensagem contém |
|---|---|
| nome fixo ou inicial que a spec não tem | o nome e a lista de nomes da spec |
| fixar o β_q do iGARCH | `derived` e o nome |
| fixar `omega` com VT | `derived` e `omega` |
| VT em iGARCH ou csGARCH | `variance targeting` e o modelo |
| csGARCH com `eta21` (φ) fixo e algum `beta` livre | `eta21` |

O `rugarch` ignora o VT no csGARCH sem avisar, e no iGARCH ele dá ω = 0 (P = 1). Aqui as duas combinações são erro. φ fixo com β livre exige Σβ ≥ φ, uma restrição que o mapa de pesos não representa; fixar φ junto com todos os β é aceito.

**Termos de peso.** O bloco de pesos (posições de hoje: sGARCH, csGARCH, iGARCH, apARCH e fGARCH `alpha..beta`; eGARCH `beta`; gjrGARCH `alpha..beta..gamma`) é visto como termos tⱼ ≥ 0, um por posição, cuja soma é a persistência P. Com π = P(z < 0) e κᵢ = `shock_moment` calculados dos parâmetros de choque correntes (que o `to_natural` já resolve antes dos pesos):

| Modelo | posição | termo tⱼ | inverso |
|---|---|---|---|
| sGARCH, csGARCH, iGARCH | αᵢ, βⱼ | o próprio valor | o próprio valor |
| eGARCH | βⱼ | βⱼ | βⱼ |
| apARCH, fGARCH | αᵢ; βⱼ | αᵢκᵢ; βⱼ | t/κᵢ; t |
| gjrGARCH, γᵢ fixo e αᵢ livre | αᵢ; γᵢ | αᵢ − max(0, −γᵢ); max(0, −γᵢ) + πγᵢ | αᵢ = t + max(0, −γᵢ) |
| gjrGARCH, os outros casos | αᵢ; γᵢ | (1 − π)αᵢ; π(αᵢ + γᵢ) | αᵢ = t_α/(1 − π); γᵢ = t_γ/π − αᵢ |

Nas duas linhas do gjrGARCH, o par soma αᵢ + πγᵢ. Uma posição é *livre* quando o parâmetro natural nela é estimado. A soma dos termos tem um teto c: `1 − 1e-8` em geral; ρ no csGARCH com `eta11` fixo; e exatamente 1 no iGARCH, onde o β_q derivado é o termo de referência.

**Mapa com fixos** (estende o `to_natural`/`to_unconstrained` do 2a–2c):

- O `to_natural` escreve primeiro todos os valores fixos em θ. Cada mapa elemento a elemento (μ, ARMA, os reais do eGARCH, distribuição, η₁/γ do apARCH, η₂, potência, ρ e φ do csGARCH, ω) só escreve entradas livres. ω e φ leem θ depois dos pesos, como hoje.
- Pesos: F = soma dos termos das posições não livres, B = c − F. Os termos livres são logistic(v)·B·softmax(w, 0) sobre as posições livres em ordem: v fica na primeira posição livre, os logits na segunda em diante, e a última posição livre é a referência (logit 0). No iGARCH não há v: os termos livres e o β_q derivado são B·softmax, com o derivado como referência. Uma só posição livre usa só v. Nenhuma posição livre: nada a mapear. Com B ≤ 0, os pesos livres viram NaN e o `loglik` sai não finito, como qualquer ponto fora do domínio.
- Sem fixos no bloco, essa é exatamente a regra de hoje (F = 0, todas as posições livres).
- O `to_unconstrained` é o inverso. As entradas de `u` em posições fixas ou derivadas valem 0 e ficam sem uso; o gradiente delas é zero, como o do β_q do iGARCH hoje.
- `on_boundary` ganha uma regra: no gjrGARCH com γᵢ fixo e αᵢ livre, αᵢ − max(0, −γᵢ) < 1e-8 marca αᵢ. Os erros-padrão deixam de fora, além das entradas de fronteira, toda entrada não estimada.

**Valores iniciais com fixos e iniciais.** A ordem dos passos de `start_values` decide quem lê o quê: primeiro os padrões de hoje; depois os iniciais do usuário e os fixos sobrescrevem a distribuição, η₁/η₂/γ do apARCH e a potência; então os α padrão do apARCH/fGARCH são divididos pelos κᵢ desses valores finais, e os iniciais e fixos do bloco de pesos sobrescrevem os seus; por fim vem ω, com a potência final. Dois ajustes só valem quando o bloco de pesos tem algum fixo ou inicial:

- No gjrGARCH com γᵢ fixo e αᵢ com valor padrão: αᵢ = max(0, −γᵢ) + 0.03/p.
- Reescala: sejam F a soma dos termos fixos, U a dos termos com inicial do usuário e D a dos termos com valor padrão. Se D > 0 e c − F − U > 0, os termos padrão são multiplicados por s = min(1, 0.95·(c − F − U)/D) e voltam para θ pelo inverso da tabela. Com c − F − U ≤ 0 nada muda, o início é inválido e o ajuste termina `FAILED`.

O φ padrão do csGARCH passa a ser min(0.03, 0.5·Σβ). Com Σβ ≥ 0.06 o resultado é 0.03 exato, então os testes de hoje não mudam.

**Variance targeting** (`ve_spec_set_variance_targeting`). Sob VT, ω é derivado: o `loglik` o calcula a cada avaliação, a partir dos resíduos εₜ da média corrente, sobre a amostra inteira e para qualquer `var_init` (o `arch` não tem VT, então não há convenção do `arch` a seguir):

- sGARCH, gjrGARCH: ω = m₂(1 − P), com m₂ = média de ε²ₜ e P = Σα + Σβ (+ πΣγ no gjrGARCH);
- eGARCH: ω = log(m₂)(1 − Σβ);
- apARCH e fGARCH: ω = m_δ(1 − P), com m_δ = média de |εₜ|^δ, δ a potência (livre ou fixa do submodelo) e P = Σαᵢκᵢ + Σβ.

São as fórmulas do `rugarch` 1.5.6 (`.sgarchLLH`, `.egarchLLH`, `.aparchLLH`, `.fgarchLLH`) sem o termo dos `vxreg`, que fica para o 2e. Fontes: Engle & Mezrich (1996), que introduzem o VT; a vinheta do `rugarch` (seção do sGARCH, eq. 10); e a documentação de `ugarchspec`, segundo a qual a variância incondicional vem "by its sample counterpart in the squared residuals during estimation", ou seja, dos resíduos correntes, como aqui. A Task 7 confirma o eGARCH contra o `max(1 − persist, ·)` do `rugarch`. No ajuste, `estimate[omega]` é esse ω calculado na estimativa.

Os erros-padrão sob VT seguem o `rugarch`: Hessiana e sanduíche da verossimilhança em que ω é função dos outros parâmetros (m₂ inclusive, pelos parâmetros da média). Eles **não** são a covariância assintótica do estimador VT em dois passos de Francq, Horváth & Zakoïan (2011, *J. Financial Econometrics* 9(4), 619–656), que soma a incerteza da variância amostral. `validation.md` registra a diferença e a referência; a covariância de dois passos fica fora do escopo e só entra se pedida.

**Todo fixo.** Uma spec sem parâmetro estimado não chama o otimizador: o ajuste avalia o modelo no ponto fixo (o `ugarchfilter` do `rugarch`). O status é `VE_FIT_CONVERGED` se o `loglik` for finito e `VE_FIT_FAILED` se não for. Os erros-padrão são todos NaN, e os critérios usam k = 0.

## Review Focus

1. **Fixos que violam as restrições juntos** (sGARCH com α₁ = 0.6 e β₁ = 0.5 fixos; gjrGARCH com γ₁ = −0.5 fixo e α₁ livre, que exige α₁ ≥ 0.5): o ajuste termina `FAILED` com NaN, sem exceção, e os outros ajustes do lote seguem normais. → testes nas Tasks 3 e 5.
2. **Warm-start com as estimativas inteiras de um ajuste anterior**, incluindo valores derivados (β_q do iGARCH, ω sob VT) e pesos de fronteira (α = 1e-12): os derivados são ignorados, o peso minúsculo vira um u finito pelo piso DBL_MIN, e o ajuste converge para o mesmo ponto. → testes nas Tasks 3 e 5.
3. **Spec reaproveitada depois de `ve_spec_set_dist`** (um `shape` fixo e a dist trocada para norm): `ve_fit_batch` e `ve_spec_validate` falham com `VE_INVALID_ARGUMENT`, e a mensagem nomeia a spec e o parâmetro. Nenhum handle é criado. → teste na Task 6.
4. **VT com persistência perto do teto e com m₂ < 1** (log negativo no eGARCH): ω segue finito, com o sinal certo (> 0 fora do eGARCH), e o `loglik` sai finito. → teste na Task 4.
5. **Spec toda fixa no mesmo lote que specs estimadas, com 1 e 4 threads:** resultado bit-idêntico ao `fit_one` isolado; a toda fixa sai `CONVERGED`, com SE NaN e critérios com k = 0. → teste na Task 6.

---

## Mapa de arquivos

```
src/core/spec.hpp                  ParameterValue; Spec::fixed, Spec::start, Spec::variance_targeting
src/model/layout.hpp/.cpp          máscaras fixed/started, given, variance_targeting; is_fixed/is_derived/is_estimated;
                                   estimated_count; spec_problem
src/model/weight_terms.hpp         termos de peso (template, interno ao model/): termos, teto, inverso, orçamento
src/model/parameter_map.*          to_natural/to_unconstrained com fixos; persistence<T>; on_boundary (gjr γ fixo); is_valid com VT
src/model/start_values.cpp         iniciais do usuário, fixos, reescala, φ do csGARCH
src/model/likelihood.*             targeted_omega<T>; ω derivado sob VT
src/fit/fit.cpp                    todo fixo sem otimizador; ω da estimativa sob VT
src/fit/standard_errors.cpp        deixa de fora as entradas não estimadas
src/fit/batch.cpp                  spec_problem no check_batch
src/capi/spec_api.cpp, include/volengine/volengine.h   set_fixed, set_start, set_variance_targeting, validate
python/src/volengine/_spec.py, _native.py              Spec(fixed=, start=, variance_targeting=)
tests/...                          espelhando src/; casos e fixtures novos em tests/reference/
```

---

### Task 1: Spec e Layout com papéis dos parâmetros

**Files:**
- Modify: `src/core/spec.hpp`, `src/model/layout.hpp`, `src/model/layout.cpp` (`criteria.cpp` já lê `estimated_count` e não muda)
- Test: `tests/unit/model/layout_test.cpp`

**Interfaces:**
- Produces:
  ```cpp
  namespace ve {
  struct ParameterValue { std::string name; double value; };
  struct Spec {   // campos do 2c, e:
      std::vector<ParameterValue> fixed;
      std::vector<ParameterValue> start;
      bool variance_targeting = false;
  };
  }
  namespace ve::model {
  struct Layout {   // campos do 2c, e:
      std::bitset<kMaxParameterCount> fixed;     // por índice
      std::bitset<kMaxParameterCount> started;
      std::array<double, kMaxParameterCount> given{};   // valor fixo ou inicial; 0 nos outros
      bool variance_targeting = false;
      bool is_fixed(std::size_t index) const;
      bool is_started(std::size_t index) const;   // inicial do usuário num parâmetro estimado
      bool is_derived(std::size_t index) const;   // β_q do iGARCH; omega sob VT
      bool is_estimated(std::size_t index) const; // nem fixo nem derivado
      std::size_t estimated_count() const;        // count() - fixos - derivados
  };
  // A regra "Specs rejeitadas", na ordem da tabela; nada quando a spec pode ser ajustada.
  std::optional<std::string> spec_problem(const Spec& spec);
  }
  ```
- `layout_of(spec)` resolve os nomes por `parameter_names` e exige `!spec_problem(spec)` (assert). Um inicial dado a um derivado não liga `started`. `has_derived()`/`derived()` continuam sendo o β_q do iGARCH.

- [ ] **Step 1: Escrever os testes**

```cpp
TEST(Layout, ResolvesFixedAndStartByName) {
  // sGARCH(1,1)-std, fixed {shape: 5}, start {alpha1: 0.1}: is_fixed(shape()), given[shape()] == 5,
  // is_started(alpha_begin()), given[alpha_begin()] == 0.1; estimated_count() == 4
}
TEST(Layout, DerivedRoles) {
  // iGARCH(1,2): is_derived(derived()); VT sGARCH(1,1): is_derived(omega()), estimated_count() == count() - 1
  // VT sGARCH com start {omega: 0.1}: !is_started(omega()) (ignorado)
}
TEST(Layout, SpecProblems) {
  // cada linha da tabela "Specs rejeitadas" devolve uma mensagem que contém o texto indicado:
  // {"shape": 5} em norm → "shape" e "mu omega alpha1 beta1"; iGARCH(1,1) fixed {beta1: 0.9} → "derived";
  // VT + fixed {omega: 0.1} → "derived"; VT em igarch e csgarch → "variance targeting";
  // csGARCH(1,1) fixed {eta21: 0.03} → "eta21"; o mesmo com beta1 também fixo → nada
}
TEST(Layout, PlainSpecUnchanged) { /* sem fixed/start/VT: máscaras vazias e estimated_count() igual ao do 2c em todo layout */ }
```

- [ ] **Step 2: Rodar e ver falhar** — `cmake --build --preset gcc-debug && ctest --preset gcc-debug -R Layout` → não compila.
- [ ] **Step 3: Implementar.**
- [ ] **Step 4: Rodar e ver passar** — suíte inteira PASS, sem mudar nenhuma constante dos testes existentes.
- [ ] **Step 5: Commit** — `git commit -m "feat: resolve fixed and start values and variance targeting into the layout"`

---

### Task 2: Mapa de parâmetros com fixos e termos de peso

**Files:**
- Create: `src/model/weight_terms.hpp`
- Modify: `src/model/parameter_map.hpp`, `src/model/parameter_map.cpp`
- Test: `tests/unit/model/parameter_map_test.cpp`

**Interfaces:**
- Consumes: `Layout::is_fixed`, `is_estimated`, `given`, `variance_targeting` (Task 1).
- Produces:
  ```cpp
  namespace ve::model {
  // P = soma dos termos de peso (tabela "Termos de peso"); NaN quando π ou κ não são computáveis.
  template <class T> T persistence(std::span<const T> theta, const Layout& layout);   // instanciado como to_natural
  // Os termos tⱼ de theta, um por posição do bloco de pesos, e o teto c.
  struct WeightTerms { std::array<double, 3 * kMaxOrder> terms{}; std::size_t count = 0; double ceiling = 0.0; };
  WeightTerms weight_terms(std::span<const double> theta, const Layout& layout);
  // Escreve em theta as entradas não fixas do bloco de pesos a partir dos termos (o inverso da tabela).
  void natural_from_terms(const WeightTerms& terms, std::span<double> theta, const Layout& layout);
  }
  ```
  `weight_terms.hpp` guarda os templates (termo por posição, teto, orçamento) que `parameter_map.cpp` e `start_values.cpp` usam; ele não sai de `src/model/`.
- Regras "Termos de peso" e "Mapa com fixos". `is_valid` não lê o ω sob VT nem as entradas derivadas.

- [ ] **Step 1: Escrever os testes**

```cpp
TEST(ParameterMap, PlainSpecsUnchanged) { /* testes do 2a–2c sem mudança, inclusive DualMatchesDouble bit a bit */ }
TEST(ParameterMap, FixedEntriesKeepTheirValue) {
  // 14 modelos × (1,1) e (2,2) × ARMA {(0,0),(1,1)} × {norm, sstd, ghyp}: para cada índice i, fixando só θᵢ
  // (de um θ interior do esquema to_natural(u) do 2b): to_natural(u qualquer)[i] == given[i] (bit a bit),
  // e o round trip θ → u → θ' fecha nas entradas livres (rel 1e-12). Pula as combinações de spec_problem.
}
TEST(ParameterMap, FixedWeightsLeaveTheirBudget) {
  // sGARCH(1,1) α₁ = 0.3 fixo: para 200 u (mt19937_64(7), N(0,3)), α₁ + β₁ < 1 - 1e-8 e β₁ ≥ 0
  // gjrGARCH(1,1)-sstd(0.8, 6), γ₁ = -0.05 fixo: α₁ ≥ 0.05 e α₁ + β₁ + π γ₁ < 1 - 1e-8
  // gjrGARCH(1,1)-sstd, α₁ = 0.02 fixo: α₁ + γ₁ ≥ 0
  // csGARCH(1,1) ρ = 0.95 fixo: α₁ + β₁ < 0.95; iGARCH(1,2) α₁ = 0.2 fixo: α₁ + β₁ + β₂ == 1 (1e-15), β₂ ≥ 0
}
TEST(ParameterMap, FixedBeyondBudgetIsNaN) { /* sGARCH α₁ = 0.6 e β₁ = 0.5 fixos, ω livre: is_valid falso; com α₁ = 0.6 fixo, β₁ livre: β₁ < 0.4 */ }
TEST(ParameterMap, AlwaysValidWithFixed) {
  // 2000 u por layout (mt19937_64(5), N(0,3)), os 14 modelos × {norm, sstd} com um fixo sorteado por layout
  // (mt19937_64(11)) no valor do θ interior: is_valid(to_natural(u)) sempre que B > 0
}
TEST(ParameterMap, DualMatchesDoubleWithFixed) { /* to_natural<Dual<16>>.value == to_natural<double> bit a bit nos layouts de FixedEntriesKeepTheirValue */ }
TEST(ParameterMap, PersistenceMatchesWeights) { /* persistence<double>(θ) == soma dos termos (1e-15) nos 14 modelos; Dual: derivadas contra CD (1e-7) */ }
TEST(ParameterMap, GjrFixedGammaBoundary) { /* γ₁ = -0.05 fixo, α₁ = 0.05 + 1e-9: on_boundary(α₁) */ }
TEST(ParameterMap, VarianceTargetingIgnoresOmega) { /* VT sGARCH: is_valid com theta[omega] = NaN e = -1; to_unconstrained escreve u[omega] = 0 */ }
```

- [ ] **Step 2: Rodar e ver falhar** — `ctest --preset gcc-debug -R ParameterMap` → não compila.
- [ ] **Step 3: Implementar.** O caminho sem fixo no bloco de pesos chama as funções de hoje sem passar pelo orçamento. É isso que mantém o `PlainSpecsUnchanged` bit a bit.
- [ ] **Step 4: Rodar e ver passar** — suíte inteira PASS.
- [ ] **Step 5: Commit** — `git commit -m "feat: map the free parameters around the fixed ones, weights within the budget left"`

---

### Task 3: Valores iniciais com iniciais do usuário e fixos

**Files:**
- Modify: `src/model/start_values.hpp` (comentário), `src/model/start_values.cpp`
- Test: `tests/unit/model/start_values_test.cpp`

**Interfaces:**
- Consumes: `Layout` (Task 1); `weight_terms`, `natural_from_terms`, `shock_moment` (Task 2). Assinatura de `start_values` sem mudança.
- Regra "Valores iniciais com fixos e iniciais".

- [ ] **Step 1: Escrever os testes**

```cpp
TEST(StartValues, PlainSpecsUnchanged) { /* testes do 2a–2c sem mudança (1e-15) */ }
TEST(StartValues, UserStartAndFixedReplaceDefaults) {
  // sGARCH(1,1)-std, start {beta1: 0.8, shape: 5}, fixed {mu: 0}: θ == {0, ω padrão, 0.05, 0.8, 5}
  // apARCH(1,1)-ged, fixed {delta: 1.5}: α₁ == 0.05 / κ₁ com κ₁ em delta = 1.5; ω == 0.05 var^0.75
}
TEST(StartValues, DefaultWeightsRescaleIntoBudget) {
  // sGARCH(1,1) fixed {alpha1: 0.3}: β₁ == 0.9·min(1, 0.95(1 - 1e-8 - 0.3)/0.9) (1e-15), is_valid
  // start {alpha1: 0.03}: β₁ == 0.9 exato (s = 1)
  // iGARCH(1,2) fixed {alpha1: 0.95}: β₁ reescalado, β₂ derivado ≥ 0, is_valid
  // gjrGARCH(1,1) fixed {gamma1: -0.05}: α₁ == 0.05 + 0.03 (antes da reescala), is_valid
  // csGARCH(1,1) fixed {alpha1: 0.9}: φ == min(0.03, 0.5 Σβ), is_valid
}
TEST(StartValues, InvalidCombinationStaysInvalid) { /* Review Focus 1: sGARCH fixed {alpha1: 0.6, beta1: 0.5}: !is_valid; fixed {alpha1: 1.2}: !is_valid */ }
TEST(StartValues, WarmStartFromEstimate) {
  // Review Focus 2: iGARCH(1,1) start {mu, omega, alpha1, beta1 (derivado)} de um θ válido: θ == o θ dado
  // nas estimadas e β₁ == 1 - α₁; sGARCH start {alpha1: 1e-12}: is_valid e to_unconstrained finito
}
```

- [ ] **Step 2: Rodar e ver falhar** — `ctest --preset gcc-debug -R StartValues` → FAIL.
- [ ] **Step 3: Implementar.**
- [ ] **Step 4: Rodar e ver passar** — suíte inteira PASS.
- [ ] **Step 5: Commit** — `git commit -m "feat: start from the user's values and fit the default weights in the budget"`

---

### Task 4: Variance targeting no `loglik`

**Files:**
- Modify: `src/model/likelihood.hpp`, `src/model/likelihood.cpp`
- Test: `tests/unit/model/likelihood_test.cpp`

**Interfaces:**
- Consumes: `persistence<T>`, `shock_moment` (Task 2); `Layout::variance_targeting` (Task 1).
- Produces:
  ```cpp
  namespace ve::model {
  // ω sob VT (regra "Variance targeting") a partir dos resíduos da média em theta. Instanciado como loglik.
  template <class T> T targeted_omega(std::span<const T> residuals, std::span<const T> theta, const Layout& layout);
  }
  ```
- Com `layout.variance_targeting`, o `loglik` calcula os resíduos, copia θ para um `std::array<T, kMaxParameterCount>` local com ω = `targeted_omega`, e segue com essa cópia (presample e recursão). Sem VT, nada muda.

- [ ] **Step 1: Escrever os testes**

```cpp
TEST(Likelihood, TargetedOmegaHandCase) {
  // r = {0.1, -0.2, 0.3, 0.05}, sem média, sample init: sGARCH(1,1) α=0.1 β=0.8 → ω = mean(r²)·0.1;
  // eGARCH(1,1) β=0.9 → ω = log(mean(r²))·0.1; apARCH(1,1)-norm δ=1.5 → ω = mean(|r|^1.5)(1 - ακ - β);
  // loglik com VT == loglik sem VT no θ com esse ω (rel 1e-14)
}
TEST(Likelihood, TargetingNearCeilingStaysFinite) {
  // Review Focus 4: sGARCH P = 1 - 2e-8 → ω > 0 e loglik finito; eGARCH em série com mean(r²) = 0.04 → ω < 0 finito, loglik finito
}
// GradientMatchesFiniteDifferences, DoubleAndDualAgreeOnValue e TermsSumToTotalAndScoresSumToGradient ganham
// VT ∈ {off, on} para SGarch, EGarch, GjrGarch, ApArch e FGarchAllGarch; a derivada na entrada omega é 0 com VT.
```

- [ ] **Step 2: Rodar e ver falhar** — `ctest --preset gcc-debug -R Likelihood` → não compila.
- [ ] **Step 3: Implementar.**
- [ ] **Step 4: Rodar e ver passar** — suíte inteira PASS; `Likelihood.MatchesPlanOneSimulatedValues` sem mudar constantes.
- [ ] **Step 5: Commit** — `git commit -m "feat: derive omega by variance targeting inside the likelihood"`

---

### Task 5: Ajuste e erros-padrão com papéis

**Files:**
- Modify: `src/fit/fit.cpp`, `src/fit/standard_errors.cpp`
- Test: `tests/unit/fit/fit_test.cpp`, `tests/unit/fit/standard_errors_test.cpp`, `tests/unit/fit/criteria_test.cpp`

**Interfaces:**
- Consumes: Tasks 1–4. `fit_one`, `standard_errors` e `FitResult` sem mudar de assinatura.
- `interior_indices` deixa de fora `!layout.is_estimated(i)`, além de `on_boundary`. Com zero parâmetros estimados, o `fit_one` não minimiza: avalia em θ = `start_values` (fixos e derivados) e segue pela regra "Todo fixo". Sob VT, `estimate[omega]` = `targeted_omega<double>` nos resíduos da estimativa.

- [ ] **Step 1: Escrever os testes**

```cpp
// T = 3000, séries do helper simulated_garch.hpp
TEST(FitRoles, FixedShapeRecoversTheRest) {
  // sGARCH(1,1)-std ν = 6 (seed 41), fixed {shape: 6}: Converged, estimate[shape] == 6 exato,
  // se_hessian[shape] e se_robust[shape] NaN, os outros finitos; ℓ̂ ≥ ℓ(θ verdadeiro) - 1e-6|ℓ|
}
TEST(FitRoles, FixedWeightFits) {
  // sGARCH(1,1)-norm (seed 42), fixed {alpha1: 0.08}: Converged, α₁ == 0.08, α₁ + β̂₁ < 1;
  // gjrGARCH(1,1)-sstd (seed 43), fixed {gamma1: 0.09}: Converged
}
TEST(FitRoles, InvalidFixedFails) { /* Review Focus 1: fixed {alpha1: 0.6, beta1: 0.5} → Failed, estimate NaN, sem exceção */ }
TEST(FitRoles, AllFixedEvaluates) {
  // sGARCH(1,1)-std, todos fixos no θ verdadeiro: Converged; loglik == model::loglik<double>(r, θ) (rel 1e-14);
  // sigma == √σ² do loglik; SE todos NaN; aic(PerObs) == -2ℓ/n (k = 0)
}
TEST(FitRoles, WarmStartReachesTheSameEstimate) {
  // Review Focus 2: iGARCH(1,1)-norm e sGARCH(1,1)-std com VT: refit com start = todas as estimativas (nomes do
  // FitResult): Converged e |θ̂' - θ̂| ≤ 1e-6·max(1, |θ̂|) nas estimadas
}
TEST(FitRoles, VarianceTargeting) {
  // sGARCH(1,1)-std VT (seed 41): Converged; estimate[omega] == targeted_omega(resíduos, θ̂) (1e-15);
  // se[omega] NaN; ℓ̂_VT ≤ ℓ̂_sem_VT + 1e-9|ℓ| (VT restringe); bic usa k = 4 (mu, alpha1, beta1, shape)
}
TEST(StandardErrors, LeavesOutNonEstimated) { /* fixos e derivados: NaN em hessian e robust; os outros iguais aos do bloco reduzido */ }
TEST(Criteria, CountEstimatedOnly) { /* k = estimated_count() com fixos e VT */ }
```

- [ ] **Step 2: Rodar e ver falhar** — `ctest --preset gcc-debug -R "FitRoles|StandardErrors|Criteria"` → FAIL.
- [ ] **Step 3: Implementar.**
- [ ] **Step 4: Rodar e ver passar** — suíte C++ inteira PASS, `-Wnrvo` sem aviso.
- [ ] **Step 5: Commit** — `git commit -m "feat: fit around fixed and derived parameters"`

---

### Task 6: Interface C e Python

**Files:**
- Modify: `include/volengine/volengine.h`, `src/capi/spec_api.cpp`, `src/fit/batch.cpp`, `python/src/volengine/_spec.py`, `python/src/volengine/_native.py`
- Test: `tests/unit/capi/spec_test.cpp`, `tests/unit/capi/batch_test.cpp`, `python/tests/test_fit.py`

**Interfaces:**
- Produces (C, só acréscimos):
  ```c
  VE_API ve_status ve_spec_set_fixed(ve_spec* s, const char* name, double value);
  VE_API ve_status ve_spec_set_start(ve_spec* s, const char* name, double value);
  VE_API ve_status ve_spec_set_variance_targeting(ve_spec* s, int32_t on);   /* 0 ou 1 */
  VE_API ve_status ve_spec_validate(const ve_spec* s);
  ```
  `name` NULL ou vazio, ou `value` não finito: `VE_INVALID_ARGUMENT` e a spec não muda. `ve_spec_validate` devolve `VE_OK` ou `VE_INVALID_ARGUMENT` com a mensagem de `spec_problem`. O `check_batch` chama `spec_problem` para cada spec antes dos jobs e devolve `"spec <i>: " + mensagem`. O header documenta, junto de cada função, a regra "Papéis dos parâmetros", as fórmulas do VT e as combinações rejeitadas.
- Python: `Spec(..., fixed: Mapping[str, float] = field(default_factory=dict), start: Mapping[str, float] = field(default_factory=dict), variance_targeting: bool = False)`. O `__post_init__` copia `fixed` e `start` para um `types.MappingProxyType(dict(...))` com valores `float`, levanta `ValueError` se algum nome estiver nos dois, e valida pelo handle nativo (os setters e depois `ve_spec_validate`). A classe define `__hash__` explícito sobre os campos, com `frozenset(fixed.items())` e `frozenset(start.items())`, para que `test_arma_list_equals_tuple_and_hashes` continue passando. A docstring de `Spec` descreve os papéis, o VT e os casos rejeitados.

- [ ] **Step 1: Escrever os testes**

```cpp
TEST(CapiSpec, FixedAndStartSetters) {
  // NULL e "" como name, value NaN e inf → INVALID_ARGUMENT; set_fixed depois de set_start no mesmo nome deixa só o fixo
  // set_variance_targeting(2) → INVALID_ARGUMENT com "variance_targeting"
}
TEST(CapiSpec, ValidateReportsSpecProblems) {
  // Review Focus 3: std, fixed {shape: 5}, depois set_dist(NORM): validate → INVALID_ARGUMENT, mensagem com "shape";
  // VT em IGARCH → mensagem com "variance targeting"
}
TEST(CapiFit, RejectsInvalidSpecInBatch) { /* Review Focus 3: a mesma spec num lote → INVALID_ARGUMENT, "spec 1: ", fits_out todo NULL */ }
TEST(CapiFit, AllFixedMixedWithEstimated) {
  // Review Focus 5: specs {sGARCH(1,1)-std todo fixo, gjrGARCH(1,1)-sstd VT, sGARCH(1,1)-norm fixed {alpha1: 0.08}}
  // × 3 séries T = 600, threads 1 e 4: estimates/loglik/sigma bit-idênticos a ve::fit_one; a toda fixa CONVERGED,
  // SE NaN, ve_aic(PER_OBS) == -2ℓ/n
}
```
```python
def test_fixed_parameter_is_kept_with_nan_se(ctx, series): ...   # Spec(dist="std", fixed={"shape": 6.0}): coef estimate 6, SE NaN
def test_start_does_not_change_the_optimum(ctx, series): ...     # start = dict(zip(f.names, f.coef().estimate)): mesma estimativa (1e-6)
def test_variance_targeting_omega_has_nan_se(ctx, series): ...
def test_unknown_fixed_name_raises(): ...                         # Spec(fixed={"shpe": 5.0}) → VolEngineError com "shpe"
def test_name_in_fixed_and_start_raises(): ...                    # ValueError
def test_spec_with_values_hashes(): ...                           # Spec(fixed={"mu": 0.0}) == e hash igual a outra igual; dict de origem mutado depois não muda a spec
```

- [ ] **Step 2: Rodar e ver falhar** — `ctest --preset gcc-debug -R "CapiSpec|CapiFit"` e pytest → FAIL.
- [ ] **Step 3: Implementar.**
- [ ] **Step 4: Rodar e ver passar** — suítes C++ e Python inteiras PASS; header C11-limpo (`c_header_check.c`); `mypy --strict` dos casos de tipo PASS.
- [ ] **Step 5: Commit** — `git commit -m "feat: expose fixed and start values and variance targeting through the C API and Python"`

---

### Task 7: Oráculos

**Files:**
- Modify: `tests/reference/cases.json`, `tests/reference/oracle_rugarch.R`, `tests/reference/oracle_arch.py`, `tests/reference/README.md`, `python/tests/test_oracles.py`, `docs/guide/validation.md`
- Regenerate (commitados): `tests/reference/fixtures/rugarch.json`, `tests/reference/fixtures/arch.json`

**Interfaces:**
- Consumes: pacote Python (Task 6). Pré-requisitos: R 4.6.2 + `rugarch` 1.5.6 e o venv com `arch` 8.0.0. Se faltar algum, pare e peça ao usuário.
- Formato do caso, só acréscimos: `"fixed": {name: float}` e `"variance_targeting": bool`, os dois opcionais. Um caso com todos os parâmetros em `fixed` é de avaliação: o `rugarch` usa `ugarchfilter`, o `arch` usa `fix()`, e a fixture leva `loglik`, `sigma` e os critérios, com `se` e `robust_se` nulos. No resto da fixture, os parâmetros fixos e derivados aparecem em `params`, com `se` nulo.
- Sem dados novos: os casos usam as séries de `grid.csv`, `grid_2b.csv` e `grid_2c.csv`. Casos (fixados):

| id | série | modelo | papéis | oráculos |
|---|---|---|---|---|
| `sgarch21_std/sgarch11_std_fixshape` | `sgarch21_std` | sGARCH(1,1)-std | `shape` = 5 | rugarch |
| `arma11_sgarch11_ged/arma11_sgarch11_ged_fixmu` | `arma11_sgarch11_ged` | ARMA(1,1)-sGARCH(1,1)-ged | `mu` = 0 | rugarch |
| `sgarch21_std/sgarch11_norm_fixalpha` | `sgarch21_std` | sGARCH(1,1)-norm | `alpha1` = 0.08 | rugarch |
| `gjrgarch11_norm/gjrgarch11_norm_fixgamma` | `gjrgarch11_norm` | gjrGARCH(1,1)-norm | `gamma1` = 0.1 | rugarch |
| `gjrgarch11_sstd/gjrgarch11_sstd_fixalpha` | `gjrgarch11_sstd` | gjrGARCH(1,1)-sstd | `alpha1` = 0.03 | rugarch |
| `aparch11_ged/aparch11_ged_fixdelta` | `aparch11_ged` | apARCH(1,1)-ged | `delta` = 1.5 | rugarch |
| `egarch11_std/egarch11_std_fixbeta` | `egarch11_std` | eGARCH(1,1)-std | `beta1` = 0.95 | rugarch |
| `csgarch11_std/csgarch11_std_fixrho` | `csgarch11_std` | csGARCH(1,1)-std | `eta11` = 0.99 | rugarch |
| `igarch11_norm/igarch11_norm_fixalpha` | `igarch11_norm` | iGARCH(1,1)-norm | `alpha1` = 0.1 | rugarch |
| `sgarch21_std/sgarch11_std_vt` | `sgarch21_std` | sGARCH(1,1)-std | VT | rugarch |
| `egarch11_std/egarch11_std_vt` | `egarch11_std` | eGARCH(1,1)-std | VT | rugarch |
| `gjrgarch11_sstd/gjrgarch11_sstd_vt` | `gjrgarch11_sstd` | gjrGARCH(1,1)-sstd | VT | rugarch |
| `aparch11_ged/aparch11_ged_vt` | `aparch11_ged` | apARCH(1,1)-ged | VT | rugarch |
| `fgarch_allgarch11_norm/fgarch_allgarch11_norm_vt` | `fgarch_allgarch11_norm` | fGARCH-ALLGARCH(1,1)-norm | VT | rugarch |
| `sgarch21_std/sgarch11_std_allfixed` | `sgarch21_std` | sGARCH(1,1)-std | todos: μ 0.05, ω 0.02, α₁ 0.08, β₁ 0.9, ν 6 | rugarch, arch |
| `gjrgarch11_sstd/gjrgarch11_sstd_allfixed` | `gjrgarch11_sstd` | gjrGARCH(1,1)-sstd | todos: μ 0.05, ω 0.01, α₁ 0.03, β₁ 0.9, γ₁ 0.09, ξ 0.85, ν 6 | rugarch |

- Regra de interior do 2b/2c, só sobre os parâmetros estimados. Caso que falhe é trocado, com o motivo em `validation.md` ("Case selection").
- Tolerâncias: as do 2a, por grandeza, nos casos estimados. Nos casos de avaliação, `loglik` e `sigma` com rel 1e-9: é a mesma conta nos dois lados, sem otimizador.
- Convenções a confirmar e documentar em `validation.md`:
  - que os critérios do `rugarch` contam só os estimados (`ipars[, "Estimate"]`), sem os fixos e sem o ω do VT;
  - que o SE dos livres do `rugarch` com `fixed.se = FALSE` vem da Hessiana reduzida, como aqui;
  - as fórmulas do ω sob VT da regra "Variance targeting", inclusive o `max(1 − persist, ·)` do eGARCH, com as referências (Engle & Mezrich 1996; vinheta do `rugarch`, eq. 10);
  - que os SE sob VT são os do `rugarch` e não os de dois passos de Francq, Horváth & Zakoïan (2011);
  - que o `fix()` do `arch` sob backcast dá o mesmo ℓ.

  Se o oráculo divergir, a correção vai no núcleo (Tasks 2–5), com os testes de gradiente mantidos, e a mudança é registrada.

- [ ] **Step 1: Gerar as fixtures** — `Rscript tests/reference/oracle_rugarch.R && python/.venv/Scripts/python tests/reference/oracle_arch.py` → um resultado por caso e oráculo; os scripts param com erro em não-convergência ou fronteira; as fixtures dos casos antigos não mudam (diff só com acréscimos).
- [ ] **Step 2: Estender o teste** — `case_spec` passa `fixed` e `variance_targeting`; `derived_mask` marca os fixos e o ω do VT; os casos de avaliação comparam `loglik`, `sigma` e os critérios.
- [ ] **Step 3: Rodar** — `VOLENGINE_LIB=<abs> python/.venv/Scripts/python -m pytest python/tests/test_oracles.py -v` → PASS; depois de cada correção no núcleo, as suítes C++ e Python inteiras.
- [ ] **Step 4: Commit** — `git commit -m "test: validate fixed parameters and variance targeting against rugarch and arch"`

---

### Task 8: Documentação

**Files:**
- Modify: `README.md`, `CHANGELOG.md`, `ARCHITECTURE.md`, `docs/performance/benchmarks.md`

**Interfaces:**
- README: um exemplo com `Spec(dist="std", fixed={"shape": 6.0})` e outro de warm-start com `start=`. CHANGELOG (`[Unreleased]`). ARCHITECTURE: os papéis no `Layout`, `weight_terms.hpp`, o ω derivado no `loglik` e o ajuste sem otimizador quando tudo é fixo. `benchmarks.md`: nota de que o 2d não muda hot path sem fixos (os `BM_*` de hoje cobrem esse caminho) e de que nenhum benchmark novo foi criado.

- [ ] **Step 1: Escrever.**
- [ ] **Step 2: Verificar** — `ctest --preset gcc-release` e pytest inteiros PASS; `check_style.py .` sai 0.
- [ ] **Step 3: Commit** — `git commit -m "docs: document fixed and start values and variance targeting"`
