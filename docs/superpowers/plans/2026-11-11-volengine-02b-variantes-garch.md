# VolEngine — Plano 2b: eGARCH, gjrGARCH, apARCH, iGARCH, csGARCH e fGARCH

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Sobre o motor genérico do plano 2a, entregar as variâncias eGARCH, gjrGARCH, apARCH, iGARCH, csGARCH e fGARCH (oito submodelos), com média ARMA e choques norm/std/ged, validadas contra `rugarch` e `arch`, expostas pela interface C e pelo Python, e medidas com benchmark, profile e leitura de assembly.

**Architecture:** Cada família de recursão vira um template sobre o tipo escalar em `src/garch/`, e `model::loglik` escolhe a recursão pelo `Model` do `Layout`. O `Layout` ganha os blocos `gamma`, `eta1`, `eta2`, a potência (`delta`/`lambda`) e o parâmetro derivado do iGARCH. A reparametrização continua impondo positividade e persistência < 1 em todas as famílias, e para isso precisa dos momentos dos choques, que vêm de `src/dist/moments` (forma fechada quando η₂ ≡ 0, quadratura quando η₂ é livre). O driver, o L-BFGS e os erros-padrão não mudam de forma: só leem o `Layout`.

**Tech Stack:** C++23, CMake + presets, GoogleTest v1.17.0, Google Benchmark v1.9.1, BS::thread_pool v5.0.0, Python ≥ 3.11 + numpy + pytest, `arch` 8.0.0, R 4.6.2 + `rugarch` 1.5.6. Medição (rodar benchmarks, AMD uProf, assembly) fica para a máquina de benchmark (decisão do usuário, 2026-11-11).

**Spec:** `docs/superpowers/specs/2026-11-10-volengine-design.md`. O plano 2a (`docs/superpowers/plans/2026-11-11-volengine-02a-motor-generico.md`) é o histórico das convenções que este plano estende.

## Roteiro

- **2a (feito):** motor genérico com AD, sGARCH(p,q), ARMA(m,n), norm/std/ged, benchmarks.
- **2b (este):** eGARCH, gjrGARCH, apARCH, iGARCH, csGARCH, fGARCH e submodelos.
- **2c:** sstd, sged, nig, ghyp, jsu. Este plano usa a simetria de norm/std/ged em dois pontos, e o 2c os generaliza: κ = ½ do gjrGARCH vira P(z < 0), e o momento assimétrico fechado da Task 2 vira quadratura para as densidades assimétricas.
- **2d:** parâmetros fixos, valores iniciais, variance targeting, `mxreg`/`vxreg`, `archm`/`archpow`, ARFIMA.

Depois: 3 diagnósticos/forecast/simulação; 4 DCC; 5 performance (L-BFGS em lote com SIMD); 6 wrapper C++, wheels, 1.0.

## Global Constraints

Valem todas as Global Constraints dos planos 1 e 2a, o `STYLEGUIDE.md` e o `ARCHITECTURE.md`. As que mais pesam aqui:

- A interface C é o único contrato estável. Nenhuma struct pública; enums `typedef int32_t` + `#define`, que só crescem; ≤ 5 parâmetros por função; nenhuma exceção atravessa a fronteira C.
- Sem estado global mutável. Resultados imutáveis. Falha de um ajuste não falha o lote.
- Ajustes bit-idênticos para qualquer número de threads na mesma máquina e ISA. O sGARCH continua dando os mesmos números do 2a: os testes `Likelihood.MatchesPlanOneSimulatedValues` e de oráculo do 2a passam sem mudar constantes nem tolerâncias.
- Convenção padrão segue o `rugarch` (`VE_VARINIT_SAMPLE`); a do `arch` é `VE_VARINIT_BACKCAST`.
- Código do `rugarch` (GPL-3) pode ser lido para entender convenções, nunca copiado ou traduzido. O `arch` é NCSA: código portado exige aviso em `THIRD-PARTY-NOTICES.md`.
- Anti-objetivo: não fazer mais do que o necessário. Nada de componente permanente do csGARCH como saída, nem persistência como função pública: não estão no escopo.
- Estilo: nunca `else`; funções ≤ 60 linhas; ≤ 5 parâmetros; aninhamento ≤ 3; ≤ 8 módulos por pasta; comentários só para restrição externa ou invariante; medições em `docs/`.
- **Retorno por valor com NRVO/RVO** (`STYLEGUIDE.md`, "Return values — RVO and NRVO"). O build GCC usa `-Wnrvo` como erro: uma função que devolve um local nomeado só devolve esse local, e os guards que devolvem outra coisa ficam numa função de fora.
- **Método de performance** (`ARCHITECTURE.md`, "Performance method"). Cada recursão nova entra com um caso `BM_Loglik` (Task 9). Rodar, fazer o profile e ler o assembly ficam para a máquina de benchmark (AMD Ryzen 7 5700X, a do `docs/performance/benchmarks.md`), na seção "Pendente na máquina de benchmark" no fim deste plano. Nesta máquina (Intel Core i7-1365U, notebook) nenhum número de desempenho é publicado. Uma otimização que custa legibilidade só entra com números de antes e depois medidos lá.
- **API Python** (decisão do usuário, 2026-11-11): estatísticas de um ajuste são membros do `Fit`, não funções do módulo. `f.loglik` é propriedade; `f.aic()`, `f.bic()`, `f.hqic()`, `f.sic()` são métodos (com `per_obs`). Os diagnósticos do plano 3 seguem o mesmo padrão (`f.ljung_box(lags=10)`). Na interface C continuam funções sobre o handle (`ve_aic(fit, ...)`).
- Commits sem trailer `Co-Authored-By`; LF. Windows/MinGW: DLL com runtime estático. Python no venv `python/.venv`, criado pelo Windows Python (`py -3.14 -m venv python/.venv`, depois `pip install -e "python[test]" arch==8.0.0`). Esta máquina ainda não tem o venv.

### Novas regras deste plano

**Modelos (C e `ve::Model`, só acréscimos):**

| Valor | C | `ve::Model` |
|---|---|---|
| 0 | `VE_MODEL_SGARCH` | `SGarch` |
| 1 | `VE_MODEL_EGARCH` | `EGarch` |
| 2 | `VE_MODEL_GJRGARCH` | `GjrGarch` |
| 3 | `VE_MODEL_APARCH` | `ApArch` |
| 4 | `VE_MODEL_IGARCH` | `IGarch` |
| 5 | `VE_MODEL_CSGARCH` | `CsGarch` |
| 6–13 | `VE_MODEL_FGARCH_GARCH`, `_TGARCH`, `_AVGARCH`, `_NGARCH`, `_NAGARCH`, `_APARCH`, `_GJRGARCH`, `_ALLGARCH` | `FGarchGarch`, `FGarchTGarch`, `FGarchAvGarch`, `FGarchNGarch`, `FGarchNaGarch`, `FGarchApArch`, `FGarchGjrGarch`, `FGarchAllGarch` |

Cada submodelo do fGARCH é um valor de modelo, não um setter. Assim não há submodelo padrão a documentar nem combinação inválida (submodelo numa spec que não é fGARCH). O Python oferece `Spec(model="fgarch", submodel="tgarch")` e traduz para o valor.

**Nomes e ordem dos parâmetros** (a ordem do `rugarch`): `mu`, `ar1..arm`, `ma1..man`, `omega`, `alpha1..alphap`, `beta1..betaq`, `gamma1..gammap`, `eta11..eta1p`, `eta21..eta2p`, `delta` ou `lambda`, `shape`. Blocos por modelo:

| Modelo | `gamma` | `eta1` | `eta2` | potência |
|---|---|---|---|---|
| sGARCH, iGARCH | — | — | — | — |
| eGARCH, gjrGARCH | p | — | — | — |
| apARCH | p | — | — | `delta` livre |
| csGARCH | — | 1 (`eta11` = ρ) | 1 (`eta21` = φ) | — |
| fGARCH-GARCH | — | — | — | λ = 2 fixo |
| fGARCH-TGARCH | — | p | — | λ = 1 fixo |
| fGARCH-AVGARCH | — | p | p | λ = 1 fixo |
| fGARCH-NGARCH | — | — | — | `lambda` livre |
| fGARCH-NAGARCH | — | — | p | λ = 2 fixo |
| fGARCH-APARCH | — | p | — | `lambda` livre |
| fGARCH-GJRGARCH | — | p | — | λ = 2 fixo |
| fGARCH-ALLGARCH | — | p | p | `lambda` livre |

No iGARCH, `beta_q` é **derivado**: tem nome e valor (1 − Σα − Σ_{j<q} βⱼ), mas não é estimado. Os critérios usam k = parâmetros estimados (`estimated_count()`), e o SE dele é NaN. O maior layout (fGARCH-ALLGARCH, ARMA(6,6), (6,6), std) tem `kMaxParameterCount = 6·kMaxOrder + 4 = 40` parâmetros. Por isso entra `Dual<40>`, e os buckets ficam 8, 16, 32, 40.

**Recursões** (εₜ resíduo da média, zₜ = εₜ/σₜ, κ e momentos sob a distribuição padronizada):

- eGARCH: log σ²ₜ = ω + Σᵢ [αᵢ zₜ₋ᵢ + γᵢ (|zₜ₋ᵢ| − E|z|)] + Σⱼ βⱼ log σ²ₜ₋ⱼ. E|z| é o da distribuição da spec, como no `rugarch`.
- gjrGARCH: σ²ₜ = ω + Σᵢ (αᵢ + γᵢ·1[εₜ₋ᵢ < 0]) ε²ₜ₋ᵢ + Σⱼ βⱼ σ²ₜ₋ⱼ.
- iGARCH: a recursão do sGARCH com βq recalculado **dentro** do `loglik` a partir das outras entradas de θ (o valor guardado em θ[βq] é ignorado). Assim ∂ℓ/∂θ[βq] = 0, e as derivadas dos demais já são as totais. Hessiana, scores e AD funcionam sem caso especial.
- Família de potência (apARCH e fGARCH, um só módulo): σₜ^λ = ω + Σᵢ αᵢ σₜ₋ᵢ^λ (|zₜ₋ᵢ − η₂ᵢ| − η₁ᵢ (zₜ₋ᵢ − η₂ᵢ))^λ + Σⱼ βⱼ σₜ₋ⱼ^λ, com δ ≡ λ (hipótese: o `rugarch` amarra δ a λ nos submodelos em que os dois são livres; a Task 8 confirma). O apARCH é essa recursão com η₁ = γ, λ = δ e η₂ = 0, e equivale a σ^δ = ω + Σ α(|ε| − γε)^δ + Σ β σ^δ.
- csGARCH: qₜ = ω + ρ qₜ₋₁ + φ (ε²ₜ₋₁ − σ²ₜ₋₁); σ²ₜ = qₜ + Σᵢ αᵢ (ε²ₜ₋ᵢ − qₜ₋ᵢ) + Σⱼ βⱼ (σ²ₜ₋ⱼ − qₜ₋ⱼ).

**Pré-amostral:**

- `VE_VARINIT_SAMPLE` (`rugarch`): para t < held = max(m, n, p, q) vale σ²ₜ = m̄ (média de ε² com os parâmetros de média correntes, como no 2a), e os estados derivados seguem: log σ²ₜ = log m̄, σₜ^λ = m̄^{λ/2}, qₜ = m̄. Daí em diante a recursão usa lags reais. Hipótese por família, confirmada na Task 8.
- `VE_VARINIT_BACKCAST` (`arch`): a recursão começa em t = 0. Um lag antes de 0 lê o valor esperado do termo, dado b (backcast fixo do 2a): σ² = ε² = q = b; 1[ε < 0]ε² = b/2; z = 0 e |z| − E|z| = 0; σ^λ = b^{λ/2}; (|z − η₂| − η₁(z − η₂))^λ = κ (o momento do choque). Para GJR e EGARCH essa é a convenção do `arch`. Para o APARCH é hipótese, que a Task 8 confirma ou registra. Para os demais modelos o `arch` não tem o modelo: é a extensão natural, sem oráculo, e diz isso em `validation.md`.

**Momentos dos choques** (simétricos; o 2c generaliza):

- E|z|^δ: norm 2^{δ/2} Γ((δ+1)/2)/√π; std (ν > δ) (ν−2)^{δ/2} Γ((δ+1)/2) Γ((ν−δ)/2) / (√π Γ(ν/2)), e +∞ para ν ≤ δ; ged λ_g^δ 2^{δ/ν} Γ((δ+1)/ν) / Γ(1/ν), com o λ_g de `ged_log_density`.
- η₂ ≡ 0: κ = E(|z| − η₁ z)^λ = ½[(1 − η₁)^λ + (1 + η₁)^λ] E|z|^λ.
- η₂ livre (AVGARCH, NAGARCH, ALLGARCH): κ = E(|z − η₂| − η₁(z − η₂))^λ por quadratura, partindo em z = η₂ (onde está o bico): ∫ sobre [η₂, ∞) e (−∞, η₂], cada metade mapeada por z = η₂ ± t/(1 − t), t ∈ [0, 1), com Gauss–Legendre de 64 nós. Se a tolerância da Task 2 não fechar com 64 nós, aumente os nós e registre em `docs/performance/benchmarks.md`. A tolerância não afrouxa.
- Um κ não finito (std com ν ≤ λ) põe o ponto fora do domínio: `loglik` não finito e `is_valid` falso.

**Persistência** P < 1 − 1e-8 (iGARCH: P = 1 exato):

- sGARCH, csGARCH: P = Σα + Σβ.
- eGARCH: P = Σβ.
- gjrGARCH: P = Σα + Σβ + ½Σγ.
- apARCH/fGARCH: P = Σ αᵢκᵢ + Σβ.
- csGARCH também: P < ρ < 1 − 1e-8 e 0 ≤ φ ≤ Σβ (hipótese sobre os limites do `rugarch`; só pesa na fronteira).

**Reparametrização** (irrestrito u → natural θ; s = sd(r)). μ, ARMA e shape como no 2a. Os **pesos** são as entradas ≥ 0 que somam P. Suas posições em u são as mesmas posições naturais do bloco: a primeira guarda v, com P = logistic(v)(1 − 1e-8), e as demais guardam os logits do softmax(w, 0).

- sGARCH, csGARCH: pesos (α, β).
- iGARCH: pesos (α, β) = softmax(w₁..w_{p+q−1}, 0), soma exatamente 1, sem v. Os logits ocupam as p+q−1 primeiras posições do bloco, e a posição de βq em u fica sem uso (u = 0, gradiente zero).
- eGARCH: pesos = β. α e γ são identidade. ω = u_ω + (1 − Σβ)·log s², porque log σ² muda de log s² quando r é escalado.
- gjrGARCH: pesos, na ordem das posições alpha, beta, gamma: (½α₁..½α_p, β₁..β_q, ½(α₁+γ₁)..½(α_p+γ_p)). Daí αᵢ = 2w_{αᵢ}, γᵢ = 2(w_{γᵢ} − w_{αᵢ}) e α + γ ≥ 0.
- apARCH/fGARCH: pesos (α₁κ₁..α_pκ_p, β), logo αᵢ = wᵢ/κᵢ. κᵢ é calculado a partir de η₁ᵢ (ou γᵢ), η₂ᵢ, λ e shape, que são mapeados antes.
- η₁ e o γ do apARCH: 2·logistic(x) − 1. η₂: identidade. δ e λ livres: exp(x).
- ω = s²·eᵘ (sGARCH, iGARCH, gjr, cs) ou s^λ·eᵘ (família de potência, com λ fixo ou livre).
- csGARCH: ρ = P + (1 − 1e-8 − P)·logistic(x_ρ); φ = Σβ·logistic(x_φ).
- `to_unconstrained` inverte cada mapa (razões de log com piso `DBL_MIN`, como no 2a; logit para η₁; κ em `double`).

**Valores iniciais** (naturais; shapes como no 2a, var = variância amostral):

| Modelo | Valores |
|---|---|
| eGARCH | ω = 0.1·log var, α = 0, γ = 0.1/p, β = 0.9/q |
| gjrGARCH | ω = 0.05·var, α = 0.03/p, γ = 0.04/p, β = 0.9/q |
| iGARCH | ω = 0.01·var, α = 0.1/p, β = 0.9/q |
| apARCH | δ = 2, γ = 0 (κ = 1), α = 0.05/p, β = 0.9/q, ω = 0.05·var |
| csGARCH | α = 0.05/p, β = 0.9/q, ρ = 0.98, φ = 0.03, ω = 0.02·var |
| fGARCH | λ = 2 se livre, η₁ = η₂ = 0 se livres, αᵢ = 0.05/(p·κᵢ), β = 0.9/q, ω = 0.05·var^{λ/2} |

**Fronteira** (fica fora da Hessiana e da matriz de scores, SE NaN):

- um peso < 1e-8; no gjrGARCH, αᵢ < 1e-8 marca αᵢ, e αᵢ + γᵢ < 1e-8 marca γᵢ;
- |η₁ᵢ| ou |γᵢ| do apARCH > 1 − 1e-8;
- no csGARCH, φ < 1e-8 marca `eta21`, e ρ − P < 1e-8 marca `eta11`;
- o βq derivado do iGARCH, sempre;
- a regra do std do 2a (1/ν < 1e-8).

**Passos da Hessiana** por `ParameterKind`:

- `Location` (μ): 1e-5·max(|θ|, s);
- `Real` (ARMA; ω, α, γ do eGARCH; γ do gjr; γ do apARCH; η₁; η₂): 1e-5·max(|θ|, 1e-3);
- `Positive` (ω dos demais, δ, λ, ρ, shape): 1e-5·θ;
- `Weight` (α, β, φ): 1e-5·max(θ, 1e-6).

**Dados insuficientes:** comprimento < `estimated_count()` + 2, ou ≤ max(m, n, p, q).

**Séries simuladas** (iguais no gerador Python e no helper C++; partem de σ² = 1, e as 1000 primeiras observações de aquecimento são descartadas; μ = 0.05):

| Série | Processo | Choques |
|---|---|---|
| `egarch11_std` | eGARCH(1,1): ω = −0.01, α = −0.08, γ = 0.15, β = 0.97 | std ν = 6 |
| `gjrgarch11_norm` | gjrGARCH(1,1): ω = 0.01, α = 0.03, γ = 0.09, β = 0.9 | norm |
| `aparch11_ged` | apARCH(1,1): ω = 0.02, α = 0.08, γ = 0.4, β = 0.9, δ = 1.3 | ged ν = 1.5 |
| `igarch11_norm` | iGARCH(1,1): ω = 0.01, α = 0.1 (β = 0.9) | norm |
| `csgarch11_std` | csGARCH(1,1): ω = 0.01, α = 0.06, β = 0.85, ρ = 0.99, φ = 0.04 | std ν = 7 |
| `fgarch_allgarch11_norm` | fGARCH-ALLGARCH(1,1): ω = 0.02, α = 0.08, β = 0.88, η₁ = 0.3, η₂ = 0.2, λ = 1.5 | norm |

## Review Focus

1. **apARCH/fGARCH com std e ν ≤ λ** (κ infinito, caudas pesadas demais para o momento): `loglik` não finito, e o ajuste termina finito (`Converged` com ν̂ > λ̂, `MaxIter` ou `Failed`), sem exceção e sem NaN vazando. → testes nas Tasks 4 e 6.
2. **eGARCH com um outlier enorme** (um retorno de 1e3 desvios): exp(log σ²) estoura, `loglik` fica não finito, a line search recua e o ajuste termina finito. → testes nas Tasks 3 e 6.
3. **iGARCH e o βq derivado:** o nome aparece, o valor é 1 − Σ dos outros, o SE é NaN só nele, os critérios contam k − 1, e os SEs dos demais são finitos. → testes nas Tasks 6 e 7.
4. **Lote misturando k = 4 (`Dual<8>`) com k = 40 (`Dual<40>`)**: cada ajuste é bit-idêntico ao `fit_one` isolado, com 1 e com 4 threads. → teste na Task 7.
5. **fGARCH com η₂ ≠ 0 e std de cauda pesada (ν = 4.5)**, onde o integrando de κ tem um bico em z = η₂ e cauda algébrica: a quadratura bate com as formas fechadas conhecidas. → teste na Task 2.

---

## Mapa de arquivos

```
src/core/spec.hpp                  Model com 14 valores; kMaxParameterCount
src/model/layout.hpp/.cpp          blocos gamma, eta1, eta2, potência; derivado; ParameterKind
src/dist/moments.hpp/.cpp          E|z|^δ, momento assimétrico, momento deslocado (quadratura)
src/garch/presample.hpp            Presample e lagged, saídos de sgarch.hpp
src/garch/egarch.hpp               recursão eGARCH
src/garch/gjrgarch.hpp             recursão gjrGARCH
src/garch/power_garch.hpp          recursão apARCH/fGARCH
src/garch/csgarch.hpp              recursão csGARCH
src/model/likelihood.cpp           despacho por modelo; iGARCH; instância Dual<40>
src/model/parameter_map.*          reparametrização, validade e fronteira de todos os modelos
src/model/start_values.cpp         valores iniciais por modelo
src/fit/fit.cpp, standard_errors.cpp, criteria.cpp   Dual<40>, passos por ParameterKind, k estimado
src/capi/spec_api.cpp, include/volengine/volengine.h  modelos novos
python/src/volengine/_spec.py      model, submodel
tests/unit/...                     espelhando src/
tests/reference/...                séries, casos e fixtures novos
tests/benchmarks/fit_benchmarks.cpp  casos novos e grade com todos os modelos
```

---

### Task 0: Estatísticas como membros do `Fit` no Python

**Files:**
- Modify: `python/src/volengine/_fit.py`, `python/src/volengine/api.py`, `python/src/volengine/__init__.py`, `python/benchmarks/bench_vs_arch.py`, `README.md`, `ARCHITECTURE.md`, `CHANGELOG.md` (a seção 7 do spec já foi atualizada junto com este plano)
- Test: `python/tests/test_fit.py`, `python/tests/test_oracles.py` e os demais testes que chamam `ve.aic`/`ve.loglik`

**Interfaces:**
- Produces (Python; a interface C não muda):
  ```python
  class Fit:
      @property
      def loglik(self) -> float: ...                          # ve_loglik
      def aic(self, *, per_obs: bool = True) -> float: ...    # ve_aic
      def bic(self, *, per_obs: bool = True) -> float: ...
      def hqic(self, *, per_obs: bool = True) -> float: ...
      def sic(self, *, per_obs: bool = True) -> float: ...
  ```
- As funções de módulo `ve.loglik`, `ve.aic`, `ve.bic`, `ve.hqic`, `ve.sic` são removidas, sem alias: a versão é 0.x, e dois caminhos para a mesma coisa é o que o projeto evita. Entra no CHANGELOG como mudança incompatível.

- [ ] **Step 1: Escrever os testes** — `test_fit_statistics_are_members`: `f.loglik == ` o valor lido direto por `ve_loglik`; `f.aic() == -2·f.loglik/n + 2k/n`, e `f.aic(per_obs=False) == -2·f.loglik + 2k` (rel 1e-14), o mesmo para bic, hqic e sic; e `not hasattr(ve, "aic")`. Troque os usos antigos nos testes existentes.
- [ ] **Step 2: Rodar e ver falhar** — `VOLENGINE_LIB=<abs> python/.venv/Scripts/python -m pytest python/tests -q` → FAIL.
- [ ] **Step 3: Implementar** e atualizar o benchmark Python e o README (exemplo `f.loglik`, `f.aic()`).
- [ ] **Step 4: Rodar e ver passar** — pytest inteiro PASS; `check_style.py .` sai 0; mypy dos casos de tipo PASS.
- [ ] **Step 5: Commit** — `git commit -m "feat!: make loglik and the information criteria members of Fit"`

---

### Task 1: Modelos, Layout estendido e `Dual<40>`

**Files:**
- Modify: `src/core/spec.hpp`, `src/model/layout.hpp`, `src/model/layout.cpp`, `src/fit/fit.cpp` (`with_dual_width`, `static_assert`), `src/model/likelihood.cpp` e `src/model/parameter_map.cpp` (instâncias `Dual<40>`)
- Test: `tests/unit/model/layout_test.cpp`

**Interfaces:**
- Produces:
  ```cpp
  namespace ve {
  enum class Model : int32_t { SGarch = 0, EGarch = 1, GjrGarch = 2, ApArch = 3, IGarch = 4, CsGarch = 5,
      FGarchGarch = 6, FGarchTGarch = 7, FGarchAvGarch = 8, FGarchNGarch = 9, FGarchNaGarch = 10,
      FGarchApArch = 11, FGarchGjrGarch = 12, FGarchAllGarch = 13 };
  constexpr std::size_t kMaxParameterCount = 6 * kMaxOrder + 4;
  }
  namespace ve::model {
  enum class ParameterKind { Location, Real, Positive, Weight };
  struct Layout {
      Model model; bool include_mean; int ar; int ma; int arch; int garch;
      int gamma; int eta1; int eta2; bool free_power; double fixed_power; Dist dist;
      // mu, ar_begin, ma_begin, omega, alpha_begin, beta_begin como no 2a, e:
      std::size_t gamma_begin() const; std::size_t eta1_begin() const; std::size_t eta2_begin() const;
      std::size_t power() const;            // só com free_power
      bool has_shape() const; std::size_t shape() const; std::size_t count() const;
      bool has_derived() const;             // iGARCH
      std::size_t derived() const;          // beta_begin() + garch - 1
      std::size_t estimated_count() const;  // count() - has_derived()
      ParameterKind kind(std::size_t index) const;
  };
  Layout layout_of(const Spec& spec);                       // tabela de blocos das Global Constraints
  std::vector<std::string> parameter_names(const Layout&);  // "delta" no apARCH, "lambda" no fGARCH
  }
  ```
- `fixed_power` vale 2.0 fora da família de potência e para GARCH/NAGARCH/GJRGARCH, e 1.0 para TGARCH/AVGARCH. `with_dual_width` despacha k ≤ 8, ≤ 16, ≤ 32, senão 40. O `static_assert` de `fit.cpp` passa a ser `kMaxParameterCount <= kWidestDual`.

- [ ] **Step 1: Escrever os testes**

```cpp
// (1,1), norm, com média: parametrizado sobre os 14 modelos
TEST_P(LayoutNames, FollowRugarchOrder) { /* tabela:
  SGarch, IGarch, FGarchGarch: mu omega alpha1 beta1
  EGarch, GjrGarch: + gamma1          ApArch: + gamma1 delta
  CsGarch: + eta11 eta21              FGarchTGarch, FGarchGjrGarch: + eta11
  FGarchAvGarch: + eta11 eta21        FGarchNGarch: + lambda
  FGarchNaGarch: + eta21              FGarchApArch: + eta11 lambda
  FGarchAllGarch: + eta11 eta21 lambda */ }
TEST(Layout, HigherOrdersRepeatPerLag) {
  // ALLGARCH p=2,q=1, ARMA(1,0), std → mu ar1 omega alpha1 alpha2 beta1 eta11 eta12 eta21 eta22 lambda shape
  // CsGarch p=2,q=2 → ... beta2 eta11 eta21 (um de cada)
}
TEST(Layout, IGarchDerivesLastBeta) { /* p=1,q=2: derived()==beta_begin()+1; estimated_count()==count()-1; outros modelos: !has_derived() */ }
TEST(Layout, WidestFitsWidestDual) { /* ALLGARCH, ARMA(6,6), (6,6), std: count()==kMaxParameterCount==40 */ }
TEST(Layout, ParameterKinds) {
  // mu Location; ar/ma Real; sGARCH omega Positive; eGARCH omega, alpha1, gamma1 Real; gjr gamma1 Real;
  // alpha/beta Weight (exceto eGARCH alpha); cs eta11 Positive, eta21 Weight; ALLGARCH eta11, eta21 Real, lambda Positive;
  // apARCH gamma1 Real, delta Positive; shape Positive
}
TEST(Layout, FixedPowers) { /* TGARCH, AVGARCH fixed_power 1; GARCH, NAGARCH, GJRGARCH 2; free_power em NGARCH, APARCH, ALLGARCH e apARCH */ }
```

- [ ] **Step 2: Rodar e ver falhar** — `cmake --build --preset gcc-debug && ctest --preset gcc-debug -R Layout` → não compila.
- [ ] **Step 3: Implementar.** Os offsets seguem a ordem de blocos da regra "Nomes e ordem". `layout_of` é uma tabela por modelo, não uma cadeia de `if`.
- [ ] **Step 4: Rodar e ver passar** — suíte inteira PASS (inclui `style` e os testes do 2a sem mudança).
- [ ] **Step 5: Commit** — `git commit -m "feat: add the GARCH variant models, their parameter blocks and Dual<40>"`

---

### Task 2: Momentos dos choques

**Files:**
- Create: `src/dist/moments.hpp`, `src/dist/moments.cpp` (nós de Gauss–Legendre)
- Modify: `CMakeLists.txt`
- Test: `tests/unit/dist/moments_test.cpp`

**Interfaces:**
- Consumes: `ad::Dual`, `ad::lgamma`, `ad::pow`; o λ_g da GED de `dist/densities.hpp`.
- Produces:
  ```cpp
  namespace ve::dist {
  template <class T> T abs_moment(Dist dist, const T& shape, const T& power);              // E|z|^power, +inf se std com nu <= power
  template <class T> T asymmetric_moment(Dist dist, const T& shape, const T& eta1, const T& power);      // eta2 ≡ 0
  template <class T> T shifted_moment(Dist dist, const T& shape, const T& eta1, const T& eta2, const T& power); // quadratura
  }
  ```
- Fórmulas: regra "Momentos dos choques". Os 64 nós e pesos de Gauss–Legendre em [0, 1] são calculados uma vez (Newton sobre o polinômio de Legendre, num `static const` local de função, que tem inicialização thread-safe). São constantes, não estado mutável. O integrando usa `dist::log_density` e é avaliado em `T`, de modo que `shifted_moment<Dual<N>>` dá as derivadas em shape, η₁, η₂ e λ.

- [ ] **Step 1: Escrever os testes**

```cpp
TEST(Moments, AbsMomentClosedForms) {
  // norm: E|z| = sqrt(2/pi) (1e-15), E z^2 = 1 (1e-14), E z^4 = 3 (1e-13)
  // std nu=6: E z^2 = 1, E z^4 = 3(nu-2)/(nu-4) = 6 (1e-12); ged nu=2 == norm para power ∈ {0.5, 1, 1.7} (1e-14)
  // ged nu ∈ {0.8, 1.5}: E z^2 = 1 (1e-12)
}
TEST(Moments, StdMomentInfiniteAtOrAbovePower) { /* std nu=3, power=3 → +inf; nu=2.5, power=2.7 → +inf; nu=3.1, power=3 finito */ }
TEST(Moments, AsymmetricMatchesQuadrature) {
  // referência: trapézio em [-60, 60] com 2e5 pontos (o de densities_test) de (|z| - eta1 z)^power f(z)
  // (norm, std nu=6, ged nu=1.5) × eta1 ∈ {-0.5, 0, 0.7} × power ∈ {0.8, 1.5, 2}: relativo 1e-6
}
TEST(Moments, ShiftedReducesToKnownForms) {
  // Review Focus 5:
  // shifted(eta2=0) == asymmetric, para norm, ged nu=1.5, std nu=8, eta1 ∈ {-0.4, 0.3}, power ∈ {1, 1.5, 2} (rel 1e-10)
  // NAGARCH: shifted(eta1=0, eta2=c, power=2) == 1 + c^2, c ∈ {-0.5, 0.3, 1.2}, em norm, std nu=4.5, ged nu=1 (rel 1e-8)
  // norm, power=1, eta1=0: E|z - c| == c(2Φ(c) - 1) + 2φ(c), c ∈ {-0.7, 0.2} (rel 1e-10)
}
TEST(Moments, DerivativesMatchFiniteDifferences) {
  // Dual<8> semeado em (shape, eta1, eta2, power) para as três funções, dists std nu=6 e ged nu=1.5:
  // |grad_i - CD_i| <= 1e-6·max(1, |grad_i|), h_i = 1e-6·max(1, |x_i|)
}
```

- [ ] **Step 2: Rodar e ver falhar** — `ctest --preset gcc-debug -R Moments` → não compila.
- [ ] **Step 3: Implementar** os três templates e os nós. Se 64 nós não fecharem as tolerâncias de `ShiftedReducesToKnownForms`, aumente os nós e registre o número em `docs/performance/benchmarks.md`.
- [ ] **Step 4: Rodar e ver passar** — suíte inteira PASS.
- [ ] **Step 5: Commit** — `git commit -m "feat: add shock moments for the asymmetric and power GARCH constraints"`

---

### Task 3: Recursões eGARCH, gjrGARCH e iGARCH no `loglik`

**Files:**
- Create: `src/garch/presample.hpp` (`Presample` e `lagged` saem de `sgarch.hpp`), `src/garch/egarch.hpp`, `src/garch/gjrgarch.hpp`
- Modify: `src/garch/sgarch.hpp`, `src/model/likelihood.cpp`
- Test: `tests/unit/garch/egarch_test.cpp`, `tests/unit/garch/gjrgarch_test.cpp`, `tests/unit/model/likelihood_test.cpp` (estendido)

**Interfaces:**
- Consumes: `Layout` (Task 1), `abs_moment` (Task 2), `garch::Presample<T>{value, held}` do 2a.
- Produces:
  ```cpp
  namespace ve::garch {
  template <class T> struct EGarchParameters { T omega; std::span<const T> alpha; std::span<const T> beta;
                                               std::span<const T> gamma; T abs_moment; };
  template <class T> void egarch_variance(std::span<const T> residuals, const EGarchParameters<T>& p,
                                          const Presample<T>& presample, std::span<T> sigma2);
  template <class T> struct GjrGarchParameters { T omega; std::span<const T> alpha; std::span<const T> beta;
                                                 std::span<const T> gamma; };
  template <class T> void gjrgarch_variance(std::span<const T> residuals, const GjrGarchParameters<T>& p,
                                            const Presample<T>& presample, std::span<T> sigma2);
  }
  ```
- `loglik` escolhe a recursão pelo `layout.model`, um `switch` fora do laço temporal. O sGARCH segue pelo mesmo caminho do 2a (mesma ordem de operações, mesmos números). No iGARCH, o β com o último termo recalculado vive num `std::array<T, kMaxOrder>` local, sem alocação.
- Pré-amostral: regra "Pré-amostral". `Presample.value` é sempre σ², e cada recursão deriva dele o próprio estado.

- [ ] **Step 1: Escrever os testes**

```cpp
// r = {0.1, -0.2, 0.3}, sem média, (1,1), sample init: m̄ = 0.14/3, held = 1, σ²₀ = m̄
TEST(EGarch, HandCase) {
  // θ: ω=-0.1, α=-0.05, β=0.9, γ=0.1, norm (E|z| = sqrt(2/pi)); z₀ = 0.1/sqrt(m̄)
  // log σ²₁ = ω + α z₀ + γ(|z₀| - E|z|) + β log m̄; z₁ = -0.2/σ₁; log σ²₂ = ω + α z₁ + γ(|z₁| - E|z|) + β log σ²₁ (1e-14)
}
TEST(EGarch, BackcastPresample) { /* held=0, b dado: log σ²₀ = ω + β log b (1e-14) */ }
TEST(GjrGarch, HandCase) {
  // θ: ω=0.1, α=0.05, β=0.8, γ=0.1: σ²₁ = 0.1 + 0.05·0.01 + 0.8 m̄; σ²₂ = 0.1 + (0.05+0.1)·0.04 + 0.8 σ²₁ (1e-14)
}
TEST(GjrGarch, BackcastPresample) { /* p=2,q=1: σ²₀ = ω + (Σα + ½Σγ + Σβ) b (1e-14) */ }
TEST(Likelihood, IGarchIgnoresStoredLastBeta) {
  // iGARCH(1,2) com θ[derived] = 0.123 dá o mesmo loglik e sigma2 que o sGARCH(1,2) com β₂ = 1 - α₁ - β₁ (rel 1e-14);
  // loglik<Dual<8>> semeado em θ natural: grad[derived] == 0.0 exato
}
TEST(Likelihood, EGarchOutlierGivesNonFinite) {
  // Review Focus 2: série de 500 obs (mt19937_64(42)) com r[100] = 1e3, eGARCH com γ=0.2 → !isfinite(loglik), sem exceção
}
// Estender os testes do 2a: GradientMatchesFiniteDifferences, DoubleAndDualAgreeOnValue e
// TermsSumToTotalAndScoresSumToGradient passam a percorrer SGarch, EGarch, GjrGarch, IGarch
// × ordens {(1,1),(2,2)} × ARMA {(0,0),(1,1)} × {norm,std,ged} × {sample,backcast}, com θ = to_natural(u),
// u = to_unconstrained(start_values) + 0.1·N(0,1) (mt19937_64(13)); até a Task 5, θ à mão por modelo.
```

- [ ] **Step 2: Rodar e ver falhar** — `ctest --preset gcc-debug -R "EGarch|GjrGarch|Likelihood"` → não compila.
- [ ] **Step 3: Implementar** as duas recursões, `presample.hpp` e o despacho.
- [ ] **Step 4: Rodar e ver passar** — suíte C++ inteira PASS; `Likelihood.MatchesPlanOneSimulatedValues` sem mudar constantes.
- [ ] **Step 5: Commit** — `git commit -m "feat: add eGARCH, gjrGARCH and iGARCH variance recursions"`

---

### Task 4: Recursões apARCH/fGARCH e csGARCH no `loglik`

**Files:**
- Create: `src/garch/power_garch.hpp`, `src/garch/csgarch.hpp`
- Modify: `src/model/likelihood.cpp`
- Test: `tests/unit/garch/power_garch_test.cpp`, `tests/unit/garch/csgarch_test.cpp`, `tests/unit/model/likelihood_test.cpp`

**Interfaces:**
- Consumes: Tasks 1–3; `asymmetric_moment`, `shifted_moment` (Task 2).
- Produces:
  ```cpp
  namespace ve::garch {
  // eta2 vazio significa eta2 ≡ 0; moments[i] = κ_i, usado no pré-amostral do backcast.
  template <class T> struct PowerGarchParameters { T omega; std::span<const T> alpha; std::span<const T> beta;
      std::span<const T> eta1; std::span<const T> eta2; T power; std::span<const T> moments; };
  template <class T> void power_garch_variance(std::span<const T> residuals, const PowerGarchParameters<T>& p,
                                               const Presample<T>& presample, std::span<T> sigma2);
  template <class T> struct CsGarchParameters { T omega; std::span<const T> alpha; std::span<const T> beta; T rho; T phi; };
  // permanent: área de trabalho do q_t, do tamanho de sigma2, alocada pelo loglik junto com as outras.
  template <class T> void csgarch_variance(std::span<const T> squared_residuals, const CsGarchParameters<T>& p,
                                           const Presample<T>& presample, std::span<T> sigma2, std::span<T> permanent);
  }
  ```
- O apARCH monta `PowerGarchParameters` com eta1 = bloco `gamma`, power = `delta` e eta2 vazio. Um submodelo de potência fixa usa `T{layout.fixed_power}`. O `loglik` calcula κᵢ (assimétrico ou deslocado, conforme o layout tiver o bloco `eta2`) e devolve não finito se algum κᵢ não for finito.

- [ ] **Step 1: Escrever os testes**

```cpp
// mesmo r, m̄ e held da Task 3
TEST(PowerGarch, ApArchHandCase) {
  // θ: ω=0.1, α=0.1, β=0.8, γ=0.3, δ=1.5: σ₁^δ = 0.1 + 0.1(0.1 - 0.3·0.1)^1.5 + 0.8 m̄^0.75;
  // σ₂^δ = 0.1 + 0.1(0.2 + 0.3·0.2)^1.5 + 0.8 σ₁^δ; sigma2 = (σ^δ)^(2/δ) (1e-14)
}
TEST(PowerGarch, AllGarchHandCase) {
  // θ: ω=0.1, α=0.1, β=0.8, η₁=0.3, η₂=0.2, λ=1.5: σ₁^λ = 0.1 + 0.1 σ₀^λ (|z₀-0.2| - 0.3(z₀-0.2))^1.5 + 0.8 σ₀^λ (1e-14)
}
TEST(PowerGarch, BackcastPresample) { /* σ₀^λ = ω + (Σ αᵢκᵢ + Σβ) b^(λ/2) (1e-14) */ }
TEST(CsGarch, HandCase) {
  // θ: ω=0.01, α=0.05, β=0.8, ρ=0.95, φ=0.03; q₀ = σ²₀ = m̄; q₁ = ω + ρ m̄ + φ(0.01 - m̄);
  // σ²₁ = q₁ + α(0.01 - m̄) + β(m̄ - m̄); q₂ e σ²₂ pela mesma regra (1e-14)
}
TEST(CsGarch, BackcastPresample) { /* q₀ = ω + ρ b; σ²₀ = q₀ (1e-14) */ }
TEST(Likelihood, SubmodelsMatchTheirModels) {
  // T=500 (mt19937_64(42)) × {norm,std,ged} × {sample,backcast}, loglik e cada sigma2 em rel 1e-12:
  // FGarchGarch(α,β) == SGarch(α,β); FGarchApArch(η₁=γ, λ=δ) == ApArch(γ,δ);
  // FGarchGjrGarch(α, η₁) == GjrGarch(α(1-η₁)², γ = 4αη₁)
}
TEST(Likelihood, PowerStdNeedsShapeAbovePower) {
  // Review Focus 1: apARCH δ=2.5, std ν=2.4 → !isfinite; ν=3 → finito; ALLGARCH λ=2.6, η₂=0.2, std ν=2.5 → !isfinite
}
// GradientMatchesFiniteDifferences, DoubleAndDualAgreeOnValue e TermsSumToTotal... passam a cobrir os 14 modelos.
```

- [ ] **Step 2: Rodar e ver falhar** — `ctest --preset gcc-debug -R "PowerGarch|CsGarch|Likelihood"` → FAIL.
- [ ] **Step 3: Implementar** as recursões e o despacho. Uso de `pow` na potência fixa: implemente primeiro com `pow` genérico. Especializar λ ∈ {1, 2} é decisão da Task 9, com profile.
- [ ] **Step 4: Rodar e ver passar** — suíte C++ inteira PASS.
- [ ] **Step 5: Commit** — `git commit -m "feat: add apARCH, fGARCH and csGARCH variance recursions"`

---

### Task 5: Reparametrização, validade, fronteira e valores iniciais de todos os modelos

**Files:**
- Modify: `src/model/parameter_map.hpp`, `src/model/parameter_map.cpp`, `src/model/start_values.cpp`, `src/fit/standard_errors.cpp` (a regra de fronteira sai daqui)
- Test: `tests/unit/model/parameter_map_test.cpp`, `tests/unit/model/start_values_test.cpp`

**Interfaces:**
- Consumes: Tasks 1, 2.
- Produces (mesmas assinaturas do 2a, agora para todos os modelos, mais):
  ```cpp
  namespace ve::model {
  bool on_boundary(const Layout& layout, std::span<const double> theta, std::size_t index);   // regra "Fronteira"
  }
  ```
- Fórmulas: regras "Persistência", "Reparametrização", "Valores iniciais" e "Fronteira". `is_valid` exige: tudo finito; ω > 0 (exceto eGARCH); pesos ≥ 0 (no gjr, α ≥ 0 e α + γ ≥ 0); P < 1 − 1e-8 (no iGARCH, α ≥ 0, β_{<q} ≥ 0 e Σα + Σβ_{<q} ≤ 1); |η₁|, |γ do apARCH| < 1; δ, λ > 0; κ finito; no csGARCH, P < ρ < 1 − 1e-8 e 0 ≤ φ ≤ Σβ; shape no domínio.

- [ ] **Step 1: Escrever os testes**

```cpp
TEST(ParameterMap, SGarchUnchanged) { /* os testes do 2a passam sem mudança (OneOneMatchesPlanOneTransform etc.) */ }
TEST(ParameterMap, RoundTripAllModels) { /* 14 modelos × {(1,1),(2,2)} × ARMA {(0,0),(1,1)} × {norm,std,ged}: θ interior → u → θ' (rel 1e-12) */ }
TEST(ParameterMap, AlwaysValid) { /* 2000 u por layout (mt19937_64(5), N(0,3)) → is_valid, em todos os modelos */ }
TEST(ParameterMap, DualMatchesDouble) { /* value_of(to_natural<Dual<40>>) == to_natural<double> bit a bit, todos os modelos */ }
TEST(ParameterMap, EGarchOmegaShiftsWithScale) { /* Standardization{0, s}: ω == u_ω + (1 - Σβ)·log(s²) (1e-15) */ }
TEST(ParameterMap, IGarchWeightsSumToOne) { /* |Σα + Σβ - 1| ≤ 1e-15; θ[derived] == 1 - Σ outros */ }
TEST(ParameterMap, PowerAlphaCarriesMoment) { /* apARCH e ALLGARCH: Σ αᵢκᵢ + Σβ == logistic(v)(1-1e-8) (1e-14) */ }
TEST(ParameterMap, CsGarchOrdering) { /* u aleatórios: P < ρ < 1 - 1e-8 e 0 < φ < Σβ */ }
TEST(ParameterMap, TinyWeightsInvertFinite) { /* gjr com α+γ = 0; cs com φ = 0; ALLGARCH com β = 0: u finito */ }
TEST(ParameterMap, BoundaryRules) {
  // gjr α₁ = 1e-9 marca alpha1; α₁+γ₁ = 1e-9 marca gamma1; ALLGARCH η₁ = 1 - 1e-9 marca eta11;
  // cs φ = 1e-9 marca eta21, ρ = P + 1e-9 marca eta11; iGARCH marca o βq derivado sempre; std ν = 1e9 marca shape
}
TEST(StartValues, FollowRule) { /* a tabela "Valores iniciais" para cada modelo (1e-15), e is_valid(start_values) para todo layout */ }
```

- [ ] **Step 2: Rodar e ver falhar** — `ctest --preset gcc-debug -R "ParameterMap|StartValues"` → FAIL.
- [ ] **Step 3: Implementar.** `standard_errors.cpp` passa a chamar `model::on_boundary`. Troque os θ à mão dos testes das Tasks 3 e 4 pelo esquema `to_natural(u)` descrito na Task 3.
- [ ] **Step 4: Rodar e ver passar** — suíte inteira PASS.
- [ ] **Step 5: Commit** — `git commit -m "feat: map, validate and start every GARCH variant in the unconstrained space"`

---

### Task 6: Driver de ajuste, erros-padrão e critérios para todos os modelos

**Files:**
- Modify: `src/fit/fit.cpp` (`has_insufficient_data` com `estimated_count()`), `src/fit/standard_errors.cpp` (`initial_step` por `Layout::kind`), `src/fit/criteria.cpp` (k = `layout_of(result.spec).estimated_count()`)
- Modify: `tests/unit/simulated_garch.hpp` (simuladores dos seis processos da tabela "Séries simuladas", implementação independente da biblioteca, com o Box–Muller portável)
- Test: `tests/unit/fit/fit_test.cpp`, `tests/unit/fit/standard_errors_test.cpp`

**Interfaces:**
- Consumes: Tasks 1–5. `fit_one`, `standard_errors` e `FitResult` mantêm as assinaturas do 2a.

- [ ] **Step 1: Escrever os testes**

```cpp
// T = 5000, um processo da tabela "Séries simuladas" por teste, seeds 31..36
TEST_P(FitModels, BeatsTrueParameters) {
  // os seis processos ajustados com o próprio modelo e dist: status Converged; ℓ(θ̂) ≥ ℓ(θ verdadeiro) - 1e-6·|ℓ|
}
TEST(FitModels, RecoversIdentifiedParameters) {
  // eGARCH, gjr, iGARCH, csGARCH, apARCH: pesos ±0.03; α, γ, η ±0.06; δ ±0.3; ρ ±0.02; φ ±0.03; ν std ±1.5, ged ±0.15
}
TEST(FitModels, GarchSubmodelFitsLikeSGarch) { /* FGarchGarch e SGarch na mesma série: |Δℓ| ≤ 1e-6·|ℓ| */ }
TEST(FitModels, ScaleInvarianceAllModels) { /* o teste de escala do 2a (r e 100·r) para os 14 modelos em (1,1)-norm, mesmas tolerâncias */ }
TEST(FitModels, DeterministicAcrossBuckets) { /* fit_one duas vezes bit-idêntico, agora também k=40 (ALLGARCH, ARMA(6,6), (6,6), std, T=600, max_iter=30) */ }
TEST(FitModels, IGarchDerivedParameter) {
  // Review Focus 3: estimate[derived] == 1 - Σ outros (1e-15); se_hessian e se_robust NaN só no derivado, os outros finitos e > 0;
  // aic(Total) == -2ℓ + 2(k - 1), bic idem com log n
}
TEST(FitModels, PowerStdLowShapeEndsFinite) {
  // Review Focus 1: apARCH-std em série simulada com std ν = 2.5: status ∈ {Converged, MaxIter, Failed}, sem exceção;
  // se Converged, ν̂ > δ̂ e estimate e loglik finitos
}
TEST(FitModels, EGarchOutlierEndsFinite) { /* Review Focus 2: série eGARCH com r[2500] = 1e3·sd: status ∈ {...}, sem exceção, finito se Converged */ }
TEST(FitModels, ShortSeriesUsesEstimatedCount) { /* iGARCH(1,1) sem média: k estimado 2, 4 obs → não é INSUFFICIENT_DATA; 3 obs → é */ }
TEST(StandardErrors, ModelBoundaries) { /* cs com φ = 0 forçado → SE[eta21] NaN, outros finitos; ALLGARCH com η₁ = 1 - 1e-9 → SE[eta11] NaN */ }
```

- [ ] **Step 2: Rodar e ver falhar** — `ctest --preset gcc-debug -R "FitModels|StandardErrors"` → FAIL.
- [ ] **Step 3: Implementar.** Se uma tolerância de recuperação não fechar, registre o valor observado no relatório da task e não afrouxe sem esse registro.
- [ ] **Step 4: Rodar e ver passar** — suíte C++ inteira PASS, e `-Wnrvo` sem aviso.
- [ ] **Step 5: Commit** — `git commit -m "feat: fit every GARCH variant with model-aware steps, boundaries and criteria"`

---

### Task 7: Interface C e Python

**Files:**
- Modify: `include/volengine/volengine.h`, `src/capi/spec_api.cpp`, `python/src/volengine/_spec.py`, `python/src/volengine/__init__.py` (se exportar tabelas)
- Test: `tests/unit/capi/spec_test.cpp`, `tests/unit/capi/batch_test.cpp`, `python/tests/test_fit.py`, `python/tests/typing_cases.py` (se a assinatura de `Spec` mudar o que o mypy vê)

**Interfaces:**
- Consumes: `ve::Model` (Task 1), `fit_one` (Task 6).
- Produces (C, só acréscimos): os `#define VE_MODEL_*` da tabela "Modelos", e no header, junto de cada valor, a lista dos parâmetros e o que é derivado. `ve_spec_create` aceita 0..13; qualquer outro valor é `VE_INVALID_ARGUMENT` com "model" na mensagem. `static_assert` de cada `VE_MODEL_*` contra `ve::Model`.
- Python: `MODELS = {"sgarch": 0, "egarch": 1, "gjrgarch": 2, "aparch": 3, "igarch": 4, "csgarch": 5}` e `SUBMODELS = {"garch": 6, "tgarch": 7, "avgarch": 8, "ngarch": 9, "nagarch": 10, "aparch": 11, "gjrgarch": 12, "allgarch": 13}`. `Spec(model="fgarch", *, submodel=None, ...)`: `submodel` é obrigatório se e só se `model == "fgarch"`. Os dois casos errados (sem submodelo no fgarch, com submodelo em outro modelo) geram `ValueError` com "submodel" na mensagem. A docstring lista os modelos e diz que `beta_q` do iGARCH é derivado.

- [ ] **Step 1: Escrever os testes**

```cpp
TEST(CapiSpec, AllModels) { /* create(0..13) OK; create(14), create(-1) → INVALID_ARGUMENT, "model" na mensagem, *out == nullptr */ }
TEST(CapiFit, MixedBucketsInOneBatch) {
  // Review Focus 4 (estende o do 2a): specs {sGARCH(1,1)-norm k=4, eGARCH(2,1)-std, ALLGARCH ARMA(6,6)-(6,6)-std k=40}
  // × 3 séries T=600, max_iter=30, threads 1 e 4: estimates/loglik/sigma bit-idênticos a ve::fit_one direto
}
TEST(CapiFit, VariantParamNames) { /* ALLGARCH(2,1)-std e csGARCH(1,1) via ve_fit_param_name, na ordem do Layout */ }
TEST(CapiFit, IGarchCriteriaCountEstimated) { /* ve_aic(TOTAL) == -2ℓ + 2·(param_count - 1) */ }
```
```python
@pytest.mark.parametrize(("model", "submodel", "extra"), [("egarch", None, ("gamma1",)), ("csgarch", None, ("eta11", "eta21")),
                                                         ("fgarch", "allgarch", ("eta11", "eta21", "lambda"))])
def test_variant_names(ctx, series, model, submodel, extra): ...   # names == ("mu","omega","alpha1","beta1", *extra)
def test_fgarch_requires_submodel(): ...                             # Spec(model="fgarch") → ValueError "submodel"
def test_submodel_only_for_fgarch(): ...                             # Spec(model="egarch", submodel="tgarch") → ValueError
def test_igarch_derived_beta_has_nan_se(ctx, series): ...            # f.se()["beta1"] NaN; os outros finitos
```

- [ ] **Step 2: Rodar e ver falhar** — `ctest --preset gcc-debug -R "CapiSpec|CapiFit"` e pytest → FAIL.
- [ ] **Step 3: Implementar.**
- [ ] **Step 4: Rodar e ver passar** — suítes C++ e Python inteiras PASS; o header continua C11-limpo; `mypy --strict` dos casos de tipo PASS.
- [ ] **Step 5: Commit** — `git commit -m "feat: expose the GARCH variants through the C API and Python"`

---

### Task 8: Oráculos para os modelos novos

**Files:**
- Modify: `tests/reference/generate_data.py`, `tests/reference/cases.json`, `tests/reference/oracle_rugarch.R`, `tests/reference/oracle_arch.py`, `tests/reference/README.md`, `python/tests/test_oracles.py`, `docs/guide/validation.md`
- Create (gerados, commitados): `tests/reference/data/grid_2b.csv`, fixtures atualizadas em `tests/reference/fixtures/`

**Interfaces:**
- Consumes: pacote Python (Task 7).
- Pré-requisitos desta máquina: R 4.6.2 com `rugarch` 1.5.6 (o usuário está instalando) e o venv com `arch` 8.0.0. Se faltar algum, pare e peça ao usuário. Não troque as versões dos oráculos sem registrar em `validation.md`.
- `cases.json` ganha `"model"` (padrão `"sgarch"` nos casos existentes, que não mudam) e `"submodel"` (só fGARCH). Os scripts traduzem para os nomes do oráculo.
- Dados: `grid_2b.csv` com as seis séries da tabela "Séries simuladas", T = 3000, de um só `numpy.random.default_rng(20261112)`, na ordem da tabela.
- Casos (fixados):
  - `rugarch`: cada série com o próprio modelo em (1,1) × {norm, std, ged}; na série `fgarch_allgarch11_norm`, os outros sete submodelos em (1,1)-norm; eGARCH(2,1)-norm em `egarch11_std`; ARMA(1,1)-gjrGARCH(1,1)-std em `arma11_sgarch11_ged` (do 2a).
  - `arch` (backcast): GJR (`GARCH(p=1, o=1, q=1)`), EGARCH (`EGARCH(p=1, o=1, q=1)`) e APARCH (`APARCH(p=1, o=1, q=1)`), cada um em (1,1) × {normal, t, ged}, na série do modelo. Conversões no `oracle_arch.py`:
    - EGARCH: `alpha[1]` do `arch` (termo em |z|) → `gamma1`; `gamma[1]` (termo em z) → `alpha1`. O `arch` centra |z| em √(2/π) para qualquer distribuição, então ω_ve = ω_arch + α_arch(E|z| − √(2/π)), e o SE de `omega` só é comparado em norm.
    - GJR: `gamma[1]` → `gamma1`.
    - APARCH: `gamma[1]` → `gamma1`, `delta` → `delta`.
  - Regra de interior (o script para e o teste confere): pesos > 1e-4, P < 0.999, |η₁|, |γ do apARCH| < 0.999, ρ − P > 1e-4, φ > 1e-4. Caso que falhe é trocado, e o motivo vai para `validation.md` ("Case selection").
- Tolerâncias: as do 2a, por grandeza.
- Convenções a confirmar e documentar em `validation.md`:
  - o pré-amostral `sample` de cada família (log σ², σ^λ, q);
  - E|z| da própria distribuição no eGARCH;
  - δ ≡ λ e os nomes do fGARCH;
  - os nomes, o valor e o SE (NA) do βq derivado do iGARCH, e o k dos critérios do `rugarch`;
  - os limites de ρ e φ do csGARCH, se algum caso tocar a fronteira;
  - o backcast do APARCH no `arch`.

  Se o oráculo divergir, a correção vai no núcleo (Tasks 3–5), com os testes de gradiente mantidos exatos, e a mudança é registrada. Se o backcast do APARCH no `arch` não for reproduzível com a regra "Pré-amostral", os casos APARCH do `arch` saem e o motivo vai para `validation.md`.

- [ ] **Step 1: Gerar dados e fixtures** — `python/.venv/Scripts/python tests/reference/generate_data.py && Rscript tests/reference/oracle_rugarch.R && python/.venv/Scripts/python tests/reference/oracle_arch.py` → um resultado por caso e oráculo; os scripts param com erro em não-convergência ou fronteira.
- [ ] **Step 2: Estender o teste** — `test_matches_oracle` parametrizado sobre (oráculo, caso), com as asserções do 2a; `test_fixture_estimates_are_interior` com a regra de interior acima.
- [ ] **Step 3: Rodar** — `VOLENGINE_LIB=<abs> python/.venv/Scripts/python -m pytest python/tests/test_oracles.py -v` → PASS; depois de cada correção no núcleo, as suítes C++ e Python inteiras.
- [ ] **Step 4: Commit** — `git commit -m "test: validate the GARCH variants against rugarch and arch"`

---

### Task 9: Benchmarks, profile, assembly e documentação

**Files:**
- Modify: `tests/benchmarks/fit_benchmarks.cpp`, `docs/performance/benchmarks.md`, `README.md`, `CHANGELOG.md`, `ARCHITECTURE.md`

**Interfaces:**
- Consumes: interface C e `ve::fit_one`, como no 2a.
- Benchmarks novos (nomes fixados; os do 2a não mudam):
  - `BM_Loglik/<spec>/{double,dual}` e `BM_FitOne/<spec>`, T = 5000, para `egarch11_std`, `gjrgarch11_norm`, `aparch11_ged`, `igarch11_norm`, `csgarch11_std`, `fgarch_tgarch11_norm` (λ fixo = 1), `fgarch_allgarch11_norm` (λ livre, η₂ por quadratura).
  - `BM_TccGridAllModels/threads:hw`: o painel de `BM_TccGrid` (20 séries × 5000) × os 14 modelos × {p, q ∈ 1..3} × ARMA {(0,0), (1,1)} × {norm, std, ged} = 15120 ajustes numa chamada `ve_fit_batch`. Contadores: ajustes/s e contagem por status e por modelo. Falha alto (`SkipWithError`) com zero trabalho ou se algum modelo tiver mais de 5% dos ajustes fora de `CONVERGED`/`MAX_ITER`.
- Nesta máquina os benchmarks só são escritos e compilados. Para conferir que nenhum falha alto, rode uma vez com `--benchmark_min_time=0.01s --benchmark_filter=BM_Loglik` e não publique nada.

- [ ] **Step 1: Escrever os benchmarks** — `cmake --preset gcc-bench && cmake --build --preset gcc-bench` → compila sem aviso; a rodada curta acima termina sem `SkipWithError`.
- [ ] **Step 2: Documentação** — README (modelos suportados e exemplo `Spec(model="fgarch", submodel="allgarch")`), CHANGELOG (`[Unreleased]`), ARCHITECTURE (módulos novos em `garch/` e `dist/`, despacho por modelo, `Dual<40>`), e em `docs/performance/benchmarks.md` uma nota dizendo que os casos novos ainda não foram medidos.
- [ ] **Step 3: Verificar** — `ctest --preset gcc-release` e pytest inteiros PASS; `check_style.py .` sai 0; o CI compila os benchmarks.
- [ ] **Step 4: Commit** — `git commit -m "bench: add benchmarks for the GARCH variants and document plan 2b"`

---

## Pendente na máquina de benchmark

Fazer na máquina do `docs/performance/benchmarks.md` (AMD Ryzen 7 5700X), depois deste plano e antes do plano 3. Não é task do subagent-driven.

1. **Rodar** — `build/gcc-bench/volengine_benchmarks --benchmark_out=build/gcc-bench/bench.json --benchmark_out_format=json --benchmark_repetitions=5 --benchmark_report_aggregates_only=true`, e atualizar `benchmarks.md` com máquina, data e commit.
2. **Profile** — AMD uProf (CPU Profile) sobre `BM_TccGridAllModels` e `BM_FitOne/fgarch_allgarch11_norm`: as cinco funções mais quentes e a fração de tempo de cada uma.
3. **Assembly** — `objdump -d -C -M intel --no-show-raw-insn` no objeto de `src/model/likelihood.cpp` do `gcc-bench`, nas instâncias `double` e `Dual<8>` das recursões novas. Registrar: chamadas a `pow`/`exp`/`log` no laço temporal, alocações no laço, cópias de `Dual`, o ISA usado (o preset não passa `-march`, espera-se SSE2), e o que `-fopt-info-vec-missed` diz não vetorizar, com o motivo.
4. **Otimizar pelos dados.** A candidata conhecida é a potência fixa (λ ∈ {1, 2}) chamando `pow` por observação. Se o profile mostrar o custo, especializar (multiplicação para 2, nada para 1, escolha fora do laço) e medir antes e depois. Se não mostrar, registrar que não compensa. Já feito no notebook (`6e52a3c`, ver abaixo); falta medir antes e depois aqui.

### Auditoria no notebook (2026-11-11)

Feita no i7-1365U, que tem núcleos híbridos e sofre throttling. Os números abaixo são indicativos e nenhum entra em `benchmarks.md`. Método: processo fixado num P-core (CPU 2), binários A e B alternados em rodadas curtas, mediana da razão pareada B/A por rodada. Num teste A/A o ruído ficou em ±7%.

Decomposição do loglik `Dual<8>` do s11_norm (T = 5000), medida num driver descartável: `sum_of_terms` levava ~50% do tempo, por causa da chamada fora de linha de `log_density` com o desvio por distribuição a cada observação. A recursão levava ~25%, e o vetor de e² (alocação mais passada) ~15%.

| Commit | Mudança | Efeito indicativo (loglik `Dual`) |
|---|---|---|
| `1b7fd77` | distribuição escolhida uma vez por série; norm e std leem z² = e²/σ² | s11_norm, s21_std −13% |
| `566baa1` | e² calculado dentro da recursão sGARCH/iGARCH, sem vetor | mais −12%; arma22_s33_std −30% |
| `053b854` | `pow(Dual, Dual)` com um `std::pow` só (`x^(e-1) = x^e / x`); laço sem desvio com as inclinações finitas | ged −14%, apARCH −20%, ALLGARCH −17% |
| `6e52a3c` | potência fixa 1 ou 2 sem `pow` | TGARCH −44% (`double` −67%) |

`BM_FitOne`, comparado com `01a0325`: TGARCH −48%, s11_norm −38%, iGARCH −30%, s21_std −28%, ALLGARCH −23%, apARCH e ged −20%. eGARCH, gjrGARCH e csGARCH ficaram dentro do ruído.

Repartição de um ajuste, medida com contadores num build descartável:
- o otimizador fica com 57–78% do tempo, em 17–65 avaliações por ajuste;
- os erros-padrão ficam com 22–43%, quase tudo na Hessiana numérica (2k gradientes).

Cada gradiente da Hessiana alocava seus buffers. `b006a58` passou a reaproveitá-los, e o `BM_FitOne` caiu mais 5 a 12%.

FTZ/DAZ não mudou o tempo de nenhum modelo, então números subnormais estão descartados como causa.

`arma22_s33_std` para em `max_iter` (500 iterações), porque o ARMA(2,2) sobre uma série sem ARMA não é identificável. No `BM_TccGrid` (1080 ajustes) são 8 casos assim, que somam ~12% do tempo do grid. O problema é de convergência, não de custo por avaliação.

Medidas e descartadas:
- divisão do `Dual` por recíproco: nenhum efeito;
- `multiply_add` fundido: a recursão escrita à mão saiu com o mesmo tempo;
- `-march=x86-64-v3`: só −5 a −10%, perto do ruído.

A fazer aqui, nesta máquina:
5. **Confirmar** os efeitos acima com `01a0325` contra a HEAD, nos mesmos benchmarks.
6. **ISA.** Medir `-march=x86-64-v3` contra o build genérico. Se compensar, decidir como entregar: preset só para benchmark, ou `target_clones` nas instâncias de `loglik` para os wheels.
7. **Candidatas restantes**, só se o profile mostrar o custo:
   - csGARCH ainda aloca o vetor de e² e o de q;
   - ALLGARCH calcula σ e σ² com dois `pow` por lag (dá para tirar σ² de σ·σ);
   - quadratura de κ com η₂ livre (320 avaliações por lag a cada chamada);
   - Hessiana numérica dos erros-padrão: são 2k gradientes; diferenças progressivas fariam com k, mas perdem precisão (só com o aval dos oráculos);
   - ajustes que param em `max_iter` (ver acima): um critério de parada mais cedo em cristas planas, se o plano 5 não cuidar disso.
