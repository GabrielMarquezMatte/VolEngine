# VolEngine — Plano 4b: Copula-DCC

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Estimar a Copula-DCC como o `cgarchfit` do `rmgarch` 1.4.3, sobre N ajustes univariados escolhidos pelo
usuário:
- cópulas mvnorm e mvt;
- correlação estática (Kendall ou ML), DCC ou aDCC;
- marginais paramétricas (CDF de cada ajuste) ou empíricas (ranks).

A Copula-DCC entra como opção do `DccSpec` do 4a. Sai no mesmo lote e no mesmo `DccFit`, com ℓ conjunta, o k do
DCC, erros-padrão em dois estágios, Rₜ e Hₜ.

**Architecture:**
- **Subpasta nova `src/dcc/copula/`:** `src/dcc/` tem 7 módulos, e a cópula traz 4.
- **Panel da cópula:** a cópula transforma cada zᵢₜ em uᵢₜ (CDF ou rank) e depois em escores εᵢₜ (Φ⁻¹, ou o quantil
  da t padronizada com forma ν). Os escores formam um `StandardizedPanel` cujo `z` é ε e cujo `sigma` é o dos
  univariados.
- **Reuso do 4a:** sobre esse panel, a parte dinâmica é exatamente o `fit_dcc` do 4a. Na mvnorm é um único ajuste;
  na mvt, ν entra por perfil: uma busca de Brent em (4, 50) envolve ajustes internos com `mshape` fixo, porque ε e
  Q̄ dependem de ν.
- **Estática:** um módulo próprio, com R constante: Kendall, ou ML com gradiente analítico. Com N = 21 a ML tem 210
  correlações, mais que as 59 do `Dual`.
- **Erros-padrão:** o `standard_errors.cpp` do 4a ganha um gancho que avalia a ℓ da cópula no θ perturbado.

**Tech Stack:** C++23, CMake + presets, GoogleTest v1.17.0, Google Benchmark v1.9.1, Boost.Math 1.87 (standalone,
só em `.cpp`: quantis `normal`/`students_t`, `brent_find_minima`), Python ≥ 3.11 + numpy + pytest, R 4.6.2 +
`rugarch` 1.5.6 + `rmgarch` 1.4.3.

**Spec:** `docs/superpowers/specs/2026-11-13-volengine-04-dcc-design.md`, seções 3 "4b", 8a e **10**, sobre a
spec-mãe `docs/superpowers/specs/2026-11-10-volengine-design.md`. Leia também o plano 4a
(`docs/superpowers/plans/2026-11-13-volengine-04a-dcc.md`): as regras "Convenções", "Critérios" e "Erros-padrão em
dois estágios" de lá valem aqui sem repetir.

## Roteiro

- **1, 2a–2f, 3a–3c e 4a (feitos).**
- **4b (este):** Copula-DCC.
- **4c:** forecast e simulação do DCC e da Copula-DCC.

Depois: 5 performance; 6 wrapper C++, wheels, 1.0.

## Global Constraints

Valem todas as Global Constraints do plano 4a, o `STYLEGUIDE.md` e o `ARCHITECTURE.md`. As que mais pesam aqui:

- **Interface C:**
  - ≤ 5 parâmetros por função; configuração nova é função nova;
  - handles imutáveis; visões zero-copy válidas enquanto o handle existir;
  - nenhuma exceção atravessa a fronteira.
- **Nenhum número do 4a muda:** com `copula = None`, todo ajuste DCC, critério, erro-padrão e Rₜ fica bit a bit
  igual, e todos os testes existentes passam sem mudar constantes nem tolerâncias.
- **Bit-identidade** entre números de threads, também no lote misto (DCC + cópula).
- **Licença:** o `rmgarch` (GPL) é lido para entender convenções e conferido numericamente. Nenhum código dele é
  copiado ou traduzido; as convenções vão para as notas em matemática.
- **Estilo:**
  - nunca `else`;
  - funções ≤ 60 linhas, ≤ 5 parâmetros, aninhamento ≤ 3;
  - ≤ 8 módulos por pasta (`src/dcc/copula/` nasce com 4);
  - Boost só em `.cpp`.
- **Ambiente e commits:**
  - commits sem `Co-Authored-By`; LF;
  - **testes só em release** (`ctest --preset gcc-release`);
  - executáveis e pytest com `/c/msys64/ucrt64/bin` primeiro no PATH e `VOLENGINE_LIB` apontando para
    `build/gcc-release/libvolengine.dll`.

### Regras da cópula (convenção padrão; o `[Rmgarch]` de cada uma sai da Task 1)

Notação: m séries, T observações, zᵢₜ = eᵢₜ/σᵢₜ de cada `Fit` (o `panel.z` do 4a).

**Marginais (u):**
- **paramétrica:** uᵢₜ = Fᵢ(zᵢₜ), pela CDF `dist::cdf` da distribuição do ajuste i, com os parâmetros de forma
  estimados. É o mesmo caminho que o GOF do plano 3c usa para obter `ShapeParameters` de um ajuste;
- **empírica:** uᵢₜ = rankᵢₜ/(T + 1), com rank 1..T por série e empates desfeitos pela ordem de t (Genest–Ghoudi–
  Rivest 1995);
- u é limitado a [2⁻⁵³, 1 − 2⁻⁵³] antes do quantil, para que um zᵢₜ extremo não dê ε infinito.
- **[Rmgarch]** (notas, seção 2): empírica rank/T; o corte é u > 0,99999 → 0,99999 e u < 1,5·2⁻⁵² → 2⁻⁵², nas duas
  transformações.

**Escores (ε):**
- **mvnorm:** εᵢₜ = Φ⁻¹(uᵢₜ);
- **mvt:** εᵢₜ = t_ν⁻¹(uᵢₜ) · √((ν − 2)/ν), o quantil da t **padronizada**. Com essa escala, a densidade mvt
  padronizada do 4a (`DccDist::MvT`) serve sem mudança. A ℓ da cópula não depende da escala, porque os jacobianos se
  cancelam.

**ℓ da cópula e ℓ reportada:**

    cₜ = log f(εₜ; Rₜ[, ν]) − Σᵢ log f₁(εᵢₜ[; ν])        (f₁ a densidade univariada normal ou t padronizada)
    ℓ  = Σᵢ ℓᵢ(univariado) + Σₜ cₜ

- `log f(εₜ; Rₜ)` é o termo do `correlation_terms` do 4a, que não inclui Σ log σ;
- a soma univariada entra também com marginais empíricas, como o `likelihood(cgarchfit)`;
- o objetivo do otimizador é −Σₜ cₜ (o termo univariado é constante).

**Dinâmica (DCC, aDCC):**
- é o 4a sobre o panel de ε: Q̄ = cov(ε), N̄ e δ pela regra do 4a, a mesma pré-amostra, as mesmas restrições e os
  mesmos inícios;
- **[Rmgarch]** (notas, seção 4): N̄ centrada e pré-amostra zero também no que é reportado, ou seja, o
  `Stage::Objective` do 4a em tudo (ℓ, Rₜ, erros-padrão). Não há a pré-amostra de uns do `dccfit`;
- **mvnorm:** um `fit_dcc` sobre o panel de ε, construído uma vez por transformação;
- **mvt com forma livre:** ν̂ = argmax_ν perfil(ν), por `boost::math::tools::brent_find_minima` em [4, 50], com 30
  bits e no máximo 100 iterações. Cada perfil(ν) reconstrói o panel de ε(ν) e roda `fit_dcc` com `mshape` fixo em ν.
  Os inícios do ajuste interno são sempre os padrão, para que o perfil seja uma função determinística de ν;
- `ponytail:` o perfil custa cerca de 30 ajustes internos. Se o benchmark pesar, trocar por um gradiente em ν por
  diferença implícita do quantil;
- **mvt com `mshape` fixo:** um único ajuste interno;
- `start` em `mshape` com cópula mvt é erro de argumento, porque a busca de Brent não usa início.

**Estática (R constante, Rₜ = R para todo t).** O `static_method` vale para as duas cópulas (notas, seção 5), com
padrão `kendall`, como no `rmgarch`:
- **`kendall`:** R = sin(πτ̂/2), com τ̂ o tau-b de Kendall amostral de cada par de colunas de u. Os ranks são os
  mesmos para z, u e ε, então R não depende de ν. Na mvt, ν vem do Brent sobre perfil(ν) com R fixo;
  - **[Rmgarch]** na mvnorm: R = cor(ε), a correlação de Pearson dos escores (o `rmgarch` chama isso de Kendall);
  - se R não for definida positiva (o Cholesky falha), o status é `Failed`, com estimativa NaN e sem exceção;
  - `ponytail:` falhar em vez de projetar; a projeção de Higham fica para quando alguém pedir.
- **`ml`:**
  - R = LLᵀ, em que a linha i de L é vᵢ/‖vᵢ‖ com vᵢ = (x_{i1}, …, x_{i,i−1}, 1). É o `.Pconstruct` do `rmgarch`:
    x são as entradas livres de A, e R = cov2cor(AAᵀ). Os m(m − 1)/2 parâmetros livres x garantem diagonal 1 e R
    definida positiva. O início é x = 0, que dá R = I;
  - **os Cᵢⱼ reportados são esses x**, não as correlações, como no `rmgarch`. A ordem é a do `lower.tri` do R,
    por coluna: C1 = x₂₁, C2 = x₃₁, C3 = x₃₂, …. As correlações estão em `R`. O `rmgarch` limita x a [−0,99; 0,99];
    o VolEngine não, nas duas convenções (notas, seção 8);
  - o L-BFGS existente minimiza −Σ cₜ / T com gradiente analítico. Com wₜ = 1 (mvnorm) ou wₜ = (ν + m)/(ν − 2 + qₜ)
    (mvt), qₜ = εₜᵀR⁻¹εₜ e S_w = Σₜ wₜ εₜεₜᵀ:

        ∂ℓ/∂R = −(T/2) R⁻¹ + ½ R⁻¹ S_w R⁻¹

    seguido da regra da cadeia por L;
  - na mvt, ν vem do Brent com a ML de R interna em cada ν.
- **Tabela:**
  - mvnorm + Kendall: vazia;
  - mvnorm + ML: `C1..C{m(m−1)/2}`;
  - mvt + Kendall: só `mshape`;
  - mvt + ML: `C1..C{m(m−1)/2}` e `mshape`.

**Critérios:** o k do 4a (univariados estimados + parâmetros estimados + m(m − 1)/2), com a matriz de correlação
contada **uma vez**: os Cᵢⱼ da estática ML ficam fora da parcela "parâmetros estimados".

**Erros-padrão em dois estágios:**
- a regra do 4a, com a ℓ reportada da cópula como f(θ) das linhas da cópula;
- uma perturbação univariada refiltra a série, recalcula zᵢ, uᵢ (paramétrica: a CDF com a forma perturbada) e ε, e
  depois Q̄, N̄ e δ;
- **empírica:** os ranks ficam fixos nas perturbações univariadas, então o bloco cruzado sai 0 e os erros-padrão
  são os do segundo estágio sozinho. `ponytail:` sem a correção semiparamétrica de Chen–Fan (2006); `validation.md`
  registra a limitação, e a correção entra quando alguém pedir;
  - **[Rmgarch]:** os ranks são recalculados a cada perturbação, como no `rmgarch` (notas, seção 6);
- **Kendall:** R não é estimada por ℓ, então só `mshape` tem erro-padrão, com R fixa;
- os Cᵢⱼ da ML têm erro-padrão nos x reportados.

**Entrada e erros:** as do 4a, mais estas combinações de `spec_problem`, todas com mensagem que nomeia a
combinação:

| Combinação | Mensagem contém |
| --- | --- |
| `static` sem cópula | `"static"` e `"copula"` |
| cópula com `fdcc` | `"fdcc"` e `"copula"` |
| cópula com `mvlaplace` | `"mvlaplace"` e `"copula"` |
| `static_method` explícito fora de `static` | `"static_method"` |
| `start` em `mshape` com cópula mvt | `"mshape"` e `"start"` |
| `static` com ordem diferente de (1, 1) | `"order"` |

O método estático tem três valores: `Default` (o Python manda quando `static_method=None`), `Kendall` e `Ml`.
`Default` vira `Kendall` na estática e é ignorado em qualquer outra combinação.

**Dados dos testes** (Task 1): `tests/reference/data/copula3.csv`, com univariados sGARCH(1,1)-std nos coeficientes
das notas, seção 1. O dcc3 do 4a tem margens quase normais, que levam ν ao teto 50.

## Review Focus

1. **zᵢₜ extremo com marginal paramétrica:** um choque de 40σ numa margem normal dá CDF 1.0 exata. O ℓ precisa
   continuar finito pelo limite de u. → teste na Task 2.
2. **Combinações inválidas:** a tabela "Entrada e erros" recusada antes de qualquer ajuste, pelo C (`VE_INVALID_ARGUMENT`) e
   pelo Python (`VolEngineError`). → testes nas Tasks 2 e 6.
3. **Kendall não definida positiva:** um lote em que uma spec falha assim tem essa spec `failed`, com estimativa NaN
   e sem exceção, e as outras specs ficam iguais às ajustadas sozinhas. → teste na Task 4.
4. **Lote misto, de 1 a 16 threads:** specs DCC e cópula juntas dão o mesmo resultado bit a bit que cada spec
   ajustada sozinha, e as specs com `copula = None` reproduzem o 4a bit a bit. → teste na Task 3.
5. **Forma da mvt no teto:** com margens quase normais (o dcc3 do 4a), ν̂ → 50 (o rmgarch estima 49,99999). O Brent tem de voltar ν̂ ≤ 50 com
   status `converged`, ℓ finita e erro-padrão finito e positivo, ou NaN documentado. → testes nas Tasks 3 e 5.

---

## Mapa de arquivos

```
docs/superpowers/plans/2026-11-13-volengine-04b-rmgarch-copula-notes.md   convenções do cgarchfit (Task 1)
src/dcc/spec.hpp                        DccModel::Static, DccCopula, StaticMethod
src/dcc/parameter_map.*                 spec_problem: combinações; nomes C1..; correlation_coefficients
src/dcc/copula/margins.hpp, .cpp        uniforms (paramétrica, empírica), limite de u
src/dcc/copula/scores.hpp, .cpp         CopulaPanel: ε(u, ν), marginal_log_density
src/dcc/copula/static_fit.hpp, .cpp     Kendall, ML de R com gradiente analítico, ℓ com R fixa
src/dcc/copula/fit.hpp, .cpp            fit_copula: dinâmica (perfil em ν) e estática; ℓ reportada
src/dcc/fit.cpp                         fit_dcc_batch despacha cópula; dcc_criterion conta C uma vez
src/dcc/standard_errors.cpp             f(θ) da cópula nas perturbações
src/capi/dcc/dcc_spec_api.cpp, include/volengine/volengine.h   setters e constantes
python/src/volengine/_dcc.py            DccSpec(copula=, static_method=, model="static")
tests/unit/dcc/copula/*_test.cpp, tests/unit/capi/dcc_test.cpp, python/tests/dcc/test_copula.py
tests/reference/copula_rmgarch.R, fixtures/rmgarch_copula.json, tests/unit/dcc/oracles/copula_oracle_test.cpp,
python/tests/oracles/test_copula_oracles.py
```

---

### Task 1: Notas das convenções do `cgarchfit`

**Files:**
- Create: `docs/superpowers/plans/2026-11-13-volengine-04b-rmgarch-copula-notes.md`, `tests/reference/copula_probe.R`

**Interfaces:**
- Produces: as notas, no formato das notas do 4a, com números de referência a 1e-12 para as Tasks 2–5:
  - **dados:** `tests/reference/data/copula3.csv`, simulado pela sonda (Ruling da Task 1);
  - **univariados:** sGARCH(1,1)-std, ARMA(0,0), `multifit` com solnp e tol 1e-10. Os coeficientes vão nas notas,
    em 15 dígitos;
  - **casos:**
    - `static_mvnorm_kendall_param`, `static_mvnorm_ml_param`, `static_mvnorm_ml_emp`;
    - `static_mvt_kendall_param`, `static_mvt_kendall_emp`, `static_mvt_ml_param`;
    - `dcc_mvnorm_param`, `dcc_mvnorm_emp`, `dcc_mvt_param`, `dcc_mvt_emp`;
    - `adcc_mvnorm_param`, `adcc_mvt_param`;
    - `dcc_mvt_param_shape8`, com `mshape` fixo em 8.

  São 13 casos.

Responder em matemática, cada item conferido numericamente, a ~1e-12, com o objeto ajustado:
1. u paramétrico e empírico, incluindo o divisor do rank e o limite ou truncamento de u;
2. ε na mvt: a escala do quantil, padronizada ou não, e se Q̄ é recalculado a cada ν;
3. Q̄, N̄ e a pré-amostra: confirmar se são as do `dccfit` sobre ε;
4. a ℓ da cópula e o termo marginal subtraído; `likelihood()` = Σ ℓᵢ + Σ cₜ;
5. o estimador da mvnorm estática. Hoje R₁₂ = 0,4745106 não bate com `cor(qnorm(u))` nem com `cov2cor`. Testar a ML
   numérica e um u truncado;
6. a mvt estática Kendall (fórmula de τ, o que acontece quando R não é definida positiva) e ML (ordem dos `C1..Cn`,
   restrições, limites da forma);
7. os erros-padrão do `matcoef` nas linhas `[Joint]`: as mesmas peças A e S do 4a e o que muda com marginais
   empíricas.

Cada diferença em relação às "Regras da cópula" deste plano é classificada pela regra 8a da spec:
- bug ou desvio de paper: vira um ponto `[Rmgarch]` do `Convention::Rmgarch`, com o número de referência;
- convenção defensável: ajusta a regra deste plano, registrada nas notas.

O script da sonda fica em `tests/reference/` para regenerar as notas.

- [x] **Step 1: Escrever o script** e rodar: `Rscript tests/reference/copula_probe.R` termina com sucesso e imprime
  os números de cada item.
- [x] **Step 2: Escrever as notas**, com a tabela final "regra do plano / `[Rmgarch]` / evidência". Se um item
  mudar uma regra deste plano, editar a regra aqui mesmo, na seção "Regras da cópula".
- [x] **Step 3: Commit** — `git commit -m "docs: rmgarch Copula-DCC conventions for plan 4b"`

---

### Task 2: Spec, marginais e escores

**Files:**
- Create: `src/dcc/copula/margins.hpp`, `.cpp`, `src/dcc/copula/scores.hpp`, `.cpp`
- Modify: `src/dcc/spec.hpp`, `src/dcc/parameter_map.hpp`, `.cpp`, `CMakeLists.txt`
- Test: `tests/unit/dcc/copula/margins_test.cpp`, `tests/unit/dcc/copula/scores_test.cpp`,
  `tests/unit/dcc/parameter_map_test.cpp`

**Interfaces:**
- Consumes: `StandardizedPanel`, `refresh_moments` e `spec_problem` do 4a; `dist::cdf`.
- Produces:
  ```cpp
  namespace ve::dcc {
  enum class DccModel : int32_t { Dcc = 0, ADcc = 1, FDcc = 2, Static = 3 };
  enum class DccCopula : int32_t { None = 0, Parametric = 1, Empirical = 2 };
  enum class StaticMethod : int32_t { Default = 0, Kendall = 1, Ml = 2 };
  struct DccSpec { /* 4a fields */ DccCopula copula = DccCopula::None; StaticMethod static_method = StaticMethod::Default; };
  // m(m-1)/2 for a static ML copula (the C1.. rows), else 0.
  std::size_t correlation_coefficients(const DccSpec& spec, std::size_t m);
  }
  namespace ve::dcc::copula {
  constexpr double kUniformFloor = 0x1p-53;      // u in [2^-53, 1 - 2^-53]
  // u, T x m row-major, of the "Marginais" rule; the caller checked panel_problem.
  std::vector<double> uniforms(std::span<const FitResult* const> fits, const StandardizedPanel& panel, DccCopula how);
  // The panel of the scores: z = epsilon(u) (MvNorm, or MvT with shape), sigma from the univariate panel, moments
  // refreshed; and sum_t sum_i log f1(epsilon_it) of the "ℓ da cópula" rule.
  struct CopulaPanel { std::shared_ptr<const StandardizedPanel> panel; double marginal_log_density = 0.0; };
  CopulaPanel copula_panel(std::span<const double> u, const StandardizedPanel& univariate, DccDist dist, double shape);
  }
  ```
- `layout_of` e os nomes do 4a seguem valendo para `Dcc`/`ADcc` com cópula. `Static` tem layout próprio:
  - `C1..Cn` só com ML na mvt;
  - `mshape` com a mvt;
  - mvnorm estática: layout vazio.

- [x] **Step 1: Escrever os testes**

```cpp
TEST(CopulaMargins, ParametricIsTheFitCdf) {
  // fit_one sGARCH(1,1)-std simulado: u[t*m+i] == dist::cdf(std, shape do fit, z) (bit a bit), limitado
}
TEST(CopulaMargins, EmpiricalRanks) {
  // z de 2 séries fixas (5 linhas, um empate): u == rank/(T+1) com o empate desfeito pela ordem de t
}
TEST(CopulaMargins, ExtremeShockStaysFinite) {
  // Review Focus 1: margem norm com z = 40: u == 1 - 2^-53; copula_panel mvnorm e mvt (nu 5): epsilon finito
}
TEST(CopulaScores, QuantilesAndMarginal) {
  // u = (0.025, 0.5, 0.975): mvnorm epsilon == -1.959963984540054, 0, 1.959963984540054 (1e-14);
  // mvt nu 5: epsilon == t_5^-1(u) sqrt(3/5) (1e-14); marginal_log_density == soma das log densidades (1e-13);
  // panel.qbar == cov(epsilon), sigma == o do univariado
}
TEST(DccParameterMap, CopulaCombinations) {
  // Review Focus 2: cada linha da tabela "Entrada e erros" dá a mensagem com as palavras da tabela;
  // nomes: static mvt ml m = 3: C1 C2 C3 mshape; static mvt kendall: mshape; static mvnorm kendall: vazio;
  // static mvnorm ml: C1 C2 C3;
  // correlation_coefficients: 3, 0, 0, 3; copula = None: tudo igual ao 4a
}
```

- [x] **Step 2: Rodar e ver falhar** — `ctest --preset gcc-release -R "Copula(Margins|Scores)|DccParameterMap"` →
  não compila.
- [x] **Step 3: Implementar.** Quantis por `boost::math::quantile(normal/students_t)` em `scores.cpp`. Os `[Rmgarch]`
  da Task 1 que tocam u ou ε ficam aqui, lidos de `spec.convention` passado pelo chamador.
- [x] **Step 4: Rodar e ver passar** — suíte release inteira PASS.
- [x] **Step 5: Commit** — `git commit -m "feat: transform DCC margins into copula scores"`

---

### Task 3: Ajuste da cópula dinâmica e lote misto

**Files:**
- Create: `src/dcc/copula/fit.hpp`, `src/dcc/copula/fit.cpp`
- Modify: `src/dcc/fit.hpp`, `src/dcc/fit.cpp`, `src/dcc/likelihood.cpp` (só se a ℓ da cópula precisar de um
  acesso novo)
- Test: `tests/unit/dcc/copula/fit_test.cpp`

**Interfaces:**
- Consumes: Task 2; `fit_dcc`, `reported_loglik` e `check_dcc_batch` do 4a.
- Produces:
  ```cpp
  namespace ve::dcc::copula {
  // What the specs of a batch share: the univariate panel, u per transformation (computed once, on demand) and
  // sum_i l_i.
  struct CopulaInputs { std::shared_ptr<const StandardizedPanel> univariate; std::vector<double> parametric, empirical;
                        double univariate_loglik = 0.0; };
  CopulaInputs copula_inputs(std::span<const FitResult* const> fits, std::span<const DccSpec> specs);
  // Copula log-likelihood of the "ℓ da cópula" rule at theta (full, natural), writing R_t unless r is empty.
  double copula_loglik(const CopulaInputs& inputs, const DccSpec& spec, std::span<const double> theta,
                       std::span<double> r = {});
  // Static or dynamic: a DccFitResult like fit_dcc's, with loglik = univariate_loglik + sum c_t, the panel of the
  // univariates (H_t from their sigma) and r.
  DccFitResult fit_copula(const CopulaInputs& inputs, const DccSpec& spec, const FitConfig& config);
  }
  ```
- `fit_dcc_batch` despacha cada spec com `copula != None` para `fit_copula`. O `CopulaInputs` é construído uma vez
  por lote, e só com as transformações que alguma spec pede.
- `dcc_criterion` passa a subtrair `correlation_coefficients(spec, m)` da parcela dos parâmetros estimados.
- Status da mvt pelo perfil:
  - `MaxIter` se o Brent esgota as 100 iterações;
  - senão, o status do ajuste interno no ν̂ (um ajuste final em ν̂, que dá estimativa, ℓ e R).

- [x] **Step 1: Escrever os testes**

```cpp
// copula3 + univariados sGARCH(1,1)-std nos coeficientes das notas da Task 1.
TEST(CopulaFit, MvnormIsDccOnScores) {
  // dcc_mvnorm_param, Standard: fit_copula == fit_dcc sobre copula_panel (estimate bit a bit);
  // loglik == univariate_loglik + (reported_loglik + sum log sigma) - marginal_log_density (rel 1e-12)
}
TEST(CopulaFit, MatchesRmgarchAtItsPoint) {
  // Convention::Rmgarch: copula_loglik no ponto do rmgarch == likelihood() das notas para dcc_mvnorm_param,
  // dcc_mvnorm_emp, dcc_mvt_param, dcc_mvt_emp, adcc_mvnorm_param, adcc_mvt_param (rel 1e-10); R_1, R_T (1e-10)
}
TEST(CopulaFit, ProfileFindsTheShape) {
  // dcc_mvt_param: nu do Brent maximiza o perfil: perfil(nu +- 1e-3) <= perfil(nu) (ou nu no teto 50 com o perfil
  // crescente); status converged; Review Focus 5: nu <= 50, loglik finita
}
TEST(CopulaFit, FixedShape) { /* dcc_mvt_param_shape8: names sem mshape; loglik == notas (Rmgarch, rel 1e-10) */ }
TEST(CopulaFit, CriteriaUseTheDccK) {
  // dcc_mvnorm_param: k == 15 + 2 + 3 (univariados std: mu omega alpha beta shape x 3)
}
TEST(DccBatch, MixedBatchSameAsAlone) {
  // Review Focus 4: 4a grid (3 modelos x 3 dists) + dcc/adcc x mvnorm/mvt x param/emp; 1, 3, 16 threads:
  // estimate, loglik, r bit a bit iguais às specs sozinhas; as do 4a iguais a um lote só de 4a
}
```

- [x] **Step 2: Rodar e ver falhar** — `ctest --preset gcc-release -R "CopulaFit|DccBatch"` → não compila.
- [x] **Step 3: Implementar.** O Brent fica em `copula/fit.cpp` (Boost só em `.cpp`).
- [x] **Step 4: Rodar e ver passar** — suíte release inteira PASS, sem mudar nenhum teste do 4a.
- [x] **Step 5: Commit** — `git commit -m "feat: fit dynamic copula DCC models in mixed batches"`

---

### Task 4: Cópula estática

**Files:**
- Create: `src/dcc/copula/static_fit.hpp`, `src/dcc/copula/static_fit.cpp`
- Modify: `src/dcc/copula/fit.cpp`
- Test: `tests/unit/dcc/copula/static_fit_test.cpp`

**Interfaces:**
- Consumes: Tasks 2–3; `linalg::cholesky_*`; `optim::lbfgs_minimize`, que recebe `f(x, g)`.
- Produces:
  ```cpp
  namespace ve::dcc::copula {
  // sin(pi tau_b / 2) of each pair of columns of u, m x m; nullopt unless positive definite.
  std::optional<std::vector<double>> kendall_correlation(std::span<const double> u, std::size_t t, std::size_t m);
  // R = L L' from the m(m-1)/2 free x of the "ML" rule, and the gradient chain dl/dx from dl/dR.
  std::vector<double> correlation_from_free(std::span<const double> x, std::size_t m);
  void free_gradient(std::span<const double> x, std::span<const double> dl_dr, std::span<double> dl_dx, std::size_t m);
  // sum_t log f(epsilon_t; R[, nu]) and, when gradient is not empty, dl/dR (m x m) of the "ML" rule.
  double static_terms(const StandardizedPanel& scores, std::span<const double> r, DccDist dist, double shape,
                      std::span<double> gradient);
  // The ML R of the scores (L-BFGS from x = 0, objective -l/T); its status.
  struct StaticCorrelation { FitStatus status; std::vector<double> r, x; };
  StaticCorrelation static_ml(const StandardizedPanel& scores, DccDist dist, double shape, const FitConfig& config);
  }
  ```
- `fit_copula` monta a estática:
  - mvnorm é `static_ml` sobre o panel de ε, sem parâmetros na tabela;
  - mvt + Kendall é o Brent sobre `static_terms` com R de Kendall;
  - mvt + ML é o Brent sobre `static_ml(ν)`.

  Os Cᵢⱼ reportados são os Rᵢⱼ naturais, na ordem da Task 1.

- [x] **Step 1: Escrever os testes**

```cpp
TEST(CopulaStatic, FreeParametersGiveCorrelations) {
  // x aleatórios, m = 5: diag(R) == 1 (1e-15), Cholesky ok; free_gradient vs diferença central (rel 1e-7)
}
TEST(CopulaStatic, AnalyticGradient) { /* static_terms: dl/dR vs diferença central, mvnorm e mvt nu 6 (rel 1e-7) */ }
TEST(CopulaStatic, KendallFormula) {
  // u de 3 séries fixas (6 linhas): R == sin(pi tau_b/2) com tau_b calculado no teste por pares (1e-15)
}
TEST(CopulaStatic, MatchesRmgarch) {
  // Rmgarch quando a Task 1 achou [Rmgarch] na estática, senão Standard: static_mvnorm_kendall_param, static_mvnorm_ml_param/emp,
  // static_mvt_kendall_param/emp, static_mvt_ml_param: loglik no ponto do rmgarch == notas (rel 1e-10);
  // estimado: R e mshape rel 1e-3 das notas, loglik rel 1e-7
}
TEST(CopulaStatic, NonPositiveDefiniteKendallFails) {
  // Review Focus 3: 3 séries com tau (0.9, 0.9, -0.9): status Failed, estimate NaN, sem exceção; num lote com
  // dcc_mvnorm_param, esta igual à sozinha (bit a bit)
}
TEST(CopulaStatic, RepeatedR) { /* r tem T blocos iguais a R; H_t == D_t R D_t */ }
```

- [x] **Step 2: Rodar e ver falhar** — `ctest --preset gcc-release -R "CopulaStatic"` → não compila.
- [x] **Step 3: Implementar.**
- [x] **Step 4: Rodar e ver passar** — suíte release inteira PASS.
- [x] **Step 5: Commit** — `git commit -m "feat: fit static Gaussian and Student copulas"`

---

### Task 5: Erros-padrão da cópula

**Files:**
- Modify: `src/dcc/standard_errors.hpp`, `src/dcc/standard_errors.cpp`
- Test: `tests/unit/dcc/copula/standard_errors_test.cpp`

**Interfaces:**
- Consumes: Tasks 2–4; `two_stage_pieces`, `sandwich`, `dcc_standard_errors` e `fits_problem` do 4a.
- Produces: as mesmas funções do 4a, agora aceitando um `DccFitResult` de cópula. Nas linhas da cópula, f(θ) é
  `copula_loglik` sobre os `CopulaInputs` reconstruídos dos univariados refiltrados (regra "Erros-padrão em dois
  estágios" deste plano).
  - Kendall: só a linha `mshape`, com R fixa;
  - mvnorm estática: tabelas vazias;
  - nenhuma assinatura pública muda.

- [x] **Step 1: Escrever os testes**

```cpp
TEST(CopulaStandardErrors, MatchRmgarch) {
  // Rmgarch, nos pontos do rmgarch: robust de dcc_mvnorm_param, dcc_mvt_param, adcc_mvnorm_param,
  // static_mvt_ml_param == [Joint] do matcoef das notas (rel 5e-3); mshape no teto: finito e > 0 (Review Focus 5)
}
TEST(CopulaStandardErrors, EmpiricalHasNoCrossBlock) {
  // dcc_mvnorm_emp: o bloco cruzado de A é 0 exato; robust == o do segundo estágio sozinho (rel 1e-12)
}
TEST(CopulaStandardErrors, OptionMatchesExplicit) { /* opção ligada == função explícita (bit a bit) em 4 specs de cópula */ }
```

- [x] **Step 2: Rodar e ver falhar** — `ctest --preset gcc-release -R "CopulaStandardErrors"` → falha.
- [x] **Step 3: Implementar.**
- [x] **Step 4: Rodar e ver passar** — suíte release inteira PASS.
- [x] **Step 5: Commit** — `git commit -m "feat: add two-stage standard errors to copula DCC fits"`

---

### Task 6: Interface C e Python

**Files:**
- Modify: `include/volengine/volengine.h`, `src/capi/dcc/dcc_spec_api.cpp`, `python/src/volengine/_dcc.py`,
  `_native.py`
- Test: `tests/unit/capi/dcc_test.cpp`, `python/tests/dcc/test_copula.py`

**Interfaces:**
- Consumes: Tasks 2–5. `ve_dcc_fit_batch` e os acessores do 4a ficam sem mudança.
- Produces:
  ```c
  #define VE_DCC_MODEL_STATIC 3
  #define VE_DCC_COPULA_NONE 0
  #define VE_DCC_COPULA_PARAMETRIC 1
  #define VE_DCC_COPULA_EMPIRICAL 2
  #define VE_DCC_STATIC_DEFAULT 0
  #define VE_DCC_STATIC_KENDALL 1
  #define VE_DCC_STATIC_ML 2
  VE_API ve_status ve_dcc_spec_set_copula(ve_dcc_spec* s, int32_t transformation);
  VE_API ve_status ve_dcc_spec_set_static_method(ve_dcc_spec* s, int32_t method);
  ```
  - Os setters só conferem o intervalo. As combinações são conferidas pelo `check_dcc_batch`, em
    `ve_dcc_fit_batch`, antes de qualquer ajuste.
  - No Python, `DccSpec` ganha `model="static"`, `copula: str | None = None` (`"parametric"`, `"empirical"`) e
    `static_method: str | None = None` (`"kendall"`, `"ml"`). Ambos entram no `__hash__`.

- [x] **Step 1: Escrever os testes**

```cpp
TEST(CapiDcc, CopulaMatchesCore) { /* lote misto de 4 specs: acessores == fit_dcc_batch (bit a bit); R e H zero-copy */ }
TEST(CapiDcc, CopulaInvalidCombinations) { /* Review Focus 2: cada linha da tabela → VE_INVALID_ARGUMENT + mensagem; setters fora do intervalo */ }
```

```python
def test_copula_grid_and_bic(ctx, series): ...        # 14 specs (static x3, dcc/adcc x mvnorm/mvt) x param/emp; bic() finito; R (T, 3, 3)
def test_copula_static_r_is_constant(ctx, series): ...  # R[t] == R[0] para todo t
def test_copula_bad_combinations_raise(ctx, series): ...  # VolEngineError com as palavras da tabela
def test_copula_standard_errors_later(ctx, series): ...
```

- [x] **Step 2: Rodar e ver falhar.**
- [x] **Step 3: Implementar.**
- [x] **Step 4: Rodar e ver passar** — suíte release, `pytest python/tests`, `check_style.py`, `mypy --strict`
  (limpo).
- [x] **Step 5: Commit** — `git commit -m "feat: expose copula DCC fits through the C API and Python"`

---

### Task 7: Oráculo `rmgarch`

**Files:**
- Create: `tests/reference/copula_rmgarch.R`, `tests/reference/fixtures/rmgarch_copula.json`,
  `tests/unit/dcc/oracles/copula_oracle_test.cpp`, `python/tests/oracles/test_copula_oracles.py`
- Modify: `tests/reference/README.md`, `docs/guide/validation.md`, `CMakeLists.txt`

**Interfaces:**
- O script R roda os 13 casos da Task 1 com `cgarchfit` (solnp, tol 1e-10, `eval.se = TRUE`). Para cada caso grava
  `{id, univariate coefs, copula coefs, llh, se [Joint], R at t = 1, T/2, T}`.
- `CopulaOracle.Rmgarch`, em C++, lê os números copiados da fixture para uma tabela no próprio teste (como o
  `DccOracle` do 4a) e confere:
  - **no ponto do rmgarch:** ℓ e R a rel 1e-10 e os erros-padrão `robust` a rel 5e-3;
  - **estimado:** objetivo ≤ o do rmgarch + 1e-7, parâmetros a rel 1e-3 e ℓ a rel 1e-7.
- No Python (convenção pública), `test_copula_matches_paper_convention[<id>]`:
  - nos casos **empíricos mvnorm**, a ℓ bate com uma avaliação direta em numpy, a rel 1e-10. A avaliação usa ranks
    por `argsort`, Φ⁻¹ por `statistics.NormalDist().inv_cdf` e a recursão do teste do 4a;
  - nos outros casos, as estimativas ficam a rel 5e-2 das do rmgarch e os erros-padrão `robust` são finitos e
    positivos;
  - o numpy não tem o quantil da t nem as CDFs univariadas, e esses casos ficam com o C++.
- `validation.md` ganha a seção "`rmgarch`, Copula-DCC":
  - o que é igual nas duas convenções;
  - cada `[Rmgarch]` da Task 1, com paper e efeito no copula3 (ℓ, BIC);
  - o estimador da mvnorm estática;
  - os erros-padrão empíricos sem Chen–Fan;
  - Kendall não definida positiva dá `failed`;
  - o perfil em ν.
- Uma divergência passa por systematic-debugging, e um xfail exige um parágrafo em `validation.md`.

- [x] **Step 1: Gerar a fixture** — `Rscript tests/reference/copula_rmgarch.R` termina e grava 13 casos.
- [x] **Step 2: Escrever os testes** e rodar → PASS.
- [x] **Step 3: Documentar.**
- [x] **Step 4: Rodar** — suíte release e `pytest python/tests` PASS.
- [x] **Step 5: Commit** — `git commit -m "test: validate copula DCC fits against rmgarch"`

---

### Task 8: Benchmarks e documentação

**Files:**
- Modify: `tests/benchmarks/fit_benchmarks.cpp`, `docs/performance/benchmarks.md`, `README.md`, `CHANGELOG.md`,
  `ARCHITECTURE.md`

**Interfaces:**
- **Grade:**
  - `BM_CopulaGrid/m:21/T:5000/threads:<1|hw>`: os univariados de `BM_DccGrid` fora da medição e 14 specs, as da
    estática (mvnorm, mvt Kendall, mvt ML) e as de dcc/adcc × mvnorm/mvt, cada uma paramétrica e empírica, sem
    erros-padrão;
  - `BM_CopulaFit/<case>/m:21`: um ajuste de cada uma das 7 formas, paramétrica;
  - `BM_CopulaStandardErrors/dcc_mvnorm_param/m:21`.

  Cinco repetições, publicadas em `benchmarks.md`. Se a mvt ML estática passar de 60 s por ajuste, registrar o
  número e o `ponytail:` do perfil; a otimização fica para o plano 5.
- **Documentação:**
  - `README.md`: um exemplo de lote misto, DCC e Copula-DCC na mesma lista, escolhido pelo `bic()`;
  - `CHANGELOG.md`: entrada "Added";
  - `ARCHITECTURE.md`: `src/dcc/copula/` e a dependência `dist` <- `dcc/copula` <- `dcc`.

- [x] **Step 1: Escrever os benchmarks** e rodar com `--benchmark_filter="BM_Copula"` → termina.
- [x] **Step 2: Medir** e publicar.
- [x] **Step 3: Documentar.**
- [x] **Step 4: Rodar** — suíte release, `pytest`, `check_style.py` → PASS.
- [x] **Step 5: Commit** — `git commit -m "bench: add copula DCC benchmarks and document plan 4b"`
