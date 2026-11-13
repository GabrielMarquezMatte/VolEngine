# VolEngine — Plano 3c: diagnósticos

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Os testes de diagnóstico que o `show()` de um ajuste do `rugarch` reporta, como funções sobre o ajuste: Ljung-Box ponderado dos resíduos padronizados e dos seus quadrados, ARCH-LM ponderado, Nyblom (individual e conjunto, com valores críticos), sign bias e Pearson ajustado (GOF). Ficam expostos pela interface C (`ve_fit_ljung_box`...) e pelo Python (`f.ljung_box(...)`...) e são validados contra o `rugarch`.

**Architecture:**
- **Testes sobre resíduos:** Ljung-Box, ARCH-LM, sign bias e GOF são funções puras de z, e e σ, que o `FitResult` já guarda. Vivem num módulo novo, `src/diagnostics/`.
- **CDFs:** o GOF precisa da CDF das 8 distribuições. Ela vai para `src/dist/cdf.*`, com forma fechada onde existe e, no nig e no ghyp, a quadratura da densidade acumulada sobre os pontos ordenados.
- **Nyblom:** precisa dos scores por observação, que o ajuste já calcula na avaliação final (para os erros-padrão) e descarta. Guardá-los pesaria T × k por ajuste. Por isso o ajuste calcula ali mesmo as estatísticas de Nyblom (k + 1 números) e as guarda no `FitResult`, sem mudar nenhum outro número.

**Tech Stack:** C++23, CMake + presets, GoogleTest v1.17.0, Google Benchmark v1.9.1, Boost.Math (já no projeto: `gamma_q`, `ibeta`, `erfc`), Python ≥ 3.11 + numpy + pytest, R 4.6.2 + `rugarch` 1.5.6.

**Spec:** `docs/superpowers/specs/2026-11-13-volengine-03-forecast-simulacao-diagnosticos-design.md` (seções 2 "3c", 3, 5–7), sobre a spec-mãe `docs/superpowers/specs/2026-11-10-volengine-design.md` (seção 3: critérios e diagnósticos são "funções sobre o resultado, não campos de struct"; seção 12: fórmulas de paper e documentação, nunca código GPL).

## Roteiro

- **1, 2a–2f, 3a, 3b (feitos).**
- **3c (este):** diagnósticos.

Depois: 4 DCC; 5 performance; 6 wrapper C++, wheels, 1.0.

## Global Constraints

Valem todas as Global Constraints dos planos 1, 2a–2f, 3a e 3b, o `STYLEGUIDE.md` e o `ARCHITECTURE.md`. As que mais pesam aqui:

- A interface C é o único contrato estável:
  - ≤ 5 parâmetros por função;
  - nenhuma exceção atravessa a fronteira C;
  - handles imutáveis;
  - saídas em arrays do chamador, como `ve_fit_std_errors`.
- **Nenhum número existente muda.** Ajuste, erros-padrão, forecast e simulação continuam iguais, e todos os testes existentes passam sem mudar constantes nem tolerâncias. O ajuste só ganha k + 1 números (Nyblom) por ajuste.
- **Fórmulas** vêm dos papers e da documentação:
  - Fisher & Gallagher (2012) para Ljung-Box e Li-Mak ponderados;
  - Nyblom (1989) e Hansen (1990, 1992) para o teste de estabilidade e a tabela de valores críticos;
  - Engle & Ng (1993) para o sign bias;
  - Vlaar & Palm (1993) para o Pearson ajustado.

  As convenções de onde o `rugarch` difere do paper foram lidas no pacote instalado (seção "Convenções") e são reproduzidas sem copiar código.
- Ficam fora:
  - diagnósticos que não estão no `show()` do `rugarch`;
  - uma função que imprime tudo como o `show()`;
  - o fallback do `rugarch` para a hessiana quando GᵀG é singular, que aqui vira NaN.
- Estilo e ambiente:
  - nunca `else`;
  - funções ≤ 60 linhas, ≤ 5 parâmetros, aninhamento ≤ 3;
  - ≤ 8 módulos por pasta. `src/diagnostics/` nasce com 4. `src/dist/` vai a 8 com `cdf`, e `src/fit/` a 6 com `nyblom`. `src/capi/results/` vai a 5, `tests/unit/capi/results/` a 4, `python/src/volengine/` a 8 e `python/tests/` a 8;
  - commits sem `Co-Authored-By`; LF;
  - **testes só em release** (`gcc-release`);
  - executáveis rodam com `/c/msys64/ucrt64/bin` primeiro no PATH;
  - `pytest` com `-k "not sgarch11_vx_std/aparch11_vx_std"`.

### Convenções (lidas no `rugarch` 1.5.6: `Weighted.Box.test`, `Weighted.LM.test`, `.weightedBoxTest`, `.nyblomTest`, `.nyblomCritical`, `.signbiasTest`, `.gofTest` e o `show` de `uGARCHfit`)

Notação:
- z = e/σ são os resíduos padronizados do ajuste, com os não finitos trocados por 0;
- n é o número de observações;
- ρ̂ₖ é a autocorrelação amostral do `acf` do R: Σₜ₌₁ⁿ⁻ᵏ (xₜ − x̄)(xₜ₊ₖ − x̄) / Σₜ₌₁ⁿ (xₜ − x̄)²;
- G(a, b) é a cauda superior de uma Gama de forma a e escala b.

**Ljung-Box ponderado.** x = z (ou z² com `squared`).
- Q = n(n + 2) Σₖ₌₁ᴸ wₖ ρ̂ₖ² / (n − k), com wₖ = (L − k + 1)/L.
- p = G(a, b), com a = ¾ (L + 1)² L / D, b = ⅔ D / (L(L + 1)) e D = 2L² + 3L + 1 − 6 L f.
- Com `squared`, f = 0, seja qual for o argumento: o `rugarch` força isso.
- Exige L ≥ 1, f ≥ 0 e L ≥ 3f − 1. Isso garante D > 0.

**ARCH-LM ponderado** (Li-Mak). x = e²/σ² (= z²).
- Q = n Σₖ₌f₊₁ᴸ wₖ ρ̂ₖ², com wₖ = (L − k + f + 1)/L.
- p = G(a, b), com a = ¾ (L + f + 1)² (L − f) / D, b = ⅔ D / (L(L + f + 1)) e D = 2L² + 3L + 2Lf + 2f² + 3f + 1.
- Exige 1 ≤ f < L.

**p-valores.** Calculados pela cauda superior direta (`gamma_q`). O `rugarch` usa `1 − pgamma` e devolve 0 quando o p é menor que ~1e-16; aqui sai o valor pequeno verdadeiro. A diferença absoluta fica abaixo de 1e-15, e `validation.md` registra isso.

**Lags do `show()`**, que a Task 6 compara e o README mostra:

| Teste | Lags e f |
| --- | --- |
| Ljung-Box em z | m + n = d (ordens ARMA, sem o ARFIMA). Lag 1 com f = 0; lag max(2, 3d − 1) com f = d; lag max(5, 5d − 1) com f = d |
| Ljung-Box em z² | d = p + q (ordens ARCH e GARCH), lags 1, max(2, 3d − 1) e max(5, 5d − 1), sempre com f = 0 |
| ARCH-LM | f = p + q, lags f + 1, f + 3 e f + 5 |

**Nyblom.**
- gₜ é o score da observação t, a derivada de ℓₜ em relação aos parâmetros **estimados** (nem fixos nem derivados), em espaço natural, na estimativa.
- Sₜ = Σₛ≤ₜ gₛ, A = Σₜ Sₜ Sₜᵀ e B = Σₜ gₜ gₜᵀ.
- A estatística individual do parâmetro i é Aᵢᵢ / (n Bᵢᵢ), e a conjunta é tr(A B⁻¹) / n.
- Com B singular (Cholesky falha), a conjunta é NaN, e uma individual com Bᵢᵢ = 0 também é NaN.
- Os valores críticos (10 %, 5 %, 1 %) para k = 1..20 parâmetros são a tabela de Hansen que o `rugarch` reporta, transcrita como dado na Task 4.
- A individual usa k = 1. Acima de 20 parâmetros não há valor crítico conjunto.

**Sign bias.** É a regressão MQO de z²ₜ sobre [1, 1(eₜ₋₁ < 0), 1(eₜ₋₁ < 0) eₜ₋₁, 1(eₜ₋₁ ≥ 0) eₜ₋₁], para t = 2..n.
- Os regressores usam o resíduo e, não z, como faz o `rugarch` (Engle & Ng usam z). `validation.md` registra isso.
- Os testes sign, negative e positive dão |t| dos três últimos coeficientes, com p bilateral de uma t de n − 1 − 4 graus de liberdade.
- O joint é o Wald b̂ᵀ V⁻¹ b̂ dos três coeficientes, com V = s² (XᵀX)⁻¹ do MQO e s² = RSS/(n − 1 − 4), e p da χ²(3).

**GOF de Pearson ajustado.**
- uᵢ = F(zᵢ) é a CDF da distribuição do ajuste nos seus parâmetros.
- Para g grupos, contam-se os uᵢ em [0, 1/g), [1/g, 2/g), ..., [(g−1)/g, 1], com o último intervalo fechado.
- E = n/g e Q = Σ (nⱼ − E)²/E, com p da χ²(g − 1).
- O `rugarch` usa `tabulate`, que **descarta os grupos vazios depois do último não vazio**. Isso é reproduzido para manter a paridade, e `validation.md` registra.
- Os grupos padrão do `show()` são 20, 30, 40 e 50.

**Erros:**
- ajuste sem ponto finito (séries vazias): `VE_INVALID_ARGUMENT` com a mensagem "the fit has no finite point";
- lag ou f fora das regras acima, `groups < 2` e n ≤ L: `VE_INVALID_ARGUMENT` citando "lag", "fitdf" ou "groups";
- sign bias com n < 6: `VE_INVALID_ARGUMENT`.

## Review Focus

1. **Ajuste `FAILED` sem ponto finito, ou série curta** (n = 10, com lag 20 ou groups 50): erro de argumento, nunca leitura fora do vetor nem divisão por zero que vire número. → testes na Task 5.
2. **z com valores não finitos ou série constante** (variância de x zero, `acf` 0/0): o resultado é NaN documentado, nunca um crash. Os não finitos viram 0, como no `rugarch`. → teste na Task 2.
3. **Nyblom com parâmetros fixos ou derivados** (VT, iGARCH, spec com tudo fixo): as individuais desses são NaN, o número de estimados vale para o valor crítico, e com zero estimados a conjunta é NaN. → testes nas Tasks 4 e 5.
4. **CDF nas caudas** (z = ±40, std com ν = 2,05, ghyp com ζ pequeno): fica em [0, 1], monotônica e finita. → teste na Task 1.
5. **O ajuste não fica mais lento:** o custo do Nyblom no fim de `fit_one` fica dentro do ruído de `BM_FitOne`. → benchmark na Task 7.

---

## Mapa de arquivos

```
src/dist/cdf.hpp, .cpp                cdf(Dist, params, z), sorted_cdf(...) (novo)
src/diagnostics/tails.hpp             gamma_upper, chi_squared_upper, student_two_sided (novo)
src/diagnostics/portmanteau.hpp, .cpp weighted_ljung_box, weighted_arch_lm, TestResult (novo)
src/diagnostics/sign_bias.hpp, .cpp   sign_bias (novo)
src/diagnostics/gof.hpp, .cpp         pearson_gof (novo)
src/fit/nyblom.hpp, .cpp              nyblom_statistics, nyblom_critical (novo); fit.cpp e fit_result.hpp
src/capi/results/diagnostics_api.cpp  ve_fit_ljung_box, ve_fit_arch_lm, ve_fit_nyblom, ve_nyblom_critical,
                                      ve_fit_sign_bias, ve_fit_gof (novo)
python/src/volengine/_diagnostics.py  TestResult, SignBias, Nyblom (novo); _fit.py, _native.py, __init__.py, api.py
tests/unit/dist/cdf_test.cpp, tests/unit/diagnostics/*, tests/unit/fit/nyblom_test.cpp,
tests/unit/capi/results/diagnostics_test.cpp, python/tests/test_diagnostics.py
tests/reference/diagnostics_rugarch.R, fixtures/rugarch_diagnostics.json, python/tests/oracles/test_diagnostics_oracles.py
```

Valores de referência dos testes unitários: foram gerados pelo `rugarch` 1.5.6, nas suas funções públicas e com `lm`, sobre os vetores abaixo, para t = 1..40:

- zₜ = sin(0,7t) + 0,3 cos(1,9t);
- σₜ = 1 + 0,2 sin(0,3t);
- eₜ = zₜ σₜ.

---

### Task 1: CDFs das 8 distribuições

**Files:**
- Create: `src/dist/cdf.hpp`, `src/dist/cdf.cpp`
- Modify: `CMakeLists.txt`
- Test: `tests/unit/dist/cdf_test.cpp`

**Interfaces:**
- Produces:
  ```cpp
  namespace ve::dist {
  // P(Z <= z) of the standardized density of make_density (the parametrization of the fit).
  double cdf(Dist dist, const ShapeParameters<double>& parameters, double z);
  // cdf at each point of sorted (ascending): nig and ghyp accumulate the quadrature between neighbours, so n points
  // cost one pass, not n integrals from -inf.
  std::vector<double> sorted_cdf(Dist dist, const ShapeParameters<double>& parameters, std::span<const double> sorted);
  }
  ```
- Formas:
  - norm: Φ (`erfc`);
  - std: t de ν graus de liberdade em z √(ν/(ν − 2)), por `ibeta`;
  - ged: ½ ± ½ P(1/ν, ½ |z/λ|^ν), por `gamma_p`;
  - sstd e sged: Fernandez–Steel sobre a CDF do núcleo, com y = `shift` + `stretch` · z. Para y < 0, F = 2/(ξ² + 1) · K(yξ); para y ≥ 0, F = 1/(ξ² + 1) + 2ξ²/(ξ² + 1) · (K(y/ξ) − ½), sendo K a CDF do núcleo de variância 1;
  - jsu: Φ(r(z)), com r de `johnson_normal_of`;
  - nig e ghyp: quadratura da densidade, que reaproveita `tail_integral` e `segment_integral` de `moments.hpp`.

- [ ] **Step 1: Escrever os testes**

```cpp
TEST(Cdf, MatchesRugarch) {
  // pdist do rugarch 1.5.6 em z = -1.5 e 0.3 (rel 1e-9; nig/ghyp abs 1e-8):
  // std shape 6: 0.057919999999999978, 0.63704250791223438
  // ged shape 1.4: 0.064408960501297186, 0.64087619258778505
  // sstd skew 1.3 shape 6: 0.039613691296787761, 0.67012366619190511
  // sged skew 0.8 shape 1.6: 0.076324648554635791, 0.59341768109990278
  // nig skew 0.3 shape 1.2: 0.043694526826888819, 0.67937614224277576
  // ghyp skew -0.2 shape 2 lambda -1.5: 0.065568350139092985, 0.61898170922993212
  // jsu skew 0.5 shape 1.8: 0.047633763880890301, 0.66242721987116615
}
TEST(Cdf, DerivativeIsTheDensity) {
  // as 8 distribuições em 2 pontos de parâmetro: (cdf(z + h) - cdf(z - h)) / 2h == exp(log_density) (rel 1e-6, h 1e-4)
  // em z = -2, -0.3, 0.7, 2.5; cdf(P(z<0)) bate com probability_below_zero em z = 0 (abs 1e-10)
}
TEST(Cdf, SortedMatchesPointwise) { /* nig e ghyp: sorted_cdf de 200 pontos == cdf ponto a ponto (abs 1e-10) */ }
TEST(Cdf, TailsStayInRange) {
  // Review Focus 4: z = -40, -8, 0, 8, 40 para cada distribuição, std 2.05, ghyp zeta 0.3: em [0, 1], monotônica, finita
}
```

- [ ] **Step 2: Rodar e ver falhar** — `ctest --preset gcc-release -R Cdf` → não compila.
- [ ] **Step 3: Implementar.**
- [ ] **Step 4: Rodar e ver passar** — suíte release inteira PASS.
- [ ] **Step 5: Commit** — `git commit -m "feat: add the distribution functions of the eight distributions"`

---

### Task 2: Ljung-Box e ARCH-LM ponderados

**Files:**
- Create: `src/diagnostics/tails.hpp`, `src/diagnostics/portmanteau.hpp`, `src/diagnostics/portmanteau.cpp`
- Test: `tests/unit/diagnostics/portmanteau_test.cpp`

**Interfaces:**
- Produces:
  ```cpp
  namespace ve::diagnostics {
  struct TestResult { double statistic; double p_value; };
  // tails.hpp: upper tails by Boost (gamma_q, ibeta), never 1 - cdf.
  double gamma_upper(double x, double shape, double scale);
  double chi_squared_upper(double x, double degrees);
  double student_two_sided(double t, double degrees);
  // Regras "Ljung-Box ponderado" e "ARCH-LM ponderado"; the caller checks lag and fitdf.
  TestResult weighted_ljung_box(std::span<const double> z, std::size_t lag, std::size_t fitdf, bool squared);
  TestResult weighted_arch_lm(std::span<const double> residuals, std::span<const double> sigma, std::size_t lag,
                              std::size_t fitdf);
  // The rules of the C API: lag >= 1, lag >= 3 fitdf - 1, n > lag (Ljung-Box); 1 <= fitdf < lag < n (ARCH-LM).
  std::optional<std::string> ljung_box_problem(std::size_t n, std::size_t lag, std::size_t fitdf, bool squared);
  std::optional<std::string> arch_lm_problem(std::size_t n, std::size_t lag, std::size_t fitdf);
  }
  ```

- [ ] **Step 1: Escrever os testes**

```cpp
TEST(Portmanteau, LjungBoxMatchesRugarch) {
  // z de 40 pontos do mapa: lag 5 fitdf 2: Q 41.287943416174535 (rel 1e-12), p < 1e-12 e > 0 (o rugarch dá 0)
  // squared lag 5 (fitdf forçado 0): 22.269312541169139, p 4.5843285358948549e-06 (rel 1e-9)
  // lag 1: 20.243295814457056, p 6.8192010626022537e-06
  // squared com fitdf 3 dá o mesmo que fitdf 0
}
TEST(Portmanteau, ArchLmMatchesRugarch) {
  // e, sigma do mapa: lag 5 fitdf 2: 23.14757005967083, p 3.4416950271953795e-06; lag 3 fitdf 1:
  // 16.096717784598464, p 8.2766564685865518e-05 (rel 1e-12 / 1e-9)
}
TEST(Portmanteau, NonFiniteAndConstant) {
  // Review Focus 2: um NaN e um inf em z contam como 0 (== mesmo z com 0 ali, bit a bit); z constante: Q e p NaN
}
TEST(Portmanteau, Problems) { /* lag 0; squared false com lag 4 fitdf 2 (4 < 5); n == lag; arch fitdf 0, fitdf == lag */ }
```

- [ ] **Step 2: Rodar e ver falhar** — `ctest --preset gcc-release -R Portmanteau` → não compila.
- [ ] **Step 3: Implementar.** O `acf` vai num helper local, e os não finitos viram 0 antes de tudo.
- [ ] **Step 4: Rodar e ver passar** — suíte release inteira PASS.
- [ ] **Step 5: Commit** — `git commit -m "feat: add the weighted Ljung-Box and ARCH-LM tests"`

---

### Task 3: Sign bias e GOF

**Files:**
- Create: `src/diagnostics/sign_bias.hpp`, `.cpp`, `src/diagnostics/gof.hpp`, `.cpp`
- Test: `tests/unit/diagnostics/sign_bias_test.cpp`, `tests/unit/diagnostics/gof_test.cpp`

**Interfaces:**
- Consumes: Task 1 (`sorted_cdf`), Task 2 (`TestResult`, caudas), `linalg` (MQO e Cholesky existentes).
- Produces:
  ```cpp
  namespace ve::diagnostics {
  struct SignBias { TestResult sign, negative, positive, joint; };  // |t| and p; joint: Wald chi2(3) and p
  SignBias sign_bias(std::span<const double> z, std::span<const double> residuals);   // n >= 6 (the caller checks)
  TestResult pearson_gof(std::span<const double> z, Dist dist, const dist::ShapeParameters<double>& parameters,
                         std::size_t groups);                                         // groups >= 2
  }
  ```

- [ ] **Step 1: Escrever os testes**

```cpp
TEST(SignBias, MatchesRugarchRegression) {
  // z, e do mapa: |t| 0.80605064564506934, 0.58096539518493029, 0.44824700499741182;
  // p 0.42565663652195873, 0.56498640324614569, 0.6567376139868587;
  // joint 0.66016010279206094, p 0.88253222361606198 (rel 1e-10)
}
TEST(Gof, MatchesRugarch) {
  // z do mapa: std shape 6, g 4: Q 3, p 0.39162517627108911; sstd skew 1.3 shape 6, g 5: 4.25, 0.3732280258334989;
  // norm, g 8: 12, 0.10055886850835885 (Q exato; p rel 1e-12)
}
TEST(Gof, DropsEmptyTopGroupsLikeRugarch) {
  // z = 10 valores em [-3, -2] com norm, g 4: só o primeiro grupo cheio; Q == (10 - 2.5)^2 / 2.5 (os 3 vazios do topo
  // saem da soma); p da chi2(3)
}
```

- [ ] **Step 2: Rodar e ver falhar** — `ctest --preset gcc-release -R "SignBias|Gof"` → não compila.
- [ ] **Step 3: Implementar.**
- [ ] **Step 4: Rodar e ver passar** — suíte release inteira PASS.
- [ ] **Step 5: Commit** — `git commit -m "feat: add the sign bias and adjusted Pearson tests"`

---

### Task 4: Nyblom no ajuste

**Files:**
- Create: `src/fit/nyblom.hpp`, `src/fit/nyblom.cpp`
- Modify: `src/fit/fit_result.hpp`, `src/fit/fit.cpp`
- Test: `tests/unit/fit/nyblom_test.cpp`

**Interfaces:**
- Produces:
  ```cpp
  namespace ve::fit {
  // Regra "Nyblom" over the columns of a T x k row-major score matrix marked in `estimated`; the others get NaN.
  struct Nyblom { std::vector<double> individual; double joint; std::size_t estimated; };
  Nyblom nyblom_statistics(std::span<const double> scores, std::size_t k, std::span<const bool> estimated);
  // Hansen's asymptotic critical values (10%, 5%, 1%) for 1..20 parameters; nullopt above 20 or for 0.
  std::optional<std::array<double, 3>> nyblom_critical(std::size_t parameters);
  }
  namespace ve { struct FitResult { /* ..., */ fit::Nyblom nyblom; }; }
  ```
- `fill_at_estimate` chama `nyblom_statistics` com os scores que já calcula e a máscara `layout.is_estimated`. Sem ponto finito, `individual` fica vazio, `joint` NaN e `estimated` 0.
- A tabela (10 %, 5 %, 1 %) por k = 1..20 é dado publicado, transcrito na ordem:

  ```
  0.353 0.470 0.748 | 0.610 0.749 1.07 | 0.846 1.01 1.35 | 1.07 1.24 1.60 | 1.28 1.47 1.88 | 1.49 1.68 2.12 |
  1.69 1.90 2.35 | 1.89 2.11 2.59 | 2.10 2.32 2.82 | 2.29 2.54 3.05 | 2.49 2.75 3.27 | 2.69 2.96 3.51 |
  2.89 3.15 3.69 | 3.08 3.34 3.90 | 3.26 3.54 4.07 | 3.46 3.75 4.30 | 3.64 3.95 4.51 | 3.83 4.14 4.73 |
  4.03 4.33 4.92 | 4.22 4.52 5.13
  ```

- [ ] **Step 1: Escrever os testes**

```cpp
TEST(Nyblom, MatchesFormula) {
  // G (40 x 2): col 0 = e - mean(e), col 1 = z^2 - mean(z^2) do mapa: individual 0.078595967479814427,
  // 0.01440865819792329; joint 0.093175052912939635 (rel 1e-12); com uma 3a coluna não estimada: NaN nela,
  // as outras iguais, estimated == 2
}
TEST(Nyblom, SingularJointIsNaN) { /* duas colunas iguais: joint NaN, individuais finitas */ }
TEST(Nyblom, CriticalValues) { /* k = 1 → {0.353, 0.47, 0.748}; k = 20 → {4.22, 4.52, 5.13}; 0 e 21 → nullopt */ }
TEST(Nyblom, StoredByTheFit) {
  // fit_one sGARCH ARMA(1,1) simulado: nyblom.individual.size() == k, todos finitos, estimated == k;
  // VT: individual[omega] NaN, estimated == k - 1; tudo fixo: estimated 0, joint NaN (Review Focus 3);
  // loglik, estimate, se_hessian e se_robust iguais bit a bit aos de antes desta task (constantes do fit_test inalteradas)
}
```

- [ ] **Step 2: Rodar e ver falhar** — `ctest --preset gcc-release -R Nyblom` → não compila.
- [ ] **Step 3: Implementar.** A conjunta usa o Cholesky de `linalg`.
- [ ] **Step 4: Rodar e ver passar.** A suíte release inteira passa. Um `BM_FitOne/s11_norm` antes e depois fica dentro do ruído, porque é O(T k²) sobre scores já calculados.
- [ ] **Step 5: Commit** — `git commit -m "feat: keep the Nyblom statistics of a fit"`

---

### Task 5: Interface C e Python

**Files:**
- Create: `src/capi/results/diagnostics_api.cpp`, `python/src/volengine/_diagnostics.py`
- Modify: `include/volengine/volengine.h`, `python/src/volengine/_fit.py`, `_native.py`, `__init__.py`, `api.py` (`__all__`), `CMakeLists.txt`
- Test: `tests/unit/capi/results/diagnostics_test.cpp`, `python/tests/test_diagnostics.py`

**Interfaces:**
- Consumes: Tasks 1–4.
- Produces:
  ```c
  /* ---- Diagnostics ----  rugarch's show(): see docs/guide/validation.md for the formulas. out: caller arrays.
     VE_INVALID_ARGUMENT for a fit with no finite point or arguments outside the rules of each test. */
  VE_API ve_status ve_fit_ljung_box(const ve_fit* f, int32_t squared, size_t lag, size_t fitdf, double* out);  /* [Q, p] */
  VE_API ve_status ve_fit_arch_lm(const ve_fit* f, size_t lag, size_t fitdf, double* out);                    /* [Q, p] */
  /* individual: ve_fit_param_count values in parameter order, NaN for the fixed and derived ones. */
  VE_API ve_status ve_fit_nyblom(const ve_fit* f, double* individual, double* joint);
  VE_API ve_status ve_nyblom_critical(size_t parameters, double* out);   /* [10%, 5%, 1%]; 1..20 */
  VE_API ve_status ve_fit_sign_bias(const ve_fit* f, double* out);  /* [t, p] x sign, negative, positive; [chi2, p] joint */
  VE_API ve_status ve_fit_gof(const ve_fit* f, size_t groups, double* out);                                    /* [Q, p] */
  ```
  ```python
  class TestResult(NamedTuple): statistic: float; p_value: float
  class SignBias(NamedTuple): sign: TestResult; negative: TestResult; positive: TestResult; joint: TestResult
  class Nyblom(NamedTuple):
      individual: dict[str, float]        # estimated parameters only
      joint: float
      individual_critical: tuple[float, float, float]           # 10%, 5%, 1% (k = 1)
      joint_critical: tuple[float, float, float] | None          # None above 20 estimated parameters
  class Fit:
      def ljung_box(self, lag: int, *, squared: bool = False, fitdf: int = 0) -> TestResult: ...
      def arch_lm(self, lag: int, *, fitdf: int) -> TestResult: ...
      def nyblom(self) -> Nyblom: ...
      def sign_bias(self) -> SignBias: ...
      def gof(self, groups: int = 20) -> TestResult: ...
  ```
- Regra "Erros". No Python, um inteiro negativo ou ≥ 2⁶⁴ dá `ValueError` antes da chamada C, e o erro da regra vem como `VolEngineError`. Os nomes do `show()` (lags e grupos padrão) vão nas docstrings e no README, não em funções.

- [ ] **Step 1: Escrever os testes**

```cpp
TEST(CapiDiagnostics, MatchCore) {
  // ve_fit_batch de 1 job sGARCH ARMA(1,1): cada função == a do núcleo sobre std_residuals/residuals/sigma
  // (bit a bit); nyblom == fit.nyblom
}
TEST(CapiDiagnostics, InvalidArguments) {
  // Review Focus 1: fit sem ponto finito ("finite"); série de 10 pontos com lag 20 ("lag"); groups 1 ("groups");
  // série de 5 pontos no sign bias; arch fitdf 0 ("fitdf"); f NULL → VE_INVALID_HANDLE; out NULL → VE_INVALID_ARGUMENT;
  // ve_nyblom_critical(0) e (21) → VE_INVALID_ARGUMENT
}
TEST(CapiDiagnostics, NyblomWithFixedParameters) { /* Review Focus 3: mu fixo: individual[0] NaN, os outros finitos */ }
```

```python
def test_diagnostics_shapes(ctx, series): ...        # TestResult, SignBias, Nyblom exportados; floats finitos
def test_nyblom_individual_names(ctx, series): ...   # chaves == parâmetros estimados (sem os fixos)
def test_diagnostics_bad_arguments_raise(ctx, series): ...   # lag -1 → ValueError; lag 0 → VolEngineError "lag"
```

- [ ] **Step 2: Rodar e ver falhar** — `ctest --preset gcc-release -R CapiDiagnostics` → não compila; `pytest python/tests/test_diagnostics.py` → `AttributeError`.
- [ ] **Step 3: Implementar.**
- [ ] **Step 4: Rodar e ver passar.** Precisam passar a suíte release inteira, o `pytest`, o `check_style.py` e o `mypy --strict`, este sem erro novo.
- [ ] **Step 5: Commit** — `git commit -m "feat: expose the diagnostics through the C API and Python"`

---

### Task 6: Oráculo `rugarch`

**Files:**
- Create: `tests/reference/diagnostics_rugarch.R`, `tests/reference/fixtures/rugarch_diagnostics.json`, `python/tests/oracles/test_diagnostics_oracles.py`
- Modify: `tests/reference/README.md`, `docs/guide/validation.md`

**Interfaces:**
- Consumes: Task 5. Também os casos e as funções de `oracle_rugarch.R` (com `source`, já que `main()` é guardado) e os helpers de `python/tests/oracles/test_oracles.py`.
- O R ajusta cada caso com oráculo `rugarch`, como `oracle_rugarch.R`, e grava `{id, ljung_box[3], ljung_box_squared[3], arch_lm[3], nyblom: {individual: {name: value}, joint}, sign_bias[4], gof[4]}`, cada teste como `[statistic, p]`. Usa as funções públicas: `Weighted.Box.test` e `Weighted.LM.test` com os lags da tabela "Lags do `show()`", `nyblom`, `signbias` e `gof(fit, c(20, 30, 40, 50))`.
- No Python:
  - **Testes sobre resíduos:** o VolEngine ajusta a spec com todos os parâmetros fixos nos coeficientes do `rugarch`. Assim z, e e σ são os do `rugarch`. Com VT, o VT fica desligado e o ω é fixado (como no 3a).
  - **Tolerâncias dos testes sobre resíduos:** estatística rel 1e-7, p abs 1e-7. GOF com Q exato, salvo um ponto na fronteira de um grupo, o que é investigado se acontecer.
  - **fGARCH:** o filtro suaviza o |·| (ruling do 3a), então esses casos usam rel 1e-4.
  - **Nyblom:** só existe com parâmetros estimados. O VolEngine ajusta a spec com `start` = coeficientes do `rugarch`, e a comparação usa rel 1e-3 (pela diferença dos pontos ótimos e pelos scores numéricos do `rugarch`). Ficam fora os casos marcados `xfail` em `test_oracles.py`.
- A seção "`rugarch`, diagnostics" de `validation.md` traz:
  - as regras da seção "Convenções";
  - as três diferenças do paper que o `rugarch` adota: e nos regressores do sign bias, os grupos vazios do topo descartados e f = 0 forçado no z²;
  - o p-valor pela cauda direta;
  - o Nyblom sem o fallback da hessiana;
  - a tabela de diferenças observadas.

- [ ] **Step 1: Escrever o script R e gerar a fixture** — `Rscript tests/reference/diagnostics_rugarch.R`.
- [ ] **Step 2: Escrever o teste** `test_diagnostics_match_rugarch[<id>]` e rodar → PASS. Uma divergência passa por systematic-debugging antes de qualquer xfail, e um xfail strict exige um parágrafo em `validation.md`.
- [ ] **Step 3: Documentar** em `validation.md` e no README de `tests/reference/`.
- [ ] **Step 4: Rodar** — `pytest python/tests` PASS.
- [ ] **Step 5: Commit** — `git commit -m "test: validate the diagnostics against rugarch"`

---

### Task 7: Benchmarks e documentação

**Files:**
- Modify: `tests/benchmarks/fit_benchmarks.cpp`, `docs/performance/benchmarks.md`, `README.md`, `CHANGELOG.md`, `ARCHITECTURE.md`

**Interfaces:**
- **Benchmarks:**
  - `BM_Diagnostics/<series>/<test>` sobre `s11_norm`, com `ljung_box` (lag 5), `arch_lm` (lag 5, f 2), `sign_bias` e `gof` (50 grupos);
  - `gof` também sobre `sgarch11_ghyp`, que custa a CDF por quadratura;
  - `BM_FitOne/s11_norm` antes e depois da Task 4 (Review Focus 5).
- Rode só esses filtros, com 5 repetições, e publique em `benchmarks.md`.
- **Documentos:**
  - `README.md`: exemplo com os testes do `show()` (`f.ljung_box(1)`, `f.ljung_box(5, fitdf=2)`, `f.arch_lm(3, fitdf=2)`, `f.nyblom()`, `f.sign_bias()`, `f.gof(20)`);
  - `CHANGELOG.md`: entrada "Added";
  - `ARCHITECTURE.md`: linhas de `src/diagnostics/`, `src/dist/cdf.*`, `src/fit/nyblom.*` e `diagnostics_api.cpp`, e a dependência `fit` <- `diagnostics` <- `capi`.

- [ ] **Step 1: Escrever os benchmarks** e rodar `--benchmark_filter="BM_Diagnostics"` → termina, todos finitos.
- [ ] **Step 2: Medir** com 5 repetições e publicar.
- [ ] **Step 3: Documentar.**
- [ ] **Step 4: Rodar** — suíte release inteira, `pytest` e `check_style.py` → PASS.
- [ ] **Step 5: Commit** — `git commit -m "bench: add diagnostics benchmarks and document plan 3c"`


## Execução — 2026-11-13

Implementado diretamente em `master`, sem worktree. O usuário autorizou a skill `implement`
no lugar das skills Superpowers ausentes. O núcleo foi dividido em frentes independentes;
a integração, a revisão e a validação final aconteceram sobre a mesma branch.

- [x] Task 1: CDFs das oito distribuições e testes.
- [x] Task 2: Ljung–Box e ARCH–LM ponderados, caudas diretas e validação de argumentos.
- [x] Task 3: sign bias e Pearson ajustado.
- [x] Task 4: Nyblom calculado sobre os scores existentes, com exclusão de fixos/derivados.
- [x] Task 5: interface C, resultados Python e testes de borda.
- [x] Task 6: fixture dos 108 casos rugarch, testes e diferenças documentadas.
- [x] Task 7: cinco repetições dos benchmarks, comparação controlada e documentação.

Validação final: 3.557 testes C++ Release passaram; Python teve 1.208 passes, 9 casos
excluídos pelo filtro prescrito e 12 xfails strict (cinco preexistentes e sete comparações
em dois casos eGARCH, por aproximação de E|z| do rugarch). `check_style.py` passou.
O `mypy --strict` mantém os mesmos seis erros do commit anterior, confirmados sobre uma
cópia dos arquivos originais; nenhum erro novo. Nenhuma constante ou tolerância de teste
existente foi alterada. Os resultados anteriores do ajuste foram preservados.

A comparação de Nyblom usa os parâmetros estimados no ponto do oráculo, aceito antes que
o otimizador os desloque (`tol=1` apenas nesse teste), e verifica sua identidade numérica.
As tolerâncias estatísticas originais foram mantidas. As exceções eGARCH, os máximos
observados e suas causas estão em `docs/guide/validation.md`. O benchmark controlado
não distinguiu desaceleração do ruído: 5,72 ms antes e 5,62 ms com Nyblom (medianas CPU).

Os passos de implementação e validação acima estão concluídos; a sequência operacional
foi adaptada ao trabalho paralelo, sem repetir artificialmente cada falha de compilação
descrita no roteiro. Os commits seguem as sete unidades do plano.
