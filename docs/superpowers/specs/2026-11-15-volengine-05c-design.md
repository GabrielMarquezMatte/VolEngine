# VolEngine — plano 5c: os outros pontos quentes (design)

Data: 2026-11-15. Continua `docs/superpowers/specs/2026-11-14-volengine-05-performance-design.md` (§3, item 5c). Tudo
o que aquela spec e a spec-mãe fixam vale aqui sem repetir:
- oráculos e tolerâncias;
- bit-identidade entre números de threads na mesma máquina e ISA;
- interface C sem assinatura nova;
- A/B na mesma sessão no `gcc-bench`;
- `STYLEGUIDE.md`.

## 1. Intenção e o porquê

A biblioteca é de **uso geral** (decisão do autor, 2026-11-15). Todo recurso exposto conta como carga real: cópulas,
nig/ghyp, ARFIMA e simulação, e não só o que o TCC do autor usa. Por isso o critério de §3 da spec do plano 5
("o que não aparecer em nenhuma carga real fica para depois") vale para todos os nove itens da lista. A ordem sai do
custo medido. Um item só fica para depois se o ganho medido for desprezível.

## 2. Divisão (decisão do autor: abordagem B)

Em três subplanos por subsistema, como 5a/5b. Cada um tem design curto, profile, A/B e oráculos verdes, e é escrito um
de cada vez.

- **5c-1, cópulas** (este documento, §3 em diante): o maior custo medido, 35,6 s num ajuste.
- **5c-2, distribuições:**
  - Bessel K sem x87 (quebra a bit-identidade, então entra mesmo sem ganho);
  - gradiente do ghyp em λ (o mesmo código);
  - quantil nig/ghyp por tabela;
  - precisão de `dist::shifted_moment` e `probability_below_zero` com ged/sged em ν 200..1000.
- **5c-3, o resto:**
  - erros-padrão do DCC no caminho tangente (1,35 s);
  - ARFIMA por FFT;
  - ziggurat na amostragem.

Abordagens consideradas e descartadas:
- **A, um plano com nove tasks:** design raso justo onde há escolhas reais;
- **C, as correções primeiro num 5c-0:** separaria o Bessel x87 do gradiente do ghyp, que mexem no mesmo arquivo.

## 3. 5c-1: o que a medida mostrou

`gcc-bench`, uma thread, m = 21, T = 5000, código em `4ed19d8`, uma execução:

| `BM_CopulaFit` | Tempo |
| --- | ---: |
| `static_mvnorm_kendall_param` | 3,79 s |
| `static_mvt_ml_param` | erro: "the DCC fit has no finite point" |
| `dcc_mvnorm_param` | 0,939 s |
| `dcc_mvt_param` | 35,6 s |

- **O ajuste interno está no `Dual`.** O perfil de ν (`src/dcc/copula/fit.cpp`, `profile_shape`) chama `fit_dcc` sobre
  o painel de scores, que é o objetivo DCC comum. Mas `has_fast_gradient` (`src/dcc/likelihood/gradient.cpp`) exclui
  `spec.copula != None`. Por isso a cópula DCC-mvnorm custa 0,94 s contra 0,145 s do DCC-mvnorm comum.
- **Os quantis não pesam.** 105 mil quantis t da Boost (`promote_double<false>`) custam ~30 ms por ν. Os ~35 pontos
  do Brent somam ~1 s dos 35,6 s. O resto são os ~35 ajustes internos de ~1 s.
- **Kendall** é O(T²) por par de séries: 210 entradas, ~3,8 s.
- **`static_mvt_ml_param` falha** desde antes do 5b (registrado em `benchmarks.md`). A causa não foi investigada.

A spec do plano 5 dizia que "o gradiente em ν elimina" o custo do perfil. A medida contradiz isso: o custo está no
ajuste interno lento, não no número de pontos.

## 4. 5c-1: objetivos

1. **Cópula DCC/aDCC/FDCC no gradiente por tangentes.** O ajuste interno de cada ponto do perfil (e o único ajuste,
   quando não há ν livre) passa a usar `ObjectiveGradient`. Espera-se que `dcc_mvnorm_param` fique perto do DCC
   comum (~0,15 s).
2. **Perfil de ν com partida a quente (decisão do autor: abordagem A).** O perfil de Brent continua, com o mesmo
   intervalo [4, 50], os mesmos 30 bits e as mesmas 100 iterações. Cada ajuste interno parte da estimativa do ponto já
   avaliado com o ν mais próximo, e o primeiro ponto parte do início padrão. Estimativa: de 35,6 s para 2–5 s.
3. **Kendall pelo algoritmo de Knight**, O(T log T) por par. As contagens são inteiras (concordantes, discordantes e
   empates em cada coluna), então τ_b sai **bit-idêntico** ao do laço O(T²). Os testes atuais devem passar sem
   tolerância nova.
4. **Corrigir `static_mvt_ml_param`.** Diagnosticar a causa primeiro e corrigir na causa. Se o defeito estiver no
   comportamento documentado da cópula estática t e não só no benchmark, ele ganha um teste.
5. **Medir o resto e registrar:**
   - `BM_CopulaStandardErrors`, `BM_CopulaGrid` e os sete `BM_CopulaFit`, com as 5 repetições adiadas do plano 4b
     (`benchmarks.md` marca "Pending");
   - o ML estático (`static_ml`).

   Otimizar algum desses só se o profile mostrar um ponto quente óbvio, com A/B.

## 5. Não-objetivos

- **Estimação conjunta de ν** com gradiente através de ε(ν) (abordagem B) e **Newton no perfil pelo teorema do
  envelope** (abordagem C). Pela estimativa de §3, B não ganha de A. C fica como próximo passo só se A deixar a
  cópula DCC-t muito acima do DCC-t comum: `ponytail:` no `profile_shape`, com os números.
- Mudar o intervalo, a precisão ou o critério do perfil.
- Interface C ou Python novas.
- Os itens de 5c-2 e 5c-3.

## 6. Anti-objetivos

- Uma estimativa de cópula que sai do oráculo `rmgarch` além das tolerâncias atuais.
- Um ν que muda por causa da partida a quente além da precisão do Brent. O máximo do perfil é o mesmo; só o caminho
  dos ajustes internos muda.
- Resultado que depende do número de threads ou de quais outras specs estão no lote. A partida a quente usa só os
  pontos do perfil da própria spec, numa ordem determinística.
- Ganho sem A/B registrado em `docs/performance/`.

## 7. Restrições

- **Valor bit a bit.** O valor de `ObjectiveGradient::evaluate` já é bit-idêntico a `objective<double>`. No painel de
  scores ele tem que continuar assim, e o gradiente tem que bater com o `Dual` a 1e-9 da maior componente, como
  no 5b. O teste do par (`DccGradient.MatchesDual`) ganha painéis de scores de cópula: mvnorm e mvt, paramétrica e
  empírica, nas duas convenções.
- **Oráculos `rmgarch`** das cópulas (plano 4b) verdes nas mesmas tolerâncias. Um xfail que passar sai do xfail.
- **Entre thread única e pool:** um ajuste de cópula num lote misto continua bit-idêntico ao mesmo ajuste sozinho.
- A/B na mesma sessão contra o build de `4ed19d8`, mais os benchmarks secundários do plano 5 (`BM_DccGrid` e os
  `BM_DccFit` não podem piorar).

## 8. Deixado ao builder

- **Como passar a partida a quente:** por `DccSpec::start` do `inner_spec` ou por um parâmetro interno do `fit_dcc`.
  O que for mais simples sem tocar a interface C.
- **"O ν mais próximo":** por |Δν| entre os pontos já avaliados. Empate vai para o de ν menor (determinístico).
- **Guardar o melhor ponto do perfil** para não reajustar o ν final em `fit_copula`. Hoje `point_at` roda de novo
  no ν que o Brent devolve. Entra se o resultado continuar o mesmo ajuste; senão, fica como está.
- **Ordem das tasks,** desde que:
  - a medição de base venha primeiro;
  - cada mudança venha com seu teste e seu A/B;
  - a correção do `static_mvt_ml` venha antes de qualquer número final da cópula estática.
- A forma do algoritmo de Knight (merge sort com contagem de inversões e os empates por blocos), em
  `src/dcc/copula/static_fit.cpp`.

## 9. Decisões tomadas pelo Claude (para conferir)

- A partida a quente vem do ponto com o ν mais próximo, não do último avaliado. O Brent alterna os lados do
  intervalo, e o mais próximo é a melhor partida.
- O ajuste com ν fixo (sem perfil) e as cópulas sem t também ganham o gradiente por tangentes. É o mesmo objetivo.
- A cópula estática (`DccModel::Static`) fica fora do gradiente por tangentes: ela não tem recursão. O ML estático
  só é medido (§4, item 5).
- O design de 5c-2 e 5c-3 é escrito depois do 5c-1, com os números em mãos.

## 10. 5c-1 feito

Medidas em `docs/performance/profile-5c.md` (5c-1 com cinco repetições, base com três, `gcc-bench`, uma thread,
m = 21, T = 5000; base em `6f7053d`).

| `BM_CopulaFit` | Base | 5c-1 |
| --- | ---: | ---: |
| `static_mvnorm_kendall_param` | 3,79 s | 0,160 s |
| `static_mvt_kendall_param` | erro | 0,894 s |
| `static_mvt_ml_param` | erro | 1,74 s |
| `dcc_mvnorm_param` | 0,945 s | 0,197 s |
| `dcc_mvt_param` | 36,4 s | 3,47 s |
| `adcc_mvt_param` | 36,6 s | 3,56 s |

O `BM_CopulaGrid` (14 specs) dá 19,8 s com uma thread e 4,53 s com todas; na base ele aborta no mesmo erro das
estáticas t. `BM_CopulaStandardErrors` fica em 4,4 s (sem mudança). `BM_DccFit`, `BM_DccGrid` e o conjunto rápido ficam
iguais dentro do ruído.

- **Causa da falha estática:** com séries independentes, o perfil de ν cresce até o teto e o Brent avalia ν = 50 exato,
  que é o melhor ponto. `fit_copula` aplicava a verificação de ν fixo fora de (4, 50) também ao ν do perfil, rejeitava
  o 50 e devolvia o ponto padrão (ℓ não finito). Agora a verificação vale só para ν fixo, e a cópula estática aceita o ν do perfil em todo o [4, 50]. As cópulas
  DCC e aDCC fixam ν num `fit_dcc` que o quer dentro de (4, 50): um ponto em 50 exato não tem ℓ finito e o máximo delas
  fica dentro do intervalo.
- **A estimativa mudou de lugar (decisão registrada):** o ν e os parâmetros das cópulas t se movem dentro da precisão do
  Brent (ν a 1e-4 relativo, ℓ a 1e-8 relativo do valor anterior); os oráculos `rmgarch` seguem verdes nas mesmas
  tolerâncias.
- **O custo restante da cópula t** é o perfil de 37 pontos (~0,1 s cada) e os quantis t (105 mil por ponto). O Newton no
  perfil pelo teorema do envelope (§5) é o próximo passo; o `ponytail:` do `profile_shape` aponta para ele.

Ficou para o autor (medido, não otimizado):
- `static_mvt_ml_param` calcula os quantis t duas vezes por ponto (no painel do ajuste e de novo no `copula_loglik`);
  passar o painel do ponto ao seu ℓ tira ~0,7 s dos 1,74 s.
- Os erros-padrão da cópula gastam 3,25 s dos 4,7 s em `inputs_at` (106 pontos perturbados). Só uma série muda por
  perturbação, e o painel de uniformes e de scores poderia recalcular só essa coluna.
- `DccFitResult` não guarda o número de iterações do ajuste; sem ele, as iterações médias do ajuste interno antes e
  depois da partida a quente não foram registradas (só os pontos do perfil: 38 → 37, e ~0,96 s → ~0,098 s por ponto).

## 11. 5c-2: distribuições

### 11.1 Intenção e o porquê

Os quatro itens de distribuições de §2, para uma biblioteca de uso geral. Duas promessas quebradas e dois custos:
- **Bit-identidade:** `dist/bessel.cpp` e `diagnostics/tails.cpp` chamam a Boost com a política padrão, que no GCC
  calcula em long double (x87); no MinGW as threads do pool rodam o x87 com outra precisão que a thread principal.
  Um ajuste nig, ghyp ou DCC-mvlaplace pode mudar com o número de threads.
- **Precisão:** `shifted_moment` e `probability_below_zero` erram com ged/sged em ν grande (1,106 contra o limite
  uniforme 1,0845 em ν = 1000, `tests/unit/sim/samplers_test.cpp`, `GedWithLargeShape`). O ν do ged não tem teto no
  ajuste (`shape = exp(x)`), e a persistência do apARCH/fGARCH usa esse momento.
- **Custo do ghyp:** o `Dual` com ordem livre faz 7 chamadas de K em long double (o valor, duas para a razão de ∂x e
  quatro do Richardson em λ).
- **Custo do quantil nig/ghyp:** ~290 µs por quantil na simulação de Copula-DCC com essas margens.

### 11.2 O que a medida mostrou

`gcc-bench`, uma thread, código em `baf5eb5`:

| Medida | Tempo |
| --- | ---: |
| `BM_LogBesselK/fixed_order` (5 mil argumentos) | 1,78 ms (356 ns por chamada) |
| `BM_LogBesselK/free_order` (`Dual`) | 25,9 ms (5,2 µs por chamada) |
| `BM_Loglik/sgarch11_ghyp/double` | 0,586 ms |
| `BM_Loglik/sgarch11_ghyp/dual` | 24,4 ms (41x o `double`) |
| `BM_FitOne/sgarch11_ghyp` | 1843 ms (nig: 86 ms) |

Spike descartável, contra a Boost em long double, v de −3,3 a 50 e x de 1e-8 a 700:

| Caminho | Por chamada | Erro relativo |
| --- | ---: | --- |
| Boost long double (hoje) | 765 ns | referência |
| Boost `double` (`promote_double<false>`) | 158 ns; ordem inteira 17,5 ns | ulps |
| Trapézio em K_v(x) = ½ ∫ exp(−x cosh t + v t) dt: valor, ∂x e ∂v numa soma | 296 ns (~39 nós) | log K ≤ 1,1e-14; ∂x ≤ 4e-15 |

O Richardson atual erra ∂v em até 3e-10 relativo com x pequeno (v = 0,1, x = 1e-8): o passo de 1e-3 encontra a
curvatura de log K em v. O trapézio não tem esse erro. Para o valor sozinho ele perde da Boost em `double`, por isso
só entra onde dá as derivadas.

### 11.3 Objetivos

1. **Toda chamada da Boost em `double`.** `promote_double<false>` no `log_bessel_k` e no `diagnostics/tails.cpp`
   (`gamma_q`, `ibeta`), e em qualquer outra chamada de função especial sem política explícita em `src/`. Os valores
   nig, ghyp e mvlaplace mudam por ulps; quem chama K fica ~5x mais rápido.
2. **Derivadas em ordem livre pelo trapézio (decisão do autor: abordagem B).** Os overloads `Dual` com ordem `T`
   (`log_bessel_k(const T&, const T&)` e o `_scaled`) tiram ∂x e ∂v de uma soma do trapézio:
   ∂x log K = −Σ cosh(t_k) w_k / Σ w_k, ∂v log K = Σ t_k w_k / Σ w_k. O valor continua o da Boost em `double`,
   igual ao do caminho `double`, então `objective<double>` e o valor do `Dual` seguem bit-idênticos. O Richardson
   (`order_slope`) sai. Os overloads de ordem fixa mantêm a razão K_(v−1)/K_v (mais barata em ordem inteira).
   Estimativa: `BM_LogBesselK/free_order` de 5,2 µs para ~0,5 µs, e o ajuste ghyp de 1843 ms para 200–400 ms.
3. **Quantil nig/ghyp por tabela e polimento (decisão do autor: opção 2).**
   - **Tabela:** cdf por margem numa grade em z, construída uma vez por chamada de simulação, acumulando segmentos
     como o `sorted_cdf`.
   - **Chute:** Hermite cúbico de z em u, com inclinação 1/f nos nós.
   - **Polimento:** um passo de Newton, com a cdf no chute = cdf do nó vizinho + um segmento curto de Gauss-Legendre.
   - **Fora da tabela** (u além da grade, nas caudas): o `solved_quantile` atual.
   - **Meta:** a precisão atual (~1e-14 relativo em z, `eps_tolerance(48)`), poucos µs por quantil.
4. **ged/sged em ν grande.** A densidade é quase plana até |z| ≈ λ 2^(1/ν) e cai numa borda de largura ~λ/ν. A
   quadratura ganha pontos de divisão nessas bordas (em cada lado do sged, com sua escala) e refinamentos na escala da
   borda, pela mesma máquina de `add_refinements`. A causa é confirmada antes por um teste vermelho; se for outra,
   corrige-se a causa.

### 11.4 Não-objetivos

- K próprio para valores: a Boost em `double` ganha do trapézio no valor.
- SIMD no trapézio: `ponytail:` com os números, se o ghyp ainda dominar algum benchmark.
- Mudar `kMaxHyperbolicShape`, a parametrização do ghyp ou o caminho de x ≥ 700 (Hankel e recorrência).
- Interface C ou Python novas.
- Os itens de 5c-3.

### 11.5 Anti-objetivos

- Ajuste nig, ghyp ou mvlaplace que mude com o número de threads (com a ISA pode, dentro da tolerância).
- Estimativa nig/ghyp fora dos oráculos `rugarch`/`rmgarch` além das tolerâncias atuais.
- Quantil por tabela menos preciso que o solve atual, ou tabela que dependa da ordem dos sorteios ou das threads.
- Ganho sem A/B registrado em `docs/performance/`.

### 11.6 Restrições

- **Módulos:** `src/dist/` já tem 8 módulos, o teto do `STYLEGUIDE.md`. O trapézio entra em `bessel.*`; a tabela em
  `cdf.*` ou no módulo de quem a usa. Nenhum módulo novo em `src/dist/`.
- **Testes:**
  - **Bit-identidade entre threads:** `DistQuantile.SameBitsOnAnyThread` ganha nig e ghyp, e um teste igual cobre
    `log_bessel_k` em ordem fracionária. Os dois ficam vermelhos antes do item 1, mostrando o defeito nesta máquina.
    Se não ficarem, registra-se isso e o item segue pela promessa.
  - **Trapézio:** ∂v e ∂x conferem com referência independente a ~1e-13 relativo, num domínio de v e x declarado no
    código. Referências: a forma fechada em v = ½ (`OrderDerivativeMatchesClosedFormAtHalf`) e diferenças em long
    double com passo pequeno, longe de x minúsculo. `Bessel.DerivativesMatchFiniteDifferences` segue verde.
  - **Quantil:** `quantile(cdf(z))` volta a z, e a tabela confere com o `solved_quantile` a ~1e-13 em nig e ghyp, com
    u nas caudas e no centro.
  - **ged/sged:** o ged com η₂ = 0 tem forma fechada ((1 − η₁)^p + (1 + η₁)^p)/2 E|z|^p, e o P(z < 0) do sged tem a
    cdf fechada do `skewed_cdf`. São os oráculos de ν = 2 a 1000. `GedWithLargeShape` volta a conferir `P(z < 0)` e
    o momento deslocado.
- **Oráculos `rugarch`/`rmgarch`** verdes nas mesmas tolerâncias: nig, ghyp, mvlaplace, cópulas com margens nig/ghyp,
  forecast e simulação.
- **A/B** na mesma sessão contra o build de `baf5eb5`:
  - `BM_LogBesselK`, `BM_Loglik` e `BM_FitOne` de nig e ghyp;
  - `BM_Diagnostics/sgarch11_ghyp/gof`;
  - `BM_DccLoglik/dcc_mvlaplace`;
  - um benchmark de quantil nig/ghyp (novo, se nenhum existir).

  O conjunto rápido não pode piorar.

### 11.7 Deixado ao builder

- **Passo e janela do trapézio.** O spike usou:
  - h = min(0,2; 0,4 σ), com σ = (x² + v²)^(−1/4);
  - a grade centrada no pico t* = asinh(|v|/x), com corte em exp(−40) do pico;
  - cosh(t_k) por recorrência de e^(t_k).

  O domínio validado vai declarado no código. Fora dele (|v| ou x além do testado) vale o mais simples e correto: o
  Richardson atual ou a mesma soma com mais nós.
- **Sinal de ∂v:** K é par em v, então ∂v log K_v = sign(v) D(|v|), com derivada 0 em v = 0.
- **A tabela:** o número de nós e a faixa em u; onde ela mora (`cdf.*` ou `dcc/future/simulate`); se o Newton
  precisa de um segundo passo; PCHIP no lugar do Hermite se este não for monótono.
- **Ordem das tasks,** desde que:
  - a medição de base venha primeiro;
  - os testes vermelhos de thread venham antes do item 1;
  - o item 1 venha antes do 2 (o 2 parte dos valores em `double`);
  - cada mudança venha com seu teste e seu A/B.

### 11.8 Decisões tomadas pelo Claude (para conferir)

- O valor de K segue da Boost, e o trapézio dá só as derivadas em ordem livre. É o refinamento de B depois do spike
  (o trapézio é 2x mais lento que a Boost em `double` no valor).
- A auditoria x87 cobre toda a `src/`, não só `bessel.cpp`: `diagnostics/tails.cpp` tem o mesmo defeito.
- O quantil fora da tabela (caudas) segue pelo solve, sem extrapolação inventada.
- O item ged/sged não teve escolha de abordagem: os pontos de divisão nas bordas seguem o padrão dos refinamentos
  existentes.

## 12. 5c-2 feito

Medidas em `docs/performance/profile-5c.md` (cinco repetições, `gcc-bench`, uma thread, base em `881e7a7`).

| Medida | Base | 5c-2 |
| --- | ---: | ---: |
| `BM_LogBesselK/free_order` | 28,9 ms | 2,20 ms |
| `BM_FitOne/sgarch11_ghyp` | 1785 ms | 141 ms |
| `BM_FitOne/sgarch11_nig` | 79,6 ms | 30,5 ms |
| `BM_Diagnostics/sgarch11_ghyp/gof` | 28,9 ms | 6,49 ms |
| quantil nig / ghyp por chamada | 291 / 496 us | 2,7 / 3,6 us (tabela) |

- **A Boost em `double`** foi o maior ganho de tempo (2 a 5x em tudo que chama K), além de ser a correção da
  bit-identidade. Os três testes de x87 (Bessel, p-valores de cauda, quantil nig/ghyp) falhavam antes. O `std::thread`
  do plano não reproduz o defeito (a thread herda a palavra de controle do x87): os testes alternam a precisão de 64
  para 53 bits na mesma thread (`tests/unit/x87_precision.hpp`).
- **O trapézio** dá ∂x e ∂v do log K numa soma (443 ns por chamada contra 5,2 us do Richardson antes e 1,08 us
  depois da Boost em `double`). Ele também é mais exato que o Richardson: o erro de ∂v contra a referência em long
  double passou de 1e-10 para 1e-12.
- **O quantil** parte da tabela e termina com Newton na própria cdf de quadratura, com as constantes que não dependem
  de z guardadas (`detail::CdfConstants`): o mesmo aritmético, os mesmos bits. O plano previa o polimento por um
  segmento curto; a cdf além de |z| = 1 é uma integral de cauda reescalada, não uma soma de segmentos, e o segmento
  não cairia na raiz do `quantile`. Ganho de 40x, não os 50 a 100x da estimativa (o denominador também caiu: a Boost em
  `double` levou o `quantile` de 291 para 109 us).
- **ged/sged em ν grande** tinha três defeitos, não um:
  - as bordas da densidade (3e-6 em ν = 20, 1,6% em ν = 1000);
  - a cdf do ged, que tomava `0,5 (|z|/escala)^ν` como 0 quando dá underflow dentro do suporte (P(z < 0) do sged de 0,61
    em vez de 0,5 de ν = 500 em diante). O oráculo do plano para P(z < 0) estava errado, e a integral numérica
    concorda com a quadratura;
  - a derivada em ν dos momentos, NaN de ν ~ 100 em diante (exp(−inf) vezes a inclinação infinita).

  Os três têm teste e estão corrigidos. Os valores do apARCH/fGARCH com ged/sged em ν acima de 6 mudam (para o valor
  certo).

Ficou para o autor:
- A soma do trapézio por SIMD (o ghyp está em 4,6x o nig; abaixo do 5x do critério).
- `solved_quantile` ainda recalcula as constantes da cdf a cada avaliação; passá-las a ele deixaria `quantile` ~3x
  mais rápido para quem chama uma vez só.
- A construção da tabela custa 1,4 a 1,9 ms por margem, em série; uma simulação com poucos caminhos e 21 margens paga
  ~35 ms.
- Os p-valores `student_two_sided` e `chi_squared_upper` mudam em ulps.

