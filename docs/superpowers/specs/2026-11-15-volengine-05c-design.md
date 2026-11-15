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
