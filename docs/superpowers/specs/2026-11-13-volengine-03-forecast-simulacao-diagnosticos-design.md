# VolEngine — plano 3: forecast, simulação e diagnósticos (design)

Data: 2026-11-13. Detalha o plano 3 do roadmap (seção 2, objetivo 3, e seção 3, "Saídas por ajuste", de
`docs/superpowers/specs/2026-11-10-volengine-design.md`, a spec-mãe). Tudo o que a spec-mãe fixa (regras de
estabilidade da interface C, convenções, oráculos, licença dos oráculos, bit-identidade entre números de threads) vale
aqui sem repetir. Escopo só univariado: forecast e simulação do DCC são do plano 4.

## 1. Intenção e o porquê

O uso real do autor é risco de uma carteira tratada como série única: ele soma os valores da carteira, ajusta um
modelo univariado e olha o **σ previsto** para os próximos dias. Para um relatório informal de "possibilidades de
retorno", **simula caminhos** de até ~60 dias a partir do ajuste e tira percentis do retorno acumulado. Das saídas do
ajuste ele usa BIC, t e p-valores, que já existem.

Os **diagnósticos** do `rugarch` entram por paridade ("ter na biblioteca"), não por uso: precisam estar certos e
validados, sem esforço de interface além do mínimo.

Por isso o plano 3 se divide em três subplanos independentes, na ordem do uso (decisão do Claude, a conferir):

- **3a — forecast** analítico n passos à frente.
- **3b — simulação** de caminhos com Philox.
- **3c — diagnósticos** do `rugarch`.

Cada subplano fecha com oráculo, documentação em `docs/guide/validation.md` e benchmarks, como os planos 2a–2f.

## 2. Objetivos

### 3a — Forecast

- A partir de um ajuste (`ve_fit`), previsão para h = 1..H passos depois do fim da amostra: σₜ₊ₕ e a média
  condicional (série prevista), como o `ugarchforecast` com `n.roll = 0`.
- Todos os modelos de variância, ordens, ARMA, distribuições e opções da spec-mãe: fixos, valores iniciais,
  variance targeting, `mxreg`, `vxreg`, `archm`, ARFIMA. Onde a previsão de um modelo depende de momentos da
  distribuição (E|z|, E[z·|z|^δ] etc. em eGARCH, apARCH, fGARCH, gjrGARCH), usar os mesmos momentos que o
  ajuste já calcula.
- Com regressores, o usuário fornece os valores futuros: matriz H×k de `mxreg` e H×k de `vxreg` (o
  `external.forecasts` do `rugarch`). Spec com regressores e sem valores futuros é erro de argumento.
- Forecast "a partir de uma especificação" não precisa de caminho próprio: um ajuste com todos os parâmetros fixos já
  é o `ugarchfilter` (plano 2d), e o forecast parte dele.
- Resultado imutável (`ve_forecast`), legível de várias threads, com visões zero-copy, como `ve_fit`.

### 3b — Simulação

- **A partir de um ajuste:** caminhos que continuam do fim da amostra (o `ugarchsim` com `startMethod =
  "sample"`): retornos e σ, P caminhos × H passos.
- **A partir de uma especificação com todos os parâmetros fixos:** caminhos a partir do estado incondicional (o
  `ugarchpath`), como a spec-mãe prevê ("parâmetros fixos, necessários para simular a partir de uma
  especificação").
- Um ajuste (ou uma spec) por chamada. **Sem grade de simulações**: o autor simula uma carteira por vez. As threads do
  `ve_context` dividem os caminhos.
- RNG Philox próprio, semente `uint64`. Resultado **bit-idêntico para qualquer número de threads** na mesma máquina e
  ISA (spec-mãe, seção 8). A semente reproduz os mesmos caminhos **só dentro da mesma versão** da biblioteca
  (decisão do autor, 2026-11-13): Philox, derivação dos fluxos e amostradores podem mudar entre versões (por
  exemplo, amostradores mais rápidos no plano 5), sem fixtures bit a bit entre versões; o CHANGELOG avisa quando os
  caminhos de uma semente mudam. Cada caminho tem seu próprio fluxo, derivado da semente e do índice do caminho, então um
  amostrador com rejeição (consumo variável) não desloca os outros caminhos.
- Amostradores para as 8 distribuições padronizadas (norm, std, sstd, ged, sged, nig, ghyp, jsu), na mesma
  parametrização do ajuste.
- Regressores na simulação: uma matriz H×k de valores futuros de `mxreg`/`vxreg`, a mesma para todos os caminhos
  (decisão do Claude; o `rugarch` aceita uma por caminho, o que ninguém aqui pediu).
- Resultado imutável (`ve_sim`) com visões zero-copy, layout caminho-major (P × H, linha = caminho), para o numpy
  ver `(paths, horizon)` sem cópia.

### 3c — Diagnósticos

Funções sobre o ajuste, como os critérios (spec-mãe, seção 3: "funções sobre o resultado, não campos de struct"),
com os testes que o `show()` de um ajuste do `rugarch` reporta:

- Ljung-Box ponderado (Fisher–Gallagher) dos resíduos padronizados e dos seus quadrados.
- ARCH-LM ponderado.
- Nyblom: estatística individual por parâmetro e conjunta, com os valores críticos assintóticos que o `rugarch`
  reporta.
- Sign bias: sign, negative, positive e joint (t ou χ² e p-valor).
- Adjusted Pearson goodness-of-fit, com o número de grupos como argumento (padrões do `rugarch`: 20, 30, 40, 50).
  Exige a CDF de cada uma das 8 distribuições.

Seguir as versões **ponderadas** que o `rugarch` reporta (decisão do Claude, para que a comparação com o oráculo
seja direta); defasagens como argumento, com os padrões do `rugarch`. As fórmulas vêm de paper e documentação, nunca
do código GPL do `rugarch` (spec-mãe, seção 12).

## 3. API Python (decisão do autor, 2026-11-13)

- Forecast é membro do ajuste: `fc = f.forecast(10)` (com `mxreg=`/`vxreg=` quando a spec os tem). Devolve um
  `Forecast` com `sigma` e `mean` como arrays numpy de tamanho H.
- Simulação é do contexto, porque usa threads: `sims = ctx.simulate(f, horizon=60, paths=10_000, seed=42)`. Aceita
  um `Fit` ou uma `Spec` com todos os parâmetros fixos. Devolve uma `Simulation` com `returns` e `sigma` de forma
  `(paths, horizon)`.
- Diagnósticos são membros do `Fit`: `f.ljung_box(...)`, `f.arch_lm(...)`, `f.nyblom()`, `f.sign_bias()`,
  `f.gof(...)` (nomes finais do builder).
- Isso substitui o esboço `ve.forecast(f, ...)` / `ve.simulate(f, ...)` da seção 7 da spec-mãe, que deve ser
  atualizada.

Na interface C continuam funções sobre handles (`ve_forecast_create`, `ve_simulate`, `ve_ljung_box`...), com
nomes do builder dentro das regras da seção 5 da spec-mãe.

## 4. Não-objetivos

- Quantil, VaR e ES como funções da biblioteca. O autor olha só o σ previsto, e percentis de caminhos simulados
  saem do numpy. Se um dia fizer falta, entra como função nova sem quebra.
- Percentis, retorno acumulado ou qualquer estatística sobre os caminhos simulados: o usuário faz com numpy.
- Grade de forecasts ou de simulações em lote (várias specs × séries numa chamada). Um ajuste por chamada; quem quiser
  vários faz um loop.
- Rolling forecast (`n.roll`, `ugarchroll`): já é não-objetivo da spec-mãe.
- Bootstrap (`ugarchboot`).
- Forecast e simulação do DCC: plano 4.
- Diagnósticos fora do `show()` do `rugarch` (por exemplo, o Ljung-Box simples do `statsmodels`): só se o usuário
  pedir.

## 5. Anti-objetivos

- Simulação que muda com o número de threads.
- Forecast ou diagnóstico que diverge do `rugarch` sem explicação documentada em `docs/guide/validation.md`.
- Mudar o resultado de um ajuste existente: o plano 3 só lê `ve_fit`, nunca altera ajuste, verossimilhança ou
  erros-padrão.
- Copiar ou traduzir código do `rugarch`.

## 6. Restrições

- Tudo da spec-mãe: C ABI sem structs públicas, handles imutáveis, só `double`, sem Eigen, dependências pela
  política da seção 10 (Boost.Math já está no projeto para Bessel K; usar outros headers dele, como função beta
  incompleta ou gama incompleta para as CDFs, segue a mesma política), estilo do repositório (`check_style.py`).
- Horizonte típico do autor: até ~60 passos; caminhos: milhares a dezenas de milhares.
- Um `ve_fit` já guarda o que o forecast precisa? Se não guarda (por exemplo, os últimos retornos, os valores
  finais da recursão de variância, os scores para o Nyblom), o builder acrescenta ao ajuste, contanto que isso não
  mude nenhum número existente nem pese em quem não usa (spec-mãe: resultados leves).

## 7. Validação

- **Forecast:** contra o `ugarchforecast` em casos dos grids existentes (`tests/reference/`), cobrindo cada modelo de
  variância, ARMA, regressores (com `external.forecasts`), `archm` e ARFIMA. O `arch` serve de segundo oráculo
  onde a previsão dele é analítica (sGARCH, gjrGARCH); onde é por simulação (eGARCH com h > 1), não.
- **Simulação:** não há como bater com o `rugarch` amostra a amostra (outro RNG). Validação estatística, com tolerância
  documentada: média simulada contra o forecast analítico do 3a, na escala em que cada modelo é linear (σ² em
  sGARCH, gjrGARCH, iGARCH, csGARCH; σᵟ em apARCH e fGARCH; log σ² em eGARCH só para h = 1, já que a previsão do
  `rugarch` para h > 1 é o plug-in, não E[log σ²] exato para todas as distribuições — o builder confirma a forma
  exata contra o oráculo); momentos e quantis de cada amostrador contra a
  distribuição (testes de aderência ficam fortes depois que o 3c traz as CDFs, então o builder decide a ordem ou usa
  momentos); e reprodutibilidade bit a bit com 1 e N threads (spec-mãe, seção 12).
- **Diagnósticos:** contra as funções do `rugarch` sobre os mesmos ajustes (os valores de `show()`: Ljung-Box e
  ARCH-LM ponderados, `nyblom`, `signbias`, `gof`).
- Benchmarks em cada subplano (preferência do autor desde o 2a): forecast por ajuste, simulação de 10.000 caminhos × 60
  passos com 1 thread e com todas, e o custo dos diagnósticos que forem caros (GOF com CDF de nig/ghyp).

## 8. Deixado ao builder

- Nomes exatos na interface C e no Python, layout interno dos handles.
- Philox (variante, número de rodadas) e a derivação do fluxo por caminho, desde que atenda à bit-identidade entre
  números de threads.
- Algoritmos dos amostradores (por exemplo, GIG para nig/ghyp) e das CDFs (integração numérica para nig/ghyp).
- Estado inicial da simulação a partir de uma spec (o estado incondicional; para iGARCH, em que ele não existe, o
  builder decide entre exigir σ inicial ou recusar, documentando).
- Os momentos de distribuição que a previsão de eGARCH/apARCH/fGARCH usa, quando não houver forma fechada.
- Ordem dos subplanos, se o builder encontrar dependência que justifique trocar (por exemplo, CDFs antes da
  validação dos amostradores).

## 9. Decisões tomadas pelo Claude (para conferir)

- Divisão em 3a forecast, 3b simulação, 3c diagnósticos, nessa ordem.
- Simulação a partir de ajuste e a partir de spec com todos os parâmetros fixos; forecast só a partir de ajuste
  (o ajuste com tudo fixo cobre o caso "spec").
- Regressores futuros: uma matriz por chamada, a mesma para todos os caminhos.
- Diagnósticos nas versões ponderadas que o `rugarch` reporta.
- Layout dos caminhos simulados: P × H, caminho por linha.
- O σ previsto é a grandeza do `ugarchforecast`, em todos os modelos: em eGARCH, apARCH e fGARCH com h > 1 ela
  não é √E[σ²ₜ₊ₕ] (eGARCH prevê log σ², apARCH/fGARCH preveem σᵟ), e `docs/guide/validation.md` diz isso. Foi o σ
  que o autor usava no TCC; √E[σ²] exigiria momentos numéricos e divergiria do oráculo.
- Com a mesma semente, os caminhos de `paths = 100` são os 100 primeiros de `paths = 1000`, e os de `horizon = 60`
  são o começo dos de `horizon = 120` (fluxo próprio por caminho, consumido em ordem de tempo).
