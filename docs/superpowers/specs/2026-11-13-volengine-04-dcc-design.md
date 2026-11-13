# VolEngine — plano 4: família DCC (design)

Data: 2026-11-13. Detalha o plano 4 do roadmap: os objetivos 2 e 3 (a parte DCC) e a seção 3, "Multivariado", da
spec-mãe `docs/superpowers/specs/2026-11-10-volengine-design.md`. Tudo o que a spec-mãe fixa vale aqui sem repetir:
- regras de estabilidade da interface C;
- handles imutáveis;
- bit-identidade entre números de threads;
- oráculos e licença (fórmulas de paper e da documentação, nunca código GPL do `rmgarch` copiado ou traduzido);
- estilo.

## 1. Intenção e o porquê

No TCC, o autor ajustava os modelos univariados de ~20 setores e do índice de mercado, todos no mesmo intervalo de
datas (T ≈ 5.000). Sobre esses resíduos, estimava **um DCC com todas as séries juntas** (N ≈ 21) para cada
variante (DCC, aDCC, FDCC) × distribuição multivariada. O melhor modelo era escolhido pelo **BIC**, e o autor
olhava também **t e p-valores** dos parâmetros. As correlações condicionais Rₜ (e Hₜ) do modelo escolhido serviam
para separar a volatilidade de mercado da volatilidade específica de cada setor. Essa conta ele fazia por fora,
sobre Rₜ/Hₜ.

O que mais importa, portanto:
- **a grade inteira de specs DCC sobre o mesmo conjunto univariado roda em segundos;**
- ℓ, critérios, erros-padrão, t e p-valores batem com o `rmgarch`;
- Rₜ e Hₜ saem como visões zero-copy.

Copula-DCC, forecast e simulação do DCC o autor não usava. Eles entram para a biblioteca ficar completa, como
pede a spec-mãe, mas depois do núcleo.

## 2. Divisão

Em três subplanos, na ordem de uso (como o plano 3):

- **4a — núcleo DCC:**
  - DCC, aDCC e FDCC com ordens (a, b);
  - distribuições mvnorm, mvt e mvlaplace;
  - lote de specs;
  - ℓ, critérios, erros-padrão em dois estágios, t e p-valores;
  - Rₜ e Hₜ;
  - oráculo `rmgarch` e benchmarks.
- **4b — Copula-DCC:**
  - cópulas mvnorm e mvt, com correlação estática, DCC e aDCC;
  - marginais paramétricas e empíricas.
- **4c — forecast e simulação do DCC (e da Copula-DCC).**

Cada subplano fecha com oráculo, documentação em `docs/guide/validation.md` e benchmarks.

## 3. Objetivos

### 4a — Núcleo DCC

- **Entrada:** N ajustes univariados **escolhidos pelo usuário**, um por série, cada um com a spec que quiser. São
  os `ve_fit` / `Fit` já feitos, de qualquer modelo, distribuição ou opção dos planos 2 e 3. A biblioteca nunca
  escolhe os modelos univariados.
  - Todos precisam cobrir o mesmo intervalo, com o mesmo T. Caso contrário, é erro de argumento (spec-mãe, painel
    regular).
  - Um ajuste sem ponto finito também é erro de argumento.
- **Passo 2:** o DCC sobre os resíduos padronizados zₜ dos univariados.
  - Modelos: DCC, aDCC e FDCC (no FDCC, o usuário dá um vetor de grupos por série), com ordens (a, b) como no
    `rmgarch`.
  - Distribuições: mvnorm, mvt (com forma estimada) e mvlaplace.
  - **FDCC entra com as três distribuições (decisão do autor, 2026-11-13).** O `rmgarch` 1.4.3 só aceita FDCC com
    a normal, então FDCC-mvt e FDCC-mvlaplace não têm oráculo. A validação delas é interna:
    - com um único grupo, o FDCC é igual ao DCC da mesma distribuição;
    - a ℓ é conferida contra uma avaliação direta da densidade.

    `validation.md` registra isso.
  - Valores fixos e iniciais, como nos univariados (paridade com `fixed.pars` / `start.pars`).
- **Lote:** várias specs DCC sobre o mesmo conjunto univariado numa chamada. O trabalho que só depende dos resíduos
  é compartilhado, e as threads do `ve_context` dividem as specs. Cada ajuste tem o próprio status; falha de
  convergência de um não falha o lote.
- **Saídas por ajuste DCC:**
  - parâmetros **só do modelo DCC** (a, b, assimetria, grupos do FDCC, forma da mvt), com erro-padrão, t e p-valor,
    pela Hessiana e robusto (decisão do autor, 2026-11-13). Os coeficientes univariados ficam nos `Fit` de origem,
    e o `DccFit` não os repete. Os erros-padrão dessas linhas são os de dois estágios do `rmgarch`, que corrigem
    pela estimação do primeiro estágio;
  - ℓ e critérios (AIC, BIC, HQIC, SIC) na convenção do `rmgarch`;
  - Rₜ e Hₜ;
  - status de convergência.
- **Erros-padrão em dois estágios:** a Hessiana conjunta da spec-mãe, nas linhas do DCC.
  - O bloco cruzado (a derivada do score do DCC em relação aos parâmetros univariados) exige refiltrar cada
    univariado em pontos perturbados. Para isso o cálculo precisa, de cada série, dos retornos, regressores, spec e
    estimativa do ajuste univariado.
  - Nenhum número de ajuste univariado existente muda.
- **Erros-padrão por opção e por função explícita** (decisão do autor, 2026-11-13):
  - **Opção ligada no ajuste DCC** (equivale ao `eval.se` do `rmgarch`): o cálculo acontece no ajuste, sobre os
    `Fit` univariados que o ajuste já recebe, e os erros-padrão ficam guardados no `DccFit`.
  - **Opção desligada:** o `DccFit` fica leve, com estimativas, ℓ, critérios, Rₜ e Hₜ. Ele não guarda dados dos
    univariados, e os acessores de erro-padrão, t e p-valor dão NaN.
  - **Depois do ajuste:** uma função explícita recebe o `DccFit` e os mesmos N `Fit` univariados do ajuste e
    devolve os erros-padrão, t e p-valores (Hessiana e robusto). Ela não altera o `DccFit`, que continua imutável.
    - Confere que os `Fit` são os do ajuste (N, T e as estimativas univariadas, que o `DccFit` guarda como
      identificação). Caso contrário, dá erro de argumento.
    - Python: por exemplo `d.coef(univariate=fits)`; nome final do builder.
  - O `DccFit` continua válido depois que os `Fit` de origem são liberados. Só a função explícita precisa deles.
  - A meta "a grade em segundos" vale para estimativas, ℓ e critérios, com a opção desligada. O custo dos
    erros-padrão é medido à parte no benchmark.
- **Layout de Rₜ e Hₜ:** T × N × N, com Rₜ contíguo por instante. Para N = 21 e T = 5.000 são ~17,6 MB por matriz
  por ajuste. Numa grade de dezenas de specs isso pesa, então o builder pode materializar Rₜ/Hₜ só quando pedidos,
  desde que a visão continue zero-copy e válida enquanto o handle existir.

### 4b — Copula-DCC

Como o `cgarchspec` do `rmgarch`:
- cópulas mvnorm e mvt;
- correlação estática (com os métodos de estimação que o `rmgarch` oferece para ela), DCC e aDCC;
- transformações das marginais paramétrica (pela CDF de cada ajuste univariado, a do plano 3c) e empírica.

A transformação `spd` fica fora (spec-mãe). Saídas e lote iguais às do 4a.

### 4c — Forecast e simulação do DCC

- Forecast n passos de Rₜ₊ₕ e Hₜ₊ₕ, como o `dccforecast` (`n.roll = 0`), sobre os forecasts univariados do 3a.
- Simulação de caminhos multivariados, como o `dccsim`, a partir do ajuste. Usa o mesmo Philox, a mesma
  bit-identidade e a mesma regra de semente do 3b (reprodutível só dentro da mesma versão).
- O mesmo para a Copula-DCC.
- Interface no padrão do 3a/3b: `d.forecast(h)` e `ctx.simulate(d, ...)`.

## 4. API (padrão dos planos anteriores)

- **Python** (spec-mãe, seção 7):
  - `d = ctx.fit_dcc(dcc_spec, univariate)` devolve um `DccFit`; uma lista de specs devolve uma lista de `DccFit`;
  - `univariate` é a sequência de N `Fit`;
  - `ve.DccSpec(model="dcc"|"adcc"|"fdcc", order=(a, b), dist="mvnorm"|"mvt"|"mvlaplace", groups=..., fixed=...,
    start=...)`;
  - o `DccFit` tem `coef()`, `loglik`, `aic()`, `bic()`, `hqic()`, `sic()`, `status` e `R` e `H` como arrays
    `(T, N, N)`.
- **C:** funções sobre handles (`ve_dcc_spec_*`, `ve_dcc_fit_batch`, acessores de `ve_dcc_fit`), com no máximo 5
  parâmetros por função e configuração nova como função nova. Os nomes exatos ficam com o builder.

## 5. Não-objetivos

- **Rotinas que separam a volatilidade de mercado da específica do setor.** O autor faz isso com numpy sobre
  Rₜ/Hₜ. Pode virar exemplo no README.
- **VAR na média, `robust`, `external.regressors` do `dccspec`** e o `realizedVol` do `dccfit`: a spec-mãe não
  os inclui.
- **Seleção automática de modelo.** O usuário compara o BIC; a biblioteca não escolhe.
- **Rolling** (`dccroll`), **GO-GARCH** e **a transformação `spd`**.
- **Diagnósticos multivariados:** o `show()` do `rmgarch` não os traz.

## 6. Anti-objetivos

- **DCC que muda com o número de threads.**
- **ℓ, critérios ou erros-padrão que divergem do `rmgarch` sem explicação documentada.** O BIC é o que o autor usa
  para escolher, e uma convenção diferente o faria escolher outro modelo.
- **Mudar qualquer número de um ajuste univariado existente.**
- **Ajuste DCC que deixa de valer quando os univariados são liberados.**

## 7. Restrições

- Tudo da spec-mãe, em particular:
  - só `double`;
  - sem Eigen;
  - Cholesky próprio (spec-mãe, seção de performance);
  - dependências pela política da seção 10.
- Escala-alvo do benchmark: N = 21, T = 5.000, a grade de 3 modelos × 3 distribuições (mais ordens, se o builder
  incluir) sobre um conjunto univariado. Medir com 1 thread e com todas, sem erros-padrão. Os erros-padrão de um
  ajuste ficam num benchmark separado.
- Oráculo: `rmgarch` 1.4.3 (instalado), em R 4.6.2.

## 8. Deixado ao builder

- Nomes na interface C e no Python, além do que a seção 4 fixa.
- O otimizador do passo 2 (o L-BFGS existente com restrições por reparametrização, como nos univariados). Também
  o tratamento de α + β < 1 e de outras restrições do `rmgarch` (`stationarity = TRUE`).
- Como calcular a Hessiana conjunta (diferenças finitas sobre a ℓ, AD ou partição em blocos), desde que bata com
  o `rmgarch`.
- Se Rₜ/Hₜ ficam materializados ou são calculados sob demanda.
- Casos do oráculo: subconjuntos dos dados de `tests/reference/` com N pequeno para os testes, e N = 21 para o
  benchmark.
- A divisão de tarefas dentro de cada subplano.

## 8a. Convenções do `rmgarch` que são bugs (decisão do autor, 2026-11-13)

Depois do plano 4a, as convenções do `rmgarch` foram cruzadas com os papers (Engle–Sheppard 2001, Cappiello–Engle–Sheppard 2006) e com o código do MFE e do ARCHModels.jl. Quatro delas são bugs ou desvios:
- a pré-amostra zero/uns;
- N̄ centrada;
- o sanduíche sem transposta;
- as restrições e o Q̄ duplo do FDCC.

**A convenção pública segue os papers.** A paridade com o `rmgarch` vira um interruptor interno do C++, que só os testes usam e que fica fora da interface C e do Python. Isso refina a seção 6: a ℓ, o BIC e os erros-padrão públicos divergem do `rmgarch` nesses quatro pontos, com explicação em `validation.md`. O detalhe está nas notas do plano 4a, seção 11.

## 9. Decisões tomadas pelo Claude (para conferir)

- Divisão em 4a núcleo, 4b Copula-DCC, 4c forecast/simulação, nessa ordem.
- Erros-padrão em dois estágios dentro do 4a, e não adiados, porque o autor olha t e p-valores.
- O padrão da opção de erros-padrão (ligada ou desligada) fica com o builder, documentado; o `DccFit` guarda as
  estimativas univariadas só como identificação para a função explícita.
- Critérios do DCC na convenção do `rmgarch`, que o oráculo confirma, para que a escolha pelo BIC seja a mesma.
- Forecast e simulação do DCC no padrão de API do 3a/3b.
