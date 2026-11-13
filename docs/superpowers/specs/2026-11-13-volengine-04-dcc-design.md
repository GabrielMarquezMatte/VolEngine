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

A transformação `spd` fica fora (spec-mãe). Saídas e lote iguais às do 4a. O detalhe está na seção 10.

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

## 10. Plano 4b — Copula-DCC (design, 2026-11-13)

### 10.1 Intenção

O autor não usa a Copula-DCC. Ela entra para a biblioteca ficar completa, como pede a spec-mãe, com paridade com o
`cgarchfit` do `rmgarch` onde ele não tem bug (regra da seção 8a). O sucesso é ter os mesmos contratos do 4a:
- bit-identidade entre números de threads;
- `DccFit` válido depois de liberar os univariados;
- lote com status por ajuste;
- Rₜ e Hₜ zero-copy.

### 10.2 Abordagens

- **A. Estender o `DccSpec` (escolhida pelo autor, 2026-11-13).** A cópula é uma opção do mesmo spec, sai no mesmo
  `ctx.fit_dcc`, `ve_dcc_fit_batch` e `DccFit`, e um lote pode misturar DCC e Copula-DCC. No C, o custo é um valor
  novo no enum de modelo e cerca de dois setters. Contra: algumas combinações inválidas só são recusadas em tempo de
  execução.
- **B. Tipos separados** (`CopulaSpec`, `ctx.fit_copula`, `CopulaFit`, handles C próprios). Descartada: a API fica
  mais explícita, mas a superfície C dobra (acessores repetidos) e DCC e cópula não entram na mesma grade.

### 10.3 O que o `rmgarch` 1.4.3 faz (sondado pelo Claude no dcc3, univariados sGARCH(1,1)-std)

- `likelihood(fit)` é a ℓ **conjunta**: a soma das ℓ univariadas mais a ℓ da cópula. Por exemplo, na mvnorm estática
  paramétrica, a soma univariada é −6426,188, o total −6122,956 e a parte da cópula +303,232.
- Não existe `infocriteria` para `cGARCHfit`.
- O `matcoef` traz as linhas univariadas, com erros-padrão conjuntos, mais as linhas `[Joint]`:
  - mvt estática por ML: `C1..C{m(m−1)/2}` e `mshape`;
  - mvt estática por Kendall: só `mshape`;
  - mvnorm estática: nenhuma linha com Kendall, `C1..C{m(m−1)/2}` com ML. A ℓ igual nos dois métodos no dcc3 foi
    coincidência de arredondamento na impressão (corrigido na Task 1 do plano 4b);
  - DCC e aDCC: `dcca`, `dccb`, `dccg` e `mshape`.
- A forma da mvt tem teto 50: o dcc3 estima 49,99999.
- O `Rt` é uma matriz única na estática e uma lista de T matrizes no DCC.
- **Identificado na Task 1 do plano 4b:** a mvnorm estática por ML estima os C (o R₁₂ = 0,4745106 que não batia);
  com "Kendall", usa cor(Φ⁻¹(u)). As convenções completas estão em
  `docs/superpowers/plans/2026-11-13-volengine-04b-rmgarch-copula-notes.md`.

### 10.4 Decisões

- **Spec** (forma fixada; nomes finais com o builder):
  - Python: `ve.DccSpec(model="static"|"dcc"|"adcc"|"fdcc", copula=None|"parametric"|"empirical",
    dist="mvnorm"|"mvt"|"mvlaplace", static_method=None|"kendall"|"ml", ...)`;
  - `copula=None` é exatamente o DCC do 4a: nenhum número do 4a muda;
  - `model="static"` só existe com cópula;
  - com cópula, só `static`, `dcc` e `adcc` e só `mvnorm` e `mvt`. O FDCC-cópula e a cópula mvlaplace ficam fora: o
    `rmgarch` não os tem e ninguém os pediu;
  - `static_method` vale para as duas cópulas estáticas, com padrão `kendall`, como no `rmgarch` (Task 1 do 4b);
  - C: um `VE_DCC_MODEL_STATIC` novo, um setter da transformação (nenhuma, paramétrica ou empírica) e um setter do
    método estático. Combinação inválida é `VE_INVALID_ARGUMENT` antes de qualquer ajuste, com mensagem que nomeia
    a combinação.
- **Marginais:**
  - paramétrica: uᵢₜ = Fᵢ(zᵢₜ), pela CDF da distribuição de cada ajuste univariado (plano 3c), com os parâmetros de
    forma estimados;
  - empírica: as pseudo-observações por ranks. A convenção pública é `rank/(T+1)` (Genest–Ghoudi–Rivest 1995). Se o
    `rmgarch` usar outra, ela é tratada pela regra da seção 8a: convenção documentada ou bug com interruptor interno.
- **ℓ:** a conjunta, como o `rmgarch` (soma das ℓ univariadas mais a ℓ da cópula). A soma univariada entra também
  com marginais empíricas.
- **Critérios:** o mesmo k do DCC (`validation.md`: parâmetros univariados estimados + parâmetros de correlação
  estimados + m(m−1)/2), para o BIC comparar DCC e Copula-DCC dentro do mesmo lote. A matriz de correlação conta
  **uma vez só**: os Cᵢⱼ da mvt estática por ML são os m(m−1)/2 e não somam de novo. Kendall e mvnorm estática também
  contam m(m−1)/2.
- **Tabela de coeficientes:** só as linhas da cópula, como no 4a: `dcca`, `dccb`, `dccg`, `mshape` e os Cᵢⱼ da
  estática por ML. A mvnorm por Kendall tem tabela vazia, e a mvt por Kendall só `mshape`.
  - Erros-padrão em dois estágios pelo mesmo mecanismo do 4a: opção no ajuste ou função explícita depois, com os
    mesmos N `Fit`.
  - Com marginais paramétricas, o bloco cruzado passa pela CDF de cada univariado.
- **Convenção pública:** os desvios da seção 8a que se aplicam à parte DCC da cópula (pré-amostra, N̄ sem centrar,
  sanduíche com transposta) seguem os papers. A paridade com o `rmgarch` fica no interruptor interno, só nos testes.
- **Rₜ e Hₜ:** sempre `(T, N, N)`, com Hₜ = DₜRₜDₜ pelos σ univariados. Na estática, a mesma matriz se repete em
  cada t. A materialização sob demanda do 4a continua permitida.
- **Lote:** a transformação para u (CDF ou ranks) é calculada uma vez por transformação e compartilhada entre as
  specs. As threads do `ve_context` dividem as specs, como no 4a.
- **Oráculo:** `cgarchfit` do `rmgarch` 1.4.3 com `solnp`:
  - estática mvnorm e mvt (Kendall e ML), DCC e aDCC, cada um com mvnorm e mvt e marginais paramétricas e
    empíricas, mais um caso com `mshape` fixo;
  - na convenção `Rmgarch`, nos testes C++, como no 4a;
  - na pública, no Python, com uma avaliação direta da ℓ.
- **Benchmark:** N = 21, T = 5.000, a grade static, dcc e adcc × mvnorm e mvt × paramétrica e empírica, com a mvt
  estática nos dois métodos. Com 1 thread e com todas, sem erros-padrão. Os erros-padrão de um ajuste ficam num
  benchmark separado, como no 4a.

### 10.5 Não-objetivos e anti-objetivos

- Não-objetivos: FDCC-cópula, cópula mvlaplace, `spd`, VAR, forma da mvt variando no tempo, forecast e simulação da
  cópula (estes vão para o 4c).
- Anti-objetivos: os do 4a, mais:
  - mudar qualquer número de um ajuste DCC do 4a;
  - um BIC de Copula-DCC que não é comparável com o do DCC no mesmo lote.

### 10.6 Deixado ao builder

- Nomes no C e no Python, incluindo os dos Cᵢⱼ (a ordem do `rmgarch` como referência).
- O estimador público da correlação da mvnorm estática e a identificação do que o `rmgarch` faz (10.3), registrados
  numa nota como `2026-11-13-volengine-04a-rmgarch-notes.md`.
- Como a correlação de Kendall, sin(πτ/2), é levada a uma matriz definida positiva quando não é, e a
  reparametrização dos Cᵢⱼ na ML estática (N = 21 dá 210 parâmetros).
- Os erros-padrão com marginais empíricas: reproduzir o `rmgarch` no interruptor interno, e na convenção pública
  escolher, com base em paper, entre a correção semiparamétrica (Chen–Fan 2006) e o mesmo cálculo do `rmgarch`, com
  a limitação documentada em `validation.md`.
- A escala dos escores da cópula sobre os quais corre a recursão: Φ⁻¹(u), e t_ν⁻¹(u) na mvt, que depende de ν e
  por isso é recalculado a cada avaliação. Fica pelos papers, conferido contra o oráculo.
- Em que momento cada combinação inválida é recusada (setter ou lote), desde que seja antes de qualquer ajuste.
- Casos exatos do oráculo e a divisão de tarefas.
