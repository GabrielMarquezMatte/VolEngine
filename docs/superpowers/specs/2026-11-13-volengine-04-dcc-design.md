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
  - **Opção desligada:** o `DccFit` fica leve, com estimativas, ℓ, critérios, Rₜ e Hₜ. Ele não guarda os dados que
    os erros-padrão exigem (retornos, regressores), e os acessores de erro-padrão, t e p-valor dão NaN. Emenda do
    4c (seção 11): ele guarda o estado final de forecast de cada série, como o `Fit` do 3a.
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

## 11. Plano 4c — forecast e simulação do DCC e da Copula-DCC (design, 2026-11-13)

### 11.1 Intenção

O autor não usa forecast nem simulação do DCC. Eles entram para a biblioteca ficar completa, como pede a spec-mãe.
O sucesso é ter os contratos do 3a/3b no multivariado:
- `d.forecast(h)` e `ctx.simulate(d, ...)` sobre qualquer `DccFit` do 4a e do 4b;
- o mesmo Philox, a mesma bit-identidade entre números de threads e a mesma regra de semente do 3b (reprodutível
  só dentro da mesma versão);
- paridade com o `dccforecast` do `rmgarch` onde ele existe e segue o paper (regra da seção 8a). O `rmgarch` é uma
  referência entre outras: paper e outras implementações vêm antes quando divergem (decisão do autor, 2026-11-13).

### 11.2 Abordagens

Onde fica o estado univariado de que forecast e simulação precisam (o `DccFit` do 4a guarda só z, σ e as
estimativas univariadas como identificação):
- **A. O `DccFit` guarda, no ajuste, o estado final de forecast de cada série (escolhida pelo autor, 2026-11-13).**
  Guarda, por série, o `ForecastState`, o layout e o θ, compartilhados entre as specs do lote via `shared_ptr`, como
  o painel. São as últimas L observações por série; o ARFIMA guarda o histórico inteiro (~0,8 MB para N = 21,
  T = 5.000, uma vez por lote). A API fica `d.forecast(h)`, como a seção 3 pede, e o `DccFit` continua válido sem
  os `Fit`.
- **B. Passar os `Fit` de novo** (`d.forecast(h, univariate=fits)`, com a checagem de `fits_problem`). Descartada:
  obriga a manter os `Fit` vivos e diverge da seção 3.

Não há conflito com os erros-padrão: eles refiltram a amostra inteira em θ perturbado e continuam pedindo os `Fit`;
forecast e simulação leem só o fim da amostra no θ estimado. A docstring diz isso.

### 11.3 O que as fontes fazem (sondado pelo Claude no dcc3, univariados sGARCH(1,1)-norm)

- **Engle–Sheppard (2001, §7)** comparam duas aproximações para h > 1, porque Eₜ[zz′] = Eₜ[R] e R = Q*⁻¹QQ*⁻¹ não
  se resolvem para frente: prever Q e normalizar, ou supor Q̄ ≈ R̄ e Eₜ[Q] ≈ Eₜ[R] e prever R direto,
  Eₜ[Rₜ₊ᵣ] = Σᵢ₌₀^{r−2} (1−a−b)R̄(a+b)ⁱ + (a+b)^{r−1}Rₜ₊₁. No Monte Carlo deles, a segunda tem menos viés e o
  erro quadrático médio praticamente igual, e os autores a recomendam.
- **`rmgarch` 1.4.3, `dccforecast`:** para h = 2..6 bate com a segunda fórmula com erro 0, com R̄ = Q̄ normalizado.
  Hₜ₊ₕ = DRD com D = diag(sigmaFor) dos univariados (erro 4e-16). Aceita h > 1 com DCC(2,1), FDCC e mvt. **Recusa
  aDCC com `n.ahead > 1`** ("asymmetric DCC specification only support n.ahead = 1 currently").
  - DCC(2,1), h ≥ 2: a mesma fórmula com a persistência total k = Σa + Σb, a partir de Rₜ₊₁ (erro 0). O termo
    a₂zₜzₜ′ já observado não entra depois do passo 1.
  - FDCC, h = 1: a recursão exata, mas com Q̄ = cov(z) e não o `Qbar` do ajuste (o Q̄ duplo da seção 8a).
  - FDCC, h ≥ 2: o Q previsto sai NA e o R não segue nem a fórmula de Hadamard (erro 1e-3) nem outra regra que o
    Claude identificou. É tratado como bug pela regra da seção 8a: sem oráculo para h ≥ 2.
- **No h = 1, o Qₜ₊₁ do `rmgarch` é a recursão exata** com Q_T, o z_T dos univariados e o Q̄ do ajuste (erro 0). A
  diferença de 5,6e-3 da primeira sonda vinha de usar `fit@mfit$stdresid`, que não é o z dos univariados.
- **O `rmgarch` não tem `cgarchforecast`**, só `cgarchsim`. O `dccsim` e o `cgarchsim` aceitam partir do ajuste
  ou de uma spec.

### 11.4 Decisões

- **Estado no `DccFit`:** abordagem A (11.2). Nenhum número de um ajuste do 4a ou do 4b muda.
- **Forecast** (`n.roll = 0`), h = 1..horizon depois da última observação:
  - univariados: o forecast do 3a de cada série, com as mesmas regras (o σ é o `sigmaFor` do 3a, que no eGARCH,
    apARCH e fGARCH não é √E[σ²] para h > 1, como no `rmgarch`);
  - h = 1: a recursão exata do modelo com zₜ, nₜ e Qₜ observados;
  - h > 1: a aproximação de Engle–Sheppard que prevê R direto (11.3), na convenção da spec. Nas ordens (a, b), a
    fórmula do (1,1) com a persistência total k = Σa + Σb, a partir de Rₜ₊₁, como o `rmgarch` (decisão do Claude,
    ao escrever o plano: o paper só trata o (1,1), e levar termo a termo misturaria zz′ observados, do espaço de Q,
    com a recursão em R; a diferença some depois de poucos passos);
  - **aDCC h > 1 (decisão do autor, 2026-11-13):** estende a mesma aproximação com Eₜ[nₜ₊ᵢnₜ₊ᵢ′] ≈ N̄. O −gN̄ e o
    gE[nn′] se cancelam do passo 2 em diante, e a persistência fica a + b. Sem oráculo; `validation.md` registra a
    aproximação e a validação interna (11.5);
  - FDCC: a mesma fórmula com (11′ − A − B) e (A + B) por elemento (Hadamard), como na recursão;
  - Copula-DCC: Rₜ₊ₕ das mesmas fórmulas sobre os escores da cópula; na estática, R fixo. Hₜ₊ₕ = DRD pelos σ
    univariados, como o Hₜ do 4b. Sem oráculo (11.3);
  - saídas: R e H `(horizon, N, N)`, σ e média `(horizon, N)`.
- **Simulação**, como o `dccsim`/`cgarchsim` a partir do ajuste:
  - em cada passo: R do modelo pelo estado do caminho, choque multivariado com essa correlação, e o passo
    univariado do 3b de cada série com o seu zᵢ (o 3b sorteia o choque dentro do passo; o builder separa o
    sorteio do passo);
  - choques: mvnorm pelo Cholesky próprio (spec-mãe), mvt e mvlaplace como misturas de escala da normal,
    padronizados para covariância R;
  - Copula-DCC: u da cópula (mvnorm ou mvt) e zᵢ = Fᵢ⁻¹(uᵢ) com a marginal do ajuste. Na paramétrica é a função
    quantil da distribuição univariada; na empírica é o quantil empírico dos zᵢ do ajuste. A recursão do caminho
    corre sobre os escores da cópula (Φ⁻¹(u) ou t_ν⁻¹(u), como no 4b), não sobre os zᵢ;
  - **função quantil das distribuições univariadas, interna** (decisão do Claude): forma fechada ou pela simétrica
    nas norm, std, ged, sstd, sged e jsu; inversão numérica da CDF do 3c na nig e na ghyp. Sem interface C nem
    Python, porque ninguém pediu VaR e seria superfície congelada sem uso;
  - o caminho p lê só `Stream(seed, p)`, e as N séries consomem dele numa ordem fixa, de modo que o resultado não
    depende do número de threads;
  - regressores futuros: as mesmas linhas para todos os caminhos, por série, como no 3b;
  - **saídas (decisão do autor, 2026-11-13):** retornos e σ `(paths, horizon, N)` sempre. R e H
    `(paths, horizon, N, N)` só com a opção pedida (Python `correlations=True`; no C, um setter novo em
    `ve_sim_options_*`). Sem ela, os acessores de R e H dão erro de argumento. Para N = 21, h = 60 e 1.000
    caminhos, são ~10 MB por saída sem a opção e ~212 MB por matriz com ela.
- **Regressores:** o estado já guarda os desvios sem o termo dos regressores. Por isso forecast e simulação aceitam
  séries com mxreg/vxreg, desde que recebam as linhas futuras de cada uma, ao contrário dos erros-padrão. Python:
  `mxreg=`/`vxreg=` como sequência de N entradas, `None` para a série sem regressor; o formato exato fica com o
  builder.
- **API:**
  - Python: `d.forecast(horizon, *, mxreg=None, vxreg=None) -> DccForecast` (R, H, sigma, mean) e
    `ctx.simulate(d, *, horizon, paths, seed, mxreg=None, vxreg=None, correlations=False) -> DccSimulation`
    (returns, sigma, R, H), como arrays zero-copy de leitura;
  - C: handles próprios para o forecast e a simulação do DCC, acessores por saída, no máximo 5 parâmetros por
    função, nomes com o builder;
  - erro de argumento quando o `DccFit` não tem ponto finito, quando faltam ou sobram regressores futuros e quando
    horizon ou paths são 0.

### 11.5 Validação e benchmark

- **Oráculo `dccforecast`** (`rmgarch` 1.4.3), nos testes C++ com o interruptor `Rmgarch`: DCC(1,1), DCC(2,1) e
  mvt com h = 1..H, e aDCC e FDCC com h = 1. Na convenção pública, no Python, a fórmula de 11.4 avaliada direto.
- **aDCC h > 1, FDCC h > 1 e Copula-DCC:** validação interna. O FDCC com um único grupo é igual ao DCC. O forecast de h = 1 é conferido contra a recursão; para h > 1, a
  média de Rₜ₊ₕ em muitos caminhos simulados fica perto do forecast, com folga, porque a aproximação tem viés
  pequeno. Com g = 0, o aDCC é igual ao DCC.
- **Simulação:** sem oráculo de caminhos (sementes diferentes). Testes:
  - bit-identidade entre números de threads;
  - a mesma semente dá os mesmos caminhos;
  - momentos: covariância empírica dos choques perto de R, média dos σ² simulados perto do forecast;
  - Copula-DCC: uniformidade dos uᵢ e marginais com a distribuição do ajuste;
  - função quantil: F(F⁻¹(u)) = u em todas as distribuições, inclusive nas formas extremas que o 3b já testa.
- **Benchmark:** N = 21, T = 5.000. Forecast h = 60 e simulação de 1.000 caminhos × 60 passos, com e sem
  `correlations`, com 1 thread e com todas, para DCC, aDCC e Copula-DCC mvt.

### 11.6 Não-objetivos e anti-objetivos

- **Não-objetivos:**
  - simulação a partir de uma spec com tudo fixo, porque exige N specs fixas e Q̄ e ninguém pediu;
  - `n.roll > 0` e `dccroll`;
  - `startMethod = "sample"`, `preQ`, `preZ` e afins;
  - função quantil, VaR ou densidade preditiva na interface pública.
- **Anti-objetivos:** os do 4a e do 4b, mais:
  - simulação que muda com o número de threads;
  - forecast que diverge do `dccforecast` sem explicação documentada nos casos que ele cobre;
  - mudar qualquer número de um `DccFit`, `Fit`, forecast ou simulação existentes.

### 11.7 Deixado ao builder

- Nomes no C e no Python, além do que 11.4 fixa.
- Como o passo univariado do 3b recebe o choque de fora sem mudar nenhum número do 3b.
- O quantil empírico (definição e interpolação), coerente com o `rank/(T+1)` da convenção pública.
- A inversão numérica da nig e da ghyp (bracket e tolerância), desde que bit-idêntica entre threads.
- Casos exatos do oráculo e a divisão de tarefas.
