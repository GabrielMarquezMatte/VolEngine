# VolEngine — documento de design

Data: 2026-11-10
Status: aprovado para planejamento após revisão

## 1. Intenção

VolEngine é uma biblioteca nativa para estimar modelos de volatilidade das famílias GARCH univariada e DCC,
chamável de qualquer linguagem por uma interface C estável.

O ponto de partida é um TCC sobre volatilidade setorial da bolsa: cerca de 20 séries setoriais com 5.000+
observações cada, estimadas em R com `rugarch`/`rmgarch` em todas as variantes, ordens p,q de 1 a 3 e todas as
distribuições, depois todos os modelos da família DCC. Mesmo com multiprocessamento, levava minutos. O objetivo é
fazer o mesmo trabalho, e cargas muito maiores, em segundos, a partir de Python primeiro e de outras linguagens
depois.

O que torna a biblioteca especial:

- **Estimação em lote como caso principal.** Muitos modelos sobre muitas séries numa só chamada, com
  multithreading e SIMD. A API de uma série é uma conveniência sobre o lote, não um segundo caminho.
- **Um único contrato estável: a interface C.** O autor teve muitas quebras de API no ExcelReader e não quer
  repetir isso. Tudo acima (wrapper C++, Python, futuros C#/R/Rust) consome essa interface; tudo abaixo (núcleo
  C++23) é interno e pode mudar livremente.
- **Controle total para otimizar.** Implementação própria onde nenhuma biblioteca faz o que é preciso ou onde
  há evidência medida de que a própria é mais rápida. Bibliotecas são aceitas enquanto forem ótimas.

## 2. Objetivos, não-objetivos e anti-objetivos

### Objetivos

1. Estimar por máxima verossimilhança os modelos da seção 3, em uma série ou em lote (lista de especificações ×
   painel de séries), devolvendo para cada ajuste: coeficientes com erro-padrão, estatística t e p-valor (matriz
   pela Hessiana e matriz robusta/sanduíche QML), log-verossimilhança, critérios de informação, diagnósticos,
   série ajustada (σₜ condicional, média condicional, resíduos, resíduos padronizados).
2. Estimar modelos DCC em lote: várias especificações DCC sobre o mesmo conjunto de ajustes univariados escolhidos
   pelo usuário.
3. Forecast n passos à frente e simulação de caminhos, univariado e DCC.
4. Resultados que batem com os oráculos (`rugarch`, `rmgarch`, `arch` do Python) dentro de tolerâncias
   documentadas.
5. Performance: lote de milhares de ajustes (escala do TCC) em segundos numa CPU desktop.
6. Interface C estável a partir da 1.0, consumida por Python (primeira linguagem) e por um wrapper C++23
   header-only.

### Não-objetivos da v1

- Stochastic Volatility (fica para depois; QML/Kalman, MCMC ou filtro de partículas é outro tipo de problema).
- GPU (fica para depois; o provedor — OpenCL, SYCL, CUDA — será escolhido com números dos kernels de CPU na mão).
- Rolling forecast / rolling estimation (`ugarchroll`). O usuário estima janela a janela e chama `forecast`.
- Bindings além de Python e C++ (C#, R, Rust vêm depois, sobre a mesma interface C).
- Precisão `float`.

### Anti-objetivos (falhas mesmo que "funcione")

- **Quebrar a interface C depois da 1.0.** Nenhuma assinatura muda, nenhum valor de enum é removido ou
  renumerado, nenhuma semântica documentada muda.
- **Divergir dos oráculos sem explicação documentada.**
- **Fazer mais do que o necessário.** A biblioteca não assume responsabilidades que o usuário resolve com o que ela
  já entrega. Exemplos: não há rolling forecast (o usuário compõe `fit` + `forecast`); não há seleção de modelo
  (`best()`), porque o critério é do usuário e um `argmin` sobre `bic()` resolve; não há leitura de arquivos nem
  cálculo de retornos a partir de preços. Antes de adicionar qualquer função, a pergunta é: "o usuário consegue
  fazer isso com o que já existe?" Se sim, não entra.

## 3. Escopo de modelos (v1)

### Univariado (paridade com `rugarch`)

- Variância: sGARCH, eGARCH, gjrGARCH, apARCH, iGARCH, csGARCH, fGARCH (com suas submodelos), ordens p,q ≥ 1.
- Média: ARMA(m,n), com ou sem constante.
- Distribuições: norm, std, sstd, ged, sged, nig, ghyp, jsu.
- Opções de especificação do `rugarch` que mudam o modelo entram: parâmetros fixos (necessários para simular a
  partir de uma especificação), valores iniciais (úteis para o usuário que faz estimação rolling com warm-start),
  variance targeting, regressores externos na média e na variância (`mxreg`/`vxreg`), ARCH-in-mean
  (`archm`/`archpow`) e ARFIMA. Os regressores fazem parte do contrato de entrada desde o início (matrizes T×k
  com strides por trabalho do lote), porque não dá para acrescentá-los depois da 1.0 sem quebra. Opções de solver
  do `rugarch` (solnp etc.) não entram.

### Multivariado (paridade com `rmgarch`)

- DCC, aDCC, FDCC, com ordens (a, b). No FDCC, o usuário fornece um vetor de grupo por série.
- Distribuições: mvnorm, mvt, mvlaplace.
- Copula-DCC (decisão do Claude): cópulas mvnorm e mvt (as que o `rmgarch` suporta; mvlaplace não), correlação
  estática, DCC e aDCC; transformações das marginais paramétrica e empírica. A transformação semiparamétrica
  (`spd`) fica fora da v1 — exige kernel + caudas GPD, que é outra biblioteca.
- Erros-padrão do DCC: paridade com o `rmgarch`, ou seja, Hessiana conjunta em dois estágios (que usa scores e
  Hessianas dos univariados), não só do passo 2. Por isso o `ve_dcc_fit` copia dos ajustes univariados o que
  precisa (resíduos, scores, Hessianas) e continua válido depois que eles são liberados. Os pontos estimados do
  passo 2 continuam dependendo só dos resíduos, e o lote de especificações DCC compartilha esse trabalho.
- Estimação em dois passos: passo 1 são os ajustes univariados **escolhidos pelo usuário** (um por série, cada um
  com a especificação que quiser); passo 2 é o DCC sobre os resíduos padronizados. Como o passo 2 só depende dos
  resíduos, várias especificações DCC sobre o mesmo conjunto univariado rodam em lote. A biblioteca nunca escolhe
  os modelos univariados.

### Saídas por ajuste

- Coeficientes: estimativa, erro-padrão, t, p-valor — matriz pela Hessiana e matriz robusta.
- Log-verossimilhança, AIC, BIC, HQIC (e SIC, se o `rugarch` o reporta separadamente).
- Diagnósticos do `rugarch`: Ljung-Box (resíduos padronizados e ao quadrado), ARCH-LM, Nyblom, sign bias,
  adjusted Pearson goodness-of-fit.
- Série ajustada: σₜ, média condicional, resíduos, resíduos padronizados; no DCC, Rₜ e Hₜ.
- Status de convergência por ajuste.

**Critérios e diagnósticos são funções sobre o resultado, não campos de struct.** Os resultados ficam leves (sem
inflar cache lines com dados que nem todo usuário consulta), e adicionar um critério novo é adicionar uma função —
nunca quebra nada.

## 4. Arquitetura

```
  usuário C++ ───► volengine.hpp  (C++23 header-only: expected, span, mdspan)
  usuário Python ─► volengine (ctypes, numpy zero-copy)
  futuros C#/R/Rust ┐
                    ▼
            interface C  (include/volengine/volengine.h)   ◄── ÚNICO contrato estável
                    ▼
            núcleo C++23 (src/, interno; muda à vontade)
```

- O núcleo é C++23 idiomático (`std::expected`, `std::span`, `std::mdspan`, `std::print`, deducing this, templates
  para despacho estático entre modelos). O styleguide é de C++ moderno, não de "C++ estilo C".
- A interface C nasce junto com a primeira API do núcleo e cresce com ela. O Python é o primeiro consumidor real e
  a valida desde o início.
- O wrapper C++ (`volengine.hpp`) é header-only sobre a interface C, como `cpp/include/xl` no ExcelReader. Quem
  usa C++ não acessa os templates internos (não pode injetar modelos próprios) — custo aceito em troca de um só
  contrato.
- `mdspan`: o MSVC (VS 2022 17.9+) e o libc++ têm; o libstdc++ só a partir do GCC 15. CI usa GCC 15+; se for
  preciso suportar GCC anterior, usa-se a implementação de referência do Kokkos como fallback.

## 5. Regras de estabilidade da interface C

Cada regra vem de uma quebra concreta do ExcelReader (guias `migrating-to-v6.md`, `migrating-to-v7.md`) ou do
FixedWidthParser (`CHANGELOG.md` 2.0.0):

| Quebra que aconteceu | Regra no VolEngine |
|---|---|
| ExcelReader v5→v6: o handle tinha um "sheet atual" (`MoveToSheet`), estado escondido que impedia paralelismo | Resultados (`ve_fit`, `ve_dcc_fit`, `ve_forecast`, `ve_sim`) são imutáveis e podem ser lidos de várias threads. O `ve_context` aceita chamadas concorrentes. Uma especificação é copiada no momento do ajuste: mudá-la depois não afeta resultados existentes. |
| v6→v7: parâmetros posicionais soltos (`headerRow`, `isDate1904, provider`) e pares `xl_open_file`/`xl_open_file_ex` | Nenhuma struct pública. Toda configuração entra por setters sobre handles opacos (`ve_spec_set_dist`, `ve_fit_options_set_max_iter`). Configuração nova = função nova. Nunca parâmetro novo, nunca `_ex`. |
| v6→v7: `CsvParallel` renomeado para `SheetParallel` ao generalizar | Nomes pelo conceito: `fit`, `forecast`, `simulate`, valem para todas as famílias. O modelo é um valor de enum (`VE_MODEL_EGARCH`), não uma função própria. Modelo novo = valor de enum novo, sem API nova. |
| v7: sistema de datas 1900 assumido em silêncio | Toda convenção em que os oráculos divergem é uma opção explícita com padrão documentado (seção 9). |
| v7: `CellType.Formula` removido | Enums só crescem por acréscimo; nenhum valor é removido ou renumerado. |
| FixedWidthParser 2.0: dois caminhos paralelos (char e UTF-8) duplicando tudo | Um caminho só: uma série é um lote de tamanho 1 na interface C. Sobrecargas de conveniência existem só nos wrappers. |
| FixedWidthParser 2.0: `ColumnParserRegistry` (estado global mutável) e tipos públicos sem uso público | Sem estado global: o thread pool e a configuração de execução vivem no `ve_context`, passado a cada chamada. Superfície pública mínima. |
| ExcelReader v6: cursor continua válido depois de `xl_close` | Handles independentes: liberar um nunca invalida outro (um `ve_dcc_fit` continua válido depois de liberar os `ve_fit` univariados que o originaram). |

Outras regras:

- Prefixo `ve_` para funções, `VE_` para enums/macros. Tipos de tamanho fixo (`int32_t`, `size_t`, `double`).
- Toda função retorna `ve_status` (enum que só cresce). Mensagem detalhada por `ve_last_error()`, thread-local.
- Lote = lista de trabalhos (índice da especificação, índice da série) sobre uma lista de especificações e um
  painel (decisão do Claude). Cobre o produto cartesiano especificações × séries e o modo pareado (especificação i
  na série i, por exemplo warm-start por série) com uma única função. Os wrappers montam o produto cartesiano.
- Num lote, cada ajuste tem o próprio status (`ve_fit_status`): falha de convergência de um ajuste não falha o
  lote. Erro de argumento (ponteiro nulo, dimensões inválidas, NaN nos dados) falha a chamada.
- Dados de entrada: ponteiro + T + N + strides (em elementos), para aceitar numpy em ordem C ou Fortran sem cópia.
- Painéis irregulares no univariado: dois arrays opcionais, início e comprimento por série (`NULL` = painel
  retangular). Internamente, ajustes da mesma especificação são ordenados por comprimento antes de virar faixas
  SIMD; faixas que terminam antes ficam mascaradas.
- DCC exige painel regular: todos os ajustes univariados de entrada cobrem o mesmo intervalo; caso contrário, erro
  de argumento.
- NaN no meio de uma série é erro de argumento. Alinhar datas é responsabilidade do usuário.
- Séries ajustadas saem como visões `const double*` + comprimento, válidas enquanto o handle existir (zero-copy
  para numpy).
- Parâmetros por índice: `ve_fit_param_count`, `ve_fit_param_name(i)`.
- Versionamento: semver; `VE_ABI_VERSION` no header e `ve_abi_version()` em runtime; abidiff no CI contra a última
  release a partir da 1.0.
- O contrato congela na 1.0, e a 1.0 só sai com o escopo inteiro da v1 (decisão do Claude): toda a superfície de
  DCC, Copula, forecast, simulação e diagnósticos é desenhada e exercitada antes de congelar. Até lá, as versões
  0.x podem quebrar.

## 6. Esboço da interface C

Ilustrativo — os nomes exatos são do builder, as regras da seção 5 não.

```c
ve_context* ctx;   ve_context_create(&ctx);
ve_context_set_threads(ctx, 0);                     /* 0 = todos os núcleos */

ve_spec* s;        ve_spec_create(VE_MODEL_EGARCH, &s);
ve_spec_set_order(s, 1, 1);
ve_spec_set_arma(s, 1, 0);
ve_spec_set_dist(s, VE_DIST_SSTD);

ve_fit_options* o; ve_fit_options_create(&o);
ve_fit_options_set_max_iter(o, 500);

/* lote: n_specs especificações × N séries → n_specs*N handles */
ve_fit_batch(ctx, specs, n_specs, data, T, N, stride_t, stride_n, o, fits_out);

ve_fit_status(fits_out[k], &st);
ve_aic(fits_out[k], &aic);
ve_fit_sigma(fits_out[k], &ptr, &len);              /* visão zero-copy */

ve_dcc_spec* d;    ve_dcc_spec_create(VE_DCC_ADCC, &d);
ve_dcc_fit_batch(ctx, dcc_specs, n_dcc, univariate_fits, N, o, dcc_out);

ve_forecast_create(fit, horizon, &fc);
ve_simulate(ctx, fit, horizon, n_paths, seed, &sim);

ve_fit_free(fit); ve_spec_free(s); ve_context_free(ctx);
```

## 7. API Python

Primeira linguagem. ctypes sobre a interface C, como `_native.py` no ExcelReader e no FixedWidthParser.

```python
import volengine as ve

ctx = ve.Context(threads=0)

f    = ctx.fit(spec, series)            # 1 spec, série 1D            → Fit
fits = ctx.fit(specs, returns)          # lista de specs, painel (T,N) → grade specs × séries

f.status, f.coef(), f.robust_coef()
ve.aic(f), ve.bic(f), ve.loglik(f), ve.ljung_box(f, lags=10)
f.sigma                                  # numpy view, zero-copy

d  = ctx.fit_dcc(dcc_spec, univariate)   # 1 DccSpec    → DccFit
ds = ctx.fit_dcc(dcc_specs, univariate)  # lista         → lista de DccFit
# univariate: N ajustes escolhidos pelo usuário, um por série

fc   = ve.forecast(f, horizon=10)
sims = ve.simulate(f, horizon=250, paths=10_000, seed=42)
```

- Sobrecargas por spec única / lista existem no Python e no `volengine.hpp`, nunca na interface C.
- Critérios e diagnósticos são funções (`ve.aic(f)`), espelhando a interface C.
- Pacote no PyPI; wheels para as plataformas da seção 11.

## 8. Performance

- **Onde está o ganho.** A recursão de variância é sequencial no tempo (σ²ₜ depende de σ²ₜ₋₁), então SIMD numa
  série isolada ajuda pouco. O ganho vem de processar muitos ajustes juntos: uma faixa SIMD por ajuste da mesma
  especificação, threads entre grupos de ajustes.
- **Otimizador próprio: L-BFGS em lote.** N problemas avançando juntos em faixas SIMD. Nenhuma biblioteca (LBFGSpp,
  NLopt) faz isso. Restrições (ω>0, α,β≥0, estacionariedade) por reparametrização (softplus/logit) — mais simples
  em lote que L-BFGS-B. A escolha final da forma de tratar restrições é do builder, desde que os resultados batam
  com os oráculos.
- **Gradiente analítico** pela mesma recursão temporal do σ² (≈ custo de uma avaliação, contra k+1 por diferenças
  finitas). Validado contra diferenças finitas nos testes.
- **Threads:** BS::thread_pool (`github.com/bshoshany/thread-pool`). Os ajustes têm durações muito diferentes
  (nig/ghyp são lentos, norm é rápido), então o trabalho é dividido em muitos blocos pequenos.
- **Cholesky próprio** com SIMD para o DCC (N×N por instante t) e para inverter a Hessiana k×k.
- **Simulação reprodutível:** RNG baseado em contador (Philox, implementação própria — o `std::philox_engine` só
  chega no C++26). Decisão do Claude sobre o alcance: ajustes e simulações são bit-idênticos para qualquer número
  de threads na mesma máquina e no mesmo conjunto de instruções. Entre ISAs (AVX2, AVX-512, NEON) e plataformas,
  só dentro de tolerância — exigir bit-identidade ali travaria o despacho do Highway e o uso de FMA.
- **Meta de performance (decisão do Claude):** a carga do TCC (20 séries × 5.000 obs × todas as variantes, ordens
  p,q 1–3 e distribuições, mais as variantes DCC) em menos de 10 s numa CPU desktop de 8 núcleos. Primeiro entra
  uma versão escalar com threads; o L-BFGS em lote com SIMD e um Bessel K próprio entram quando o benchmark
  mostrar onde o tempo está.
- **Benchmarks** com Google Benchmark, incluindo um cenário com a escala do TCC (20 séries × 5.000 obs × todas as
  variantes/ordens/distribuições + DCC) como referência de performance.

## 9. Convenções explícitas

Onde `rugarch`, `rmgarch` e `arch` divergem, a convenção é uma opção com padrão documentado:

- Inicialização de σ²₀ (o `rugarch` e o `arch` usam formas diferentes).
- Normalização dos critérios de informação (o `rugarch` divide por n; o `arch` não).
- Tipo de erro-padrão (os dois são sempre calculados; a opção decide o que os wrappers mostram por padrão, se
  aplicável).
- Parametrização do sstd e de outras distribuições assimétricas.

Padrão (decisão do Claude): seguir o `rugarch`/`rmgarch`, que é o que o autor usava no TCC. A convenção do `arch`
fica disponível como opção.

## 10. Dependências

Política: usar biblioteca enquanto ela for ótima; implementação própria quando nenhuma faz o que é preciso ou
quando há evidência medida de que a própria é mais rápida.

| Área | Escolha |
|---|---|
| SIMD | Google Highway (despacho em runtime AVX2/AVX-512/NEON). Intrínsecos diretos só em kernel com benchmark que prove ganho. |
| Threads | BS::thread_pool (single header) |
| Bessel K (nig, ghyp) | Boost.Math (só esse header); `std::cyl_bessel_k` não existe no libc++. Implementação própria se benchmark mostrar ganho. |
| Otimizador | Próprio (L-BFGS em lote) |
| Álgebra linear | Própria (Cholesky pequeno com SIMD); sem Eigen |
| Densidades | Próprias (fórmulas fechadas) |
| RNG | Próprio (Philox) |
| Testes / benchmarks | GoogleTest, Google Benchmark |
| Build | CMake + `CMakePresets.json`, FetchContent |
| Python | ctypes + numpy |

## 11. Plataformas, distribuição, licença

- Plataformas (decisão do Claude, iguais às wheels do FixedWidthParser): Windows x64, macOS arm64 e x64, Linux
  glibc e musl em x64 e arm64.
- CI em MSVC, Clang e GCC 15+.
- Pacote Python no PyPI; biblioteca nativa anexada às GitHub Releases.
- Licença MIT, como os outros projetos do autor. `CITATION.cff` incluído (uso acadêmico).

## 12. Validação

- **Oráculos:** `rugarch` e `rmgarch` (R) e `arch` (Python). O `arch` não tem DCC, então o DCC é validado só
  contra o `rmgarch`; o univariado tem dois oráculos independentes.
- **Fixtures** geradas por scripts R e Python versionados em `tests/reference/`, com dados fixos e as saídas dos
  oráculos guardadas no repositório. Os testes C++ e Python comparam contra elas.
- **Tolerâncias** documentadas por grandeza (loglik, parâmetros, erros-padrão, σₜ, critérios). Os otimizadores dos
  oráculos param em pontos ligeiramente diferentes, então a log-verossimilhança é a comparação primária: o
  VolEngine deve atingir loglik ≥ a do oráculo menos a tolerância. Divergência de parâmetros acima da tolerância
  exige explicação documentada.
- **Licença dos oráculos:** `rugarch`/`rmgarch` são GPL-3. O código deles pode ser lido para entender convenções
  (σ²₀, parametrizações, Nyblom, bins do GOF), mas nunca copiado ou traduzido. Em zona cinzenta, a implementação
  vem de paper e documentação. O `arch` usa licença NCSA (permissiva, compatível com MIT); código portado dele exige
  manter o aviso de copyright em `THIRD-PARTY-NOTICES.md`. As fixtures (saídas numéricas) podem ser commitadas.
- Gradiente analítico testado contra diferenças finitas.
- Simulação testada para reprodutibilidade com 1 e com N threads.

## 13. Estrutura de pastas

Referência, não cópia, do ExcelReader e do FixedWidthParser. A diferença é que aqui o núcleo é C++, então
`include/` e `src/` ficam na raiz e a interface C é uma camada, não um binding.

```
VolEngine/
  include/volengine/       volengine.h (interface C, o contrato)
  src/                     núcleo C++23: garch/, dcc/, optim/, dist/, simd/, capi/ ...
  cpp/                     volengine.hpp (wrapper header-only) + testes/pacote CMake
  python/                  pacote volengine (src/, tests/, benchmarks/, pyproject.toml)
  tests/
    unit/                  testes C++ do núcleo
    reference/             scripts R/Python dos oráculos + fixtures
    benchmarks/
  docs/                    guide/, performance/, cpp-api/, python-api/
  cmake/  CMakeLists.txt  CMakePresets.json
  .github/workflows/       ci, abi-check, release, benchmarks
  ARCHITECTURE.md  STYLEGUIDE.md  CONTRIBUTING.md  CHANGELOG.md
  CITATION.cff  LICENSE  README.md  SECURITY.md
```

Os nomes de subpastas de `src/` são sugestão; o builder decide.

## 14. Deixado ao builder

- Nomes exatos de funções, enums e handles, respeitando a seção 5.
- Organização interna do núcleo (hierarquia de tipos, templates, despacho estático entre modelos).
- Tratamento das restrições no otimizador, critérios de parada, valores iniciais padrão.
- Formato das fixtures dos oráculos.
- Ordem das fases de implementação. Sugestão: primeiro um modelo (sGARCH + norm) de ponta a ponta — núcleo →
  interface C → Python → comparação com oráculo — e só depois a amplitude de modelos, distribuições e o DCC.
- Conteúdo do STYLEGUIDE.md (C++23 moderno, coerente com a seção 4).

## 15. Decisões tomadas pelo Claude (para conferir)

- Nenhuma struct pública na interface C; configuração só por setters.
- `ve_context` como dono do thread pool (sem estado global).
- Padrão das convenções segue `rugarch`/`rmgarch`; `arch` como opção.
- Valores iniciais e variance targeting entram na especificação.
- NaN no meio de uma série é erro de argumento.
- Lote como lista de trabalhos (especificação, série), cobrindo cartesiano e pareado.
- Copula-DCC sem a transformação `spd` na v1; cópulas mvnorm/mvt.
- Erros-padrão do DCC com Hessiana conjunta em dois estágios (paridade com `rmgarch`).
- Bit-identidade só entre números de threads na mesma máquina/ISA.
- Meta: carga do TCC em < 10 s numa CPU de 8 núcleos; escalar + threads primeiro, SIMD em lote depois.
- 1.0 só com o escopo inteiro da v1; 0.x pode quebrar.
- Só `double`.
- Plataformas iguais às do FixedWidthParser; CI com GCC 15+.
- Philox próprio para simulação.
- Reparametrização para restrições, salvo o builder encontrar motivo para outra abordagem.
