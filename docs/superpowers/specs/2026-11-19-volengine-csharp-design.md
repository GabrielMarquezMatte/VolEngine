# Binding C# (.NET 10) — design

2026-11-19. Pedido do autor antes do SV (spec 2026-11-17). Decisões do autor nesta conversa: alvo **só `net10.0`**
(nada de netstandard2.0 nem .NET Framework); **API C# completa**, com a cobertura do Python (não um binding fino);
objetivo duplo: ampliar o alcance da biblioteca e usar num serviço do autor; tempo de vida **simples, um `using`**
(nada de persistir ou recarregar fits, nada de cache: o que o serviço precisar a mais, o serviço implementa);
declarações nativas **escritas à mão**, verificadas por teste contra o contrato (abordagem 1). O que está marcado
"minha escolha" foi decidido por mim e pode ser corrigido.

## Intenção

Quem usa .NET não tem GARCH/DCC rápido por máxima verossimilhança. O C ABI 1.x é o único contrato estável; o C# é
mais um consumidor do header, como o Python (ctypes) e o wrapper C++ (header-only). Nenhuma função nova da C API.
A biblioteca é de uso geral: o serviço do autor é um consumidor, não dimensiona nada.

## Metas

- **Cobertura**: tudo o que o Python expõe, com os mesmos nomes de conceito (camada sobre as 130 funções do
  contrato): contexto e threads; `Spec`, `FitOptions`; ajuste em grade spec × série com ranges e regressores
  (mxreg/vxreg); `Fit` (status, nobs, nomes, estimativa, tabela de coeficientes com SE hessiana/robusta, loglik e
  critérios aic/bic/hqic/sic com `perObs`, sigma, resíduos, resíduos padronizados, fitted, Ljung-Box, ARCH-LM, sign
  bias, GOF, Nyblom, forecast com regressores do horizonte); simulação a partir de fit ou de spec com tudo fixo;
  `DccSpec` (DCC/aDCC/FDCC, mvnorm/mvt/mvlaplace, Copula-DCC paramétrica/empírica, estática kendall/ml), ajuste em
  lote, SE de duas etapas (na opção ou depois), forecast e simulação do DCC (com `correlations`); versão do ABI.
- **Pacote**: um `VolEngine` no nuget.org, versão = versão do release (`VE_VERSION`), com as 7 bibliotecas nativas
  que o `wheels.yml` já produz em `runtimes/<rid>/native/` — os nomes da matriz (`linux-x64`, `linux-arm64`,
  `linux-musl-x64`, `linux-musl-arm64`, `osx-arm64`, `osx-x64`, `win-x64`) já são RIDs do .NET. Publicado pelo
  `release.yml` na mesma tag, a partir dos artefatos `native-*` (nada recompila). Primeiro pacote na tag 1.1.0, com
  o EWMA; **a 1.1.0 espera o binding inteiro** (decisão do autor), então a divisão em planos não precisa entregar
  pacote no meio. O SV vai para a 1.2.0.
- **Publicação** (decisão do autor): Trusted Publishing do nuget.org (OIDC, sem segredo, como o PyPI), passo
  `NuGet/login`. Antes da tag, o autor confere que o ID `VolEngine` está livre (ou o reserva), registra a policy para
  `release.yml` e informa o username do nuget.org que o passo pede.
- **Semver da API C#** (decisão do autor): a API pública é estável desde 1.1.0 e só cresce em todo o 1.x, como o
  ABI. Verificação automática: `PublicApiAnalyzers` (arquivos `PublicAPI.Shipped.txt`/`Unshipped.txt`) e
  `EnablePackageValidation` com baseline no pacote anterior a partir da 1.1.0. Toda forma pública (tipos de resultado,
  `Fit` dentro do lote, ranks dos arrays, sobrecargas) fica fechada antes da tag.
- **Nome da biblioteca nativa** (minha escolha): dentro do pacote o arquivo se chama `libvolengine.{dll,so,dylib}`
  e as declarações usam `[LibraryImport("libvolengine")]` — o .NET acrescenta só a extensão e o Windows não
  acrescenta prefixo, então o mesmo nome resolve nos três sistemas, e é o `/name:libvolengine.dll` do `.lib`. O
  asset `volengine-native-win-x64.dll` é renomeado no empacotamento.
- **Biblioteca local**: variável de ambiente `VOLENGINE_LIB` (a mesma do Python) com o caminho de um build local,
  via `NativeLibrary.SetDllImportResolver`; sem ela, a resolução padrão do NuGet. Testes e CI usam a variável.
- **Verificação do ABI na carga**: na primeira chamada, `ve_abi_version()` diferente de 1 lança, como o Python.
- **AOT/trimming**: `[DisableRuntimeMarshalling]` (tudo é blittable: handles `nint`, `int`, `nuint`, `double*`,
  `byte*` UTF-8), `IsAotCompatible=true`, nullable habilitado, warnings como erro. Um teste publica um consumidor
  mínimo com Native AOT num SO do CI e o executa (minha escolha; prova o `IsAotCompatible`).

## Forma da API (minha escolha, salvo o marcado)

- **Erros**: exceções. `VolEngineException` com `Status` (enum dos `VE_*` de status) e a mensagem de
  `ve_last_error`, lida logo após a chamada na mesma thread (a mensagem é thread-local; nenhuma chamada é `async`,
  então não há troca de thread no meio). Argumento inválido do lado C# (horizonte negativo, nome com NUL, dimensões
  que não batem) é `ArgumentException` antes de chamar o nativo, como o `ValueError` do Python.
- **Tempo de vida** (decisão do autor: um `using`): cada objeto que segura um handle nativo é `IDisposable` sobre um
  `SafeHandle` próprio (o finalizador do `SafeHandle` evita vazamento se alguém esquecer o `Dispose`). Usar um objeto
  já liberado é `ObjectDisposedException`. O resultado de um lote (`FitGrid`, `DccFitBatch` ou nome que o builder
  escolher) é **um** `IDisposable` que libera todos os fits dele; um `Fit` tirado dele vale enquanto o lote vive.
  Um ajuste de uma série só devolve um `Fit` direto.
- **Arrays de saída**: cópias gerenciadas e **jagged** (`double[]`, `double[][]` para caminhos de simulação,
  `double[][][]` para correlações/covariâncias T × m × m e `double[][][][]` para a simulação do DCC com
  `correlations`), não views sobre memória nativa: continuam válidas depois do `Dispose` e serializam direto com
  System.Text.Json (inclusive o gerador de fonte do AOT), que recusa arrays multidimensionais (minha escolha, depois da
  revisão).
- **Entrada**: uma série como `ReadOnlySpan<double>`; um painel como `double[,]` T × N (ordem C, fixado e passado
  sem cópia ao `ve_panel_create`), com `starts`/`lengths` opcionais; e uma sobrecarga com
  `IReadOnlyList<double[]>` (séries de tamanhos diferentes, copiadas para um painel com ranges), a forma natural de um
  serviço que recebe séries por requisição. Regressores na mesma forma de painel por spec que o Python aceita.
- **Specs**: `sealed record` imutáveis (`Spec`, `FitOptions`, `DccSpec`, opções de simulação) com propriedades
  `init` e os padrões do Python (`p = q = 1`, `Distribution.Norm`, `IncludeMean = true`, `MaxIter = 500`,
  `Tol = 1e-6`, ...). `Fixed`/`Start` são `IReadOnlyDictionary<string, double>`, com igualdade de record por valor
  (como o `__hash__` do Python). Como um inicializador de objeto não roda código depois de todas as propriedades, a
  validação nativa (`ve_spec_validate` e afins) roda no primeiro uso (fit, simulação) e num `Validate()` público
  para quem quer falhar cedo; o Python valida na construção.
- **Enums** em vez das strings do Python: `Model` com um membro por valor `VE_MODEL_*` (o fGARCH vira
  `FGarchTGarch` etc., como no ABI, em vez do par model/submodel do Python), `Distribution`, `VarInit`,
  `DccModel`, `DccDistribution`, `Copula`, `StaticMethod`, `FitStatus`, `StandardErrors` (hessian/robust). Valores
  iguais aos `VE_*`; o teste do contrato confere cada um.
- **Threads**: o C# promete exatamente o que o header promete, nem mais nem menos (minha escolha, depois da
  revisão): sobre um mesmo `Context`, as funções `ve_context_*` (exceto o free) e o ajuste em lote univariado
  (`ve_fit_batch`) podem rodar ao mesmo tempo de várias threads; leituras de resultados imutáveis também. O header não
  diz isso de `ve_simulate_*`, `ve_dcc_fit_batch` e `ve_dcc_standard_errors`, então a documentação XML do C# também
  não diz, e o C# não põe lock (serializaria o serviço). Estender a garantia é trabalho do nativo (testes + comentário
  no header), fora deste design. Chamadas síncronas; o paralelismo é o pool do `ve_context`. Quem precisa de `Task`
  chama `Task.Run`.
- **Nomes** seguem as convenções .NET (PascalCase, `Aic(perObs: true)`, `Coef(StandardErrors.Hessian)`), com os
  mesmos conceitos e nomes de parâmetro nativos (`"alpha1"`, `"shape"`) que o Python. A documentação XML dos membros
  públicos copia as regras do docstring do Python (o que é derivado, o que é recusado, convenções de ordem p/q).

## Validação

- **Contrato**: um teste lê `include/volengine/volengine.abi` e falha se uma `function` do contrato não tem
  declaração na classe `Native` (por reflexão sobre os métodos `[LibraryImport]`), se a assinatura C# não
  corresponde aos tipos da linha (tabela pequena de tipos C → C#), ou se o valor de um `define VE_*` difere do membro
  de enum ou da constante C# correspondente. É o que torna seguro escrever as declarações à mão.
- **Cobertura**: um teste que chama todas as funções do contrato pelo caminho da API pública, como o
  `abi_coverage` do C++ (6b).
- **Comportamento**: testes da camada (erros e mensagens, `Dispose`/uso depois de liberado, ranges, regressores,
  recusas, threads concorrentes sobre um `Context`), no estilo de `cpp/tests/`.
- **Paridade numérica** (minha escolha): poucos casos de `tests/reference/fixtures` (um rugarch univariado, um
  forecast, um DCC) com as tolerâncias de `docs/guide/validation.md`, só para provar que dados e parâmetros passam
  certos pela fronteira; a numérica em si já é validada no nativo e no Python.
- Framework de teste: xUnit (minha escolha; o builder pode trocar por MSTest se houver motivo).
- **CI**: workflow novo `csharp.yml` nos 3 SOs (Linux, macOS, Windows), como o `cpp.yml`: compila o nativo como o
  `cpp.yml` já faz, roda `dotnet test` com `VOLENGINE_LIB`, e o teste do AOT. O `release.yml` empacota e publica o
  NuGet (Trusted Publishing do nuget.org se o autor registrar; senão uma API key em secret) e testa o `.nupkg` num
  consumidor limpo, como o pip install em venv limpo do 6a. A publicação no NuGet roda depois do GitHub Release, como
  o `publish-pypi`: uma falha nela não bloqueia o release nem o PyPI (minha escolha).
- **Benchmark** (convenção do projeto: benchmark em todo plano): BenchmarkDotNet num projeto fora do pacote, um
  `BM_FitOne` equivalente e um lote, comparados ao Google Benchmark do C, para mostrar que o binding fica dentro do
  ruído (como o 6b mostrou para o C++). Sem comparação com outra biblioteca .NET.

## Layout

`csharp/` na raiz, ao lado de `cpp/` e `python/`: `csharp/VolEngine.slnx`, `csharp/src/VolEngine/` (o pacote),
`csharp/tests/VolEngine.Tests/`, `csharp/benchmarks/`, consumidores de teste (AOT e pacote). Guia
`docs/guide/csharp.md` (como `python.md` e `cpp.md`), seção no README e no `ARCHITECTURE.md` (camada nova acima da
C API, ao lado do Python). Os nomes de pasta e arquivo dentro de `csharp/` ficam com o builder.

## Abordagens consideradas

1. **Declarações nativas à mão + teste contra o contrato (escolhida pelo autor).** Consistente com o Python
   (`PROTOTYPES` à mão) e o C++; tipos C# bons já na declaração; o teste do contrato pega o que faltar quando o ABI
   crescer (o SV acrescenta `VE_MODEL_SV` e `VE_MODEL_SV_LEVERAGE`).
2. Gerar as declarações a partir do `volengine.abi` (`abi_contract.py csharp`): o ABI novo apareceria sozinho, mas é
   mais um gerador e as assinaturas sairiam cruas; os enums continuariam à mão. Descartada.
3. ClangSharp PInvokeGenerator: dependência externa com libclang para o que o contrato já normaliza. Descartada.
4. C++/CLI sobre o wrapper C++: só Windows. Descartada.
5. netstandard2.0 (ou multi-target): alcançaria .NET Framework/Excel-DNA, ao custo de polyfills, `.targets` para
   copiar a DLL e sem `IsAotCompatible`. Descartada pelo autor.

## Não-metas

- Persistir, serializar ou recarregar um fit; cache de fits (o serviço implementa se precisar).
- API `async`/`Task`.
- netstandard2.0, .NET Framework, Unity/Mono, RIDs além dos 7 que o release já produz (win-arm64 etc.).
- Tipos de dados de terceiros (Math.NET, DataFrame) na API pública.
- Função nova na C API para o binding.

## Anti-metas

- Um binding que dependa de algo que só o Python faz (lógica numérica no C#): a camada só converte e chama o nativo.
- Views de memória nativa que fiquem inválidas depois do `Dispose` sem o usuário perceber.
- Um pacote que precise de compilador ou de passo manual para achar a biblioteca nativa.
- Declarações que divirjam do contrato sem o teste falhar.

## Restrições

STYLEGUIDE nas regras independentes de linguagem (sem `else`, retorno cedo, ≤ 60 linhas por função, ≤ 5
parâmetros, aninhamento ≤ 3, ≤ 8 módulos por pasta, comentários sem achados); verificação automática dessas regras
para C# fica a critério do builder. ABI 1.x só cresce e o C# não muda o contrato. Testes nativos só no preset
release; MinGW com `/c/msys64/ucrt64/bin` primeiro no PATH ao rodar localmente; SDK .NET 10 já instalado na máquina
(10.0.403).

## O que fica com o builder

Nomes exatos dos tipos de resultado de lote e de simulação, organização interna das pastas, o conjunto exato de
sobrecargas de `Context.Fit`/`Simulate` (cobrindo as formas acima), como o `csharp.yml` obtém o nativo, e se vale um
checador de estilo para C#.

## Sugestão de divisão em planos (o planejador decide)

- C#-a: projeto, `Native` + enums + teste do contrato, carga da biblioteca, erros, contexto, specs, ajuste em lote,
  `Fit` completo, forecast, simulação, diagnósticos, testes, `csharp.yml`, benchmark.
- C#-b: DCC inteiro (spec, lote, SE, forecast, simulação), teste de cobertura das 130 funções, AOT, empacotamento
  NuGet no `release.yml`, guia, README/ARCHITECTURE, `PublicAPI.Shipped.txt` fechado. Depois dele, a tag 1.1.0.
