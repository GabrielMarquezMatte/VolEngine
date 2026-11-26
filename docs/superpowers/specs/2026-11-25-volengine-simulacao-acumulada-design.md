# Simulação acumulada — desenho

Data: 2026-11-25. Versão alvo: 1.3.0 (ABI 1, aditiva), no mesmo release do cancelamento.

## Intenção

Quem simula para medir risco no horizonte quase nunca quer os passos: quer, por caminho, a soma da série simulada até
um ou mais prazos (o log-retorno acumulado, quando a série é de log-retornos). Hoje `ve_simulate_*` devolve todos os
passos (caminhos × horizonte, ou caminhos × horizonte × séries no DCC, com o sigma do mesmo tamanho ao lado), e quem
chama soma e descarta. O DataPanel, por exemplo, limita caminhos × horizonte × séries a 5 milhões de valores só para a
saída completa caber na memória.

O objetivo é uma opção da simulação que devolve só as somas nos prazos pedidos: a memória passa a ser
caminhos × K × séries, independente do horizonte, e o VaR de vários prazos sai dos mesmos caminhos.

## Objetivos

- Uma lista de prazos (checkpoints) no `ve_sim_options`, usada pelas três simulações: `ve_simulate_fit`,
  `ve_simulate_spec` e `ve_simulate_dcc` (e as variantes `_with_cancel`).
- Com a lista, a simulação guarda por caminho só a soma da série nos prazos: caminhos × K (univariada) ou
  caminhos × K × séries (DCC). A saída completa e o sigma por passo não são alocados.
- O valor no prazo k é, bit a bit, a soma da esquerda para a direita dos k primeiros passos da saída completa com a
  mesma semente.
- Bindings: wrapper C++, C# e Python expõem a lista e o resultado.

## Não-objetivos

- Outras reduções por caminho (máximo, drawdown, sigma acumulado, variância realizada). Ficam para quando houver
  pedido; a opção é sobre a soma da série.
- Compor retornos simples. A biblioteca soma o que simula, na escala do ajuste; converter para retorno simples
  (`exp(soma) - 1` para log-retornos) é de quem chama.
- R e H por caminho nos prazos.

## Anti-objetivos

- Mudar a saída de quem não passa a lista: sem ela, tudo fica como hoje, bit a bit.
- Uma soma que difira da soma dos passos da saída completa (ordem de soma diferente, compensação de Kahan): a
  referência de teste e a do DataPanel é a soma direta dos passos.
- Quebrar a ABI 1.

## Abordagens

- **Escolhida: a lista de prazos no `ve_sim_options`.** Uma função nova configura o modo, as simulações continuam as
  mesmas entradas, e cada caminho mantém uma soma corrente gravada nos prazos.
- Funções de simulação novas (`ve_simulate_*_cumulative`): dobrariam as seis entradas (com as `_with_cancel`) só para
  mudar a forma da saída. Descartada.
- Somar nos bindings: não economiza a memória nativa, que é o objetivo. Descartada.
- Só o fim do horizonte (K = 1): é o caso particular da lista; o usuário escolheu a lista para ter vários prazos dos
  mesmos caminhos.

## C ABI

```c
/* Checkpoints of the horizon: the simulation keeps, per path (and per series in a DCC), only the sum of the
   simulated series over steps 1..k for each k, as the left-to-right sum of the first k steps of the full output with
   the same seed. ks are strictly increasing, from 1, and the last is the horizon of o; otherwise
   VE_INVALID_ARGUMENT, with the option unchanged. n = 0 (ks may be NULL) clears it: back to the full output.
   NULL ks with n > 0 is VE_INVALID_ARGUMENT. Since 1.3.0. */
VE_API ve_status ve_sim_options_set_checkpoints(ve_sim_options* o, const size_t* ks, size_t n);

/* With checkpoints: paths x K values, row-major (row p is path p), valid until sim is freed. Without them
   VE_INVALID_ARGUMENT. Since 1.3.0. */
VE_API ve_status ve_sim_cumulative(const ve_sim* sim, const double** data, size_t* len);
/* The checkpoints of the simulation (K values), or len = 0 without them. Since 1.3.0. */
VE_API ve_status ve_sim_checkpoints(const ve_sim* sim, const size_t** data, size_t* len);

/* paths x K x m, row-major. Same rules as ve_sim_cumulative. Since 1.3.0. */
VE_API ve_status ve_dcc_sim_cumulative(const ve_dcc_sim* sim, const double** data, size_t* len);
VE_API ve_status ve_dcc_sim_checkpoints(const ve_dcc_sim* sim, const size_t** data, size_t* len);
```

Regras:

- Com checkpoints, `ve_sim_returns`, `ve_sim_sigma`, `ve_dcc_sim_returns` e `ve_dcc_sim_sigma` dão
  `VE_INVALID_ARGUMENT` (a saída completa não existe), como as views de correlação sem `correlations`.
- `ve_sim_paths`, `ve_sim_horizon`, `ve_dcc_sim_paths`, `ve_dcc_sim_horizon` e `ve_dcc_sim_series` continuam valendo.
- Checkpoints com `correlations` ligado é `VE_INVALID_ARGUMENT` no `ve_simulate_dcc` (minha decisão: o modo existe
  para economizar memória, e R/H por passo anularia isso).
- O teste de "too large" passa a usar paths × K × m (× 1 na univariada) no modo acumulado.
- O `ve_sim_options_create` muda o horizonte só na criação, então a regra "o último é o horizonte" é conferida no
  setter, e de novo na simulação (defesa contra um handle montado de outro jeito no futuro).
- Contrato: `volengine.abi` ganha as cinco linhas; nada muda nas existentes.

## Núcleo

- `sim::Request` e `dcc::future::DccRequest` ganham `std::vector<std::size_t> checkpoints` (vazio = saída completa).
- `sim::Simulation` e `DccSimulation` ganham `checkpoints` e `cumulative`. No modo acumulado, `returns` e `sigma`
  ficam vazios.
- Por caminho, uma soma corrente em `double` (por série no DCC), somando o retorno do passo h logo depois de
  calculado, na mesma ordem dos passos; no passo `checkpoints[j]`, grava a soma em `cumulative[p][j]`. O caminho segue
  lendo só `Stream(seed, p)`, então o resultado não depende do número de threads.
- Cancelamento: o mesmo ponto de conferência por caminho que já existe.

## Wrapper C++

- `SimOptions` ganha `std::vector<std::size_t> checkpoints` (vazio = saída completa), aplicado em `make_handle`.
- `Simulation::cumulative()` e `Simulation::checkpoints()`, e o mesmo em `DccSimulation`, como `Result<...>` das
  views, no padrão das outras views.

## C#

- `SimulationOptions` e `DccSimulationOptions` ganham `IReadOnlyList<int>? Checkpoints { get; init; }`.
- `Simulation.Cumulative` (`double[][]?`, Paths × K) e `DccSimulation.Cumulative` (`double[][][]?`,
  Paths × K × séries), nulos sem checkpoints, como `DccSimulation.Correlation` sem correlações; e
  `IReadOnlyList<int> Checkpoints` nos dois (vazio sem eles). Com checkpoints, `Returns` e `Sigma` são arrays vazios
  (comprimento 0): torná-los anuláveis mudaria o tipo público que já foi publicado.
- O limite de `Array.MaxLength` que hoje vale para Horizon × Paths passa a valer para Paths × K no modo acumulado,
  porque é isso que vira array gerenciado.
- `PublicAPI.Unshipped.txt` com as linhas novas.

## Python

- `ctx.simulate(..., checkpoints=[1, 5, 10])`, uma palavra-chave dos TypedDicts de simulação (`FutureKeywords` e
  `DccSimulateKeywords`), não um parâmetro a mais (limite de 5 parâmetros).
- `Simulation.cumulative` (ndarray paths × K) e `DccSimulation.cumulative` (paths × K × séries), e `checkpoints`
  (tupla). Com checkpoints, `returns` e `sigma` levantam `VolEngineError`, como `correlation` sem `correlations=True`.

## Testes

- Núcleo/C ABI: com a mesma semente, `cumulative[p][j]` é, bit a bit, a soma dos `checkpoints[j]` primeiros passos da
  saída completa, nas três simulações (fit, spec, DCC, com regressores em uma delas); igual para 1 e 4 threads.
- Validação: n = 0 limpa e volta à saída completa (`VE_OK`); lista não crescente, começando em 0, com o último
  diferente do horizonte, ou `NULL` com n > 0 dá `VE_INVALID_ARGUMENT` e deixa a opção como estava; checkpoints com
  correlações dá `VE_INVALID_ARGUMENT`.
- Acessores: views completas dão `VE_INVALID_ARGUMENT` no modo acumulado, `ve_sim_checkpoints` devolve a lista e
  len = 0 sem ela.
- Bindings: o mesmo resultado bit a bit contra a soma dos passos da saída completa, em C++, C# e Python.

## Documentação

- Uma seção nos guias de C++, C# e Python, com o exemplo de VaR em vários prazos a partir dos mesmos caminhos.
- `CHANGELOG.md`, seção `[Unreleased]`, Added.
- Os guias ganham a frase sobre a escala (item 6 das propostas): o motor é invariante à escala, os dados podem entrar
  em fração.

## Deixado para quem constrói

- Onde ficam os campos novos dos resultados internos e como a soma é escrita no laço dos passos, desde que a ordem de
  soma seja a dos passos.
- O nome exato dos membros nos bindings, se o padrão de cada um pedir outro.
