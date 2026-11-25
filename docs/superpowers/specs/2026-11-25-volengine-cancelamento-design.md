# Cancelamento cooperativo — desenho

Data: 2026-11-25. Versão alvo: 1.3.0 (ABI 1, aditiva).

## Intenção

O DataPanel chama `Fit`, `FitDcc` e `Simulate` a partir da API em C# (`RiskEngine`) com tempo limite. Hoje a chamada
nativa não pode ser interrompida: quando o tempo estoura, a resposta é abandonada, mas a vaga do semáforo (um DCC por
vez) só é liberada quando a chamada termina, e um cálculo abandonado segura todos os outros. O batch em Python quer o
mesmo para interromper um ajuste com prazo.

O objetivo é um sinal de cancelamento cooperativo, criado por quem chama e passado a cada chamada longa, que a faz
devolver um status próprio em pouco tempo.

## Objetivos

- Cancelar `ve_fit_batch`, `ve_dcc_fit_batch`, `ve_dcc_standard_errors`, `ve_simulate_fit`, `ve_simulate_spec` e
  `ve_simulate_dcc`.
- Um status de chamada novo, `VE_CANCELLED`, distinto de qualquer status de ajuste (não convergência continua sendo
  `VE_FIT_MAX_ITER`/`VE_FIT_FAILED` por ajuste).
- Latência limitada pelo trabalho entre dois pontos de conferência: uma iteração do otimizador, uma etapa dos
  erros-padrão ou um caminho da simulação.
- Bindings: C# com `CancellationToken`, Python com um objeto de cancelamento, wrapper C++ com `std::stop_token`.
- As seis funções atuais marcadas como obsoletas, para remoção na ABI 2.

## Não-objetivos

- Resultados parciais. Um lote cancelado não devolve os ajustes que já terminaram (decisão do usuário: tudo ou nada,
  basta parar).
- Prazo embutido no handle. Quem chama agenda o sinal (`CancelAfter` no C#, `threading.Timer` no Python).
- Cancelar chamadas curtas (forecast, diagnósticos, acessores).
- Mudar o comportamento das funções atuais: até a ABI 2 elas continuam iguais, só ganham o aviso de obsoleta.

## Anti-objetivos

- Quebrar a ABI 1: nenhuma assinatura muda, nenhum valor existente é renumerado.
- Um handle que vaze para outras chamadas: o sinal vale só para as chamadas que o recebem.
- Deixar threads do pool rodando trabalho cancelado depois que a chamada devolveu.

## Abordagens

1. **Variantes novas com o handle como último parâmetro antes da saída (escolhida pelo usuário).** O cancelamento é
   estado de uma chamada, então entra explícito na chamada. Custa seis funções novas mais as três do handle.
2. **Setter nas opções** (`ve_fit_options_set_cancel` etc.). Menos funções, mas mistura configuração reutilizável com
   estado de uma chamada: um objeto de opções compartilhado entre threads cancelaria chamadas que não deviam. E
   `ve_dcc_standard_errors` não recebe opções. Descartada.
3. **Cancelar pelo contexto** (para todas as chamadas do contexto) ou **handle no thread de quem chama**. A primeira
   obriga um contexto (e um pool) por requisição; a segunda não chega às threads do pool. Descartadas.

## C ABI

```c
#define VE_CANCELLED 6

typedef struct ve_cancel ve_cancel;
VE_API ve_status ve_cancel_create(ve_cancel** out);
VE_API ve_status ve_cancel_request(ve_cancel* c);
VE_API void ve_cancel_free(ve_cancel* c);

VE_API ve_status ve_fit_batch_with_cancel(ve_context* ctx, const ve_batch* b, const ve_cancel* c, ve_fit** out);
VE_API ve_status ve_dcc_fit_batch_with_cancel(ve_context* ctx, const ve_dcc_batch* b, const ve_cancel* c,
                                              ve_dcc_fit** out);
VE_API ve_status ve_dcc_standard_errors_with_cancel(ve_context* ctx, const ve_dcc_fit* f, const ve_fit* const* fits,
                                                    size_t m, const ve_cancel* c, ve_dcc_fit** out);
VE_API ve_status ve_simulate_fit_with_cancel(ve_context* ctx, const ve_fit* f, const ve_sim_options* o,
                                             const ve_cancel* c, ve_sim** out);
VE_API ve_status ve_simulate_spec_with_cancel(ve_context* ctx, const ve_spec* s, const ve_sim_options* o,
                                              const ve_cancel* c, ve_sim** out);
VE_API ve_status ve_simulate_dcc_with_cancel(ve_context* ctx, const ve_dcc_fit* f, const ve_sim_options* o,
                                             const ve_dcc_future* x, const ve_cancel* c, ve_dcc_sim** out);
```

Semântica:

- `c == NULL` é "não cancelável": a variante faz exatamente o que a função antiga faz. As antigas passam a delegar
  para as novas com `NULL`.
- `ve_cancel_request` é segura de qualquer thread, a qualquer momento, inclusive durante uma chamada que usa o handle,
  e idempotente. Um handle sinalizado continua sinalizado (como `CancellationToken`); não há reset.
- O mesmo handle pode ser passado a várias chamadas, simultâneas ou não; sinalizá-lo cancela todas.
- Liberar o handle enquanto uma chamada o usa é comportamento indefinido (documentado no header).
- `VE_CANCELLED` significa que a chamada parou sem resultado: `out` fica com `NULL` (cada um dos `n` handles, no
  `ve_fit_batch` e no `ve_dcc_fit_batch`), o mesmo contrato das outras falhas, que já zeram a saída na entrada.
  `ve_last_error` diz "cancelled".
- Se o sinal chegar depois do último ponto de conferência, a chamada devolve `VE_OK` com o resultado completo. Quem
  chama não pode assumir que um pedido de cancelamento sempre produz `VE_CANCELLED`.
- Os argumentos são validados antes do primeiro ponto de conferência: uma chamada com argumento inválido devolve o
  mesmo erro com ou sem o handle sinalizado (decisão do Claude, para o erro ser determinístico).
- Quando a chamada devolve, nenhuma thread do pool continua trabalhando nela.

Obsoletas: as seis funções atuais ganham `VE_DEPRECATED("use ve_..._with_cancel(..., NULL, ...)")`, uma macro no
header que vira `__attribute__((deprecated(msg)))` no GCC/Clang e `__declspec(deprecated(msg))` no MSVC, em C e em
C++, e some com `VE_NO_DEPRECATED_WARNINGS` definido. Ela vem antes de `VE_API`, para `scripts/abi_contract.py` ler a
declaração como hoje (a linha do contrato não muda). Elas saem na ABI 2. Na ABI 2
os nomes `_with_cancel` ficam como estão, para não haver uma segunda quebra só de nome (decisão do Claude).

`VE_ABI_VERSION` continua 1. `volengine.abi` ganha as entradas novas.

## Núcleo

As três famílias já passam pelos mesmos pontos, então o cancelamento é uma exceção interna:

- Um tipo interno de sinal (um `std::atomic<bool>`; leitura `relaxed` basta) e uma exceção `ve::Cancelled`.
- `capi::guard` captura `ve::Cancelled` antes de `std::exception` e devolve `VE_CANCELLED`.
- `run_in_blocks` já repassa a primeira exceção depois de todos os blocos iniciados terminarem, o que garante que
  nenhuma thread fica rodando depois da volta.
- Pontos de conferência:
  - o laço do L-BFGS (`optim::Minimizer`), uma vez por iteração. Ele é usado no univariado, no DCC e na cópula
    estática ML. O sinal entra pelo `LbfgsOptions` ou pelo objetivo;
  - o início de cada job do lote univariado e de cada spec do lote DCC;
  - os erros-padrão: antes de começar e dentro do laço por coluna/etapa dos erros-padrão do DCC, que podem pesar tanto
    quanto o ajuste;
  - cada caminho da simulação, univariada e DCC.
- O sinal chega ao núcleo como um ponteiro que pode ser nulo. Sem sinal, o custo é uma comparação com nulo por ponto de
  conferência.

## Wrapper C++ (`cpp/include/volengine`)

- Passa a chamar as variantes `_with_cancel`.
- Os métodos de `Context` que fazem as seis chamadas ganham um parâmetro `std::stop_token stop = {}`. Quando
  `stop.stop_possible()`, o wrapper cria um `ve_cancel`, registra um `std::stop_callback` que chama
  `ve_cancel_request` e o passa à chamada. `VE_CANCELLED` vira `Status::cancelled` no `ve::Error`.
- Header-only, então o parâmetro com padrão não quebra quem compila contra ele.

## C#

- Sobrecargas novas em `Context.Fit` (as duas), `Context.FitDcc` (as duas), `Context.StandardErrors` e
  `Context.Simulate` (as três): os parâmetros atuais, com os mesmos padrões, mais `CancellationToken cancellationToken
  = default` no fim. As sobrecargas atuais continuam, delegando com `CancellationToken.None`, mas perdem os valores
  padrão (todos os parâmetros passam a obrigatórios). É o padrão de compatibilidade dos analisadores de API pública
  (RS0026/RS0027): o binário compilado contra a 1.x continua achando o método antigo, e o código-fonte passa a usar o
  novo. Elas **não** recebem `[Obsolete]`: em .NET a forma sem token é idiomática (decisão do Claude). Na v2 do pacote
  as sobrecargas antigas saem.
- Com um token cancelável: `ThrowIfCancellationRequested()` antes da chamada, criar o handle, `token.Register(...)`
  chamando `ve_cancel_request`, e no `VE_CANCELLED` lançar `OperationCanceledException(token)`. O registro é desfeito
  antes de liberar o handle.
- `NativeMethods` ganha as funções novas; as antigas deixam de ser usadas. `PublicAPI.Unshipped.txt` e
  `ContractCheck` são atualizados.

## Python

- `ve.Cancel`: um objeto que tem o handle, com `cancel()` (a ABI não tem consulta do estado, e ninguém pediu uma).
- Um argumento só de palavra-chave `cancel: Cancel | None = None` em `Context.fit`, `fit_dcc`, `simulate` e no
  caminho dos erros-padrão do DCC.
- `VE_CANCELLED` vira uma exceção `ve.Cancelled`, subclasse de `VolEngineError`.
- A biblioteca é carregada com `ctypes.CDLL`, que solta o GIL durante a chamada, então `cancel()` pode ser chamado de
  outra thread (por exemplo, um `threading.Timer`) enquanto o ajuste roda.

## Testes

- Para cada uma das seis variantes: handle sinalizado antes da chamada devolve `VE_CANCELLED`, com `out` em `NULL`.
  Handle `NULL` e handle não sinalizado dão o mesmo resultado, bit a bit, que a função antiga.
- Um teste por família com o sinal disparado de outra thread durante um trabalho grande: a chamada devolve
  `VE_CANCELLED` muito antes do tempo do trabalho completo. Use margens folgadas para o teste não ficar instável.
- Argumento inválido com handle sinalizado devolve o erro de argumento, não `VE_CANCELLED`.
- As funções obsoletas mantêm um teste de delegação cada, compilado com o aviso de obsoleta desligado. O restante dos
  testes internos passa para as variantes novas, porque o projeto compila com `-Werror`.
- C# e Python: um teste de cancelamento por família e um de token/handle já cancelado.

## Documentação

- O comentário do header, cada guia (`docs/guide/cpp.md`, `csharp.md`, `python.md`) com uma seção de cancelamento e o
  `CHANGELOG` (1.3.0: adicionado e obsoleto).

## Deixado para quem constrói

- Onde exatamente o sinal entra no L-BFGS (`LbfgsOptions` ou o objetivo) e como ele desce até os erros-padrão.
- A granularidade dos pontos de conferência dentro dos erros-padrão do DCC.
- Os nomes internos (tipo do sinal, exceção) e o texto exato das mensagens.
