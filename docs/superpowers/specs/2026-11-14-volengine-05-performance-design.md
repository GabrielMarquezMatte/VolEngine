# VolEngine — plano 5: performance (design)

Data: 2026-11-14. Detalha o plano 5 do roadmap: a seção 8 ("Performance") e o objetivo 5 da spec-mãe
`docs/superpowers/specs/2026-11-10-volengine-design.md`. Tudo o que a spec-mãe fixa vale aqui sem repetir:
- regras de estabilidade da interface C;
- bit-identidade entre números de threads na mesma máquina e ISA;
- oráculos e tolerâncias;
- licença (nada de código GPL copiado);
- estilo (`STYLEGUIDE.md`).

## 1. Intenção e o porquê

O diferencial da biblioteca é **velocidade** numa biblioteca **poliglota**: um núcleo nativo, chamado de Python
primeiro e depois de C++, C#, R e Rust pela mesma interface C. A meta não é cruzar uma linha e parar. É ficar **o
mais rápido possível sem perder legibilidade** (decisão do autor, 2026-11-14).

Código complexo pode ser legível. O `STYLEGUIDE.md` já cobre isso: performance e legibilidade andam juntas, e um
caminho quente medido pode ler pior, com a medição registrada em `docs/`. SIMD não é, por si, ilegível.

**Primeiro medir, depois otimizar.** O autor quer benchmarks rápidos e de escopo pequeno. Rodadas de segundos, não
de horas, que isolem uma parte do código, para trabalhar com resultado na mão.

## 2. Medida de referência

- **Carga principal:** `BM_TccGrid`, a grade de 1080 ajustes.
  - 20 séries × T = 5000;
  - {p, q ∈ 1..3} × {ARMA(0,0), ARMA(1,1)} × {norm, std, ged};
  - todas as threads da máquina.
- **Piso:** abaixo de 10 s (decisão do autor, 2026-11-14). Hoje está em 10,7 s com 16 threads e 68,5 s com 1, no
  Ryzen 7 5700X (`docs/performance/benchmarks.md`). Passar dos 10 s não encerra o plano.
- **Duas leituras do mesmo número:**
  - 1 thread mede o custo do algoritmo;
  - todas as threads mede também o balanceamento do lote.
- **Medidas secundárias,** para não piorar o que já existe:
  - `BM_DccGrid`;
  - `BM_FitOne` dos casos de `named_specs`;
  - Python contra `arch` (`docs/performance/python-vs-arch.md`).

**Build medido (decisão do Claude):** o preset `gcc-bench`, sem `-march=native`. As wheels e os binários
distribuídos são genéricos (x86-64 base), então o número que vale é o do build genérico. Um ganho que só aparece
com `-march=native` não conta. SIMD entra por despacho em tempo de execução. Um número com `native-release` pode
aparecer como informação extra, marcado como tal.

## 3. Divisão

Em três subplanos, escritos um de cada vez. O desenho de 5b e 5c é revisto com os números de 5a na mão; esta seção
fixa só a ordem e o que entra em cada um.

- **5a — medição e o que o profile mostrar no caminho escalar.**
  - O ferramental da seção 4.
  - Um profile da grade de 1080, publicado em `docs/performance/`.
  - As otimizações escalares que o profile apontar. Candidatos, a confirmar pelo profile:
    - menos avaliações por iteração na busca linear;
    - critério de parada;
    - valores iniciais;
    - ordem dos jobs no pool (mais caros primeiro, para encurtar a cauda);
    - alocações dentro do laço do otimizador;
    - custo do `Dual<N>`.
  - A correção do L-BFGS que trava em ridge (decisão do Claude: entra aqui porque é o mesmo caminho e muda o número
    de iterações). Ele aceita um passo plano ou falho quando max|g| ≤ √gtol·escala. Casos conhecidos:
    - os xfails estritos ALLGARCH skewed, ARFIMA csGARCH e `aparch11_vx_std`;
    - `arma22_s33_std` em `MAX_ITER`;
    - FDCC-mvnorm "convergido" no ponto inicial (`grid5`).
  - A corrida do `std::lgamma` em `signgam` (decisão do Claude: entra aqui porque está no caminho quente das
    densidades e é segurança entre threads).
  - A opção de erros-padrão da seção 5 (decisão do autor, 2026-11-14). O trabalho de inferência no ponto estimado
    custa, por estimativa do revisor, 20–30% da grade.
- **5b — SIMD.**
  - A forma sai do profile de 5a:
    - **faixas por ajuste:** L-BFGS em lote, uma faixa por ajuste da mesma spec, como a spec-mãe previa;
    - **vetorização dentro de um ajuste:** as operações do `Dual<N>` e os laços em t que já são independentes.
  - Google Highway é a dependência candidata (spec-mãe §10). Entra se 5a mostrar que o ganho está aí.
- **5c — os outros pontos quentes,** fora da grade de 1080, por ordem de custo medido:
  - **ghyp:** o gradiente custa 38 avaliações, porque o Bessel K recebe λ como `Dual`. Bessel K próprio ou
    derivada analítica em λ;
  - **Bessel K x87:** depende do número de threads (`promote_double<false>` em `dist/bessel.cpp`). Quebra a
    bit-identidade prometida, então entra mesmo sem ganho de tempo;
  - **perfil t da cópula:** ~37 s; o gradiente em ν o elimina;
  - **Kendall:** O(T²); trocar pelo algoritmo de Knight;
  - **quantil nig/ghyp:** ~290 µs por chamada; tabela monótona por margem;
  - **ARFIMA:** a diferença fracionária é O(T²); trocar por FFT;
  - **amostragem:** ziggurat;
  - **erros-padrão do DCC:** 3 s;
  - **precisão:** `dist::shifted_moment` / `probability_below_zero` com ged/sged em ν 200..1000.

  O ganho de cada um é medido antes; o que não aparecer em nenhuma carga real fica para depois, com `ponytail:`.

Cada subplano fecha com:
- oráculos verdes;
- antes/depois medido;
- `docs/performance/benchmarks.md` atualizado.

## 4. Ferramental de medição (5a)

O objetivo é responder em segundos onde está o tempo e se uma mudança ajudou.

- **Profile da grade (executável próprio, fora do Google Benchmark).**
  - Roda a grade de 1080 uma vez, ou um recorte por filtro: specs, séries ou threads.
  - Mede cada ajuste dentro da thread que o executa. Por ajuste, registra:
    - tempo;
    - iterações do L-BFGS;
    - avaliações da função com gradiente;
    - tentativas da busca linear;
    - status.
  - Agrega por spec, por modelo, por distribuição e por ordem, ordenado pelo tempo total.
  - Mede o aproveitamento do pool: a soma dos tempos por ajuste dividida por (threads × tempo real). Isso separa
    custo do algoritmo de cauda e desbalanceamento.
  - Saída em tabela no terminal, e em CSV ou JSON para comparar duas rodadas.
- **Benchmarks pequenos (Google Benchmark).** Cada um com menos de ~1 s por repetição. O conjunto rápido inteiro,
  com 3 repetições, em cerca de 1 minuto.
  - Uma versão reduzida da grade, representativa da de 1080, para o antes/depois de mudanças no otimizador. A
    composição sai do profile: as specs que dominam o tempo e uma amostra das outras.
  - Contadores de iterações e de avaliações em `BM_FitOne`.
  - Benchmarks de componente onde o profile apontar. Exemplos: a busca linear, uma iteração do L-BFGS, a
    recursão de variância isolada, a densidade isolada.
- **Contagem interna.** Iterações e avaliações vêm de contadores internos do núcleo (decisão do Claude), não da
  interface C.
- **Profile por amostragem.** AMD uProf (instalado na máquina do autor) sobre um preset com símbolos e frame
  pointers. O preset é novo, `gcc-profile`, com as mesmas flags de otimização do `gcc-bench`, para o profile medir
  o mesmo código.
  - Se o uProf não ler os símbolos DWARF do MinGW-gcc, vale outro profiler ou outro compilador só para localizar o
    ponto quente (decisão do Claude). Por exemplo, clang com `-gcodeview`.
  - O profile diz **onde** está o tempo. **Quanto** se ganhou vem sempre do `gcc-bench`.
- **Comandos documentados.** Em `docs/performance/`: o conjunto rápido, o profile da grade, o profile com uProf. Um
  comando curto para cada.
- **A/B na mesma sessão.** Toda otimização registra antes e depois medidos em sequência, na mesma sessão da
  máquina, com o conjunto rápido. Nunca contra uma tabela antiga: o ruído entre dias, nesta máquina, passa do
  desvio-padrão.

## 5. Restrições

- **Correção vem primeiro.** Os testes de oráculo (`rugarch`, `rmgarch`, `arch`) continuam verdes nas mesmas
  tolerâncias. A comparação primária é a da spec-mãe: ℓ do VolEngine ≥ ℓ do oráculo menos a tolerância.
  - Um xfail estrito que passar a passar sai do xfail.
  - Uma mudança no otimizador pode mover parâmetros dentro das tolerâncias. Em 0.x isso não é quebra.
- **Bit-identidade entre números de threads** na mesma máquina e ISA continua valendo.
  - O resultado de um job não depende de quantas threads rodam, nem de quais outros jobs estão no lote. Vale também
    no SIMD em lote: a faixa de um ajuste não pode ser afetada pelas faixas vizinhas.
  - Entre ISAs (despacho do Highway, FMA), só dentro de tolerância, como na spec-mãe.
  - Fica para o design de 5b se um job sozinho (`ve_fit`) e o mesmo job dentro de um grupo de faixas SIMD precisam
    ser bit-idênticos entre si. A resposta decide se o caminho vetorial e o fallback escalar podem usar exp/log e FMA
    diferentes.
- **Interface C: só uma adição.** Nenhuma assinatura existente muda.
  - A adição é `ve_fit_options_set_standard_errors(o, on)`, no padrão de `ve_dcc_options_set_standard_errors`
    (decisão do autor, 2026-11-14). Ela liga ou desliga o trabalho de inferência no ponto estimado: erros-padrão
    pela Hessiana numérica, erros-padrão robustos, scores e Nyblom.
  - O padrão é ligado, então o comportamento atual não muda. Desligado, os acessores desse trabalho devolvem um
    erro de argumento que diz como ligá-lo. ℓ, critérios, série ajustada e estado do forecast continuam sempre
    calculados.
  - O Python expõe a mesma opção.
  - O uso previsto: a grade roda com a opção desligada, e o modelo escolhido é reajustado com ela ligada.
  - `BM_TccGrid` é medido das duas formas.
  - Qualquer outra assinatura nova volta ao autor.
- **Gradiente:** o caminho do gradiente (`Dual<N>` em modo forward) só muda depois do profile (decisão do autor,
  2026-11-14). Se o profile mostrar que ele domina, o autor recebe os números e as opções antes de qualquer
  código. As opções são:
  - um `Dual` mais barato;
  - AD reverso genérico;
  - adjunto escrito à mão.
- **`tol` mantém o significado documentado** (decisão do Claude):
  - critério max|g| ≤ tol·max(1, |f|), com padrão 1e-6 e `max_iter` 500;
  - a correção do ridge muda só a aceitação com √gtol;
  - ganho em parada vem de não gastar iterações depois que o critério já foi atingido, não de afrouxar o critério.
- **Mudanças no otimizador são validadas além da grade** (decisão do Claude):
  - nos casos de `BM_FitOne` de `named_specs`, em `BM_DccGrid` e nos oráculos;
  - uma piora acima do ruído em qualquer um deles bloqueia a mudança, até o autor aceitar a troca com os números
    na mão.

  A grade é de um único processo gerador (sGARCH(1,1)-std). Ajustar o otimizador só a ela pode piorar ARMA(2,2),
  eGARCH ou DCC.
- **Legibilidade:**
  - o `STYLEGUIDE.md` vale inteiro;
  - SIMD fica atrás de tipos e funções com nome, nunca intrínsecos espalhados;
  - um ganho que lê pior exige a medição registrada em `docs/` (regra de Priorities do styleguide).
- **Dependências** entram quando um benchmark mostra ganho, com licença compatível com MIT e via FetchContent, como
  as atuais.
- **Plataformas:** GCC, Clang e MSVC compilam.
  - Todo caminho SIMD tem fallback escalar.
  - O despacho é em tempo de execução, não `-march=native`.

## 6. Não-objetivos

- GPU e `float` (spec-mãe). Mudança de assinatura existente da interface C.
- Otimizar o que não aparece em nenhuma carga medida.
- Reescrever a camada Python. Só entra se o profile mostrar que o overhead de ctypes pesa num caso real.

## 7. Anti-objetivos

- **Ganho sem medição.** Toda mudança de performance tem antes/depois registrado.
- **Ganho que só existe com `-march=native`,** reportado como se fosse o da biblioteca distribuída.
- **Velocidade comprada com divergência dos oráculos,** ou com perda de bit-identidade entre threads.
- **Intrínsecos espalhados pelo núcleo,** ou código que só o autor da otimização entende.
- **Benchmark que só roda em horas** como única forma de saber se uma mudança ajudou.

## 8. Deixado ao builder

- Nome, formato de saída e flags do executável de profile da grade.
- Composição exata da grade reduzida e dos benchmarks de componente, a partir do profile.
- Quais otimizações escalares entram em 5a e em que ordem, pelo profile.
- Como corrigir o critério de parada do L-BFGS, desde que os casos conhecidos da seção 3 passem e nenhum oráculo
  piore.
- Onde ficam os contadores internos de iterações e avaliações.

## 9. Decisões tomadas pelo Claude (para conferir)

- O número oficial é o do build genérico (`gcc-bench`); `-march=native` é só informação extra.
- A correção do L-BFGS em ridge e a corrida do `lgamma` entram em 5a; o Bessel K x87 entra em 5c, junto com ghyp.
- Contadores de iterações e avaliações são internos, sem mudar a interface C.
- `tol` mantém o significado e o padrão documentados.
- Mudança no otimizador que piore acima do ruído um caso fora da grade é bloqueada até o autor aceitar.
- Se o uProf não ler os símbolos do MinGW-gcc, outro profiler ou compilador serve só para localizar o ponto quente.
- A bit-identidade entre um job sozinho e o mesmo job num grupo SIMD fica para o design de 5b.
- Novo preset `gcc-profile` para o uProf, com as mesmas flags de otimização do `gcc-bench`.
- 5b e 5c são redesenhados com os números de 5a antes de virar plano.
