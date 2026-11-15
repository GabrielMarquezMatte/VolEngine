# VolEngine — plano 5b: SIMD (design)

Data: 2026-11-14. Detalha o 5b do design-mãe `docs/superpowers/specs/2026-11-14-volengine-05-performance-design.md`.
Tudo o que ele fixa continua valendo aqui:
- restrições da §5 (oráculos, bit-identidade entre números de threads, interface C, legibilidade, dependências,
  plataformas);
- medição A/B da §4;
- não-objetivos e anti-objetivos.

Vem depois do 5a e do gradiente adjunto (`2026-11-14-volengine-05-adjoint-design.md`).

## 1. Intenção e o porquê

O objetivo é o mesmo do plano 5: o mais rápido possível sem perder legibilidade. O que o autor quer derrubar é o
**pipeline do TCC de ponta a ponta, com todas as threads** (decisão do autor, 2026-11-14):
- a grade univariada de 1080 ajustes (`BM_TccGrid`): 2,72 s;
- depois, a grade do DCC sobre as 21 séries (`BM_DccGrid/threads:hw`, 9 specs): 2,35 s.

O DCC é quase metade do total e não ganhou nada desde o 4a. Ele está limitado pelo caminho crítico: 9 jobs para 16
threads, e o ajuste mais longo (FDCC) leva ~1,5 s.

## 2. O que já foi medido

Spike descartável sobre o DCC, m = 21, T = 5000, 1 thread, `gcc-bench`:

- **O otimizador é ~99% de cada ajuste do DCC.** `reported_loglik` leva 11 ms e o estado final 2 ms.
- **Poucas avaliações, cada uma cara.** Um ajuste faz de 15 a 26 avaliações. Uma avaliação com gradiente custa ~28 ms
  no DCC e ~57 ms no FDCC, contra 11 ms sem gradiente (~2,5×).
- **A largura do `Dual` não importa.** Com `Dual<2>` ou `Dual<4>` no lugar de `Dual<8>`, o tempo ficou igual (0,44 s
  contra 0,46 s). O custo está na aritmética por t, não nos lanes.
- **Por t, duas metades de tamanho parecido:**
  - a recursão de Q_t e a normalização para R_t: ~3m² operações, elemento a elemento, independentes entre os m²
    elementos;
  - o Cholesky, o solve e a forma quadrática: ~m³/6 multiplicações com somas e ~m²/2 divisões.

No univariado, depois do adjunto, a avaliação coberta é ~metade forward e ~metade reverse. O caro são as
transcendentais da densidade (`log1p`, `log`, `pow`), uma vez por t. Na std, o reverse recalcula `log1p`, ~27% do
tempo do adjunto. As recursões (σ²_t, ARMA) são sequenciais em t.

## 3. Abordagens

- **SIMD dentro de uma avaliação — escolhida pelo autor (2026-11-14).**
  - Vetoriza o que é independente dentro da avaliação: os elementos da matriz no DCC; as observações t no laço da
    densidade univariada.
  - Não exige otimizador em lote nem threads aninhadas.
- **Threads dentro de um ajuste do DCC.** Usaria as 7 threads ociosas no FDCC. A recursão sequencial limita a um
  pipeline (recursão → Cholesky), com ganho ≤ 2× e o pool mais complexo. Fica para depois, se a medida pedir.
- **Faixas por ajuste** (L-BFGS em lote, a ideia original da spec-mãe). Descartada: as 9 specs do DCC são todas
  diferentes, e o otimizador em lockstep com máscaras é o mais caro de ler e de testar.
- **`Dual` mais estreito no DCC.** Medido e descartado: não muda o tempo (§2).

## 4. Desenho

### 4.1 Dependência e organização

- **Google Highway** (Apache-2.0 ou BSD-3), via FetchContent como as outras dependências.
  - Versão fixa: 1.2.0 (decisão do Claude), que é anterior a esta data.
  - Despacho em tempo de execução com `HWY_DYNAMIC_DISPATCH`; build genérico x86-64, sem `-march=native`.
  - O fallback escalar é o target escalar/EMU128 do próprio Highway.
- **Os kernels moram num módulo próprio, `src/simd/`** (decisão do Claude).
  - Os nomes são do domínio: por exemplo `simd::cholesky_factor`, `simd::dcc_update`, `simd::student_terms`.
  - O resto do código chama funções sobre `std::span<double>` e nunca vê um tipo do Highway.
  - Intrínsecos e tags do Highway só aparecem dentro de `src/simd/` (anti-objetivo da §7 do design-mãe).

### 4.2 DCC: valor

Por t, com kernels vetoriais sobre `double`:
- a recursão de Q_t (DCC, aDCC, FDCC) e a normalização R_t = diag(Q)^-½ Q diag(Q)^-½, elemento a elemento;
- Cholesky de R_t, solve e forma quadrática, com os produtos internos e as atualizações de linha vetoriais;
- log |R_t| a partir da diagonal do fator.

A densidade por t (mvnorm, mvt, mvlaplace) continua escalar: são uma ou duas transcendentais por t, contra ~m³
operações de matriz. O Bessel K do mvlaplace continua na Boost.

### 4.3 DCC: gradiente — forward na recursão, reverse dentro de t (decisão do autor, 2026-11-14)

O objetivo do DCC deixa de rodar em `Dual<N>`, cujo layout AoS (valor e lanes intercalados por elemento) impede SIMD
sobre a matriz.

- **Forward na recursão:** k matrizes tangentes dQ_t^(i), uma por parâmetro livre, cada uma uma matriz `double`.
  - A recursão de Q_t é linear em Q_{t-1}, então dQ_t^(i) segue a mesma recursão, mais o termo direto de ∂/∂θ_i.
  - Usa os mesmos kernels elemento a elemento do valor.
- **Reverse dentro de t:** a densidade de t é puxada uma vez para ∂ℓ_t/∂R_t, com
  d log|R| = ⟨R⁻¹, dR⟩ e d(zᵀR⁻¹z) = −wᵀ dR w (w = R⁻¹z), e depois pela normalização para ∂ℓ_t/∂Q_t.
  - A inversa R⁻¹ sai do fator de Cholesky já calculado.
- **Contração:** ∂ℓ_t/∂θ_i = ⟨∂ℓ_t/∂Q_t, dQ_t^(i)⟩ + o termo direto da forma. Cada parâmetro custa O(m²) por t, sem
  Cholesky tangente.
- **O `Dual` do DCC continua** como referência dos testes e como fallback do que este caminho não cobrir.

### 4.4 Univariado: densidade vetorial e forward fundido

- **Um único laço vetorial para norm, std e ged** (decisão do autor, 2026-11-14), em todo modelo que passa por
  `sum_of_terms`, com ou sem adjunto (csGARCH, família power, ARFIMA e regressores incluídos).
  - Usa `Log`, `Log1p` e `Exp` do Highway; para a ged, y^p = exp(p·log y).
  - Nos modelos sem adjunto, o ℓ em `double` diverge em ulps do valor do `Dual` do otimizador; os testes desse par
    passam a usar tolerância.
- **O forward do adjunto é fundido** (decisão do Claude, lida na escolha da abordagem 1, que o incluía; abre a exceção
  ao anti-objetivo §7 do design do adjunto). Valor e parciais de cada termo saem do mesmo kernel vetorial, e o reverse
  não recalcula transcendentais.
  - As recursões (média, variância) continuam sendo o código de `loglik<double>`.
  - Só o laço da densidade passa a ter duas formas: só o valor, e valor mais parciais. As duas ficam no mesmo kernel,
    e o valor de ambas sai bit-idêntico (teste).
- **As recursões em t continuam escalares:** sequenciais, não vetorizam dentro de um ajuste.
- **O que não tem adjunto** (família power, csGARCH, densidades assimétricas, nig, ghyp, jsu) continua no `Dual`,
  sem mudança.

### 4.5 Bit-identidade (decisão do autor, 2026-11-14: opção a)

- **Mantém-se:**
  - na mesma máquina, os resultados não dependem do número de threads, porque o target do despacho é um só por
    máquina;
  - o ℓ do adjunto é bit-idêntico ao de `loglik<double>`, porque os dois usam o mesmo kernel.
- **Muda:**
  - **o `Dual` e o `double` não batem mais em ulps.** O valor do `Dual` (fallback, scores) continua com o libm, e os
    caminhos `double` usam as transcendentais e a ordem de soma do Highway. Os testes que exigiam igualdade exata
    entre os dois (`DoubleAndDualAgreeOnValue`, os bit-exatos dos erros-padrão) passam a usar tolerância, com o
    valor medido registrado;
  - **entre máquinas de ISA diferente** (SSE4, AVX2, AVX-512), só dentro de tolerância, o que o design-mãe §5 já
    permite. Mudam o número de lanes, a ordem das somas, os polinômios e o FMA.
- **O `ad::Dual` continua independente de SIMD.**

## 5. Verificação

- **Kernels contra referência escalar, em todo target disponível na máquina** (o Highway permite testar cada target
  compilado). Cada kernel é comparado com uma implementação escalar direta, com a tolerância em ulps documentada por
  kernel; inclui os restos (m ou T que não são múltiplos do número de lanes).
- **Gradiente tangente do DCC contra o `Dual`,** como teste de propriedade:
  - DCC, aDCC e FDCC × mvnorm, mvt e mvlaplace;
  - vários θ, entre eles persistência perto de 1 e m pequeno (2, 3) e grande (21);
  - tolerância relativa à norma ∞ do gradiente, como no adjunto.
- **Univariado:**
  - o teste de propriedade do adjunto continua: gradiente contra o `Dual` em 1e-10, e ℓ do adjunto bit-idêntico a
    `loglik<double>`;
  - o valor de `loglik<double>` contra o `Dual`, dentro da tolerância registrada.
- **Oráculos** (rugarch, rmgarch, arch) verdes nas tolerâncias atuais, e bit-identidade entre números de threads
  verde.
- **Plataformas:** GCC, Clang e MSVC compilam e passam.
- **Primeira task: verificar a toolchain** (decisão do autor, 2026-11-14).
  - Fato a verificar: o MinGW-GCC pode não realinhar a pilha para os targets AVX2/AVX-512 do Highway (GCC bug
    54412).
  - Se só SSE4 rodar no `gcc-bench`, a decisão de qual build dá o número oficial volta ao autor antes de seguir.
- **Portão de medida cedo** (decisão do autor, 2026-11-14): logo depois dos kernels de valor do DCC, um A/B da
  avaliação com gradiente (caminho novo contra `Dual`). Se o caminho novo não vencer, o 5b fecha com o SIMD no valor
  e o gradiente do DCC fica no `Dual`.
- **Os erros-padrão do DCC** usam diferenças centrais de `correlation_terms<double>` (`dcc_scores`), não o `Dual`.
  O valor vetorial muda esses números: medir o ruído e dar tolerância aos testes bit-exatos.
- **Medida (A/B na mesma sessão):**
  - antes dos kernels, o profile do DCC no uProf, para confirmar a divisão recursão versus Cholesky;
  - depois `BM_DccFit`, `BM_DccGrid` (1 thread e todas), `BM_TccGrid` (1 e todas, com e sem erros-padrão),
    `BM_Loglik`, `BM_FitOne`, `BM_DccStandardErrors`, `BM_CopulaFit` e Python contra `arch`;
  - o número que conta é o do build genérico com despacho; `-march=native` só como informação.

## 6. Não-objetivos

- Faixas por ajuste (L-BFGS em lote) e threads dentro de um ajuste.
- Vetorizar as recursões sequenciais em t.
- Reverse na recursão do DCC ao longo de t: o reverse fica dentro de cada t (§4.3).
- SIMD no que continua no `Dual` (modelos e densidades sem adjunto).
- `float` e GPU.

## 7. Anti-objetivos

- Intrínsecos ou tipos do Highway fora de `src/simd/`.
- Um kernel sem teste contra a referência escalar em todos os targets disponíveis.
- Um gradiente tangente que diverge do `Dual` em silêncio.
- Ganho que só aparece com `-march=native`.
- Perder a bit-identidade entre números de threads na mesma máquina.

## 8. Deixado ao builder

- Os nomes e as assinaturas dos kernels; a organização interna de `src/simd/` (respeitando o limite de 8 módulos por
  pasta).
- O layout das matrizes tangentes (k matrizes separadas, ou um bloco intercalado por linha), o blocking do Cholesky e
  como obter R⁻¹ do fator, escolhidos pela medida.
- Se o FDCC precisa de um kernel próprio para a matriz de coeficientes por grupo.
- Se os erros-padrão do DCC passam ao caminho tangente ou ficam no `Dual`.
- Se as cópulas DCC herdam os kernels: elas compartilham a recursão.
- A ordem das tasks, desde que o profile venha primeiro e cada kernel entre com seu teste e seu A/B.

## 9. Decisões tomadas pelo Claude (para conferir)

- Highway 1.2.0, com os kernels isolados em `src/simd/`.
- O `Dual` do DCC fica como referência dos testes e como fallback.
- O forward fundido do adjunto entra aqui, lido da escolha da abordagem 1.
- Para a ged, y^p = exp(p·log y) com as funções do Highway.
- A densidade por t do DCC e o Bessel K continuam escalares.

## 10. Questões da revisão, respondidas pelo autor (2026-11-14)

1. Gradiente do DCC: forward na recursão, reverse dentro de t (§4.3).
2. Portão de medida cedo para o gradiente do DCC (§5).
3. Um único laço vetorial de densidade para todos os modelos (§4.4).
4. Toolchain verificada na primeira task (§5).

## 11. Feito (2026-11-15)

Resultados em `docs/performance/profile-5b.md` e `docs/performance/benchmarks.md`. O pipeline do TCC com todas as
threads (`BM_TccGrid` + `BM_DccGrid`) caiu de 5,05 s para 1,88 s (0,37).

- **Toolchain (§5):** o AVX2 roda no MinGW-GCC (o GCC bug 54412 não aparece: um milhão de chamadas em todo alinhamento
  de pilha, em todo target); o AVX3 compila, mas esta CPU não o executa. O MSVC precisou de `/wd4701` (um aviso do
  Boost no passo de geração de código, com `/WX`).
- **Profile (§5):** Cholesky e solve ~80% do objetivo em `Dual`, recursão ~15%.
- **Kernels de valor (§4.2):** em vez de produtos internos por linha da matriz (que mediu 2x mais lento que o
  escalar), as lanes vetoriais levam *observações t diferentes* (as R_t são independentes dado Q_t); a recursão é
  elemento a elemento e sem fusão, igual ao escalar, bit a bit.
- **Gradiente do DCC (§4.3), portão cedo:** passou com folga (`BM_DccGrid/threads:1` a 0,21 do tempo, limite 0,9). As
  tangentes dos pesos saem do `weights_of` instanciado em `Dual` (o mesmo código serve DCC, aDCC e FDCC), o pullback
  por t vai a Q_t sem Cholesky tangente, e o valor é o de `objective<double>` bit a bit.
- **Univariado (§4.4):** um laço vetorial de norm, std e ged em todo modelo, fundido com as parciais do adjunto; o
  ℓ do adjunto continua bit-idêntico ao de `loglik<double>`.
- **Bit-identidade (§4.5):** `double` e `Dual` diferem em ulps (tolerância de 1e-14 relativa, medida, nos testes
  do par); na mesma máquina, resultados independentes do número de threads; entre ISAs, dentro de tolerância.
- **Plataformas (§5):** GCC e MSVC compilam e passam; o Clang não está instalado nesta máquina.
- **Fora do que foi feito:** as cópulas DCC têm objetivo próprio, ainda em `Dual`, e não ganham (`BM_CopulaFit`
  inalterado); e os erros-padrão do DCC ficam em diferenças centrais do valor `double` (0,47 do tempo, só pelo kernel
  de valor), sem passar ao caminho tangente.
