# VolEngine — plano 5: gradiente adjunto (design)

Data: 2026-11-14. Faz parte do plano 5 (performance) e segue o design-mãe
`docs/superpowers/specs/2026-11-14-volengine-05-performance-design.md`. Tudo o que ele fixa vale aqui: as
restrições da §5, a medição A/B da §4, os não-objetivos e os anti-objetivos. Fica entre o 5a (feito) e o 5b (SIMD).

## 1. Intenção e o porquê

Depois do 5a, o gradiente em modo forward (`ad::Dual<N>`) continua sendo quase todo o tempo de um ajuste. O
otimizador gasta 89% do ajuste, e cada avaliação dele calcula f e ∇f com `Dual`, que custa de 3× a 4× uma avaliação
em `double`. O autor escolheu a opção (3) do design-mãe: um design próprio para o gradiente reverso, decidido com
medida (2026-11-14).

A meta continua a do plano 5: o mais rápido possível sem perder legibilidade.

## 2. O que o spike mediu

O spike era descartável e foi desfeito. Ele mediu uma avaliação da log-verossimilhança com gradiente, com
T = 5000, sGARCH(p,q)-ARMA(m,n)-std, build `gcc-bench`, mediana de 3, em µs:

| Caso | k | `double` (sem ∇) | `Dual` (hoje) | `Dual` + AVX2 | Tape genérica | Adjunto à mão |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| s11 arma00 | 5 | 106 | 281 | 280 | 552 | 142 |
| s11 arma11 | 7 | 114 | 306 | 261 | 639 | 174 |
| s22 arma11 | 9 | 137 | 424 | 344 | 757 | 205 |
| s33 arma11 | 11 | 150 | 472 | 378 | 892 | 230 |

- Os gradientes da tape e do adjunto batem com o `Dual` até 1e-12 relativo.
- Com o adjunto ligado ao L-BFGS, nos 360 ajustes std da grade (1 thread, sem erros-padrão), o tempo foi de
  8,62 s para 4,71 s, 0,55 do tempo. Os 360 terminaram `CONVERGED`, com as mesmas avaliações (razão 0,999), e a
  maior mudança de ℓ foi de 4e-8.
- A cadeia até o espaço u continuou pelo `Dual` de `to_natural`, O(k²) por avaliação, e não pesou.

## 3. Abordagens

- **Tape genérica (AD reverso), descartada pela medida.**
  - Um tipo `ad::Var` com fita por thread entrava no mesmo `T` dos templates, sem código por modelo.
  - Saiu ~2× mais lento que o `Dual`: ~165 mil nós por avaliação, cerca de 4 MB, mais que o L2.
  - Bibliotecas do mesmo tipo:
    - Adept (Apache-2.0) só reduziria o custo por nó; no máximo empataria com o `Dual`;
    - CoDiPack, XAD e CppAD têm licença incompatível com MIT.
- **SIMD no `Dual` (AVX2), descartado como alternativa.** Ganha de 0 a 20%, contra ~2× do adjunto. Pode voltar no
  5b para o que continuar no `Dual`.
- **Adjunto à mão, por componente, com o `Dual` como fallback (escolhida).**
  - Avaliação com gradiente ≈ 1,3–1,5× uma em `double`, ~2× mais rápida que o `Dual`.
  - Custo: código de derivada por componente, que precisa de verificação automática contra o `Dual` para não
    divergir em silêncio.

## 4. Desenho

### 4.1 Forma

O gradiente de ℓ em θ (espaço natural) é calculado em dois passes:
1. **Forward:** a própria `model::loglik<double>`, com as saídas `residuals` e `sigma2` (e `deviations` quando
   preciso) gravadas.
   - O valor de ℓ sai bit-idêntico ao do caminho `double` de hoje, e o forward não ganha código novo.
2. **Reverse:** os componentes na ordem inversa de `evaluate`. Cada um recebe os adjuntos das suas saídas e
   acumula os das entradas e de θ:
   1. densidade: ∂ℓ_t/∂e_t, ∂ℓ_t/∂σ²_t e ∂ℓ/∂(forma);
   2. recursão de variância: propaga ∂/∂σ²_t (ou ∂/∂log σ²_t no eGARCH) para trás em t. Acumula ∂/∂(ω, α, β, γ),
      ∂/∂e_{t-i} e o adjunto do presample;
   3. presample: com `VarInit::Sample`, o valor depende de e (média de e²), então devolve 2e_t/T vezes o seu adjunto
      para cada e_t; com `Backcast` é constante;
   4. média: a recursão ARMA invertida em t, que acumula ∂/∂(μ, ar, ma).

Cada `backward` mora no mesmo arquivo do forward que deriva, logo abaixo dele, com a fórmula da derivada no
comentário (`garch/sgarch.hpp`, `garch/gjrgarch.hpp`, `garch/egarch.hpp`, `mean/arma.hpp`, `dist/symmetric.hpp`). A
composição fica num arquivo novo do `model`, ao lado de `likelihood.cpp`.

**Constantes em forma.** As constantes que dependem só da forma reutilizam as funções templated de hoje com um
`Dual<1>` na forma, sem derivada à mão: a constante de log da std e da ged, a escala da ged e o E|z| do eGARCH.
Custo O(1) por avaliação.

### 4.2 Fronteira da fase 1 (decisão do autor, 2026-11-14)

O adjunto cobre:
- **Variância:** sGARCH, iGARCH (β_q = 1 − Σα − Σ_{j<q} β_j), gjrGARCH e eGARCH, qualquer (p, q) ≤ `kMaxOrder`;
- **Densidades:** norm, std e ged;
- **Média:** ARMA(m, n), com ou sem média;
- **Presample:** `VarInit::Sample` e `Backcast`.

Fica no `Dual`, sem mudar de comportamento:
- família power/fGARCH, apARCH, csGARCH;
- sstd, sged, nig, ghyp, jsu;
- regressores na média ou na variância, ARCH-in-mean e ARFIMA;
- variance targeting.

Ordem prevista depois da fase 1, por custo medido em cargas reais: família power/fGARCH, csGARCH, densidades
assimétricas e jsu, regressores, ARCH-in-mean, ARFIMA. O ghyp entra com o Bessel do 5c. Cada uma é um plano curto,
não parte deste.

Uma única função decide a cobertura: `has_adjoint(const Layout&, VarInit)`. É uma lista explícita do que está
coberto, não do que falta. Uma combinação nova cai no `Dual` por padrão.

### 4.3 Onde entra

- **No objetivo do L-BFGS** (`NegativeMeanLoglik` em `fit/fit.cpp`):
  - quando `has_adjoint`, θ e a Jacobiana dθ/du saem de `to_natural<Dual<N>>`, como hoje;
  - ℓ e ∂ℓ/∂θ saem do adjunto;
  - ∇_u = Jᵀ ∂ℓ/∂θ;
  - o resto (escala, não-finito → +∞ com gradiente zero) não muda.
- **No `LoglikGradient`** (gradiente em θ natural, chamado 2k vezes pela Hessiana numérica dos erros-padrão): usa o
  adjunto quando coberto. O trabalho de inferência também fica mais barato.
- **Fica no `Dual`:** scores por observação (OPG, erros-padrão robustos, Nyblom), porque o modo reverso dá só o
  gradiente da soma. Também ficam o estágio de correlação do DCC e as cópulas (k pequeno, fora do escopo).
- **Erros-padrão do DCC:** os blocos univariados da Hessiana (`fit::loglik_hessian`, chamado de
  `dcc/standard_errors.cpp`) passam pelo `LoglikGradient` e herdam o adjunto (decisão do Claude). Mudam nos últimos
  dígitos; os oráculos do rmgarch continuam verdes, e `BM_DccStandardErrors` entra no A/B.
- **Interface C:** nada muda.

### 4.4 Domínio e valores não finitos

- O adjunto só roda quando o forward devolve ℓ finito. Fora do domínio (σ² ≤ 0 ou não finito, forma fora do
  domínio), o caminho é o de hoje: +∞ no objetivo, `false` no `LoglikGradient`.
- Um gradiente não finito com ℓ finito é tratado como hoje trata o `Dual`: o `LoglikGradient` devolve `false`; o
  L-BFGS recebe o vetor como está. Exemplos: |z| = 0 na ged com potência < 1; z = 0 no |z| do eGARCH, onde o `Dual`
  usa sinal 0.
- As derivadas nos pontos não diferenciáveis seguem a convenção do `Dual` em cada caso:
  - `abs` com sinal 0 em 0;
  - no gjr, o indicador 1[e < 0] tem derivada 0;
  - na `pow` da ged, a mesma regra de `pow_slope`.

## 5. Verificação

- **Rede de segurança obrigatória, numa suíte própria (teste de propriedade):**
  - para cada combinação coberta por `has_adjoint` — modelo × densidade × (com e sem média) × ARMA ∈ {(0,0), (1,1),
    (2,2)} × (p, q) ∈ {(1,1), (2,1), (1,2), (3,3)} × `VarInit` —, compara ℓ e ∂ℓ/∂θ do adjunto com o
    `Dual<kMaxParameterCount>` em vários θ, entre eles perto das bordas: persistência ~0,999, forma pequena e
    grande, ν > 10 (ramo Stirling da std);
  - no gradiente, |adjunto − Dual| ≤ 1e-10 · max(1, ‖∇ do Dual‖∞) por componente (decisão do Claude);
    - a escala é a da norma do gradiente, não a de cada componente: perto do ótimo, de ν grande ou de
      persistência ~0,999, há componentes perto de zero por cancelamento de termos grandes, e a ordem diferente
      das somas faria um teste por componente falhar com os dois gradientes corretos;
  - ℓ bit-idêntico ao `loglik<double>`;
  - séries curtas, em que held ≥ T e o presample cobre tudo.
- **Cobertura:** um teste confirma que `has_adjoint` é falso fora da fronteira da §4.2 (para um exemplar de cada
  item que fica no `Dual`).
- **Ajuste:**
  - os oráculos (rugarch, arch) continuam verdes nas tolerâncias atuais;
  - os testes de bit-identidade entre números de threads continuam verdes;
  - na grade (`compare-profile`), os status e os ℓ ficam iguais aos de hoje dentro do esperado para ulps de
    gradiente. O spike mostrou mudança máxima de 4e-8 em ℓ.
  - Pela multimodalidade já registrada, um ajuste sobreparametrizado pode trocar de ótimo; se acontecer, vai para o
    autor com os números, como na memória 40.
- **Medida (A/B na mesma sessão, design-mãe §4):**
  - `BM_Loglik` ganha a variante `/adjoint` nos casos cobertos;
  - grade de 1080 com 1 e 16 threads, com e sem erros-padrão;
  - `BM_FitOne` dos casos nomeados;
  - `BM_DccGrid`, que não deve mover;
  - Python contra `arch`.
  - Resultados em `docs/performance/` (novo `profile-adjoint.md` ou seção nova de `profile-5a.md`, à escolha do
    builder) e `benchmarks.md`.

## 6. Não-objetivos

- AD reverso genérico (medido, §3).
- Adjunto dos scores por observação: eles continuam no `Dual`.
- DCC e cópulas.
- Mudar o otimizador, a parametrização ou a interface C.
- Cobrir nesta fase o que a §4.2 deixa no `Dual`.

## 7. Anti-objetivos

- **Um adjunto que diverge do forward em silêncio:** toda combinação coberta tem o teste de propriedade da §5.
- **Cobertura implícita**, em que uma combinação nova cai num adjunto que não a conhece: `has_adjoint` é uma lista
  explícita.
- **Duas implementações do forward:** o forward é sempre `loglik<double>`; só o reverse é código novo.
- **Derivadas ilegíveis:** cada `backward` tem a fórmula no comentário e segue o styleguide inteiro (≤ 60 linhas, ≤ 5
  parâmetros, aninhamento ≤ 3, sem `else`).

## 8. Deixado ao builder

- O nome e a assinatura exata das funções `backward` e da composição. A composição recebe `Data`, θ, `Layout`,
  `VarInit` e os buffers, e devolve ℓ e ∂ℓ/∂θ.
- Onde ficam os buffers de adjuntos: reaproveitados por objeto, como os do `NegativeMeanLoglik`, sem alocação por
  avaliação.
- Como o reverse obtém o que o forward não grava: o presample é recalculado; o log σ² do eGARCH é recalculado ou
  gravado.
- A ordem de implementação dentro da fase 1. Sugestão: sGARCH-norm/std/ged com ARMA primeiro, que é a grade, depois
  iGARCH, gjr e eGARCH.

## 9. Decisões tomadas pelo Claude (para conferir)

- O forward reaproveita `loglik<double>`, para o valor de ℓ ficar bit-idêntico e não existir um segundo forward.
- As constantes que dependem só da forma usam `Dual<1>` sobre as funções existentes, sem derivada à mão.
- O `LoglikGradient` da Hessiana numérica também usa o adjunto; os scores ficam no `Dual`.
- Variance targeting fica fora da fase 1: ω depende da média de e² e da persistência, e ninguém pediu.
- Tolerância do teste de propriedade: 1e-10 relativa à norma ∞ do gradiente, contra 1e-12 a 4e-13 observados no
  spike.
- Os blocos univariados dos erros-padrão do DCC herdam o adjunto pelo `LoglikGradient`.

## 10. Feito (2026-11-14)

Plano `docs/superpowers/plans/2026-11-14-volengine-05-adjunto.md`, commits `bec4f04..` em `master`. Números em
`docs/performance/profile-5a.md` (seção "Adjoint gradient") e `docs/performance/benchmarks.md`.

- **Grade de 1080:** 1 thread 31,1 s → 22,0 s (0,71); todas as threads 4,24 s → 2,72 s; sem erros-padrão 2,13 s.
  Todos `CONVERGED`, ℓ muda no máximo 3,4e-8, avaliações iguais (razão 1,001).
- **Python contra `arch`:** 2,46× → 2,77×.
- **Abaixo da estimativa da §2 (0,55):** o adjunto custa ~2× uma avaliação `double`, não 1,3–1,5× como no spike.
  O reverse recalcula os transcendentais que o forward `loglik<double>` já fez: `log1p` na std (~27% do adjunto),
  `pow`/`log` na ged. Por distribuição na grade: norm 0,63, std 0,70, ged 0,81.
- **Decisão pendente do autor:** um forward fundido com as parciais dos termos (o valor seguiria bit-idêntico, com
  teste) tiraria esse recálculo, ao custo de uma segunda implementação dos termos da densidade, o que hoje é
  anti-objetivo da §7.
- `src/model/` passou de 8 módulos: os internos da verossimilhança foram para `src/model/likelihood/` (`terms.hpp`,
  `parameters.hpp`, `adjoint.{hpp,cpp}`).

