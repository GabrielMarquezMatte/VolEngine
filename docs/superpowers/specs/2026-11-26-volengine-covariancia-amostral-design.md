# Covariância amostral e Ledoit-Wolf — desenho

Data: 2026-11-26. Versão alvo: 1.3.0 (ABI 1, aditiva), no mesmo release do cancelamento e dos acumulados.

## Intenção

O VolEngine monta modelos de risco, não só GARCH. A covariância sem modelo de um painel de retornos é o ponto de
partida de quase todo risco de carteira, e hoje o DataPanel a mantém por conta própria
(`Services/Risk/Estimators/Covariances.cs`). Com os estimadores no motor, o batch, a API e qualquer outro usuário
partem da mesma definição, testada contra `numpy` e `sklearn`.

É o primeiro módulo sem relação com o GARCH: pasta própria (`src/cov/`), prefixo próprio (`ve_cov_*`), mesma
biblioteca e mesma C ABI.

## Definição

Painel X com T datas e N séries; x̄_j a média da série j (soma da esquerda para a direita, dividida por T) e
Z = X − x̄ o painel centrado.

- **Amostral:** S = ZᵀZ / (T − 1), como `numpy.cov(X, rowvar=False)`.
- **Ledoit-Wolf** (2004, alvo μ·I), como `sklearn.covariance.ledoit_wolf(X, assume_centered=False)`:
  - E = ZᵀZ / T (divisor T, não T − 1) e μ = tr(E) / N;
  - β = (Σ_ij [(Z∘Z)ᵀ(Z∘Z)]_ij / T − ‖E‖²_F) / (N·T);
  - δ = ‖E − μI‖²_F / N, calculado como a soma dos quadrados da diferença (nunca negativo; exatamente 0 com uma
    série, em que a intensidade do sklearn é 0);
  - intensidade s = min(β, δ) / δ, e s = 0 quando δ = 0 (todas as séries constantes, ou uma só série);
  - resultado (1 − s)·E + s·μ·I, com s em [0, 1].
- As duas matrizes saem simétricas bit a bit: calcula-se o triângulo superior e espelha-se.

## Objetivos

- Duas funções puras na C ABI, sobre o `ve_panel` que já existe (strides incluídos).
- Bindings C++, C# e Python.

## Não-objetivos

- Outros estimadores (OAS, Ledoit-Wolf com correlação constante, EWMA sobre o painel): entram depois como funções
  novas, se alguém pedir.
- `assume_centered`, pesos por data, dados faltantes.
- Threads, contexto e cancelamento: a conta é O(T·N²) e roda numa thread; uma variante com `ve_context*` pode entrar
  depois sem quebrar a ABI, se alguém medir que precisa.

## Anti-objetivos

- Uma matriz que não seja simétrica bit a bit.
- Aceitar em silêncio um painel com intervalos por série, NaN ou infinito.

## Abordagens

- **Escolhida: funções puras com buffer do chamador**, como `ve_fit_estimates`: o chamador já sabe N pelo painel,
  sem handle e sem free.
- Handle `ve_cov` com views: mesmo padrão dos resultados de ajuste, mas um free e um tipo opaco sem ganho, já que
  nada é calculado sob demanda nem vive além da chamada. Descartada.
- Funções com `ve_context*` (threads e cancelamento): só compensa com milhares de séries. Adiada.

## C ABI

```c
/* Covariance estimators over a T x N panel of returns (series in columns), with each series' mean removed. `out`
   holds N * N doubles and receives the matrix row-major, exactly symmetric. The panel must be rectangular (no
   ve_panel_set_ranges), with t >= 2, n >= 1 and every value finite; otherwise VE_INVALID_ARGUMENT and `out` is left
   unchanged. Since 1.3.0. */
/* The sample covariance Z'Z / (t - 1), as numpy.cov(rowvar=False). */
VE_API ve_status ve_cov_sample(const ve_panel* panel, double* out);
/* The Ledoit-Wolf (2004) shrinkage of Z'Z / t towards mu * I, mu the mean variance, as
   sklearn.covariance.ledoit_wolf(assume_centered=False); *shrinkage receives the intensity, in [0, 1]. */
VE_API ve_status ve_cov_ledoit_wolf(const ve_panel* panel, double* out, double* shrinkage);
```

- Painel nulo: `VE_INVALID_HANDLE`; `out` ou `shrinkage` nulos: `VE_INVALID_ARGUMENT`.
- Contrato: `volengine.abi` ganha as duas linhas.

## Núcleo

- `src/cov/` com funções puras sobre `PanelView`, sem saber da C ABI; a validação do painel fica na camada C.
- Reaproveitar o que `src/linalg/` já tiver para produtos internos, se couber.

## Bindings

- C++: `cpp/include/volengine/cov.hpp`, `ve::cov::sample(const Panel&) -> Result<std::vector<double>>` e
  `ve::cov::ledoit_wolf(const Panel&) -> Result<LedoitWolf>`, com `LedoitWolf{std::vector<double> covariance;
  double shrinkage;}`.
- C#: classe estática `Covariances` com `Sample(double[,] returns)` e `LedoitWolf(double[,] returns)`, entrada
  datas × séries como os regressores de `Fit`, saída `double[][]` (N linhas de N) como `DccFit.Covariance`, e
  `LedoitWolf` devolve um record `(double[][] Covariance, double Shrinkage)`.
- Python: subpacote `volengine.cov` com `sample(returns)` → ndarray (N, N) e `ledoit_wolf(returns)` →
  `(ndarray, float)`; entrada array 2-D (T, N), qualquer stride, como os painéis atuais.

## Testes

- Oráculo: um script em `tests/reference/` escreve `fixtures/sklearn_cov.json` com `numpy.cov` e
  `sklearn.covariance.ledoit_wolf` sobre `data/copula3.csv` (3 séries, T = 2000) e sobre um painel largo gerado com
  semente fixa (N perto de T, para a intensidade ficar longe de 0 e de 1). Os testes Python comparam com tolerância
  relativa 1e-12 e não precisam do sklearn.
- Núcleo (C++): uma série dá intensidade 0 e a variância de E; séries constantes dão δ = 0, intensidade 0 e matriz
  nula; simetria bit a bit; painel com strides (coluna) igual ao contíguo; valores à mão num painel 3 × 2.
- C ABI: intervalos por série, t < 2, n = 0, NaN, infinito e ponteiros nulos dão o status certo e não tocam `out`.
- Bindings: os valores batem com a C ABI.

## Documentação

- Os três guias ganham uma seção curta "Covariance estimators", com a definição e a correspondência com numpy e
  sklearn.
- `CHANGELOG.md`, `[Unreleased]`, Added.

## Deixado para quem constrói

- O tamanho do painel largo do oráculo, desde que os testes fiquem rápidos.
- Onde os nomes C# e Python ficam dentro do limite de 8 módulos por pasta.
