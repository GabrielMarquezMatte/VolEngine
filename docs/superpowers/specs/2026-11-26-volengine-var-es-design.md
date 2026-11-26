# VaR e ES — desenho

Data: 2026-11-26. Versão alvo: 1.3.0 (ABI 1, aditiva), no mesmo release da covariância e dos acumulados.

## Intenção

O VolEngine monta modelos de risco. O VaR e o ES são a medida final que quase todo uso desses modelos produz, e hoje
o DataPanel os calcula por conta própria (`Services/Risk/Estimators/LossQuantiles.cs`), com o quantil da t vindo do
MathNet. Com eles no motor, o batch, a API e qualquer outro usuário obtêm o mesmo número, e o paramétrico usa a
mesma distribuição que o ajuste estimou (as 8 do motor), não só normal ou t.

Segundo módulo fora do GARCH: pasta própria (`src/risk/`), prefixo próprio (`ve_risk_*`).

## Definição

Confiança α em (0, 1), cauda p = 1 − α. VaR e ES são positivos como perda, em unidades do retorno.

- **Empírico** (histórico e caminhos): um painel T × N e pesos w (N valores). Cada linha vira o retorno simples da
  carteira, R = Σ_j w_j s_j, com s_j = e^{r_j} − 1 quando a entrada é log-retorno e s_j = r_j quando é retorno
  simples. q é o quantil de R em p com a interpolação linear do `numpy.quantile` (posição (T − 1)·p sobre os valores
  ordenados); VaR = −q e ES = −(média dos R ≤ q). Pesos não são normalizados: a carteira é Σ w_j s_j como dada.
- **Paramétrico**: o retorno é μ + σ·z, com z a distribuição padronizada do motor (média 0, variância 1; `dist`,
  `skew`, `shape`, `ghlambda` com os mesmos domínios do ajuste). q_p é o quantil de z em p (`ve::dist::quantile`) e
  m_p = E[z | z ≤ q_p] = (1/p) ∫_{−∞}^{q_p} z f(z) dz. VaR = −(μ + σ q_p) e ES = −(μ + σ m_p).
  - Normal e t com fórmula fechada: m_p = −φ(q_p)/p; na t padronizada (escala s = √((ν − 2)/ν), t_p o quantil da t
    de Student), m_p = −s · f_ν(t_p)(ν + t_p²) / ((ν − 1) p).
  - As outras seis, por quadratura de z f(z) na semirreta (Boost.Math, já linkada).
  - Sem conversão de log para simples: o quantil e o ES são do retorno que o modelo descreve, como no DataPanel.

## Objetivos

- Duas funções puras na C ABI; bindings C++, C# e Python.
- O acumulado da simulação (`ve_sim_cumulative`, `ve_dcc_sim_cumulative`) entra no empírico sem cópia, como painel com
  strides; a documentação mostra como.

## Não-objetivos

- Calcular σ = √(wᵀΣw) ou ler a distribuição de um ajuste: quem chama faz a conta e passa os valores.
- Funções de distribuição públicas (`ve_dist_quantile`, `ve_dist_cdf`) e um handle de distribuição: entram depois, se
  alguém pedir ou medir lentidão do nig/ghyp em laço.
- Outras definições de quantil empírico, pesos por data, VaR de componentes ou marginal, backtests.

## Anti-objetivos

- Um parâmetro de convenção de sinal: o sinal é fixo e documentado.
- Aceitar em silêncio NaN, intervalos por série, α fora de (0, 1) ou parâmetros fora do domínio.

## Abordagens

- **Escolhida: função plana para o paramétrico** (`dist`, `skew`, `shape`, `ghlambda`, μ, σ, α), uma chamada sem
  handle; os parâmetros que a distribuição não usa são ignorados, como no spec.
- Handle `ve_dist` validado uma vez: abriria funções de distribuição públicas e guardaria a tabela de quantis do
  nig/ghyp, ao custo de um tipo opaco e um free sem pedido. Adiada.
- Ler a distribuição de um `ve_fit*`: não serve quem só tem um spec, e sugeriria exatidão no horizonte h > 1, em que a
  soma dos choques não segue a distribuição do ajuste. Descartada.
- Histórico e caminhos numa função só sobre `ve_panel`: são a mesma conta.

## C ABI

```c
/* ---- Risk measures (ve_risk_*) ----
   VaR and ES at confidence `confidence` in (0, 1) (0.99: the 1% tail), both positive as losses, in units of the
   return. On an error *var and *es are left unchanged. Since 1.3.0. */
#define VE_RETURNS_LOG 0
#define VE_RETURNS_SIMPLE 1
/* Empirical, over a T x N panel (dates or paths in rows): row t is the portfolio's simple return
   sum_j weights[j] * s_tj, s = exp(r) - 1 for VE_RETURNS_LOG and s = r for VE_RETURNS_SIMPLE. VaR is minus its
   linear-interpolated quantile at 1 - confidence (numpy.quantile's default), ES minus the mean of the returns at or
   below it. The panel must be rectangular with t >= 1 and finite values; weights holds N finite values. */
VE_API ve_status ve_risk_empirical(const ve_panel* returns, const double* weights, int32_t kind, double confidence,
                                   double* var, double* es);
/* Parametric: the return is mean + sigma * z, z the standardized VE_DIST_* density with the fit's skew, shape and
   ghlambda (ignored where the density has none). VaR = -(mean + sigma q), ES = -(mean + sigma E[z | z <= q]), q the
   quantile of z at 1 - confidence. mean finite, sigma >= 0 finite, parameters in the density's domain. */
VE_API ve_status ve_risk_parametric(int32_t dist, double skew, double shape, double ghlambda, double mean, double sigma,
                                    double confidence, double* var, double* es);
```

- Painel nulo: `VE_INVALID_HANDLE`. `weights`, `var` ou `es` nulos, `kind` desconhecido, `dist` fora de
  `VE_DIST_NORM .. VE_DIST_JSU` e os demais casos acima: `VE_INVALID_ARGUMENT`.
- Contrato: `volengine.abi` ganha as duas funções e as duas constantes.

## Núcleo

- `src/risk/` com funções puras: o empírico sobre `PanelView` + pesos + tipo de retorno, o paramétrico sobre
  `Dist` + `ShapeParameters` + μ, σ, α; a média condicional da cauda m_p fica aqui (`src/dist/` já tem 8 módulos).
- Validação na camada C (`src/capi/risk/`).

## Bindings

- C++ (`cpp/include/volengine/risk.hpp`): `struct RiskMeasure { double var; double es; };`,
  `enum class ReturnKind { log, simple };`, `struct Distribution { Dist dist = Dist::norm; double skew; double shape;
  double ghlambda; };`, `ve::risk::empirical(const Panel&, std::span<const double> weights, ReturnKind, double
  confidence) -> Result<RiskMeasure>` e `ve::risk::parametric(const Distribution&, double mean, double sigma, double
  confidence) -> Result<RiskMeasure>`.
- C#: classe estática `Risk` com `Empirical(double[,] returns, double[] weights, ReturnKind kind, double confidence)`
  e `Parametric(Distribution distribution, double mean, double sigma, double confidence)`, ambas devolvendo
  `RiskMeasure(double Var, double Es)`; `Distribution` é um record com `Dist`, `Skew`, `Shape` e `GhLambda`.
- Python: subpacote `volengine.risk` com `empirical(returns, weights, *, kind="log", confidence)` e
  `parametric(mean, sigma, *, confidence, dist="norm", skew=None, shape=None, ghlambda=None)`, ambas devolvendo
  `(var, es)`; um parâmetro `None` vai como NaN e um que a distribuição exige dá `VolEngineError`.

## Testes

- Oráculo R: um script em `tests/reference/` escreve `fixtures/rugarch_risk.json` com, para cada uma das 8
  distribuições (os casos de `quantile_cases` de `tests/unit/dist/cdf_test.cpp` servem de base) e α ∈ {0,95, 0,99,
  0,999}, `qdist` do rugarch e a média da cauda por `integrate(function(z) z * ddist(...), -Inf, q, rel.tol = 1e-12)`
  dividida por p. Os testes comparam o VaR e o ES do motor (com μ e σ dados) a esses valores; a tolerância fica com
  quem constrói, a partir da precisão do quantil e da quadratura.
- Núcleo: normal e t batem com as fórmulas fechadas (e o quantil normal de 0,975 conhecido); a quadratura reproduz a
  fórmula fechada da t quando usada nela; o ES ≥ VaR; σ = 0 dá VaR = ES = −μ; empírico com valores à mão (ex.: 5
  retornos simples, α = 0,8) e igual ao `numpy.quantile` em Python; log e simples dão o mesmo resultado quando a
  entrada simples é e^r − 1; painel com strides igual ao contíguo; painel de caminhos DCC lido do acumulado da
  simulação por strides.
- C ABI: cada recusa listada dá o status certo e não toca `*var` e `*es`.
- Bindings: os valores batem com a C ABI e com a fixture.

## Documentação

- Os três guias ganham "Risk measures", com o exemplo do acumulado da simulação lido por strides.
- `docs/guide/validation.md` e `tests/reference/README.md`: o oráculo novo.
- `CHANGELOG.md`, `[Unreleased]`, Added.

## Deixado para quem constrói

- A regra de quadratura e seu limite de erro, desde que o ES bata com a fixture.
- Onde os nomes C# e Python ficam dentro do limite de 8 módulos por pasta.
