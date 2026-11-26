# Previsão acumulada no horizonte — desenho

Data: 2026-11-26. Versão alvo: 1.3.0 (ABI 1, aditiva), no mesmo release do cancelamento e da simulação acumulada.

## Intenção

Quem mede risco num prazo de h dias precisa da distribuição do retorno acumulado R_h = r_{T+1} + … + r_{T+h}, não
dos passos. Hoje a previsão devolve média e σ (univariada) ou média, σ, R e H (DCC) por passo, e quem chama monta o
acumulado. O DataPanel soma as médias e as covariâncias H₁ … H_h. Isso só está certo quando a média não tem memória:
com ARMA (ou ARFIMA) na média, um choque de hoje continua aparecendo nos retornos seguintes, e a variância do
acumulado leva esses pesos.

O objetivo é que a previsão já traga a média e a variância (covariância no DCC) do retorno acumulado para todo prazo
de 1 ao horizonte, calculadas no motor, para que o batch e a API usem a mesma definição.

## Definição

A média de cada série segue x_t = r_t − nível_t, com φ(L)(1 − L)^d x_t = θ(L) ε_t (d = 0 sem ARFIMA), e o nível
contém a constante, os regressores da média e o termo GARCH-in-mean. Sejam ψ₀ = 1, ψ₁, … os pesos da representação
MA(∞), x_t = Σ ψ_j ε_{t−j}, e a(n) = ψ₀ + … + ψₙ. Com σ²_{T+k} a variância prevista do choque no passo k:

- média acumulada no prazo h: M_h = Σ_{k=1..h} μ_{T+k}, a soma (da esquerda para a direita) das médias previstas;
- variância acumulada no prazo h: V_h = Σ_{k=1..h} a(h − k)² σ²_{T+k};
- no DCC, entre as séries i e j: C_h,ij = Σ_{k=1..h} a_i(h − k) a_j(h − k) H_{T+k,ij}, com os pesos de cada série.

Sem ARMA e sem ARFIMA, a(n) = 1 e V_h = Σ σ², C_h = Σ H: a soma que o DataPanel faz hoje.

Os pesos ψ: ψ_j = θ_j + Σ_{i=1..min(j,p)} φ_i ψ_{j−i} (θ_j = 0 para j > q) para a parte ARMA, convoluídos com os
coeficientes de (1 − L)^{−d} (g₀ = 1, g_j = g_{j−1}(j − 1 + d)/j) quando há ARFIMA.

Aproximações, documentadas:

- σ²_{T+k} é o quadrado do σ que a previsão já devolve. Para os modelos em que esse σ não é a raiz de E[σ²] (eGARCH,
  apARCH, fGARCH com potência livre), V_h herda a aproximação da própria previsão.
- Com GARCH-in-mean, o σ futuro na média é aleatório; a fórmula usa o σ previsto, como a média prevista já usa, e
  ignora a variância e a covariância que essa aleatoriedade acrescenta.
- No DCC, H_{T+k} para k ≥ 2 é a previsão de H (com a aproximação de R_h que ela já tem).

## Objetivos

- Views novas nos handles de previsão que já existem, para h = 1 .. horizonte.
- Univariado e DCC, com regressores e com ARMA/ARFIMA.
- Bindings C++, C# e Python.

## Não-objetivos

- Quantis, VaR ou ES do acumulado (item 5 das propostas, módulo próprio).
- Prazos escolhidos: a previsão já é por passo e o acumulado sai para todos eles.
- Corrigir as aproximações listadas acima.

## Anti-objetivos

- Mudar qualquer valor que a previsão já devolve.
- Uma covariância acumulada que não seja simétrica bit a bit.

## Abordagens

- **Escolhida: views nos handles de previsão**, para todo h. O custo é O(h²) por série (O(h² m²) no DCC) e a
  memória é horizonte × m × m, pequena.
- Uma função `ve_forecast_aggregate(fc, h)` para um prazo só: mais chamadas e mais superfície de API para
  economizar uma memória que já é pequena. Descartada.
- Somar nos bindings: duplicaria a definição em três linguagens, que é justamente o que o item quer evitar.

## C ABI

```c
/* The forecast of the cumulative return R_h = r_{T+1} + ... + r_{T+h} for h = 1..horizon (horizon values each, valid
   until fc is freed): its mean (the left-to-right sum of ve_forecast_mean) and its standard deviation
   sqrt(sum_k a(h-k)^2 sigma_{T+k}^2), with a(n) = psi_0 + ... + psi_n from the MA(inf) weights of the ARMA/ARFIMA
   mean (a = 1 without one). sigma_{T+k} is ve_forecast_sigma; with GARCH-in-mean the randomness of the future sigma
   in the mean is ignored. Since 1.3.0. */
VE_API ve_status ve_forecast_cumulative_mean(const ve_forecast* fc, const double** data, size_t* len);
VE_API ve_status ve_forecast_cumulative_sigma(const ve_forecast* fc, const double** data, size_t* len);

/* The same for a DCC forecast: horizon x m (mean) and horizon x m x m (covariance, sum_k a_i a_j H_k, exactly
   symmetric), row-major, valid until fc is freed. The covariance is computed once, on first access. Since 1.3.0. */
VE_API ve_status ve_dcc_forecast_cumulative_mean(const ve_dcc_forecast* fc, const double** data, size_t* len);
VE_API ve_status ve_dcc_forecast_cumulative_covariance(const ve_dcc_forecast* fc, const double** data, size_t* len);
```

- Contrato: `volengine.abi` ganha as quatro linhas.
- O univariado calcula na criação (O(h²)); a covariância DCC no primeiro acesso, com `std::once_flag`, como a
  covariância do ajuste DCC já faz.

## Núcleo

- Uma função pura que dá os pesos a(0..h−1) de uma série a partir do spec e das estimativas (`forecast` já tem o
  layout e os blocos de AR, MA e d), em `src/forecast/`.
- `forecast::Forecast` ganha `cumulative_mean` e `cumulative_sigma`, calculados em `forecast(...)`.
- O DCC calcula `cumulative_mean` (soma das médias de cada série) na criação e a covariância a partir de H e dos pesos
  de cada série no primeiro acesso: só o triângulo superior é calculado e espelhado, para a simetria exata.
- A média acumulada é a soma da esquerda para a direita das médias, para bater bit a bit com essa soma.

## Bindings

- C++: `Forecast::cumulative_mean()`, `Forecast::cumulative_sigma()`, `DccForecast::cumulative_mean()`,
  `DccForecast::cumulative_covariance()`, no padrão das views atuais.
- C#: `Forecast.CumulativeMean` e `Forecast.CumulativeSigma` (`double[]`), `DccForecast.CumulativeMean`
  (`double[][]`, horizonte × m) e `DccForecast.CumulativeCovariance` (`double[][][]`, horizonte × m × m), copiados como
  as propriedades atuais.
- Python: `Forecast.cumulative_mean` e `Forecast.cumulative_sigma` (ndarray de horizonte), `DccForecast.cumulative_mean`
  (horizonte, m) e `DccForecast.cumulative_covariance` (horizonte, m, m).

## Testes

- Sem ARMA: média acumulada igual bit a bit à soma da esquerda para a direita das médias; variância igual a Σ σ²
  (tolerância relativa 1e-14) no univariado e covariância igual a Σ H (1e-14) no DCC.
- Univariado com ARMA(1,1) e com ARFIMA, num sGARCH (em que σ² previsto é E[σ²] exato): a variância acumulada bate
  com a variância amostral dos acumulados da simulação (`ve_sim_options_set_checkpoints`), com muitos caminhos,
  dentro de 5 erros-padrão, em vários prazos.
- DCC com séries AR(1) na média: a covariância acumulada é igual (1e-14) a Σ_k a_i(h − k) a_j(h − k) H_k calculada no
  teste a partir do H da previsão e de a(n) = (1 − φ^{n+1})/(1 − φ). Não há Monte Carlo no DCC, porque o H previsto
  para k ≥ 2 já é uma aproximação e a simulação não precisa batê-lo.
- Pesos: ARMA(1,0) dá ψ_j = φ^j; MA(1) dá ψ = (1, θ, 0, …); ARFIMA puro dá os coeficientes de (1 − L)^{−d}.
- Simetria: `cumulative_covariance` é simétrica bit a bit.
- Bindings: os valores batem com a C ABI, e a média acumulada com a soma das médias da previsão.

## Documentação

- Os três guias ganham uma seção curta sobre a previsão acumulada, com a definição e as aproximações.
- `CHANGELOG.md`, `[Unreleased]`, Added.

## Deixado para quem constrói

- Onde a função dos pesos fica dentro de `src/forecast/` e como é reaproveitada pelo DCC.
- O número de caminhos e de prazos dos testes de Monte Carlo, desde que fiquem rápidos no `gcc-release`.
