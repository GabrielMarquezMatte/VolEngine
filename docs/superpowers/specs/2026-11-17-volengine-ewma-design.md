# EWMA (RiskMetrics) — design

2026-11-17. Pedido do autor: adicionar EWMA como modelo próprio (caminho 2: `VE_MODEL_EWMA`), SV fica para depois.
O autor dispensou as perguntas; as decisões abaixo marcadas "minha escolha" podem ser corrigidas.

## Intenção

EWMA é o filtro de volatilidade mais usado na prática (RiskMetrics) e o que um usuário procura pelo nome. Ele é o
iGARCH(1,1) com omega = 0: sigma2_t = lambda sigma2_{t-1} + (1 - lambda) e_{t-1}^2. Um modelo com nome, parâmetro
`lambda` e documentação próprios é melhor que o truque "iGARCH com omega fixo em 0", que hoje nem é aceito
(omega > 0 é exigido em todo modelo menos o eGARCH).

## Metas

- `VE_MODEL_EWMA` (= 14) no ABI, que só cresce: `ve::model::ewma` no C++, `model="ewma"` no Python.
- Parâmetros: `mu` (e o resto da média), `omega` (derivado, 0), `alpha1` (derivado, 1 - lambda), `lambda`
  (estimado por ML), parâmetros da distribuição. Derivados como o `beta_q` do iGARCH: aparecem na tabela com valor e
  erro-padrão NaN, não contam nos critérios, não podem ser fixados.
- `lambda` estimado por padrão, como todo parâmetro da biblioteca (minha escolha); RiskMetrics é
  `fixed={"lambda": 0.94}`. Início do otimizador em 0.94 (o de RiskMetrics e do arch).
- Domínio de `lambda`: [0, 1], o mesmo dos pesos do iGARCH (minha escolha; o mapa irrestrito alcança (0, 1)).
- Tudo que o iGARCH(1,1) aceita vale: ARMA, mxreg, vxreg, archm, ARFIMA, as 8 distribuições, fixed/start,
  `VE_VARINIT_SAMPLE`/`BACKCAST` (o backcast é o do arch, o que dá o oracle), forecast (plano em sigma2), simulação a
  partir de fit, diagnósticos, Nyblom, margens de DCC e cópula.
- Oracle: `arch` 8.0 (`ConstantMean` + `EWMAVariance(lam=None)` e `EWMAVariance(0.94)`), com backcast.

## Não-metas

- Ordem diferente de (1, 1): rejeitada ("the ewma model has order (1, 1)").
- Variance targeting e `ve_simulate_spec`: rejeitados, como no iGARCH (não há variância incondicional).
- Inicialização própria de RiskMetrics (e_1^2, janela de 75): as duas de `VE_VARINIT_*` bastam.
- EWMA multivariado (covariância RiskMetrics): decisão à parte.

## Abordagens

1. **Peso derivado generalizado (escolhida).** O iGARCH já tem um peso derivado (beta_q = 1 - os outros) tratado
   no layout, no mapa de parâmetros, na verossimilhança, no adjoint e no início. O EWMA é o mesmo com o peso derivado
   em alpha1 e omega derivado em 0. As checagens "é iGARCH" do código de pesos passam a "tem peso derivado"
   (`Layout::has_derived`), e o cálculo do derivado passa a ser "1 - os outros pesos do bloco" em vez de "o último
   beta". O iGARCH fica bit-idêntico (mesma ordem de operações). Consolida em vez de duplicar.
2. Recursão e mapa próprios do EWMA. Duplica o caminho do sGARCH (verossimilhança, adjoint, forecast) por um
   parâmetro; descartada.
3. EWMA como iGARCH com omega fixo internamente. Deixa `lambda` derivado (sem erro-padrão) ou exige inverter os nomes;
   descartada.

## Restrições

STYLEGUIDE (sem `else`, ≤ 60 linhas, ≤ 5 parâmetros, ≤ 8 módulos por pasta); ABI 1.0 congelado só cresce
(`scripts/abi_contract.py write`); testes de oracle em `python/tests/oracles`; benchmark: um caso EWMA no quick bench.
