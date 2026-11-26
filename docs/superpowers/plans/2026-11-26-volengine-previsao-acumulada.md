# Previsão acumulada — plano de implementação

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** as previsões univariada e DCC trazem, para h = 1 .. horizonte, a média e o σ (univariado) ou a covariância
(DCC) do retorno acumulado R_h, com os pesos ψ da média ARMA/ARFIMA; C ABI e bindings C++, C# e Python.

**Architecture:** uma função pura dá os pesos acumulados a(0..h−1) de uma série. `forecast::Forecast` ganha
`cumulative_mean` e `cumulative_sigma`, calculados na criação. `DccForecast` ganha `cumulative_mean` e os pesos de cada
série; a covariância acumulada é uma função do núcleo chamada no primeiro acesso pela C ABI.

**Tech Stack:** C++23 (GCC/MinGW, gtest), C ABI, wrapper C++ header-only, C# (.NET 10), Python (ctypes, numpy, pytest,
mypy).

**Spec:** [docs/superpowers/specs/2026-11-26-volengine-previsao-acumulada-design.md](../specs/2026-11-26-volengine-previsao-acumulada-design.md)

## Global Constraints

- ABI 1 aditiva; `volengine.abi` só ganha as quatro linhas (`python scripts/abi_contract.py write`, depois `check`).
- Nomes exatos da C ABI: `ve_forecast_cumulative_mean`, `ve_forecast_cumulative_sigma` (`const ve_forecast*, const
  double**, size_t*`), `ve_dcc_forecast_cumulative_mean`, `ve_dcc_forecast_cumulative_covariance` (`const
  ve_dcc_forecast*, const double**, size_t*`). Comentários com o texto da seção "C ABI" do spec e "Since 1.3.0.".
- Definição (spec, "Definição"): M_h = soma da esquerda para a direita das médias previstas; V_h = Σ_{k=1..h}
  a(h − k)² σ²_{T+k}; C_h,ij = Σ_{k=1..h} a_i(h − k) a_j(h − k) H_{T+k,ij}; a(n) = ψ₀ + … + ψₙ; ψ_j = θ_j + Σ_{i=1..min(j,p)}
  φ_i ψ_{j−i}, convoluído com g₀ = 1, g_j = g_{j−1}(j − 1 + d)/j quando há ARFIMA. Sem ARMA/ARFIMA, a(n) = 1.
- Nenhum valor que a previsão já devolve muda.
- A covariância acumulada é simétrica bit a bit (triângulo superior calculado e espelhado).
- Limites do STYLEGUIDE (funções ≤ 60 linhas, ≤ 5 parâmetros, ≤ 8 módulos por pasta, sem `else`); o teste `style` da
  raiz roda em todo `ctest`. `python/src/volengine` e `python/tests` já estão em 8 módulos.
- Comandos: `cmake --build --preset gcc-release` e `ctest --preset gcc-release -j 8` (PowerShell); wrapper com
  `bash cpp/scripts/ci_check.sh` (`/c/msys64/ucrt64/bin` primeiro no PATH, `VOLENGINE_NATIVE_LIB` na DLL do
  `gcc-release`); C# `VOLENGINE_LIB=... dotnet test --solution csharp/VolEngine.slnx -c Release`; Python
  `VOLENGINE_LIB=... uv run --project python pytest python/tests -q` e `uv run --project python mypy python/src`.
- `abi_coverage` falha entre a Tarefa 3 e a Tarefa 4.
- Branch `develop`, commits no estilo do histórico, sem trailer `Co-Authored-By`.

## Review Focus

1. **Horizonte 1**: M_1 = μ_{T+1}, σ acumulado = σ_{T+1}, C_1 = H_1, com ou sem ARMA. Teste na Tarefa 1 e na Tarefa 2.
2. **AR perto da raiz unitária** (φ = 0.99): os pesos crescem até perto de h, sem NaN nem infinito para horizontes
   de algumas centenas. Teste na Tarefa 1.
3. **Séries com médias diferentes no DCC** (uma AR(1), uma sem média, uma MA(1)): cada par usa os pesos das suas
   duas séries. Teste na Tarefa 2.
4. **GARCH-in-mean e regressores na média**: a média acumulada inclui o termo in-mean e os regressores, igual à soma
   das médias da previsão; a variância usa a(n) = 1 quando não há ARMA. Teste na Tarefa 1.
5. **Primeiro acesso concorrente à covariância DCC**: duas threads lendo `ve_dcc_forecast_cumulative_covariance` ao
   mesmo tempo recebem o mesmo ponteiro e os mesmos valores. Teste na Tarefa 3.

---

### Task 1: Núcleo univariado

**Files:**
- Create: `src/forecast/cumulative.hpp`, `src/forecast/cumulative.cpp` (listados no `CMakeLists.txt` junto dos outros
  `src/forecast`)
- Modify: `src/forecast/forecast.hpp` (`Forecast::cumulative_mean`, `Forecast::cumulative_sigma`),
  `src/forecast/forecast.cpp` (preenchidos em `forecast(...)`)
- Test: `tests/unit/forecast/forecast_test.cpp`

**Interfaces:**
- Produces:
  - `std::vector<double> ve::forecast::cumulative_weights(const model::Layout& layout, std::span<const double> theta,
    std::size_t horizon)`: a(0..horizon−1).
  - `forecast::Forecast::cumulative_mean`, `forecast::Forecast::cumulative_sigma` (`std::vector<double>`, horizonte).
  - `double ve::forecast::cumulative_variance(std::span<const double> weights, std::span<const double> variances,
    std::size_t h)`: Σ_{k=1..h} weights[h−k]² variances[k−1], usada também pelo DCC na diagonal.

- [ ] **Step 1: Write the failing tests** em `forecast_test.cpp`:
  - `CumulativeWeights.ArOneIsGeometric`: AR(1) φ = 0.6, sem MA: a(n) = Σ_{j≤n} 0.6^j, `EXPECT_NEAR` 1e-15
    relativo, n < 30.
  - `CumulativeWeights.MaOne`: MA(1) θ = 0.4: a = (1, 1.4, 1.4, …).
  - `CumulativeWeights.FractionalIntegration`: ARFIMA(0, d, 0) d = 0.3: a(n) é a soma parcial de g_j.
  - `CumulativeWeights.NearUnitRoot` (Review Focus 2): φ = 0.99, horizonte 500: todos finitos e crescentes.
  - `CumulativeForecast.WithoutArmaIsThePlainSum`: sGARCH sem ARMA: `cumulative_mean[h−1]` bit a bit igual à soma
    da esquerda para a direita de `mean[0..h−1]`; `cumulative_sigma[h−1]²` igual a Σ σ² (1e-14 relativo).
  - `CumulativeForecast.HorizonOne` (Review Focus 1): com ARMA(1,1), `cumulative_mean[0] == mean[0]` e
    `cumulative_sigma[0] == sigma[0]` bit a bit.
  - `CumulativeForecast.InMeanAndRegressors` (Review Focus 4): um ajuste com `archm = 1` e `mxreg = 1`: a média
    acumulada é a soma das médias; o σ acumulado ao quadrado é Σ σ².
  - `CumulativeForecast.MatchesTheSimulation`: ajuste ARMA(1,1)-sGARCH-norm de uma série simulada; `sim::simulate`
    do mesmo ajuste com 200000 caminhos e `checkpoints = {1, 5, 20}`; em cada prazo, a variância amostral dos
    acumulados está a menos de 5 erros-padrão de `cumulative_sigma²` (erro-padrão da variância amostral estimado pelos
    próprios caminhos) e a média amostral a menos de 5 erros-padrão de `cumulative_mean`. O mesmo com ARFIMA(1, d, 0).
- [ ] **Step 2: Run to verify they fail.** `cmake --build --preset gcc-release`. Expected: erro de compilação.
- [ ] **Step 3: Implement** as três interfaces.
- [ ] **Step 4: Run.** `ctest --preset gcc-release -j 8 -R "Cumulative|Forecast"`, depois a suíte inteira. Expected:
  PASS.
- [ ] **Step 5: Commit** `feat(forecast): cumulative mean and sigma over the horizon`

### Task 2: Núcleo DCC

**Files:**
- Modify: `src/dcc/future/forecast.hpp` (`DccForecast::cumulative_mean` horizonte × m, `DccForecast::weights`
  horizonte × m, `cumulative_covariance`), `src/dcc/future/forecast.cpp`
- Test: `tests/unit/dcc/future/forecast_test.cpp`

**Interfaces:**
- Consumes: `cumulative_weights` (Tarefa 1).
- Produces:
  - `DccForecast::cumulative_mean`, `DccForecast::weights` (`std::vector<double>`, `[h * m + i]` = a_i(h)),
    preenchidos em `forecast_dcc`.
  - `std::vector<double> ve::dcc::future::cumulative_covariance(const DccForecast& forecast)`: horizonte × m × m,
    `[(h−1) m² + i m + j]` = C_h,ij, calculado só para i ≤ j e espelhado.

- [ ] **Step 1: Write the failing tests:**
  - `DccCumulative.WithoutArmaIsThePlainSum`: DCC sobre `dcc3_fits()` (média constante): covariância acumulada igual
    a Σ H (1e-14 relativo) e média acumulada igual bit a bit à soma das médias de cada série.
  - `DccCumulative.MixedMeans` (Review Focus 3): três séries ajustadas com AR(1), sem ARMA e MA(1) (use
    `mixed_panel.hpp` ou fixe os coeficientes no spec de cada uma); C_h,ij igual (1e-14) a
    Σ_k a_i(h − k) a_j(h − k) H_{k,ij} calculado no teste a partir de `forecast.h` e de `cumulative_weights` de cada
    série; C_1 = H_1 (Review Focus 1).
  - `DccCumulative.ExactlySymmetric`: C_h,ij e C_h,ji iguais bit a bit para todo h, i, j.
- [ ] **Step 2: Run to verify they fail.** Expected: erro de compilação.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run.** `ctest --preset gcc-release -j 8 -R DccCumulative` e a suíte inteira. Expected: PASS.
- [ ] **Step 5: Commit** `feat(dcc): cumulative mean and covariance of the DCC forecast`

### Task 3: C ABI

**Files:**
- Modify: `include/volengine/volengine.h` (as quatro declarações, com o texto do spec), `include/volengine/volengine.abi`
- Modify: `src/capi/handles.hpp` (`ve_dcc_forecast` ganha `mutable std::once_flag cumulative_once` e
  `mutable std::vector<double> cumulative_covariance`)
- Modify: `src/capi/results/forecast_api.cpp`, `src/capi/dcc/dcc_forecast_api.cpp`
- Modify: `CHANGELOG.md` (`[Unreleased]`, Added, uma linha da C ABI)
- Test: `tests/unit/capi/results/forecast_test.cpp`, `tests/unit/capi/dcc_test.cpp`

**Interfaces:**
- Consumes: Tarefas 1 e 2.
- Produces: as quatro funções da Global Constraints.

- [ ] **Step 1: Write the failing tests:**
  - `CapiForecastCumulative.MatchesCore`: `ve_forecast_cumulative_mean`/`_sigma` iguais bit a bit a
    `forecast::forecast(...).cumulative_*` do mesmo ajuste; `len` = horizonte; `NULL` no handle dá
    `VE_INVALID_HANDLE`, em `data`/`len` dá `VE_INVALID_ARGUMENT`.
  - `CapiDccForecastCumulative.MatchesCore`: média e covariância iguais bit a bit às do núcleo; `len` = horizonte × m
    e horizonte × m × m.
  - `CapiDccForecastCumulative.ConcurrentFirstAccess` (Review Focus 5): duas `std::jthread` chamam
    `ve_dcc_forecast_cumulative_covariance` no mesmo handle recém-criado; os dois ponteiros são iguais e os valores
    iguais aos do núcleo.
- [ ] **Step 2: Run to verify they fail.** Expected: erro de compilação.
- [ ] **Step 3: Implement.** `python scripts/abi_contract.py write`.
- [ ] **Step 4: Verify.** `python scripts/abi_contract.py check`, build, `ctest --preset gcc-release -j 8`.
  Expected: tudo PASS.
- [ ] **Step 5: Commit** `feat(capi): cumulative forecast views`

### Task 4: Wrapper C++

**Files:**
- Modify: `cpp/include/volengine/fit.hpp` (`Forecast::cumulative_mean()`, `Forecast::cumulative_sigma()`),
  `cpp/include/volengine/dcc.hpp` (`DccForecast::cumulative_mean()`, `DccForecast::cumulative_covariance()`)
- Modify: `docs/guide/cpp.md` (seção "Cumulative forecast": definição curta, os pesos ψ e as aproximações do spec)
- Test: `cpp/tests/future_test.cpp`, `cpp/tests/dcc_future_test.cpp`

**Interfaces:**
- Consumes: Tarefa 3.
- Produces: `Result<std::span<const double>>` nos quatro métodos, no padrão de `sigma()`.

- [ ] **Step 1: Write the failing tests:** `Forecast.CumulativeIsTheSumWithoutArma` (média acumulada = soma das
  médias bit a bit; σ acumulado² = Σ σ², 1e-14) e `DccForecast.CumulativeCovarianceShape` (tamanho horizonte × m × m,
  diagonal do prazo 1 = diagonal de H₁).
- [ ] **Step 2: Run to verify they fail.** `bash cpp/scripts/ci_check.sh`. Expected: erro de compilação.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run.** `bash cpp/scripts/ci_check.sh` e `uv run --project python pytest python/tests/packaging -q`.
  Expected: PASS, inclusive `abi_coverage`.
- [ ] **Step 5: Commit** `feat(cpp): cumulative forecast views in the wrapper`

### Task 5: C#

**Files:**
- Modify: `csharp/src/VolEngine/Interop/Native/NativeMethods.Results.cs` e `NativeMethods.Dcc.cs` (as quatro funções)
- Modify: `csharp/src/VolEngine/Results/Forecast.cs` (`double[] CumulativeMean`, `double[] CumulativeSigma`),
  `csharp/src/VolEngine/Dcc/DccForecast.cs` (`double[][] CumulativeMean`, `double[][][] CumulativeCovariance`)
- Modify: `csharp/src/VolEngine/PublicAPI.Unshipped.txt`
- Modify: `docs/guide/csharp.md` (a mesma seção) e `csharp/tests/VolEngine.Tests/Guide/GuideSnippetsTests.cs`
- Test: `csharp/tests/VolEngine.Tests/Results/ForecastTests.cs`, `csharp/tests/VolEngine.Tests/Dcc/DccForecastTests.cs`

**Interfaces:**
- Consumes: Tarefa 3. As propriedades novas são copiadas na construção, como `Sigma` e `Mean`.

- [ ] **Step 1: Write the failing tests:** `Cumulative_mean_is_the_left_sum_of_the_mean` (bit a bit, univariado e DCC),
  `Cumulative_sigma_without_arma_is_the_root_of_the_summed_variance` (1e-14) e
  `Cumulative_covariance_is_symmetric` (bit a bit).
- [ ] **Step 2: Run to verify they fail.** Expected: erro de compilação.
- [ ] **Step 3: Implement**, com as linhas novas no `PublicAPI.Unshipped.txt`.
- [ ] **Step 4: Run.** Expected: PASS, sem avisos, inclusive `ContractTests` e `CoverageTests`.
- [ ] **Step 5: Commit** `feat(csharp): cumulative forecasts`

### Task 6: Python

**Files:**
- Modify: `python/src/volengine/_native.py` (as quatro funções em `PROTOTYPES`)
- Modify: `python/src/volengine/_forecast.py` (`Forecast.cumulative_mean`, `Forecast.cumulative_sigma`,
  `DccForecast.cumulative_mean` (horizonte, m), `DccForecast.cumulative_covariance` (horizonte, m, m))
- Modify: `docs/guide/python.md` (a mesma seção; o `tests/docs` roda o snippet)
- Modify: `CHANGELOG.md` (as linhas de C++, C# e Python da previsão acumulada)
- Test: `python/tests/test_forecast.py`, `python/tests/dcc/test_dcc_forecast.py`

**Interfaces:**
- Consumes: Tarefa 3. As views seguem o padrão de `sigma` e `H` (somente leitura, presas ao dono).

- [ ] **Step 1: Write the failing tests:** média acumulada = soma da esquerda para a direita das médias (comparação de
  bits com `.view(np.uint64)`), σ acumulado² = Σ σ² sem ARMA (1e-14), covariância DCC simétrica bit a bit e com a
  forma (horizonte, m, m).
- [ ] **Step 2: Run to verify they fail.** Expected: FAIL (`has no attribute 'cumulative_mean'`).
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run.** pytest inteiro, mypy e `python tests/style/check_style.py .`. Expected: PASS.
- [ ] **Step 5: Commit** `feat(python): cumulative forecasts`
