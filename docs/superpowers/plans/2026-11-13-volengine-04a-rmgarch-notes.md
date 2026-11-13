# rmgarch 1.4.3 DCC conventions (R 4.6.2): formula-level summary for reimplementation

I got everything here by printing rmgarch/rugarch internals and checking each claim numerically against fitted
objects. The C++ kernels (`dccnormC1/C2`, `dccstudentC1/C2`, `dcclaplaceC1/C2`, `fdccnormC1/C2`) ship only as
binaries, so their behaviour was checked by black-box tests: my own recursion matches rmgarch's per-observation
likelihoods, R_t and Q_t to about 1e-15. rmgarch is GPL. Everything below is written in my own words and math.
Scripts and outputs are in the same scratchpad folder: `run1.R`/`run1.out` (fits), `verify1..4.R` (checks),
`dcc3.csv` (the data).

Notation: m series, T observations, t = 1..T. Parameter orders are (P, Q) = `dccOrder = c(P, Q)`: P lags of the
z z' (and asymmetry) term and Q lags of Q_t. Data rows are time.

## 1. From univariate fits to z_t

- `dccfit(spec, data, fit = multifit)`: when a `uGARCHmultifit` is passed, the univariate models are **not
  re-estimated**. rmgarch takes `residuals(fit)` (eps_t = y_t - mu_t) and `sigma(fit)` (the conditional sd) from
  each fit and sets **z_it = eps_it / sigma_it**. No re-standardisation and no demeaning. Without `fit`, it runs
  `multifit` first with `ufit.control = {stationarity, scale}` taken from `fit.control`. If any univariate fit does
  not converge, it returns the multifit object instead of failing.
- `plik_i` is the univariate log-likelihood of series i (the sum of the per-observation log densities). It is
  stored in `mfit$plik` (DCC/aDCC only; the FDCC path does not set it) and is never added to the reported DCC
  log-likelihood.
- The univariate **distribution** (norm/std/ged...) only matters for estimating mu_t and sigma_t. The DCC
  log-likelihood swaps the univariate densities for the multivariate one (section 3).
- The univariate estimates, covariance matrices and scores are reused for the two-stage standard errors
  (section 6).
- The fit needs at least 100 observations after `out.sample`.

## 2. Recursions

### 2.1 Shared quantities

- **Qbar = sample covariance of z**: demeaned, divisor (T-1) (R's `cov`). It is **not** z'z/T and **not** a
  correlation matrix (diagonals are about 1.004 in the example).
- **Asymmetry indicator:** I_it = 1 if z_it < 0, otherwise 0 (z = 0 gives 0). n_t = I_t ∘ z_t (element-wise).
- **Nbar = sample covariance of n** (demeaned, divisor T-1). For DCC, Nbar = 0 and n is set to 0.
- **FDCC estimation only:** Qbar_E = z'z/T, raw second moment, no demeaning. See 2.4 for the inconsistency.

### 2.2 DCC / aDCC, order (P, Q); aDCC has P asymmetry coefficients g_1..g_P

    Q_t = (1 - Σa_j - Σb_k) Qbar - (Σg_j) Nbar
          + Σ_{j=1..P} a_j z_{t-j} z_{t-j}'
          + Σ_{j=1..P} g_j n_{t-j} n_{t-j}'
          + Σ_{k=1..Q} b_k Q_{t-k}
    R_t = diag(Q_t)^{-1/2} Q_t diag(Q_t)^{-1/2}

**Presample** (mx = max(P, Q) rows before t = 1):
- Q_s = Qbar for every s ≤ 0.
- The presample z_s (and n_s) differ between the two stages:
  - **Estimation objective (stage-2 optimiser, "LLH1"):** z_s = 0 and n_s = 0, so the z z' and n n' terms
    vanish for s ≤ 0.
  - **Everything reported** (llh, per-observation likelihoods, R_t, Q_t, H_t, scores and Hessian for the
    standard errors, "LLH2"): z_s = **(1, ..., 1)** and n_s = (1, ..., 1). So for P = Q = 1,
    Q_1 = (1-a-b) Qbar + a·J + b·Qbar, where J is the all-ones matrix (aDCC adds -g Nbar + g J).
  - Consequence: the reported estimates maximise the zero-presample criterion, but the reported llh and R_t use
    the ones-presample. In the example, the zero-presample llh at the estimates is -6082.9178 and the reported
    one is -6082.9409.
  - Verified: the ones-presample matches `rcor` to 4e-16 for DCC(1,1), DCC(2,1), aDCC(1,1) and aDCC(1,2). The
    zero-presample is off by 0.03.

### 2.3 Stationarity constraints

- DCC: Σa + Σb < 1.
- aDCC: Σa + Σb + δ Σg < 1, where δ = the largest eigenvalue of Qbar^{-1/2} Nbar Qbar^{-1/2}. Qbar^{-1/2} is the
  inverse of the eigen-decomposition square root of Qbar.

### 2.4 FDCC (Billio-Caporin flexible DCC): G groups, order (1,1) only

- Groups: the vector `groups` (length m, values 1..G) defines a 0/1 selection matrix S (m×G). Per-series
  coefficient vectors are α = S a_grp and β = S b_grp.
- Recursion, element-wise:

      AA = α α',  BB = β β',  C = 1 - AA - BB   (1 = all-ones matrix)
      Q_t = C ∘ Qbar + AA ∘ (z_{t-1} z_{t-1}') + BB ∘ Q_{t-1}

  The **coefficients enter as products** α_i α_j and β_i β_j. With one group, FDCC (a_F, b_F) equals DCC with
  a = a_F² and b = b_F².
- Presample: Q_0 = Qbar. z_0 = 0 in estimation, z_0 = 1 (ones vector) in reporting.
- Qbar differs between the two stages:
  - **Estimation uses Qbar_E = z'z/T.** This is also what `mfit$Qbar` stores for FDCC.
  - **Reporting** (llh, R_t, standard errors) recomputes **Qbar = cov(z)**.
  - Verified: R_t matches with cov + ones-presample to 7e-16. With z'z/T + ones-presample it is off by 1.3e-4.
- **FDCC constraint as coded:**
  - G > 1: for **every** quadruple (g, h, k, l) of groups, a_g a_h + b_k b_l ∈ [0, 1]. That is G⁴ inequality
    constraints, mismatched indices included.
  - **G = 1: the constraint is a + b ≤ 1, not a² + b² ≤ 1 (bug).** One-group FDCC therefore cannot reach the
    DCC optimum: the example gives a = 0.2330, b = 0.7670 (a + b = 1 - 1.3e-8) and llh -6106.16, against DCC's
    -6082.94. "FDCC with one group = DCC" holds for the model, not for rmgarch's fitted FDCC.

## 3. Log-likelihood

Notation:
- q_t = z_t' R_t^{-1} z_t
- L_t = log|R_t|
- s_t = Σ_i log σ_it

### 3.1 Reported per-observation log-likelihood (`mfit$log.likelihoods`, length T, positive-convention log density)

ℓ_t = log f(z_t; R_t) - s_t. This is the exact log density of ε_t with covariance H_t = D_t R_t D_t, where
D_t = diag(σ_t). R_t uses the ones-presample.

- **mvnorm:** log f = -½ (m log 2π + L_t + q_t).
- **mvt** (shape ν; standardised t with covariance R_t, scale matrix R_t(ν-2)/ν):

      log f = lnΓ((ν+m)/2) - lnΓ(ν/2) - (m/2) log(π(ν-2)) - ½ L_t - ((ν+m)/2) log(1 + q_t/(ν-2))

  Verified to 7e-15. The unscaled-t variant is off by 0.2.
- **mvlaplace** (symmetric multivariate Laplace with covariance R_t), with v = 1 - m/2:

      log f = log 2 - (m/2) log 2π - ½ L_t + (v/2) log(q_t/2) + log K_v(√(2 q_t))

  K is the modified Bessel function of the second kind (K_v = K_{-v}). Verified to 3e-15.

**Reported llh = Σ_t ℓ_t** (`likelihood(fit)` equals `mfit$llh`). It is a **full** log-likelihood:
- It is not the univariate sum plus a correlation part when the univariate densities are not normal.
- For a normal univariate with mvnorm it equals Σ plik_i + ½ Σ_t (z_t'z_t - L_t - q_t).
- In the example: -6426.0930 + 343.1521 = -6082.9409.

### 3.2 Stage-2 objective that is minimised (LLH1)

The objective uses the zero-presample R_t, and z from the univariate fits only (no refit):
- mvnorm: ½ Σ_t (L_t + q_t). No 2π constant and no s_t.
- mvt: -Σ_t log f_t(z_t; R_t, ν), with all constants; no s_t.
- mvlaplace: -Σ_t log f_t(z_t; R_t), with all constants; no s_t.
- FDCC (normal only): as mvnorm, with Qbar_E = z'z/T.

Example value at the DCC/mvnorm optimum: 1911.09985833825.

**Penalty:**
- With the `nlminb`/`lbfgs` solvers, if persistence ≥ 1 (or the C routine returns a non-finite value), the
  objective returns the last finite value plus 10% of its absolute value.
- With solnp/gosolnp (default solnp), that penalty is off. The persistence function is passed instead as an
  inequality constraint in [0, 1].

## 4. Parameters: names, order, bounds, starts, fixed values

Order in the internal vector:
- **DCC/aDCC:** dcca1..dccaP, dccb1..dccbQ, dccg1..dccgP (aDCC), mshape (mvt).
- **FDCC:** `fdcca[g=1,P=1]`..`fdcca[g=G,P=1]`, `fdccb[g=1,Q=1]`..`fdccb[g=G,Q=1]`, mshape. The mvt shape is
  allocated, but the fit fails.

| param | start | LB | UB |
|---|---|---|---|
| dcca_j | 0.05/P | 2.22e-16 (machine eps) | 1 |
| dccb_k | 0.9/Q | eps | 1 |
| dccg_j | 0.05/P | eps | 1 |
| mshape | 5 | 4 | 50 |
| fdcca[g] | 0.01 | 0 | 0.999 |
| fdccb[g] | 0.9 | 0 | 1 |

- **Optimisation:** box bounds plus the inequality constraint from section 2. Default solver solnp with rho = 1,
  outer.iter = 50, inner.iter = 1800 (1000 if the user passes any control), delta = 1e-7, tol = 1e-8.
  - The solver works in natural space (no reparametrisation). Estimates can end on a bound: mshape = 49.99999
    in the example with normal data.
  - With nlminb, only the box plus the penalty apply.
- **`fixed.pars`** (named list in `dccspec`): the parameter is removed from the estimated set and kept at the
  given value. It then disappears from `coef(fit, "dcc")`, from `matcoef` and from the parameter count k.
  - Example: with mvt and `fixed.pars = list(mshape = 8)`, matcoef has only dcca1 and dccb1, and k = 17.
- **`start.pars`**: replaces the start value. It is matched by exact name for dcc*, and by prefix for mshape.
- **All DCC parameters fixed:**
  - with `eval.se = FALSE`, a `DCCfilter` is returned;
  - with `eval.se = TRUE`, the fixed parameters are treated as "estimated" only for the standard errors, with
    no optimisation.
- **Coefficient names in `coef(fit)` / matcoef:** `[<colname>].<univariate par>` for each series in column order
  (rugarch order, e.g. mu, omega, alpha1, beta1; only the parameters estimated in that univariate fit), then
  `[Joint]dcca1`, `[Joint]dccb1`, ... Unnamed data columns get the names `Asset_i`.

## 5. mvt and mvlaplace specifics

- mvt: one shape ν for all series, in [4, 50], start 5. Section 3.1 gives the exact standardised density. ν has
  no reparametrisation and no extra constraint. With near-normal data ν goes to the upper bound 50.
- mvlaplace: no extra parameter. Section 3.1 gives the density with Bessel K of order 1 - m/2.
- FDCC is fitted only with mvnorm (section 9).

## 6. Standard errors (two-stage, Engle-Sheppard style), exactly as computed

The full parameter vector θ = (θ_1, ..., θ_m, θ_D) has n_p entries: every univariate parameter estimated in each
series (in series order), then the n_D estimated DCC parameters.

**Matrix A** (n_p × n_p):
- **Univariate diagonal blocks** A_ii = (cvar_i)^{-1}. cvar_i is the univariate fit's `fit$cvar`, the inverse of
  rugarch's numerical (numDeriv Richardson) Hessian of the **negative** univariate log-likelihood. So
  A_ii ≈ the rugarch Hessian of -ℓ_i (positive definite).
- **Off-diagonal univariate blocks and the univariate-row/DCC-column block: 0.**
- **DCC rows** (all n_p columns) = the Hessian of the **reported total log-likelihood (+ℓ, section 3.1)** as a
  function of all of θ:
  - Each evaluation re-filters every univariate model at the perturbed univariate parameters (rugarch filter
    with the same spec), recomputes z, **recomputes Qbar = cov(z) and Nbar**, and runs the ones-presample
    recursion.
  - Finite differences, forward:
    - h_i = ε^{1/3} max(|θ_i|, 1), with ε = 2.22e-16 and h_i adjusted as (θ_i + h_i) - θ_i;
    - f0 = f(θ), g_i = f(θ + h_i e_i);
    - H_ij = [f(θ + h_i e_i + h_j e_j) - g_i - g_j + f0] / (h_i h_j), for j a DCC index and i ≤ j; symmetrised.
  - Note the sign: these rows are the Hessian of +ℓ (negative definite in the DCC block), while the univariate
    blocks are Hessians of -ℓ.

**Scores S** (T × n_p):
- **Univariate columns** = the rugarch `fit$scores` of each series: the numerical (Richardson) Jacobian of the
  per-observation **negative** log-likelihood with respect to that series' parameters.
- **DCC columns** = central differences of the reported per-observation **+ℓ_t** (section 3.1), with the
  univariate parameters held at their estimates:
  - step h_k = max(|θ_k|/2, 0.01) · ε^{1/3};
  - S_tk = [ℓ_t(θ + h_k e_k) - ℓ_t(θ - h_k e_k)] / (2 h_k).
- The DCC score columns do not average to 0 (mean -1.0e-3 for dcca1 in the example), because the estimates
  optimise the zero-presample objective.

**Covariance:**

    B = cov(S)  (demeaned, divisor T-1)
    Ā = A / T
    V = Ā^{-1} B Ā^{-1} / T        <- the second factor is Ā^{-1}, NOT its transpose
    se = sqrt(|diag V|)
    t = θ / se
    p = 2(1 - Φ(|t|))

- Reproduced exactly: the recomputed V matches `mfit$cvar` with relative difference 0.
- Reading the DCC rows: take the last n_D rows of matcoef (`[Joint]...`).
- **There is no separate non-robust or "robust" DCC matcoef.** `mfit$matcoef` is this sandwich and is the only
  standard-error table. The univariate rows in matcoef come from the same joint V and differ slightly from the
  univariate fits' own SEs.
- For comparison, in the example (dcca1, dccb1):

  | form | se(dcca1) | se(dccb1) |
  |---|---|---|
  | rmgarch Ā⁻¹BĀ⁻¹ | 0.00627467 | 0.01326756 |
  | textbook Ā⁻¹BĀ⁻ᵀ (same with sign-consistent rows) | 0.00639279 | 0.01334010 |
  | inverse of -H_DD only | 0.00684873 | 0.01556525 |

- If `eval.se = FALSE`, matcoef has NA in the standard-error, t and p columns, and `cvar`/`scores` are NULL.
- The parallel path is identical.
- Unlike VolEngine's univariate robust standard errors (S'S, uncentred), B here is **centred and uses T-1**.

## 7. Information criteria

`infocriteria(fit)` reports Akaike, Bayes, Shibata and Hannan-Quinn, divided by T (per observation):

    k = (number of estimated univariate parameters over all series) + n_D (estimated DCC params incl. mshape)
        + m(m-1)/2  (the off-diagonal Qbar entries counted as parameters)
    AIC  = (-2ℓ + 2k)/T
    BIC  = (-2ℓ + k log T)/T
    SIC  = -2ℓ/T + log((T + 2k)/T)
    HQIC = (-2ℓ + 2k log log T)/T

- ℓ is the reported full llh (section 3.1). k uses the matcoef row count, so fixed parameters are excluded.
- Example DCC/mvnorm: k = 12 + 2 + 3 = 17. FDCC with 2 groups: k = 12 + 4 + 3 = 19. Checked.
- A VAR mean would add its coefficients (out of scope).

## 8. Outputs

- `rcor(fit)`:
  - an array **[m, m, T]**, with **time in the 3rd dimension**;
  - t = 1 is the first data row (no presample entries), and dimnames[[3]] are the dates;
  - R_1 is the ones-presample value (section 2.2);
  - `rcor(fit, type = "Q")` gives Q_t;
  - `output = "matrix"` gives T × m(m-1)/2, with lower-triangle columns ordered column-major ("B:A", "C:A",
    "C:B").
- `rcov(fit)` is an array [m, m, T] with **H_t = D_t R_t D_t**, where D_t = diag(σ_t) from the univariate
  fits' `sigma` (matches to 0). `sigma(dccfit)` equals the univariate sigma.
- `mfit$stdresid` (not z!) = ε_t H_t^{-1/2}, using the eigen-decomposition square root of H_t.
- With `out.sample`, only the first T = n - out.sample rows are used and reported.

## 9. Refused combinations and quirks

- **FDCC + mvt and FDCC + mvlaplace:** `dccspec` accepts them, but `dccfit` fails with "number of items to
  replace is not a multiple of replacement length". There is no FDCC likelihood for them, so there is no oracle.
- **FDCC order other than (1,1):** `dccspec` refuses it ("FDCC model only allows (1,1)").
- **FDCC with the nlminb/lbfgs solvers:**
  - the stationarity check compares a logical "any(...)" with 1, so the penalty always fires;
  - the fit returns NA estimates and an empty llh.
- **FDCC group-index check:** the contiguity check never fires (same logical-comparison pattern). For example,
  groups c(1,3) give G = 2 and series with label 3 get an all-zero row of S, so α = β = 0 for that series.
- **FDCC with G = 1:** the constraint is a + b ≤ 1 (section 2.4).
- **Estimation and reporting use different presamples** (0 vs 1), and FDCC also uses a different Qbar
  (z'z/T vs cov).
- **The DCC parameter limits are boxes [eps, 1]**, not a reparametrisation. aDCC with order (1,2) on this data
  gives dccb2 ≈ 1e-5 and huge SEs (1.10, 1.02).
- **aDCC + mvlaplace works** (dccg ≈ 1e-7 here). **DCC orders (P, Q) > 1 work** for DCC and aDCC.
- **Score mixing:** the A rows and score columns mix -ℓ (univariate) and +ℓ (DCC) conventions, and V uses Ā⁻¹ on
  both sides (section 6). Reproduce this literally to match.
- **When `fit=` is given**, the DCC recursion uses that fit's residuals and sigma directly. The Hessian
  re-filters using the multispec stored in the DCC spec, so the DCC spec's uspec must match the fits' specs.

## 10. Example (test oracle hint)

**Data and univariate fits:**
- Data: `dcc3.csv` in this scratchpad, T = 1500, m = 3, columns A, B, C, generated by `run1.R` with
  set.seed(42).
  - Simulated as DCC(0.05, 0.9) with Rbar off-diagonals (.5, .3, .4) and GARCH(1,1) normal margins, plus 0.01.
  - xts index from 2000-01-01, daily.
- Univariate models: `ugarchspec(mean armaOrder (0,0) with mean, sGARCH(1,1), norm)` fitted with `multifit`;
  dccfit uses `fit = mf` and solver solnp.

Univariate estimates (mu, omega, alpha1, beta1):

| series | mu | omega | alpha1 | beta1 | llh |
|---|---|---|---|---|---|
| A | -0.00891999125885 | 0.01947914006937 | 0.07282178352751 | 0.90657081832066 | -2001.64239768346 |
| B | -0.02778587421953 | 0.06144179210890 | 0.08407703380533 | 0.84625809773984 | -1991.03468862373 |
| C | -0.00997194689148 | 0.02565498174998 | 0.05037006752284 | 0.93387233803435 | -2433.41590886058 |

The univariate llh sum is -6426.09299517.

**DCC/mvnorm (1,1):**
- dcca1 = 0.0431630001835819, dccb1 = 0.9142701826310855.
- llh = **-6082.94087772754**. Stage-2 objective (LLH1, zero-presample) at the optimum = 1911.09985833825.
- AIC 8.13325450364, BIC 8.19347100136, Shibata 8.13300143181, HQ 8.15568733467.
- se(dcca1) = 0.00627467301294802 (t 6.87892422354, p 6.0307e-12).
- se(dccb1) = 0.01326755808404980 (t 68.9102076538).
- Qbar = cov(z), symmetric: diagonal (1.003953652140, 0.998490213085, 1.005193385645); (1,2) 0.475223590240,
  (1,3) 0.294556722873, (2,3) 0.340855529165.
- R_1: r12 = 0.497294779040641, r13 = 0.323589762766861, r23 = 0.368658387124292.
- Q_1 diagonal: (1.003783000651985, 0.998555380017464, 1.004969223539578).
- First per-observation ℓ_t: -3.87527571695479, -2.9806184777153, -5.51904233840933.
- Captured DCC Hessian block (of +ℓ) on the (a, b) columns: [[-48087.363, -15786.133], [-15786.133, -9309.785]].

**Other variants** (same data and univariate fits; llh, then the DCC parameters as estimate (se)):

| variant | llh | DCC parameters | AIC / BIC |
|---|---|---|---|
| DCC/mvt | -6083.64532606 | dcca1 0.04306741677 (0.00637055), dccb1 0.91353820409 (0.01371052), mshape 49.99998817 (10.0908) | 8.13552710141 / 8.19928574605 |
| aDCC/mvnorm | -6082.91639426 | dcca1 0.04243321505 (0.00695453), dccb1 0.91398307950 (0.01347095), dccg1 0.00240804736 (0.01040230) | 8.13455519234 / 8.19831383699 |
| FDCC/mvnorm, groups (1,1,2) | -6082.56656677 | a[g1] 0.19004678480 (0.02580607), a[g2] 0.24108698852 (0.04327988), b[g1] 0.96463050567 (0.01296845), b[g2] 0.94296237137 (0.01708654) | 8.13542208903 / 8.20272288060 |
| DCC/mvlaplace | -6364.40615222 | dcca1 0.03872917063 (0.00679018), dccb1 0.90870637605 (0.01707081) | 8.50854153629 / 8.56875803401 |
| DCC(2,1)/mvnorm | -6082.93058943 | dcca1 0.04060218 (0.01671719), dcca2 0.00292010 (0.01712745), dccb1 0.91353221 (0.01345161) | — |
| FDCC one group | -6106.16437055 | a 0.23300904, b 0.76699094 (constraint bug) | — |
| DCC/mvt, mshape fixed at 8 | -6132.56175655 | dcca1 0.04016082 (0.00650193), dccb1 0.91135622 (0.01554983) | AIC 8.19941567540 |

- aDCC/mvnorm: Nbar diagonal (0.32925751520, 0.36020626676, 0.33486534108); off-diagonals 0.14533523897,
  0.07781415473, 0.09450954717. Persistence including δ is 0.957363545732.
- FDCC/mvnorm: mfit$Qbar = z'z/T, with diagonal (1.003321306019, 0.997830832278, 1.004547927389).
