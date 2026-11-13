# rmgarch 1.4.3 Copula-DCC conventions (R 4.6.2): formula-level summary for plan 4b

Read from `cgarchfit` and its helpers to understand conventions, then confirmed numerically: an independent
evaluation in `tests/reference/copula_probe.R`, written from the formulas below and not from rmgarch code, reproduces
`likelihood(fit)` of all 13 cases to |diff| <= 9e-12. No rmgarch code is copied or translated. Notation as in the
plan 4a notes: m series, T observations, z_it = e_it / sigma_it of each univariate fit.

## 1. Data and univariate fits

- `tests/reference/data/copula3.csv`: simulated by the probe (seed 20261113), T = 2000, m = 3. Three sGARCH(1,1)
  series with standardized-t(7) margins joined by a Student-t(6) copula with DCC(0.04, 0.93) on Qbar =
  [[1, .5, .3], [.5, 1, .4], [.3, .4, 1]]. dcc3 is not used: its margins are near-normal, so the copula shape runs
  to the 50 ceiling and rmgarch's standard errors for the Kendall t copula fail (singular A).
- Univariate: sGARCH(1,1)-std, ARMA(0,0), `multifit`, solnp, tol 1e-10. Sum of univariate llh
  -8664.90170655746.

| | A | B | C |
| --- | --- | --- | --- |
| mu | -0.0181865826694867 | 0.0145422387458537 | 0.00843170064471263 |
| omega | 0.0188827760630150 | 0.0560886489944989 | 0.04917700882790111 |
| alpha1 | 0.0542214285957902 | 0.0957365591243709 | 0.06558797670577575 |
| beta1 | 0.9267446862078732 | 0.8661253621409556 | 0.89906386979536290 |
| shape | 6.9885886575033203 | 5.3111792779986891 | 6.68792316409210841 |

## 2. Margins (u)

- parametric: u_it = F_i(z_it), `pdist` of the fit's distribution with its estimated skew, shape and lambda.
- empirical: u_it = `ecdf(z_i)(z_it)` = rank_it / T (not T + 1).
- then, for both: u > 0.99999 becomes 0.99999, and u < 1.5 eps becomes eps (eps = 2^-52). The cut is asymmetric:
  on the upper side it bites at 1 - 1e-5, so the largest empirical u is always 0.99999.

## 3. Scores (epsilon) and the copula log-likelihood

- mvnorm: epsilon = qnorm(u). mvt: epsilon = qdist("std", u, shape = nu), the **standardized** t quantile; it
  depends on nu, so Qbar, Nbar and Kendall's R are recomputed from epsilon(nu) at every evaluation.
- c_t = log f(epsilon_t; R_t[, nu]) - sum_i log f1(epsilon_it[; nu]), with every constant: f the mvnorm, or the
  standardized mvt of plan 4a with R_t as its correlation (`lgamma((nu+m)/2) - lgamma(nu/2) - m/2 log(pi (nu-2)) -
  1/2 log|R| - (nu+m)/2 log(1 + q/(nu-2))`); f1 the standard normal or standardized-t density.
- `likelihood(fit)` = sum_i l_i (univariate) + sum_t c_t, also with empirical margins.

## 4. Dynamic copula (time.varying = TRUE)

- Qbar = cov(epsilon) (centred, T - 1). aDCC: n = 1(epsilon < 0) epsilon, Nbar = cov(n) (centred, T - 1); delta
  from that Nbar (the 4a notes' rule).
- Presample epsilon_s = n_s = 0 for s <= 0, Q_s = Qbar, in the objective **and** in everything reported (llh, R_t,
  standard errors). Unlike `dccfit`, there is no presample of ones.
- Constraint and bounds as `dccfit`: a, b, g in [eps, 1], sum a + sum b (+ delta sum g) < 1; mshape in [4, 50].
- Starts a 0.05, b 0.9, g 0.05, mshape 5.

## 5. Static copula (time.varying = FALSE)

`method` picks the correlation estimator for **both** copulas:

| copula | method | R | [Joint] rows |
| --- | --- | --- | --- |
| mvnorm | Kendall (default) | cor(epsilon) (Pearson; the name is misleading) | none |
| mvnorm | ML | from_free(C) by ML | C1..C{m(m-1)/2} |
| mvt | Kendall (default) | sin(pi tau_b / 2), tau_b of epsilon (= of u: ranks) | mshape |
| mvt | ML | from_free(C) by ML, jointly with nu | C1..C{m(m-1)/2}, mshape |

- from_free(C): A lower-triangular with unit diagonal and C in R's `lower.tri` order (column-major: C1 = A21,
  C2 = A31, C3 = A32 for m = 3), R = cov2cor(A A'). Row i of A over its norm is row i of the Cholesky factor of R.
  The reported C are these free entries, not correlations. Boxes C in [-0.99, 0.99], start 0.
- When Kendall's (or cor's) R has a negative eigenvalue, rmgarch adds the projection `.makeposdef` (eigenvalues
  lifted to 2 m max|lambda| eps) with a warning.

## 6. Standard errors

The same two-stage sandwich as `dccfit` (plan 4a notes, section 6): A with the univariate blocks inverse(cvar) and
the copula rows the forward-difference Hessian of the full llh (refiltering the univariates, recomputing u,
epsilon, Qbar, Nbar); S with the univariate scores and central-difference copula scores, h = max(|theta|/2, 0.01)
eps^(1/3); V = A^-1 B A^-1 / T, no transpose. Empirical margins are refiltered too: the ranks of the perturbed z
are recomputed, so a perturbation that swaps two ranks shows in the finite differences. mvnorm Kendall has no
[Joint] row and no copula block.

## 7. Reference numbers (copula3, rmgarch's estimates)

llh = `likelihood(fit)`; R entries are R21, R31, R32.

| id | llh | [Joint] estimate (se robust) | R_1 | R_1000 | R_2000 |
| --- | --- | --- | --- | --- | --- |
| static_mvnorm_kendall_param | -8117.1869165044 | — | .513710062071992 .398508626536416 .406936683559792 | = R_1 | = R_1 |
| static_mvnorm_ml_param | -8117.1868564310 | C1 .598672396279854 (.0286473454512667), C2 .449366756278866 (.0280579284266360), C3 .265832497450360 (.0269658866816793) | .513657953074682 .398341381350193 .406795476599054 | | |
| static_mvnorm_ml_emp | -8115.7480726458 | C1 .597226254277687 (.0291107096657751), C2 .450164982005494 (.0273423435915265), C3 .267833429564139 (.0268768904219042) | .512743670621971 .398769267011434 .408159448768339 | | |
| static_mvt_kendall_param | -8012.1841333659 | mshape 4.7793578808066 (.3902939004744) | .510975878847646 .389184431763653 .402305107376601 | | |
| static_mvt_kendall_emp | -8015.1511339642 | mshape 4.95613482263086 (.392780601957729) | .510975878847646 .389184431763653 .402305107376601 | | |
| static_mvt_ml_param | -8012.0736645395 | C1 .594217920400736 (.0272004100371396), C2 .436936698871691 (.0269005277361837), C3 .277810026573495 (.0253538144732677), mshape 4.790827028666584 (.4064715629142370) | .510836135301799 .388010185709091 .410293857644785 | | |
| dcc_mvnorm_param | -7999.2557546252 | dcca1 .0473819588956065 (.00688260476128296), dccb1 .9175354409917849 (.01506025946251671) | .513710062071992 .398508626536416 .406936683559792 | .542055905211159 .243814997355676 .142280592260793 | .155972386680157 .088123875634007 .207603432323858 |
| dcc_mvnorm_emp | -8000.3387982860 | dcca1 .0469471761609685 (.00704190213083246), dccb1 .9179675558919811 (.01539842075082857) | .513374041590776 .399416241144631 .408805414514274 | .542471816068691 .244683660349199 .149864669396081 | .154850759329173 .0880884649009998 .21091235366846 |
| dcc_mvt_param | -7916.6409789490 | dcca1 .0444504443322488 (.0058219978418709), dccb1 .9194035843020166 (.0121850853069757), mshape 5.8841836499319067 (.4971728250148147) | .518184331109798 .412627644979759 .438797045747512 | .538995915675457 .293665921136297 .168695842937856 | .144506744820049 .144239270083496 .239873143273593 |
| dcc_mvt_emp | -7922.6218623579 | dcca1 .0451190098670297 (.00629871570871223), dccb1 .9172060812961198 (.01287396420998735), mshape 6.3934861824246285 (.51738958619896758) | .516726508249743 .414282949503144 .445589776318746 | .535304409700781 .289680200120334 .182618695813444 | .149082826076376 .144239174414481 .24885537152601 |
| adcc_mvnorm_param | -7999.0810297674 | dcca1 .0453138390863231 (.00800333265116977), dccb1 .9178306244174149 (.01490734007035691), dccg1 .0048824360257134 (.00975448437638895) | .5138191159492 .398532802736662 .406978936073836 | .543515030706206 .255728591665038 .15239787292293 | .170666902988211 .102095366922369 .213960418915813 |
| adcc_mvt_param | -7916.6409793054 | dcca1 .0444504473500459 (.00692009848883715), dccb1 .919403559315639 (.01218639325288598), dccg1 4.75615550848946e-08 (.00913182661616523), mshape 5.88418954448619 (.49737239024369195) | .518184334381777 .412627632339022 .438796992462346 | .538995932542449 .293665925607916 .168695845279408 | .144506866673249 .144239252917887 .239873140428432 |
| dcc_mvt_param_shape8 | -7921.3163263539 | dcca1 .0463737883675772 (.00608616991829144), dccb1 .9188279296008739 (.01281214954238004) | .518051176560416 .408981225401481 .426668133095522 | .541739218276258 .275555606117706 .152817329501554 | .139279869296678 .11975745905962 .222571891227835 |

## 8. Verdict (rule 8a of the plan 4 spec)

| Item | rmgarch | Public (Standard) | Kind | Evidence |
| --- | --- | --- | --- | --- |
| empirical u | rank/T | rank/(T+1) | **[Rmgarch]** deviation | Genest-Ghoudi-Rivest 1995: rank/(T+1) keeps u inside (0,1); rank/T puts the maximum at 1, hidden by the cut |
| cut of u | [eps, 0.99999], asymmetric | [2^-53, 1 - 2^-53] | **[Rmgarch]** deviation | the upper cut at 1 - 1e-5 changes every u above it; a symmetric cut at the double limit only avoids infinite quantiles |
| presample | zeros, objective and reported | plan 4a Standard (z_s z_s' = Qbar, n_s n_s' = Nbar) | **[Rmgarch]**, the 4a point (Stage::Objective) | plan 4a notes 11.2 |
| Nbar | centred | uncentred | **[Rmgarch]**, the 4a point | plan 4a notes 11.5 |
| sandwich | no transpose | A^-1 B A^-T / T | **[Rmgarch]**, the 4a point | plan 4a notes 11.4 |
| empirical standard errors | finite differences through recomputed ranks | cross block 0 (ranks held fixed) | **[Rmgarch]** deviation | ranks are piecewise constant in theta; their finite differences are 0 or spikes, neither an estimate (Chen-Fan 2006 give the proper correction, not implemented) |
| mvnorm static "Kendall" | cor(epsilon) | sin(pi tau_b / 2) | **[Rmgarch]** deviation | for any elliptical copula tau = (2/pi) arcsin(rho) (Lindskog-McNeil-Schmock 2003); cor(epsilon) is a moment estimator under another name |
| C boxes | [-0.99, 0.99] | unconstrained (any real x gives a valid R) | not reproduced; documented | interior in every oracle case |
| Kendall R not positive definite | projection with a warning | status Failed (plan rule) | not reproduced; documented | no oracle case hits it |
| copula llh, standardized-t scores, Qbar = cov(epsilon), mvt Kendall, from_free, mshape in [4, 50] | — | same | defensible convention, kept | section 3 reproduces rmgarch to 1e-11 |
