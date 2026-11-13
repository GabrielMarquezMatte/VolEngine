# Fits every case of cases.json that lists rugarch and writes the oracle fixture.
# Run from the repository root: Rscript tests/reference/oracle_rugarch.R

suppressPackageStartupMessages({
    library(rugarch)
    library(jsonlite)
})

reference <- file.path("tests", "reference")
cases_path <- file.path(reference, "cases.json")
data_paths <- file.path(reference, "data", c("sgarch11_norm.csv", "grid.csv", "grid_2b.csv", "grid_2c.csv",
                                                "grid_2e.csv", "grid_2f.csv"))
regressors_path <- file.path(reference, "data", "regressors_2e.csv")
fixture_path <- file.path(reference, "fixtures", "rugarch.json")

load_series <- function(paths) {
    columns <- list()
    for (path in paths) {
        table <- read.csv(path, check.names = FALSE)
        for (name in colnames(table)) {
            values <- table[[name]]
            columns[[name]] <- values[!is.na(values)]
        }
    }
    columns
}

models <- c(sgarch = "sGARCH", egarch = "eGARCH", gjrgarch = "gjrGARCH", aparch = "apARCH", igarch = "iGARCH",
            csgarch = "csGARCH", fgarch = "fGARCH")

# The columns a case names, as the matrix rugarch's external.regressors takes; NULL when it names none.
regressor_matrix <- function(columns, regressors) {
    if (is.null(columns)) {
        return(NULL)
    }
    as.matrix(regressors[, unlist(columns), drop = FALSE])
}

# The starting values the case sets. A start value for archm makes ugarchfit take the pre-sample residuals from the
# mean without the archm term (VolEngine's e0) instead of running its auxiliary fit without archm first. archm and
# every vxreg start at 0.01 and not at 0 (rugarch's default for vxreg, and the value the plan names for archm): from
# 0 the solver stops there on the flat likelihoods, at the lower bound of vxreg and where the Hessian step, a fraction
# of the parameter, vanishes (see docs/guide/validation.md).
start_pars <- function(case) {
    start <- list()
    if (!is.null(case$archm)) {
        start$archm <- 0.01
    }
    for (k in seq_along(case$vxreg)) {
        start[[paste0("vxreg", k)]] <- 0.01
    }
    start
}

# rugarch's garchOrder is c(ARCH order, GARCH order): VolEngine's (p, q).
case_spec <- function(case, regressors) {
    variance <- list(model = models[[case$model]], garchOrder = c(case$p, case$q),
                     variance.targeting = isTRUE(case$variance_targeting),
                     external.regressors = regressor_matrix(case$vxreg, regressors))
    if (!is.null(case$submodel)) {
        variance$submodel <- toupper(case$submodel)
    }
    archm <- !is.null(case$archm)
    ugarchspec(
        variance.model = variance,
        mean.model = list(armaOrder = unlist(case$arma), include.mean = case$include_mean, archm = archm,
                          arfima = isTRUE(case$arfima),
                          archpow = if (archm) case$archm else 1,
                          external.regressors = regressor_matrix(case$mxreg, regressors)),
        distribution.model = case$dist,
        start.pars = start_pars(case),
        fixed.pars = if (is.null(case$fixed)) list() else case$fixed
    )
}

# A case with every parameter fixed is an evaluation: ugarchfilter, no optimizer.
is_evaluation <- function(spec) {
    pars <- spec@model$pars
    all(pars[pars[, "Include"] == 1, "Fixed"] == 1)
}

# rugarch lists the fixed parameters in place but appends the variance-targeting omega to coef(); VolEngine's
# order is rugarch's parameter table.
in_model_order <- function(values, fit) {
    position <- match(names(values), rownames(fit@model$pars))
    stopifnot(!anyNA(position))
    values[order(position)]
}

# rugarch's robust.se.coef is a Newey-West sandwich (Bartlett kernel, floor(1.2 T^(1/3)) lags). The QML
# sandwich H^-1 S'S H^-1 is rebuilt from rugarch's own numerical Hessian and per-observation scores. The scores
# cover the estimated parameters only: the derived iGARCH beta_q gets NA, as in matcoef.
qml_robust_se <- function(fit) {
    hessian_inverse <- solve(fit@fit$hessian)
    covariance <- hessian_inverse %*% crossprod(fit@fit$scores) %*% hessian_inverse
    se <- setNames(rep(NA_real_, length(coef(fit))), names(coef(fit)))
    se[colnames(fit@fit$scores)] <- sqrt(diag(covariance))
    as.list(se)
}

# The parameters the optimizer moved: neither fixed nor derived (the iGARCH beta_q, the variance-targeting omega).
estimated_names <- function(fit) {
    ipars <- fit@fit$ipars
    rownames(ipars)[ipars[, "Estimate"] == 1]
}

block <- function(params, name) {
    params[grepl(sprintf("^%s[0-9]+$", name), names(params))]
}

# The weights are the non-negative entries that sum to the persistence P, named by their position. A power-family
# weight is alpha * kappa, taken from rugarch's persistence, so these cases have p = 1. The gjrGARCH weights use
# pi = P(z < 0), which rugarch integrates into its persistence P = sum(alpha) + sum(beta) + pi sum(gamma); with
# gamma_i fixed and alpha_i estimated they are alpha_i - max(0, -gamma_i) and max(0, -gamma_i) + pi gamma_i.
interior_weights <- function(case, params, persistence, estimated) {
    alpha <- block(params, "alpha")
    beta <- block(params, "beta")
    if (case$model == "egarch") {
        return(beta)
    }
    if (case$model == "gjrgarch") {
        gamma <- block(params, "gamma")
        below_zero <- (persistence - sum(alpha) - sum(beta)) / sum(gamma)
        split <- names(alpha) %in% estimated & !(names(gamma) %in% estimated)
        floor <- pmax(0, -gamma)
        alpha_weight <- ifelse(split, alpha - floor, (1 - below_zero) * alpha)
        gamma_weight <- ifelse(split, floor + below_zero * gamma, below_zero * (alpha + gamma))
        return(c(setNames(alpha_weight, names(alpha)), beta, setNames(gamma_weight, names(gamma))))
    }
    if (case$model %in% c("aparch", "fgarch")) {
        stopifnot(case$p == 1)
        return(c(alpha1 = persistence - sum(beta), beta))
    }
    c(alpha, beta)
}

# Only the estimated parameters are held to the rule. Every estimated weight > 1e-4 and P < 0.999 (iGARCH: P = 1
# by construction); |eta1| and the apARCH |gamma| < 0.999; csGARCH: rho - P > 1e-4 and phi > 1e-4.
interior_failures <- function(case, params, persistence, estimated) {
    weights <- interior_weights(case, params, persistence, estimated)
    weights <- weights[names(weights) %in% estimated]
    failures <- weights[weights <= 1e-4]
    if (case$model != "igarch" && persistence >= 0.999) {
        failures <- c(failures, persistence = persistence)
    }
    asymmetry <- c(block(params, "eta1"), block(params, "gamma"))
    asymmetry <- asymmetry[names(asymmetry) %in% estimated]
    if (case$model %in% c("aparch", "fgarch")) {
        failures <- c(failures, asymmetry[abs(asymmetry) >= 0.999])
    }
    if (case$model == "csgarch") {
        failures <- c(failures, block(params, "eta1")[params[["eta11"]] - persistence <= 1e-4],
                      block(params, "eta2")[params[["eta21"]] <= 1e-4])
    }
    failures
}

# The skewed distributions: |rho| < 0.999 (nig, ghyp), xi in (0.05, 20) (sstd, sged), 1 / shape > 1e-4 and
# |ghlambda| < 10. A distribution, archm, arfima or regressor parameter must also stay off rugarch's own bounds (its LB
# and UB columns; vxreg is bounded to [0, 100], arfima to [1e-8, 0.5]).
distribution_failures <- function(case, fit) {
    params <- coef(fit)
    params <- params[names(params) %in% estimated_names(fit)]
    failures <- c()
    if (case$dist %in% c("nig", "ghyp") && "skew" %in% names(params)) {
        failures <- c(failures, params["skew"][abs(params[["skew"]]) >= 0.999])
    }
    if (case$dist %in% c("sstd", "sged") && "skew" %in% names(params)) {
        failures <- c(failures, params["skew"][params[["skew"]] <= 0.05 || params[["skew"]] >= 20])
    }
    if ("shape" %in% names(params)) {
        failures <- c(failures, params["shape"][1 / params[["shape"]] <= 1e-4])
    }
    if ("ghlambda" %in% names(params)) {
        failures <- c(failures, params["ghlambda"][abs(params[["ghlambda"]]) >= 10])
    }
    names_dist <- names(params)[grepl("^(skew|shape|ghlambda|archm|arfima|mxreg[0-9]+|vxreg[0-9]+)$", names(params))]
    bounds <- fit@model$pars[names_dist, c("LB", "UB"), drop = FALSE]
    on_bound <- abs(params[names_dist] - bounds[, "LB"]) <= 1e-6 * pmax(1, abs(bounds[, "LB"])) |
        abs(params[names_dist] - bounds[, "UB"]) <= 1e-6 * pmax(1, abs(bounds[, "UB"]))
    c(failures, params[names_dist][on_bound])
}

check_interior <- function(case, fit, persistence) {
    params <- coef(fit)
    failures <- c(interior_failures(case, params, persistence, estimated_names(fit)),
                  distribution_failures(case, fit))
    if (length(failures) > 0) {
        stop(sprintf("rugarch estimate on case %s is on the boundary: %s", case$id,
                     paste(names(failures), signif(failures, 4), sep = " = ", collapse = ", ")))
    }
}

# rugarch's default Hessian is Richardson-extrapolated from a first step of 10% of each parameter (hess.d = 0.1),
# which is biased where the likelihood is ill-conditioned (two GARCH lags) or not smooth (ged with shape < 2).
# A first step of 0.1% removes that bias. sged, fitted with shape < 2 at a kink that is not at z = 0, needs 0.01%.
# The regressor and archm cases take a first step of 1%: the step is a fraction of each parameter, and with a
# coefficient near zero (mu next to archm, a mean regressor, an apARCH gamma) 0.1% is smaller than the numerical noise
# (docs/guide/validation.md). The ARFIMA cases take 1% for the same reason: the eGARCH alpha1 sits near zero.
hessian_step <- function(case) {
    if (case$dist == "sged") {
        return(1e-4)
    }
    if (uses_regressors(case) || isTRUE(case$arfima)) {
        return(1e-2)
    }
    1e-3
}

uses_regressors <- function(case) {
    !is.null(case$mxreg) || !is.null(case$vxreg) || !is.null(case$archm)
}

# The regressor and archm cases tighten solnp's tolerance (default 1e-8) to 1e-10: with the default, rugarch stops up to
# 1e-5 from the optimum along the flat mu-archm direction (docs/guide/validation.md).
solver_control <- function(case) {
    if (uses_regressors(case)) {
        return(list(tol = 1e-10))
    }
    list()
}

# The fixed point evaluated: infocriteria() of a uGARCHfilter counts no parameter. No standard error.
filter_case <- function(case, spec, series) {
    filtered <- ugarchfilter(spec, series[[case$series]])
    params <- in_model_order(coef(filtered), filtered)
    none <- as.list(setNames(rep(NA_real_, length(params)), names(params)))
    criteria <- infocriteria(filtered)
    list(
        case = case$id,
        series = case$series,
        loglik = likelihood(filtered),
        params = as.list(params),
        se = none,
        robust_se = none,
        qml_robust_se = none,
        persistence = persistence(filtered)[[1]],
        sigma = as.numeric(sigma(filtered)),
        aic = criteria["Akaike", 1],
        bic = criteria["Bayes", 1],
        hqic = criteria["Hannan-Quinn", 1],
        sic = criteria["Shibata", 1]
    )
}

fit_case <- function(case, series, regressors) {
    spec <- case_spec(case, regressors)
    if (is_evaluation(spec)) {
        return(filter_case(case, spec, series))
    }
    fit <- ugarchfit(spec, series[[case$series]], solver = "hybrid", solver.control = solver_control(case),
                     numderiv.control = list(hess.d = hessian_step(case)))
    if (convergence(fit) != 0) {
        stop(sprintf("rugarch did not converge on case %s", case$id))
    }
    # csGARCH reports (transitory, permanent) persistence; P is the transitory one, alpha + beta.
    persistence <- persistence(fit)[[1]]
    check_interior(case, fit, persistence)
    # infocriteria() rows: Akaike, Bayes, Shibata, Hannan-Quinn, all per observation.
    criteria <- infocriteria(fit)
    list(
        case = case$id,
        series = case$series,
        loglik = likelihood(fit),
        params = as.list(in_model_order(coef(fit), fit)),
        se = as.list(in_model_order(fit@fit$matcoef[, " Std. Error"], fit)),
        robust_se = as.list(in_model_order(fit@fit$robust.matcoef[, " Std. Error"], fit)),
        qml_robust_se = in_model_order(qml_robust_se(fit), fit),
        persistence = persistence,
        sigma = as.numeric(sigma(fit)),
        aic = criteria["Akaike", 1],
        bic = criteria["Bayes", 1],
        hqic = criteria["Hannan-Quinn", 1],
        sic = criteria["Shibata", 1]
    )
}

main <- function() {
    cases <- Filter(function(case) "rugarch" %in% unlist(case$oracles), read_json(cases_path))
    series <- load_series(data_paths)
    regressors <- read.csv(regressors_path, check.names = FALSE)
    fixture <- list(
        oracle = "rugarch",
        oracle_version = as.character(packageVersion("rugarch")),
        cases = vapply(cases, function(case) case$id, ""),
        results = lapply(cases, fit_case, series = series, regressors = regressors)
    )
    dir.create(dirname(fixture_path), recursive = TRUE, showWarnings = FALSE)
    write_json(fixture, fixture_path, auto_unbox = TRUE, digits = I(17), pretty = TRUE, na = "null")
}

# Sourced by forecast_rugarch.R for its helpers: main() runs only under Rscript on this file.
if (sys.nframe() == 0) {
    main()
}
