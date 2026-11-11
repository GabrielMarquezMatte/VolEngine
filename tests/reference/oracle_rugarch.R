# Fits every case of cases.json that lists rugarch and writes the oracle fixture.
# Run from the repository root: Rscript tests/reference/oracle_rugarch.R

suppressPackageStartupMessages({
    library(rugarch)
    library(jsonlite)
})

reference <- file.path("tests", "reference")
cases_path <- file.path(reference, "cases.json")
data_paths <- file.path(reference, "data", c("sgarch11_norm.csv", "grid.csv", "grid_2b.csv"))
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

# rugarch's garchOrder is c(ARCH order, GARCH order): VolEngine's (p, q).
case_spec <- function(case) {
    variance <- list(model = models[[case$model]], garchOrder = c(case$p, case$q))
    if (!is.null(case$submodel)) {
        variance$submodel <- toupper(case$submodel)
    }
    ugarchspec(
        variance.model = variance,
        mean.model = list(armaOrder = unlist(case$arma), include.mean = case$include_mean),
        distribution.model = case$dist
    )
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

block <- function(params, name) {
    params[grepl(sprintf("^%s[0-9]+$", name), names(params))]
}

# The weights are the non-negative entries that sum to the persistence P. A power-family weight is alpha * kappa,
# taken from rugarch's persistence, so these cases have p = 1.
interior_weights <- function(case, params, persistence) {
    alpha <- block(params, "alpha")
    beta <- block(params, "beta")
    if (case$model == "egarch") {
        return(beta)
    }
    if (case$model == "gjrgarch") {
        return(c(alpha / 2, beta, (alpha + block(params, "gamma")) / 2))
    }
    if (case$model %in% c("aparch", "fgarch")) {
        stopifnot(case$p == 1)
        return(c(alpha1_kappa1 = persistence - sum(beta), beta))
    }
    c(alpha, beta)
}

# Every weight > 1e-4 and P < 0.999 (iGARCH: P = 1 by construction); |eta1| and the apARCH |gamma| < 0.999;
# csGARCH: rho - P > 1e-4 and phi > 1e-4.
interior_failures <- function(case, params, persistence) {
    weights <- interior_weights(case, params, persistence)
    failures <- weights[weights <= 1e-4]
    if (case$model != "igarch" && persistence >= 0.999) {
        failures <- c(failures, persistence = persistence)
    }
    asymmetry <- c(block(params, "eta1"), block(params, "gamma"))
    if (case$model %in% c("aparch", "fgarch")) {
        failures <- c(failures, asymmetry[abs(asymmetry) >= 0.999])
    }
    if (case$model == "csgarch") {
        failures <- c(failures, block(params, "eta1")[params[["eta11"]] - persistence <= 1e-4],
                      block(params, "eta2")[params[["eta21"]] <= 1e-4])
    }
    failures
}

check_interior <- function(case, params, persistence) {
    failures <- interior_failures(case, params, persistence)
    if (length(failures) > 0) {
        stop(sprintf("rugarch estimate on case %s is on the boundary: %s", case$id,
                     paste(names(failures), signif(failures, 4), sep = " = ", collapse = ", ")))
    }
}

# rugarch's default Hessian is Richardson-extrapolated from a first step of 10% of each parameter (hess.d = 0.1),
# which is biased where the likelihood is ill-conditioned (two GARCH lags) or not smooth (ged with shape < 2).
# A first step of 0.1% removes that bias.
fit_case <- function(case, series) {
    fit <- ugarchfit(case_spec(case), series[[case$series]], solver = "hybrid",
                     numderiv.control = list(hess.d = 1e-3))
    if (convergence(fit) != 0) {
        stop(sprintf("rugarch did not converge on case %s", case$id))
    }
    # csGARCH reports (transitory, permanent) persistence; P is the transitory one, alpha + beta.
    persistence <- persistence(fit)[[1]]
    check_interior(case, coef(fit), persistence)
    # infocriteria() rows: Akaike, Bayes, Shibata, Hannan-Quinn, all per observation.
    criteria <- infocriteria(fit)
    list(
        case = case$id,
        series = case$series,
        loglik = likelihood(fit),
        params = as.list(coef(fit)),
        se = as.list(fit@fit$matcoef[, " Std. Error"]),
        robust_se = as.list(fit@fit$robust.matcoef[, " Std. Error"]),
        qml_robust_se = qml_robust_se(fit),
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
    fixture <- list(
        oracle = "rugarch",
        oracle_version = as.character(packageVersion("rugarch")),
        cases = vapply(cases, function(case) case$id, ""),
        results = lapply(cases, fit_case, series = series)
    )
    dir.create(dirname(fixture_path), recursive = TRUE, showWarnings = FALSE)
    write_json(fixture, fixture_path, auto_unbox = TRUE, digits = I(17), pretty = TRUE, na = "null")
}

main()
