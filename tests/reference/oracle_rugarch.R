# Fits every case of cases.json that lists rugarch and writes the oracle fixture.
# Run from the repository root: Rscript tests/reference/oracle_rugarch.R

suppressPackageStartupMessages({
    library(rugarch)
    library(jsonlite)
})

reference <- file.path("tests", "reference")
cases_path <- file.path(reference, "cases.json")
data_paths <- file.path(reference, "data", c("sgarch11_norm.csv", "grid.csv"))
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

# rugarch's garchOrder is c(ARCH order, GARCH order): VolEngine's (p, q).
case_spec <- function(case) {
    ugarchspec(
        variance.model = list(model = "sGARCH", garchOrder = c(case$p, case$q)),
        mean.model = list(armaOrder = unlist(case$arma), include.mean = case$include_mean),
        distribution.model = case$dist
    )
}

# rugarch's robust.se.coef is a Newey-West sandwich (Bartlett kernel, floor(1.2 T^(1/3)) lags). The QML
# sandwich H^-1 S'S H^-1 is rebuilt from rugarch's own numerical Hessian and per-observation scores.
qml_robust_se <- function(fit) {
    hessian_inverse <- solve(fit@fit$hessian)
    covariance <- hessian_inverse %*% crossprod(fit@fit$scores) %*% hessian_inverse
    as.list(setNames(sqrt(diag(covariance)), names(coef(fit))))
}

check_interior <- function(id, params) {
    weights <- params[grepl("^(alpha|beta)[0-9]+$", names(params))]
    if (min(weights) <= 1e-4 || sum(weights) >= 0.999) {
        stop(sprintf("rugarch estimate on case %s is on the boundary: %s", id,
                     paste(names(weights), signif(weights, 4), sep = " = ", collapse = ", ")))
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
    check_interior(case$id, coef(fit))
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
    write_json(fixture, fixture_path, auto_unbox = TRUE, digits = I(17), pretty = TRUE)
}

main()
