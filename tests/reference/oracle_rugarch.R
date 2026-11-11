# Fits every reference series with rugarch and writes the oracle fixture.
# Run from the repository root: Rscript tests/reference/oracle_rugarch.R

suppressPackageStartupMessages({
    library(rugarch)
    library(jsonlite)
})

data_path <- file.path("tests", "reference", "data", "sgarch11_norm.csv")
fixture_path <- file.path("tests", "reference", "fixtures", "rugarch_sgarch11_norm.json")

sgarch_spec <- function() {
    ugarchspec(
        variance.model = list(model = "sGARCH", garchOrder = c(1, 1)),
        mean.model = list(armaOrder = c(0, 0), include.mean = TRUE),
        distribution.model = "norm"
    )
}

# rugarch's robust.se.coef is a Newey-West sandwich (Bartlett kernel, floor(1.2 T^(1/3)) lags). The QML
# sandwich H^-1 S'S H^-1 is rebuilt from rugarch's own numerical Hessian and per-observation scores.
qml_robust_se <- function(fit) {
    hessian_inverse <- solve(fit@fit$hessian)
    covariance <- hessian_inverse %*% crossprod(fit@fit$scores) %*% hessian_inverse
    as.list(setNames(sqrt(diag(covariance)), names(coef(fit))))
}

fit_series <- function(name, values) {
    fit <- ugarchfit(sgarch_spec(), values, solver = "hybrid")
    if (convergence(fit) != 0) {
        stop(sprintf("rugarch did not converge on series %s", name))
    }
    # infocriteria() rows: Akaike, Bayes, Shibata, Hannan-Quinn, all per observation.
    criteria <- infocriteria(fit)
    list(
        series = name,
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
    table <- read.csv(data_path, check.names = FALSE)
    series <- colnames(table)
    results <- lapply(series, function(name) {
        values <- table[[name]]
        fit_series(name, values[!is.na(values)])
    })
    fixture <- list(
        oracle = "rugarch",
        oracle_version = as.character(packageVersion("rugarch")),
        series = series,
        results = results
    )
    dir.create(dirname(fixture_path), recursive = TRUE, showWarnings = FALSE)
    write_json(fixture, fixture_path, auto_unbox = TRUE, digits = I(17), pretty = TRUE)
}

main()
