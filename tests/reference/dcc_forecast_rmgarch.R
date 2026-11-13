# Run from the repository root: Rscript tests/reference/dcc_forecast_rmgarch.R
# The oracle of the DCC forecast (plan 4c): rmgarch's dccforecast (n.roll = 0) on the dcc3 panel.
suppressPackageStartupMessages({
    library(rmgarch)
    library(jsonlite)
})

reference <- file.path("tests", "reference")

panel_fit <- function(file) {
    data <- as.matrix(read.csv(file.path(reference, "data", file)))
    data <- data[complete.cases(data), , drop = FALSE]
    spec <- ugarchspec(variance.model = list(model = "sGARCH", garchOrder = c(1, 1)),
                       mean.model = list(armaOrder = c(0, 0)), distribution.model = "norm")
    specs <- multispec(replicate(ncol(data), spec))
    fit <- multifit(specs, data, solver = "solnp", solver.control = list(tol = 1e-10))
    stopifnot(all(vapply(fit@fit, function(x) x@fit$convergence == 0, TRUE)))
    list(data = data, spec = specs, fit = fit)
}

# The aDCC and the FDCC forecast one step only: rmgarch refuses n.ahead > 1 for the aDCC and gives NA Q for the
# FDCC from h = 2 (docs/guide/validation.md).
cases <- list(
    list(id = "dcc_mvnorm", model = "DCC", order = c(1, 1), dist = "mvnorm", ahead = 10),
    list(id = "dcc21_mvnorm", model = "DCC", order = c(2, 1), dist = "mvnorm", ahead = 10),
    list(id = "dcc_mvt", model = "DCC", order = c(1, 1), dist = "mvt", ahead = 10),
    list(id = "adcc_mvnorm", model = "aDCC", order = c(1, 1), dist = "mvnorm", ahead = 1),
    list(id = "fdcc_mvnorm", model = "FDCC", order = c(1, 1), dist = "mvnorm", ahead = 1)
)

forecast_case <- function(panel, case) {
    groups <- c(rep(1L, ncol(panel$data) - 1), 2L)
    spec <- dccspec(panel$spec, dccOrder = case$order, model = case$model, distribution = case$dist, groups = groups)
    fit <- dccfit(spec, data = panel$data, fit = panel$fit, solver = "solnp", solver.control = list(tol = 1e-10))
    stopifnot(fit@mfit$convergence == 0)
    params <- coef(fit, type = "dcc")
    names(params) <- sub("^\\[Joint\\]", "", names(params))
    names(params) <- sub("^fdcca\\[g=([0-9]+),P=1\\]$", "fdcca\\1", names(params))
    names(params) <- sub("^fdccb\\[g=([0-9]+),Q=1\\]$", "fdccb\\1", names(params))
    fc <- dccforecast(fit, n.ahead = case$ahead)
    correlation <- rcor(fc)[[1]]
    covariance <- rcov(fc)[[1]]
    sigma <- sigma(fc)[, , 1, drop = FALSE]
    steps <- seq_len(case$ahead)
    list(id = case$id, model = case$model, order = case$order, dist = case$dist, groups = groups, ahead = case$ahead,
         params = as.list(params),
         R = lapply(steps, function(h) unname(correlation[, , h])),
         H = lapply(steps, function(h) unname(covariance[, , h])),
         sigma = lapply(steps, function(h) unname(sigma[h, , 1])))
}

main <- function() {
    panel <- panel_fit("dcc3.csv")
    results <- lapply(cases, function(case) {
        message(case$id)
        forecast_case(panel, case)
    })
    fixture <- list(oracle = "rmgarch", oracle_version = as.character(packageVersion("rmgarch")),
                    rugarch_version = as.character(packageVersion("rugarch")), R_version = R.version.string,
                    panel = "dcc3.csv",
                    univariate = lapply(panel$fit@fit, function(fit) as.list(coef(fit))), results = results)
    write_json(fixture, file.path(reference, "fixtures", "rmgarch_dcc_forecast.json"),
               auto_unbox = TRUE, digits = I(17), pretty = TRUE, na = "null")
}

main()
