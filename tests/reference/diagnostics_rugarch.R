# Public diagnostics of rugarch fits; formulas are documented in docs/guide/validation.md.
# Run from the repository root: Rscript tests/reference/diagnostics_rugarch.R
source(file.path("tests", "reference", "oracle_rugarch.R"))

diagnostics_path <- file.path(reference, "fixtures", "rugarch_diagnostics.json")
weighted_box <- getFromNamespace("Weighted.Box.test", "rugarch")
weighted_lm <- getFromNamespace("Weighted.LM.test", "rugarch")

test_pair <- function(test) {
    unname(c(test$statistic, test$p.value))
}

nyblom_result <- function(fit) {
    if (inherits(fit, "uGARCHfilter")) {
        return(list(individual = list(), joint = NA_real_))
    }
    result <- nyblom(fit)
    individual <- setNames(as.numeric(result$IndividualStat), rownames(result$IndividualStat))
    list(individual = as.list(individual), joint = unname(result$JointStat))
}

diagnostics_case <- function(case, series, regressors) {
    message(case$id)
    spec <- case_spec(case, regressors)
    fit <- diagnostics_fit(case, spec, series[[case$series]])
    z <- as.numeric(residuals(fit, standardize = TRUE))
    e <- as.numeric(residuals(fit))
    variance <- as.numeric(sigma(fit))^2
    d <- sum(unlist(case$arma))
    f <- case$p + case$q
    lags <- c(1, max(2, 3 * d - 1), max(5, 5 * d - 1))
    squared_lags <- c(1, max(2, 3 * f - 1), max(5, 5 * f - 1))
    sign <- signbias(fit)
    pearson <- gof(fit, c(20, 30, 40, 50))
    list(
        id = case$id,
        params = as.list(in_model_order(coef(fit), fit)),
        ljung_box = lapply(seq_along(lags), function(i) {
            fitdf <- c(0, d, d)[i]
            test_pair(weighted_box(z, lag = lags[i], type = "Ljung-Box", fitdf = fitdf,
                                   sqrd.res = FALSE, weighted = TRUE))
        }),
        ljung_box_squared = lapply(squared_lags, function(lag) {
            test_pair(weighted_box(z, lag = lag, type = "Ljung-Box", fitdf = 0,
                                   sqrd.res = TRUE, weighted = TRUE))
        }),
        arch_lm = lapply(f + c(1, 3, 5), function(lag) {
            test_pair(weighted_lm(e, h.t = variance, lag = lag, fitdf = f))
        }),
        nyblom = nyblom_result(fit),
        sign_bias = lapply(seq_len(4), function(i) unname(as.numeric(sign[i, 1:2]))),
        gof = lapply(seq_len(4), function(i) unname(as.numeric(pearson[i, 2:3])))
    )
}

diagnostics_fit <- function(case, spec, series) {
    if (is_evaluation(spec)) {
        return(ugarchfilter(spec, series))
    }
    fit <- ugarchfit(spec, series, solver = "hybrid", solver.control = solver_control(case),
                     numderiv.control = list(hess.d = hessian_step(case)))
    stopifnot(convergence(fit) == 0)
    fit
}

diagnostics_main <- function() {
    cases <- Filter(function(case) "rugarch" %in% unlist(case$oracles), read_json(cases_path))
    series <- load_series(data_paths)
    regressors <- read.csv(regressors_path, check.names = FALSE)
    results <- lapply(cases, diagnostics_case, series = series, regressors = regressors)
    fixture <- list(oracle = "rugarch", oracle_version = as.character(packageVersion("rugarch")), results = results)
    write_json(fixture, diagnostics_path, auto_unbox = TRUE, digits = I(17), pretty = TRUE, na = "null")
}

diagnostics_main()
