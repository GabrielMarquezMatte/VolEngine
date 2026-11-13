# Forecasts every rugarch case of cases.json at rugarch's own estimate (fixtures/rugarch.json) and writes the
# forecast fixture. The model is fixed at that point and filtered on the first n = T - 25 observations; the last 25
# rows of the regressors are the external forecasts. Comparing at the oracle's point separates the forecast rules
# from the optimizers. Run from the repository root: Rscript tests/reference/forecast_rugarch.R

source(file.path("tests", "reference", "oracle_rugarch.R"))

horizon <- 25
forecast_fixture_path <- file.path(reference, "fixtures", "rugarch_forecast.json")

# The parameters the point fixes: all of rugarch's estimate but the derived iGARCH beta_q. Under variance targeting
# omega is fixed at the targeted value of the fit and targeting is off (ugarchfilter rejects a fully fixed point with
# targeting on): targeting only chooses omega, and the forecast rules do not see how.
point_parameters <- function(case, params) {
    derived <- character(0)
    if (case$model == "igarch") {
        derived <- paste0("beta", case$q)
    }
    params[!(names(params) %in% derived)]
}

forecast_case <- function(case, result, series, regressors) {
    r <- series[[case$series]]
    n <- length(r) - horizon
    point <- case
    point$fixed <- point_parameters(case, result$params)
    point$variance_targeting <- FALSE
    in_sample <- regressors[seq_len(n), , drop = FALSE]
    ahead <- regressors[(n + 1):length(r), , drop = FALSE]
    spec <- case_spec(point, in_sample)
    forecast <- ugarchforecast(spec, data = r[seq_len(n)], n.ahead = horizon,
                               external.forecasts = list(mregfor = regressor_matrix(case$mxreg, ahead),
                                                         vregfor = regressor_matrix(case$vxreg, ahead)))
    list(
        case = case$id,
        n = n,
        horizon = horizon,
        fixed = point$fixed,
        sigma = as.numeric(sigma(forecast)),
        mean = as.numeric(fitted(forecast))
    )
}

forecast_main <- function() {
    cases <- Filter(function(case) "rugarch" %in% unlist(case$oracles), read_json(cases_path))
    results <- read_json(fixture_path)$results
    by_case <- setNames(results, vapply(results, function(result) result$case, ""))
    series <- load_series(data_paths)
    regressors <- read.csv(regressors_path, check.names = FALSE)
    fixture <- list(
        oracle = "rugarch",
        oracle_version = as.character(packageVersion("rugarch")),
        horizon = horizon,
        results = lapply(cases, function(case) forecast_case(case, by_case[[case$id]], series, regressors))
    )
    write_json(fixture, forecast_fixture_path, auto_unbox = TRUE, digits = I(17), pretty = TRUE, na = "null")
}

forecast_main()
