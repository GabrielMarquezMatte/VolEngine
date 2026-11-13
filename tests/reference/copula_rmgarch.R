# Run from the repository root: Rscript tests/reference/copula_rmgarch.R
# Reads data/copula3.csv (written once by copula_probe.R; never regenerated here) and writes
# fixtures/rmgarch_copula.json with rmgarch's Copula-DCC fits of plan 4b.
suppressPackageStartupMessages({
    library(rmgarch)
    library(jsonlite)
})

reference <- file.path("tests", "reference")
data_file <- "copula3.csv"

data <- as.matrix(read.csv(file.path(reference, "data", data_file)))
uspec <- ugarchspec(variance.model = list(model = "sGARCH", garchOrder = c(1, 1)),
                    mean.model = list(armaOrder = c(0, 0)), distribution.model = "std")
mspec <- multispec(replicate(ncol(data), uspec))
mfit <- multifit(mspec, data, solver = "solnp", solver.control = list(tol = 1e-10))
stopifnot(all(vapply(mfit@fit, function(x) x@fit$convergence == 0, TRUE)))

case <- function(id, copula, time_varying, transformation, method = "ML", asymmetric = FALSE, fixed = NULL) {
    list(id = id, copula = copula, method = method, time_varying = time_varying, transformation = transformation,
         asymmetric = asymmetric, fixed = fixed)
}

cases <- list(
    case("static_mvnorm_kendall_param", "mvnorm", FALSE, "parametric", "Kendall"),
    case("static_mvnorm_ml_param", "mvnorm", FALSE, "parametric"),
    case("static_mvnorm_ml_emp", "mvnorm", FALSE, "empirical"),
    case("static_mvt_kendall_param", "mvt", FALSE, "parametric", "Kendall"),
    case("static_mvt_kendall_emp", "mvt", FALSE, "empirical", "Kendall"),
    case("static_mvt_ml_param", "mvt", FALSE, "parametric"),
    case("dcc_mvnorm_param", "mvnorm", TRUE, "parametric"),
    case("dcc_mvnorm_emp", "mvnorm", TRUE, "empirical"),
    case("dcc_mvt_param", "mvt", TRUE, "parametric"),
    case("dcc_mvt_emp", "mvt", TRUE, "empirical"),
    case("adcc_mvnorm_param", "mvnorm", TRUE, "parametric", asymmetric = TRUE),
    case("adcc_mvt_param", "mvt", TRUE, "parametric", asymmetric = TRUE),
    case("dcc_mvt_param_shape8", "mvt", TRUE, "parametric", fixed = list(mshape = 8))
)

fit_case <- function(one) {
    fixed <- if (is.null(one$fixed)) list() else one$fixed
    spec <- cgarchspec(mspec, dccOrder = c(1, 1), asymmetric = one$asymmetric, fixed.pars = fixed,
                       distribution.model = list(copula = one$copula, method = one$method,
                                                 time.varying = one$time_varying,
                                                 transformation = one$transformation))
    fit <- cgarchfit(spec, data, fit = mfit, fit.control = list(eval.se = TRUE))
    stopifnot(fit@mfit$convergence == 0)
    table <- fit@mfit$matcoef
    joint <- grep("^\\[Joint\\]", rownames(table))
    names_joint <- sub("^\\[Joint\\]", "", rownames(table)[joint])
    rt <- fit@mfit$Rt
    indices <- c(1L, nrow(data) %/% 2L, nrow(data))
    at <- function(t) if (is.list(rt)) unname(rt[[t]]) else unname(rt)
    list(id = one$id, copula = one$copula, method = one$method, time_varying = one$time_varying,
         transformation = one$transformation, asymmetric = one$asymmetric, fixed = one$fixed,
         params = as.list(setNames(table[joint, 1], names_joint)),
         se = as.list(setNames(table[joint, 2], names_joint)),
         llh = as.numeric(likelihood(fit)), times = indices, R = lapply(indices, at))
}

results <- lapply(cases, function(one) {
    message(one$id)
    fit_case(one)
})
fixture <- list(oracle = "rmgarch", oracle_version = as.character(packageVersion("rmgarch")),
                rugarch_version = as.character(packageVersion("rugarch")), R_version = R.version.string,
                file = data_file, nobs = nrow(data), columns = colnames(data),
                univariate = lapply(mfit@fit, function(fit) as.list(coef(fit))),
                univariate_llh = sum(sapply(mfit@fit, function(fit) fit@fit$LLH)), results = results)
write_json(fixture, file.path(reference, "fixtures", "rmgarch_copula.json"),
           auto_unbox = TRUE, digits = I(17), pretty = TRUE, na = "null")
