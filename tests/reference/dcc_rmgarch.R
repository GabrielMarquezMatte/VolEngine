# Run from the repository root: Rscript tests/reference/dcc_rmgarch.R
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
    list(data = data, spec = specs, fit = fit, file = file)
}

case_grid <- function() {
    cases <- list()
    for (model in c("DCC", "aDCC", "DCC21")) {
        for (dist in c("mvnorm", "mvt", "mvlaplace")) {
            cases[[length(cases) + 1]] <- list(model = model, dist = dist)
        }
    }
    c(cases, list(list(model = "FDCC", dist = "mvnorm"),
                  list(model = "DCC", dist = "mvt", fixed = list(mshape = 8))))
}

fit_case <- function(panel, case, panel_id) {
    model <- case$model
    order <- c(1, 1)
    if (model == "DCC21") {
        model <- "DCC"
        order <- c(2, 1)
    }
    groups <- c(rep(1L, ncol(panel$data) - 1), 2L)
    spec <- dccspec(panel$spec, dccOrder = order, model = model, distribution = case$dist,
                    groups = groups, fixed.pars = case$fixed)
    fit <- dccfit(spec, data = panel$data, fit = panel$fit, solver = "solnp",
                   solver.control = list(tol = 1e-10), fit.control = list(eval.se = TRUE))
    stopifnot(fit@mfit$convergence == 0)
    params <- coef(fit, type = "dcc")
    original_names <- paste0("[Joint]", sub("^\\[Joint\\]", "", names(params)))
    names(params) <- sub("^\\[Joint\\]", "", names(params))
    names(params) <- sub("^fdcca\\[g=([0-9]+),P=1\\]$", "fdcca\\1", names(params))
    names(params) <- sub("^fdccb\\[g=([0-9]+),Q=1\\]$", "fdccb\\1", names(params))
    indices <- c(1L, nrow(panel$data) %/% 2L, nrow(panel$data))
    suffix <- ""
    if (!is.null(case$fixed)) {
        suffix <- "_fixedshape"
    }
    table <- fit@mfit$matcoef
    errors <- table[match(original_names, rownames(table)), 2]
    stopifnot(all(is.finite(errors)))
    list(id = paste0(panel_id, "_", case$model, "_", case$dist, suffix), panel = panel_id,
         model = case$model, dist = case$dist, order = order, groups = groups,
         fixed = case$fixed, params = as.list(params), llh = as.numeric(likelihood(fit)),
         criteria = as.list(setNames(as.numeric(infocriteria(fit)), c("aic", "bic", "sic", "hqic"))),
         se = as.list(setNames(errors, names(params))),
         times = indices, R = lapply(indices, function(t) unname(rcor(fit)[, , t])))
}

main <- function() {
    panels <- list(dcc3 = panel_fit("dcc3.csv"), grid5 = panel_fit("grid_2c.csv"))
    results <- list()
    for (id in names(panels)) {
        for (case in case_grid()) {
            message(id, " ", case$model, " ", case$dist)
            results[[length(results) + 1]] <- fit_case(panels[[id]], case, id)
        }
    }
    metadata <- lapply(panels, function(panel) list(file = panel$file, nobs = nrow(panel$data),
                      columns = colnames(panel$data),
                      univariate = lapply(panel$fit@fit, function(fit) as.list(coef(fit)))))
    fixture <- list(oracle = "rmgarch", oracle_version = as.character(packageVersion("rmgarch")),
                    rugarch_version = as.character(packageVersion("rugarch")), R_version = R.version.string,
                    panels = metadata, results = results)
    write_json(fixture, file.path(reference, "fixtures", "rmgarch_dcc.json"),
                auto_unbox = TRUE, digits = I(17), pretty = TRUE, na = "null")
}

main()
