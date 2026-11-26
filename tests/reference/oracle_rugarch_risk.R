# Writes fixtures/rugarch_risk.json: for each standardized density and confidence, the quantile q at
# p = 1 - confidence and the tail mean E[z | z <= q] = integrate(z * ddist(z), -Inf, q) / p. VolEngine's VaR and ES
# of mean + sigma * z are -(mean + sigma * quantile) and -(mean + sigma * tail_mean). q is the root of
# integrate(ddist, -Inf, q) = p started from rugarch's qdist: for ghyp, qdist's own pdist misses p by 1e-7 to 1e-6.
# Run from the repository root: Rscript tests/reference/oracle_rugarch_risk.R

suppressPackageStartupMessages({
    library(rugarch)
    library(jsonlite)
})

# The densities of tests/unit/risk/measures_test.cpp; skew, shape and lambda in rugarch's parametrization, which is
# VolEngine's (NA where the density has none).
densities <- list(
    list(dist = "norm", skew = NA, shape = NA, ghlambda = NA),
    list(dist = "std", skew = NA, shape = 8, ghlambda = NA),
    list(dist = "std", skew = NA, shape = 4.5, ghlambda = NA),
    list(dist = "ged", skew = NA, shape = 0.6, ghlambda = NA),
    list(dist = "ged", skew = NA, shape = 1.4, ghlambda = NA),
    list(dist = "sstd", skew = 1.4, shape = 8, ghlambda = NA),
    list(dist = "sged", skew = 0.7, shape = 1.5, ghlambda = NA),
    list(dist = "nig", skew = -0.4, shape = 1.5, ghlambda = NA),
    list(dist = "ghyp", skew = 0.3, shape = 2, ghlambda = -1.5),
    list(dist = "ghyp", skew = 0.3, shape = 2, ghlambda = 2),
    list(dist = "jsu", skew = 0.8, shape = 1.5, ghlambda = NA)
)
confidences <- c(0.95, 0.99, 0.999)

# rugarch's defaults stand in for the parameters a density does not read.
or_default <- function(value, default) if (is.na(value)) default else value

case_of <- function(density, confidence) {
    p <- 1 - confidence
    skew <- or_default(density$skew, 1)
    shape <- or_default(density$shape, 5)
    lambda <- or_default(density$ghlambda, -0.5)
    f <- function(z) ddist(density$dist, z, mu = 0, sigma = 1, skew = skew, shape = shape, lambda = lambda)
    start <- qdist(density$dist, p, mu = 0, sigma = 1, skew = skew, shape = shape, lambda = lambda)
    mass <- function(z) integrate(f, -Inf, z, rel.tol = 1e-13, subdivisions = 1000L)$value - p
    q <- uniroot(mass, c(start - 0.01, start + 0.01), tol = 1e-15)$root
    integrand <- function(z) z * f(z)
    tail <- integrate(integrand, -Inf, q, rel.tol = 1e-12, subdivisions = 1000L)$value / p
    list(dist = density$dist, skew = density$skew, shape = density$shape, ghlambda = density$ghlambda,
         confidence = confidence, quantile = q, tail_mean = tail)
}

results <- list()
for (density in densities) {
    for (confidence in confidences) {
        results[[length(results) + 1]] <- case_of(density, confidence)
    }
}
fixture <- list(rugarch_version = as.character(packageVersion("rugarch")), r_version = R.version.string,
                results = results)
path <- file.path("tests", "reference", "fixtures", "rugarch_risk.json")
writeLines(toJSON(fixture, auto_unbox = TRUE, digits = NA, na = "null", pretty = FALSE), path)
cat("wrote", length(results), "cases to", path, "\n")
