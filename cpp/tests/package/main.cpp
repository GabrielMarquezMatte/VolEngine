// A downstream project: simulates a GARCH series, fits it back and checks that the fit converged.
#include <cmath>
#include <cstdio>
#include <volengine/volengine.hpp>

int main()
{
    const auto context = ve::Context::create(1);
    if (!context)
    {
        std::fprintf(stderr, "FAIL: %s\n", context.error().message.c_str());
        return 1;
    }
    const ve::Spec truth{.fixed = {{"mu", 0.05}, {"omega", 0.01}, {"alpha1", 0.08}, {"beta1", 0.9}}};
    const auto simulation = context->simulate(truth, ve::SimOptions{.horizon = 2000, .paths = 1, .seed = 42});
    if (!simulation)
    {
        std::fprintf(stderr, "FAIL: %s\n", simulation.error().message.c_str());
        return 1;
    }
    const auto fit = context->fit(ve::Spec{}, simulation->returns().value());
    if (!fit || fit->status() != ve::FitStatus::converged || !std::isfinite(fit->loglik().value()))
    {
        std::fprintf(stderr, "FAIL: the fit did not converge\n");
        return 1;
    }
    std::printf("fitted %zu observations, loglik %.4f\n", fit->nobs().value(), fit->loglik().value());
    return 0;
}
