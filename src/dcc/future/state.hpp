#pragma once

#include "core/spec.hpp"
#include "dcc/panel.hpp"
#include "dcc/recursion.hpp"
#include "dcc/spec.hpp"
#include "forecast/state.hpp"

#include <cstddef>
#include <span>
#include <vector>

namespace ve::dcc::future
{

// A series as forecast and simulation read it (plan 4c): its spec, estimate and final state.
struct SeriesEnd
{
    Spec spec;
    std::vector<double> estimate;
    forecast::ForecastState state;
};

// The correlation recursion after observation rows - 1, at Stage::Reported of the spec's convention:
// theta every DCC parameter (natural, layout order, fixed ones included); qbar and nbar packed (nbar aDCC only);
// q the last spec.q matrices Q, packed, newest first; scores the last spec.p rows of the scores, newest first;
// static_r the m x m R of a static copula (q and scores empty). Empty theta and static_r: the fit has no finite
// point (its r is null). An all-fixed spec has an empty table but a full theta.
struct DccEnd
{
    std::vector<double> theta, qbar, nbar, q, scores, static_r;
};

// The end after the first rows observations of the scores, rows <= scores.t and at least the lags of the spec; an
// empty DccEnd when theta is outside the domain of the recursion.
DccEnd dcc_end(const StandardizedPanel& scores, const DccSpec& spec, std::span<const double> theta,
               std::size_t rows);

// Q and R one step after an end, then the next ones as scores arrive: next() gives the R of the coming step and
// push() takes the scores of that step. The sums run in the order of CorrelationRecursion, so the bits of next()
// are those of the fit's r.
class CorrelationStepper
{
public:
    CorrelationStepper(const DccEnd& end, const DccSpec& spec, std::size_t m);

    // R of the next step, m x m row-major.
    std::span<const double> next();
    // The scores of the step next() just gave (m values).
    void push(std::span<const double> scores);

private:
    void add_news(std::size_t lag);
    void correlation();

    std::size_t m_;
    std::size_t count_;
    bool constant_;
    detail::Weights<double> weights_;
    std::vector<double> q_ring_;
    std::vector<double> score_ring_;
    std::vector<double> q_;
    std::vector<double> r_;
    std::vector<double> z_products_;
    std::vector<double> n_products_;
    std::vector<double> n_;
};

}
