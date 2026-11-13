#pragma once

#include "core/spec.hpp"
#include "dcc/dcc3_panel.hpp"
#include "dcc/future/forecast.hpp"
#include "fit/fit.hpp"
#include "fit/fit_result.hpp"
#include "model/data.hpp"

#include <cmath>
#include <cstddef>
#include <span>
#include <vector>

namespace ve::testing
{

constexpr std::size_t kMixedHorizon = 40;

// A panel of an ARFIMA series, a series with a mean regressor and a csGARCH series on the 1500 rows of dcc3, with
// the regressor of the first kMixedHorizon rows after the sample.
struct MixedPanel
{
    std::vector<FitResult> fits;
    std::vector<double> x;
};

inline const MixedPanel& mixed_panel()
{
    static const MixedPanel panel = []
    {
        MixedPanel out;
        const std::vector<std::vector<double>> columns = read_dcc3();
        out.x.resize(kDcc3Rows + kMixedHorizon);
        for (std::size_t t = 0; t < out.x.size(); ++t)
        {
            out.x[t] = 0.1 * std::sin(0.3 * static_cast<double>(t));
        }
        out.fits.push_back(fit_one(columns[0], Spec{.ar = 1, .ma = 1, .arfima = true}, FitConfig{}));
        model::Data with_mean{columns[1], {std::span<const double>(out.x).first(kDcc3Rows), 1}, {}};
        out.fits.push_back(fit_one(with_mean, Spec{.mxreg = 1}, FitConfig{}));
        out.fits.push_back(fit_one(columns[2], Spec{.model = Model::CsGarch}, FitConfig{}));
        return out;
    }();
    return panel;
}

// The regressors of the first horizon steps after the sample, one entry per series.
inline dcc::future::SeriesRegressors mixed_future(const MixedPanel& panel, std::size_t horizon)
{
    dcc::future::SeriesRegressors future(panel.fits.size());
    future[1].mean = {std::span<const double>(panel.x).subspan(kDcc3Rows, horizon), 1};
    return future;
}

}
