#include "capi/results/regressor_rows.hpp"

#include "capi/error.hpp"
#include "capi/handles.hpp"

namespace ve::capi
{

namespace
{

std::vector<double> copy_rows(const PanelView& view, std::size_t horizon)
{
    std::vector<double> rows(horizon * view.n);
    for (std::size_t t = 0; t < horizon; ++t)
    {
        for (std::size_t k = 0; k < view.n; ++k)
        {
            rows[t * view.n + k] = view.at(t, k);
        }
    }
    return rows;
}

ve_status shape_error(const std::string& name, std::size_t horizon, std::size_t columns)
{
    return set_error(VE_INVALID_ARGUMENT, name + " must be a panel of " + std::to_string(horizon) + " rows x " +
                                              std::to_string(columns) + " columns without ranges");
}

ve_status presence(bool given, int columns, const std::string& name)
{
    if (columns == 0 && given)
    {
        return set_error(VE_INVALID_ARGUMENT, name + " was given but the spec has no " + name);
    }
    if (columns != 0 && !given)
    {
        return set_error(VE_INVALID_ARGUMENT,
                         name + " must not be null: the spec has " + std::to_string(columns) + " " + name);
    }
    return VE_OK;
}

}

ve_status future_rows(const ve_panel* panel, int columns, std::size_t horizon, const std::string& name,
                      std::vector<double>& rows)
{
    ve_status status = presence(panel != nullptr, columns, name);
    if (status != VE_OK || columns == 0)
    {
        return status;
    }
    const PanelView& view = panel->view;
    std::size_t count = static_cast<std::size_t>(columns);
    if (view.has_ranges() || view.t != horizon || view.n != count)
    {
        return shape_error(name, horizon, count);
    }
    rows = copy_rows(view, horizon);
    return VE_OK;
}

ve_status store_rows(const ve_panel* panel, std::size_t horizon, const std::string& name, StoredRows& rows)
{
    if (panel == nullptr)
    {
        rows = StoredRows{};
        return VE_OK;
    }
    const PanelView& view = panel->view;
    if (view.has_ranges() || view.t != horizon)
    {
        return set_error(VE_INVALID_ARGUMENT,
                         name + " must be a panel of " + std::to_string(horizon) + " rows without ranges");
    }
    rows = StoredRows{true, view.n, copy_rows(view, horizon)};
    return VE_OK;
}

ve_status check_stored(const StoredRows& rows, int columns, std::size_t horizon, const std::string& name)
{
    ve_status status = presence(rows.given, columns, name);
    if (status != VE_OK || columns == 0)
    {
        return status;
    }
    if (rows.columns != static_cast<std::size_t>(columns))
    {
        return shape_error(name, horizon, static_cast<std::size_t>(columns));
    }
    return VE_OK;
}

}
