#pragma once

#include "core/panel.hpp"

#include <cstddef>
#include <string>
#include <vector>
#include <volengine/volengine.h>

namespace ve::capi
{

// The forecast's rule: the panel matches the `columns` the spec has (NULL when it has none) with `horizon` rows
// and no ranges; copied row-major into rows.
ve_status future_rows(const ve_panel* panel, int columns, std::size_t horizon, const std::string& name,
                      std::vector<double>& rows);

// Regressors a simulation option keeps before it knows the spec.
struct StoredRows
{
    bool given = false;
    std::size_t columns = 0;
    std::vector<double> values{};
};

// Copies a panel of `horizon` rows without ranges; NULL stores none.
ve_status store_rows(const ve_panel* panel, std::size_t horizon, const std::string& name, StoredRows& rows);

// Whether the stored rows match the `columns` the spec has, with the messages of future_rows.
ve_status check_stored(const StoredRows& rows, int columns, std::size_t horizon, const std::string& name);

}
