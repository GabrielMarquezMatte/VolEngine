#ifndef VOLENGINE_CPP_INPUT_HPP
#define VOLENGINE_CPP_INPUT_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <utility>
#include <vector>
#include <volengine/error.hpp>

namespace ve
{

// The enumerators are the lower-case suffixes of the C macros; VE_DIST_STD is Dist::student.
enum class Model : std::int32_t
{
    sgarch = VE_MODEL_SGARCH,
    egarch = VE_MODEL_EGARCH,
    gjrgarch = VE_MODEL_GJRGARCH,
    aparch = VE_MODEL_APARCH,
    igarch = VE_MODEL_IGARCH,
    csgarch = VE_MODEL_CSGARCH,
    fgarch_garch = VE_MODEL_FGARCH_GARCH,
    fgarch_tgarch = VE_MODEL_FGARCH_TGARCH,
    fgarch_avgarch = VE_MODEL_FGARCH_AVGARCH,
    fgarch_ngarch = VE_MODEL_FGARCH_NGARCH,
    fgarch_nagarch = VE_MODEL_FGARCH_NAGARCH,
    fgarch_aparch = VE_MODEL_FGARCH_APARCH,
    fgarch_gjrgarch = VE_MODEL_FGARCH_GJRGARCH,
    fgarch_allgarch = VE_MODEL_FGARCH_ALLGARCH,
    ewma = VE_MODEL_EWMA,
};

enum class Dist : std::int32_t
{
    norm = VE_DIST_NORM,
    student = VE_DIST_STD,
    ged = VE_DIST_GED,
    sstd = VE_DIST_SSTD,
    sged = VE_DIST_SGED,
    nig = VE_DIST_NIG,
    ghyp = VE_DIST_GHYP,
    jsu = VE_DIST_JSU,
};

enum class VarInit : std::int32_t
{
    sample = VE_VARINIT_SAMPLE,
    backcast = VE_VARINIT_BACKCAST,
};

using NamedValues = std::vector<std::pair<std::string, double>>;

namespace detail
{

using SpecHandle = Owned<ve_spec, ve_spec_free>;
using OptionsHandle = Owned<ve_fit_options, ve_fit_options_free>;

// Sets each named value on a spec of either kind (`set` is ve_spec_set_fixed, ve_dcc_spec_set_start, ...).
template <class Spec>
Result<void> apply_named(Spec* spec, const NamedValues& values, ve_status (*set)(Spec*, const char*, double))
{
    for (const auto& [name, value] : values)
    {
        if (auto applied = check(set(spec, name.c_str(), value)); !applied)
        {
            return applied;
        }
    }
    return {};
}

} // namespace detail

// What to fit: the defaults are those of the Python package (sGARCH(1,1) with a mean and normal innovations).
// `fixed` and `start` give parameters by the names of Fit::param_names.
struct Spec
{
    Model model = Model::sgarch;
    int p = 1;
    int q = 1;
    int arma_ar = 0;
    int arma_ma = 0;
    Dist dist = Dist::norm;
    bool include_mean = true;
    int mxreg = 0;
    int vxreg = 0;
    int archm = 0;
    bool arfima = false;
    bool variance_targeting = false;
    NamedValues fixed{};
    NamedValues start{};

    // The reason in the error is that of ve_spec_validate.
    Result<void> validate() const
    {
        auto handle = make_handle();
        if (!handle)
        {
            return detail::failure(handle);
        }
        return detail::check(ve_spec_validate(handle->get()));
    }

    // A new native spec. The setters check their own ranges, so this fails for p or q outside 1 .. VE_MAX_ORDER.
    Result<detail::SpecHandle> make_handle() const
    {
        ve_spec* raw = nullptr;
        if (auto created = detail::check(ve_spec_create(static_cast<std::int32_t>(model), &raw)); !created)
        {
            return detail::failure(created);
        }
        detail::SpecHandle handle{raw};
        auto applied = apply_structure(raw)
                           .and_then([&] { return apply_extensions(raw); })
                           .and_then([&] { return detail::apply_named(raw, fixed, ve_spec_set_fixed); })
                           .and_then([&] { return detail::apply_named(raw, start, ve_spec_set_start); });
        if (!applied)
        {
            return detail::failure(applied);
        }
        return handle;
    }

private:
    Result<void> apply_structure(ve_spec* spec) const
    {
        return detail::check(ve_spec_set_order(spec, p, q))
            .and_then([&] { return detail::check(ve_spec_set_arma(spec, arma_ar, arma_ma)); })
            .and_then([&] { return detail::check(ve_spec_set_dist(spec, static_cast<std::int32_t>(dist))); })
            .and_then([&] { return detail::check(ve_spec_set_include_mean(spec, include_mean ? 1 : 0)); });
    }

    Result<void> apply_extensions(ve_spec* spec) const
    {
        return detail::check(ve_spec_set_mxreg(spec, mxreg))
            .and_then([&] { return detail::check(ve_spec_set_vxreg(spec, vxreg)); })
            .and_then([&] { return detail::check(ve_spec_set_archm(spec, archm)); })
            .and_then([&] { return detail::check(ve_spec_set_arfima(spec, arfima ? 1 : 0)); })
            .and_then(
                [&] { return detail::check(ve_spec_set_variance_targeting(spec, variance_targeting ? 1 : 0)); });
    }
};

struct FitOptions
{
    int max_iter = 500;
    double tol = 1e-6;
    VarInit var_init = VarInit::sample;
    // Off skips the Hessian, robust standard errors, scores and Nyblom: the accessors that need them then fail.
    bool standard_errors = true;

    Result<detail::OptionsHandle> make_handle() const
    {
        ve_fit_options* raw = nullptr;
        if (auto created = detail::check(ve_fit_options_create(&raw)); !created)
        {
            return detail::failure(created);
        }
        detail::OptionsHandle handle{raw};
        auto applied = detail::check(ve_fit_options_set_max_iter(raw, max_iter))
                           .and_then([&] { return detail::check(ve_fit_options_set_tolerance(raw, tol)); })
                           .and_then([&]
                               { return detail::check(ve_fit_options_set_var_init(raw, static_cast<std::int32_t>(var_init))); })
                           .and_then([&]
                               { return detail::check(ve_fit_options_set_standard_errors(raw, standard_errors ? 1 : 0)); });
        if (!applied)
        {
            return detail::failure(applied);
        }
        return handle;
    }
};

// A matrix of series: rows are observations, columns are series. It borrows the data, which must outlive every call
// that uses the panel.
class Panel
{
public:
    // Row-major (C order): element (t, j) is data[t * cols + j].
    static Result<Panel> create(std::span<const double> data, std::size_t rows, std::size_t cols)
    {
        return create_strided(data, rows, cols, static_cast<std::ptrdiff_t>(cols), 1);
    }

    // Element (t, j) is data[t * stride_rows + j * stride_cols]. Negative strides are refused: a span cannot say where
    // the elements before its start are.
    static Result<Panel> create_strided(std::span<const double> data, std::size_t rows, std::size_t cols,
        std::ptrdiff_t stride_rows, std::ptrdiff_t stride_cols)
    {
        if (auto extent = check_extent(data.size(), rows, cols, stride_rows, stride_cols); !extent)
        {
            return detail::failure(extent);
        }
        ve_panel* raw = nullptr;
        if (auto created = detail::check(ve_panel_create(data.data(), rows, cols, &raw)); !created)
        {
            return detail::failure(created);
        }
        Handle handle{raw};
        if (auto strided = detail::check(ve_panel_set_strides(raw, stride_rows, stride_cols)); !strided)
        {
            return detail::failure(strided);
        }
        return Panel{std::move(handle), rows, cols};
    }

    // Series j uses the rows starts[j] .. starts[j] + lengths[j] - 1. Both spans need one entry per column.
    Result<void> set_ranges(std::span<const std::size_t> starts, std::span<const std::size_t> lengths)
    {
        if (starts.size() < cols_ || lengths.size() < cols_)
        {
            return std::unexpected(Error{Status::invalid_argument, "ranges need one entry per column"});
        }
        return detail::check(ve_panel_set_ranges(handle_.get(), starts.data(), lengths.data()));
    }

    std::size_t rows() const noexcept
    {
        return rows_;
    }

    std::size_t cols() const noexcept
    {
        return cols_;
    }

    ve_panel* native() const noexcept
    {
        return handle_.get();
    }

private:
    using Handle = detail::Owned<ve_panel, ve_panel_free>;

    Panel(Handle handle, std::size_t rows, std::size_t cols) noexcept
        : handle_(std::move(handle)), rows_(rows), cols_(cols)
    {
    }

    static Result<void> check_extent(std::size_t size, std::size_t rows, std::size_t cols, std::ptrdiff_t stride_rows,
        std::ptrdiff_t stride_cols)
    {
        if (rows == 0 || cols == 0)
        {
            return {};
        }
        if (stride_rows < 0 || stride_cols < 0)
        {
            return std::unexpected(Error{Status::invalid_argument, "negative strides are not supported"});
        }
        // Each term below size keeps the sum from wrapping, so a huge stride cannot pass the check.
        const auto row_span = static_cast<std::size_t>(stride_rows);
        const auto col_span = static_cast<std::size_t>(stride_cols);
        if ((rows > 1 && row_span > (size - 1) / (rows - 1)) || (cols > 1 && col_span > (size - 1) / (cols - 1)) ||
            (rows - 1) * row_span + (cols - 1) * col_span >= size)
        {
            return std::unexpected(Error{Status::invalid_argument, "the data is smaller than the shape of the panel"});
        }
        return {};
    }

    Handle handle_;
    std::size_t rows_;
    std::size_t cols_;
};

namespace detail
{

// The C API takes a null panel for "no regressors".
inline const ve_panel* native_or_null(const Panel* panel) noexcept
{
    if (panel == nullptr)
    {
        return nullptr;
    }
    return panel->native();
}

} // namespace detail

// Specs, one panel and the jobs (which spec on which series) of one ve_fit_batch call. It copies the specs and the
// options; the panel is borrowed.
class Batch
{
public:
    static Result<Batch> create()
    {
        ve_batch* raw = nullptr;
        if (auto created = detail::check(ve_batch_create(&raw)); !created)
        {
            return detail::failure(created);
        }
        return Batch{Handle{raw}};
    }

    Result<void> set_specs(std::span<const Spec> specs)
    {
        std::vector<detail::SpecHandle> handles;
        std::vector<const ve_spec*> pointers;
        for (const Spec& spec : specs)
        {
            auto handle = spec.make_handle();
            if (!handle)
            {
                return detail::failure(handle);
            }
            pointers.push_back(handle->get());
            handles.push_back(std::move(*handle));
        }
        return detail::check(ve_batch_set_specs(handle_.get(), pointers.data(), pointers.size()));
    }

    Result<void> set_panel(const Panel& panel)
    {
        return detail::check(ve_batch_set_panel(handle_.get(), panel.native()));
    }

    // Job i fits spec[i] on series[i].
    Result<void> set_jobs(std::span<const std::size_t> spec, std::span<const std::size_t> series)
    {
        if (spec.size() != series.size())
        {
            return std::unexpected(Error{Status::invalid_argument, "spec and series need one entry per job"});
        }
        return detail::check(ve_batch_set_jobs(handle_.get(), spec.data(), series.data(), spec.size()));
    }

    Result<void> set_options(const FitOptions& options)
    {
        auto handle = options.make_handle();
        if (!handle)
        {
            return detail::failure(handle);
        }
        return detail::check(ve_batch_set_options(handle_.get(), handle->get()));
    }

    // One panel per job, or none (an empty span, or a null entry). The regressors are borrowed like the data.
    Result<void> set_regressors(std::span<const Panel* const> mxreg, std::span<const Panel* const> vxreg)
    {
        if (!mxreg.empty() && !vxreg.empty() && mxreg.size() != vxreg.size())
        {
            return std::unexpected(Error{Status::invalid_argument, "mxreg and vxreg need one entry per job"});
        }
        const auto mean = natives(mxreg);
        const auto variance = natives(vxreg);
        std::size_t jobs = std::max(mxreg.size(), vxreg.size());
        if (jobs == 0)
        {
            // Both empty clears the regressors: null arrays with the job count of the batch.
            const auto count = job_count();
            if (!count)
            {
                return std::unexpected(count.error());
            }
            jobs = *count;
        }
        return detail::check(ve_batch_set_regressors(handle_.get(), data_or_null(mean), data_or_null(variance), jobs));
    }

    Result<std::size_t> job_count() const
    {
        std::size_t count = 0;
        if (auto read = detail::check(ve_batch_job_count(handle_.get(), &count)); !read)
        {
            return detail::failure(read);
        }
        return count;
    }

    const ve_batch* native() const noexcept
    {
        return handle_.get();
    }

private:
    using Handle = detail::Owned<ve_batch, ve_batch_free>;

    explicit Batch(Handle handle) noexcept : handle_(std::move(handle))
    {
    }

    static std::vector<const ve_panel*> natives(std::span<const Panel* const> panels)
    {
        std::vector<const ve_panel*> pointers;
        for (const Panel* panel : panels)
        {
            if (panel == nullptr)
            {
                pointers.push_back(nullptr);
                continue;
            }
            pointers.push_back(panel->native());
        }
        return pointers;
    }

    // The C API takes a null array for "no job has this kind of regressor".
    static const ve_panel* const* data_or_null(const std::vector<const ve_panel*>& pointers) noexcept
    {
        if (pointers.empty())
        {
            return nullptr;
        }
        return pointers.data();
    }

    Handle handle_;
};

} // namespace ve

#endif
