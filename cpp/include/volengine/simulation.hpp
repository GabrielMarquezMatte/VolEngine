#ifndef VOLENGINE_CPP_SIMULATION_HPP
#define VOLENGINE_CPP_SIMULATION_HPP

#include <cstddef>
#include <cstdint>
#include <span>
#include <volengine/error.hpp>
#include <volengine/fit.hpp>
#include <volengine/input.hpp>

namespace ve
{

// `horizon` steps of `paths` paths. The same seed, source and library version give the same paths on any machine with
// the same ISA and the same SIMD target (chosen at run time) and for any number of
// threads; the first k paths of a simulation are those of any longer one.
struct SimOptions
{
    std::size_t horizon = 1;
    std::size_t paths = 1;
    std::uint64_t seed = 0;
    // Keep R and H per path (DCC simulations only).
    bool correlations = false;
    // The regressors of the horizon, the same for every path: panels of `horizon` rows. Null for none.
    const Panel* mxreg = nullptr;
    const Panel* vxreg = nullptr;

    Result<detail::Owned<ve_sim_options, ve_sim_options_free>> make_handle() const
    {
        ve_sim_options* raw = nullptr;
        if (auto created = detail::check(ve_sim_options_create(horizon, paths, seed, &raw)); !created)
        {
            return detail::failure(created);
        }
        detail::Owned<ve_sim_options, ve_sim_options_free> handle{raw};
        auto applied =
            detail::check(
                ve_sim_options_set_regressors(raw, detail::native_or_null(mxreg), detail::native_or_null(vxreg)))
                .and_then([&] { return detail::check(ve_sim_options_set_correlations(raw, correlations ? 1 : 0)); });
        if (!applied)
        {
            return detail::failure(applied);
        }
        return handle;
    }
};

// Simulated returns and conditional standard deviations, paths x horizon values each, row-major (row p is path p). It
// copies what it needs: it stays valid after the context, fit, spec and options are destroyed.
class Simulation
{
public:
    explicit Simulation(ve_sim* raw) noexcept : handle_(raw)
    {
    }

    Result<std::size_t> paths() const
    {
        return count(ve_sim_paths);
    }

    Result<std::size_t> horizon() const
    {
        return count(ve_sim_horizon);
    }

    Result<std::span<const double>> returns() const
    {
        return detail::view(handle_.get(), ve_sim_returns);
    }

    Result<std::span<const double>> sigma() const
    {
        return detail::view(handle_.get(), ve_sim_sigma);
    }

private:
    Result<std::size_t> count(ve_status (*function)(const ve_sim*, std::size_t*)) const
    {
        std::size_t value = 0;
        if (auto read = detail::check(function(handle_.get(), &value)); !read)
        {
            return detail::failure(read);
        }
        return value;
    }

    detail::Owned<ve_sim, ve_sim_free> handle_;
};

} // namespace ve

#endif
