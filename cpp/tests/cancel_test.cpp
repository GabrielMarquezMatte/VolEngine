#include "fixture_data.hpp"

#include <bit>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <stop_token>
#include <thread>
#include <vector>
#include <volengine/volengine.hpp>

namespace
{

using namespace std::chrono_literals;

// A grid many times the work that fits in kSignalDelay, so a stop at kSignalDelay lands inside the call.
constexpr std::size_t kLongGrid = 200;
constexpr auto kSignalDelay = 50ms;

const ve::NamedValues kFixedParams{{"mu", 0.01}, {"omega", 0.05}, {"alpha1", 0.1}, {"beta1", 0.85}};

const std::vector<double>& series()
{
    static const std::vector<double> column = ve::testing::read_csv_columns("sgarch11_norm.csv")[0];
    return column;
}

template <class T>
void expect_cancelled(const ve::Result<T>& result)
{
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().status, ve::Status::cancelled);
}

std::stop_token stopped()
{
    std::stop_source source;
    source.request_stop();
    return source.get_token();
}

std::vector<std::uint64_t> bits(const std::vector<double>& values)
{
    std::vector<std::uint64_t> out;
    for (double value : values)
    {
        out.push_back(std::bit_cast<std::uint64_t>(value));
    }
    return out;
}

TEST(Cancel, StoppedTokenGivesCancelled)
{
    const auto context = ve::Context::create(1).value();
    const auto fits = ve::testing::dcc3_univariate_fits(context);
    const auto dcc = context.fit_dcc(ve::DccSpec{}, fits, ve::DccOptions{.standard_errors = false}).value();
    const ve::SimOptions options{.horizon = 5, .paths = 3, .seed = 1};
    const std::vector<ve::DccSpec> grid(2);

    expect_cancelled(context.fit(ve::Spec{}, series(), {}, stopped()));
    expect_cancelled(context.fit_dcc(ve::DccSpec{}, fits, {}, stopped()));
    expect_cancelled(context.fit_dcc(grid, fits, {}, stopped()));
    expect_cancelled(context.standard_errors(dcc, fits, stopped()));
    expect_cancelled(context.simulate(fits[0], options, stopped()));
    expect_cancelled(context.simulate(ve::Spec{.fixed = kFixedParams}, options, stopped()));
    expect_cancelled(context.simulate(dcc, options, nullptr, stopped()));
}

TEST(Cancel, DefaultTokenUnchanged)
{
    const auto context = ve::Context::create(1).value();
    const auto plain = context.fit(ve::Spec{}, series()).value();
    const auto with_token = context.fit(ve::Spec{}, series(), {}, std::stop_token{}).value();
    std::stop_source never;
    const auto with_source = context.fit(ve::Spec{}, series(), {}, never.get_token()).value();
    EXPECT_EQ(bits(with_token.estimates().value()), bits(plain.estimates().value()));
    EXPECT_EQ(bits(with_source.estimates().value()), bits(plain.estimates().value()));
}

TEST(Cancel, StopFromAnotherThread)
{
    const auto context = ve::Context::create(2).value();
    const std::vector<ve::Spec> grid(kLongGrid);
    const auto panel = ve::Panel::create(series(), series().size(), 1).value();
    std::stop_source source;
    std::jthread signal(
        [&source]
        {
            std::this_thread::sleep_for(kSignalDelay);
            source.request_stop();
        });
    const auto fits = context.fit(grid, panel, {}, source.get_token());
    expect_cancelled(fits);
}

} // namespace
