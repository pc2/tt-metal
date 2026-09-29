// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <ranges>
#include <span>
#include <vector>

#include "hostdev/streaming_profiler_sync.h"
#include "impl/streaming_profiler/clock_map.hpp"
#include "impl/streaming_profiler/capture_context.hpp"

namespace tt::tt_metal::streaming_profiler {

// The x mean accumulates in long double, since x may be a count far from its spread.
struct LineFit {
    double x_mean = 0.0, y_mean = 0.0, slope = 0.0;
};
template <
    std::ranges::forward_range Points,
    std::invocable<std::ranges::range_reference_t<const Points>> X,
    std::invocable<std::ranges::range_reference_t<const Points>> Y>
LineFit fit_line(const Points& pts, X x, Y y) {
    long double sx = 0;
    double sy = 0.0;
    size_t n = 0;
    for (const auto& p : pts) {
        sx += std::invoke(x, p);
        sy += std::invoke(y, p);
        n++;
    }
    LineFit f{.x_mean = static_cast<double>(sx / static_cast<long double>(n)), .y_mean = sy / static_cast<double>(n)};
    double sxx = 0.0, sxy = 0.0;
    for (const auto& p : pts) {
        const double dx = std::invoke(x, p) - f.x_mean;
        sxx += dx * dx;
        sxy += dx * (std::invoke(y, p) - f.y_mean);
    }
    f.slope = sxy / sxx;
    return f;
}

// Edge i says x[ends[i][0]] - x[ends[i][1]] = value[i], an empty end being the ground, which is fixed at 0.
struct Potential {
    std::vector<double> x;
    std::optional<size_t> unreached;
};
Potential solve_potential(
    std::span<const std::array<std::optional<size_t>, 2>> ends, std::span<const double> value, size_t unknowns);

static_assert(1'000'000'000 % kernel_profiler::kEthRefclkHz == 0);
inline constexpr int64_t kNsPerRefclk = 1'000'000'000 / kernel_profiler::kEthRefclkHz;

// One link round in the sender's refclk ticks: the midpoint of its stamps, and the receiver's offset from it there.
struct RoundPoint {
    double mid = 0.0, off = 0.0;
};

// A chip's refclk onto the root's.
struct RootXf {
    double scale = 1.0, shift = 0.0;
    bool ok = false;
    double operator()(double refclk) const { return scale * refclk + shift; }
};

template <std::predicate<size_t> Usable>
std::vector<bool> reached_from_root(std::span<const CaptureContext::Link> links, size_t devices, Usable usable) {
    std::vector<bool> reached(devices, false);
    reached[0] = true;
    for (bool grew = true; grew;) {
        grew = false;
        for (size_t li = 0; li < links.size(); li++) {
            const CaptureContext::Link& L = links[li];
            if (reached[L.dev_a] != reached[L.dev_b] && usable(li)) {
                reached[L.dev_a] = reached[L.dev_b] = true;
                grew = true;
            }
        }
    }
    return reached;
}

// Chains each link's line (null for a link with none) onto the root chip; a chip no line reaches is not ok.
std::vector<RootXf> compose_on_root(const CaptureContext& ctx, std::span<const LineFit* const> lines);

// LINK samples feed the link solver, refclk against refclk, so DVFS on either wall clock cannot enter the link solve.
class SyncEngine {
public:
    // Starts a capture on `ctx`, whose every device has a path over its links to the root (device index 0), publishing
    // into `map`.
    SyncEngine(const CaptureContext& ctx, ClockMap& map);
    ~SyncEngine();
    SyncEngine(const SyncEngine&) = delete;
    SyncEngine& operator=(const SyncEngine&) = delete;

    void on_record(uint32_t dev, uint32_t core, const kernel_profiler::SyncRecord& rec);
    // Publishes what the batch's records added to the map, and returns whether anything was.
    bool on_batch_end();
    void on_capture_end();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace tt::tt_metal::streaming_profiler
