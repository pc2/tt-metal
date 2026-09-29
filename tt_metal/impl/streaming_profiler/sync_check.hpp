// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "impl/streaming_profiler/sync_engine.hpp"

namespace tt::tt_metal::streaming_profiler {

// Measures the synced timeline against an independent reference: the ruler's refclk readings on each chip, placed
// through the clock map, against the held-out link rounds solved onto the root. Logs the chip-to-chip error and each
// chip's AICLK over the capture at finish().
class SyncCheck {
public:
    struct Reading {
        double r;
        int64_t w8;
        float weight = 1;
    };
    struct Round {
        size_t li;
        RoundPoint p;
    };
    struct Batch {
        explicit Batch(size_t devices = 0) : readings(devices), offsets(devices), aiclk(devices) {}
        std::vector<Round> rounds;
        std::vector<std::vector<Reading>> readings;
        std::vector<std::optional<double>> offsets;
        // Each chip's AICLK in GHz at root refclk ticks, one entry per instant on a line.
        std::vector<std::vector<std::pair<double, double>>> aiclk;
    };

    using PlotFn = std::function<void(const std::string&, std::span<const std::pair<double, double>>)>;

    SyncCheck(const CaptureContext& ctx, const ClockMap& map);
    ~SyncCheck();
    SyncCheck(const SyncCheck&) = delete;
    SyncCheck& operator=(const SyncCheck&) = delete;

    // Hands the batch to the check's thread and leaves it empty.
    void submit(Batch& b);
    // Stops the check's thread and logs the report.
    void finish();
    // The worst error per millisecond, pooled and per chip, as Tracy plot series.
    void plots(const PlotFn& plot) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace tt::tt_metal::streaming_profiler
