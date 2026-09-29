// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include <tt-metalium/experimental/streaming_profiler.hpp>

namespace tt::tt_metal::streaming_profiler {

inline constexpr size_t kProcessorCount = static_cast<size_t>(experimental::streaming_profiler::Processor::ERISC1) + 1;
inline constexpr std::array<const char*, kProcessorCount> kProcessorNames = {
    "BRISC", "NCRISC", "TRISC_0", "TRISC_1", "TRISC_2", "ERISC_0", "ERISC_1"};

// Immutable once the receiver starts.
struct CaptureContext {
    struct Device {
        std::vector<experimental::streaming_profiler::Core> lanes;
        std::vector<uint32_t> core_xy;  // core index -> packed NoC (y << 16) | x
        uint32_t chip_id = 0;
        // Per core: the tracker's wall tick minus that core's. Every tile keeps its own wall clock on the one AICLK,
        // so each is one integer for the capture.
        std::vector<int64_t> tile_offset;
        int64_t ruler_offset = 0;
    };
    std::vector<Device> devices;
    struct Link {
        uint32_t dev_a = 0, dev_b = 0;
        uint32_t core_a = 0, core_b = 0;
        CoreCoord eth_a, eth_b;
    };
    std::vector<Link> links;
    bool sync_check = false;
};

}  // namespace tt::tt_metal::streaming_profiler
