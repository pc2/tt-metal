// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <vector>

namespace tt::tt_metal::streaming_profiler {

template <typename Key>
struct ClockNode {
    Key at{};
    double value = 0.0;
    double tangent = 0.0;
};
using SyncNode = ClockNode<int64_t>;
using HostNode = ClockNode<double>;

struct ClockBases {
    int64_t root_refclk = 0, tsc = 0;
};

template <typename Key>
struct SeriesCursor {
    Key a = std::numeric_limits<Key>::max();
    Key b = std::numeric_limits<Key>::lowest();
    Key origin{};
    double value = 0.0;
    double slope = 0.0;
    bool holds(Key t) const noexcept { return t >= a && t <= b; }
    double at(Key t) const noexcept { return value + slope * static_cast<double>(t - origin); }
};

// Each chip's series maps its wall ticks onto the root chip's refclk, and the host series maps root refclk ticks onto
// the host TSC. One thread writes, and reads never lock or block it (BroadcastRing::read_at).
class ClockMap {
public:
    // At 24 B a node, 1.5 GB at most per chip. A chip takes a node at least every millisecond (kPointTicks in
    // kernels/eth_clock_model.cpp), so a series holds about 18.6 hours of a steady clock.
    static constexpr uint32_t kSeriesNodes = 1u << 26;

    // Placement state for one ClockMap, owned and used by a single thread.
    class Reader {
    public:
        Reader() = default;

    private:
        friend class ClockMap;
        explicit Reader(size_t devices) : chips_(devices) {}
        std::vector<SeriesCursor<int64_t>> chips_;
        SeriesCursor<double> host_;
        int64_t tsc_base_ = 0;
    };

    ClockMap(size_t devices, uint32_t series_nodes, ClockBases bases);
    ~ClockMap();
    ClockMap(const ClockMap&) = delete;
    ClockMap& operator=(const ClockMap&) = delete;
    Reader reader() const;

    // A node at or before its series' last node is dropped.
    void append(uint32_t dev, SyncNode node);
    void append_host(HostNode node);
    // Ends the chip's series, which releases every record held for it.
    void finish(uint32_t dev);
    int64_t root_base() const noexcept;
    int64_t tsc_base() const noexcept;

    std::optional<double> root_offset(Reader& r, uint32_t dev, int64_t wall, double frac) const noexcept;
    int64_t place_host(Reader& r, uint32_t dev, int64_t wall) const {
        if (const std::optional<int64_t> tsc = on_cursors(r.chips_[dev], r.host_, r.tsc_base_, wall)) {
            return *tsc;
        }
        return place_slow(r, dev, wall);
    }
    std::optional<double> host_tsc_offset(Reader& r, double root) const noexcept;
    bool has_host_nodes() const noexcept;
    // The wall tick below which the chip's records have final host times.
    int64_t cover_ticks(uint32_t dev) const noexcept;

    // One instruction, where std::llround is a call.
    static int64_t round_nearest(double x) noexcept { return static_cast<int64_t>(std::nearbyint(x)); }

private:
    struct Impl;
    static std::optional<int64_t> on_cursors(
        const SeriesCursor<int64_t>& cc, const SeriesCursor<double>& hc, int64_t tsc_base, int64_t wall) {
        if (cc.holds(wall)) {
            const double root = cc.at(wall);
            if (hc.holds(root)) {
                return tsc_base + round_nearest(hc.at(root));
            }
        }
        return std::nullopt;
    }
    int64_t place_slow(Reader& r, uint32_t dev, int64_t wall) const;
    std::unique_ptr<Impl> impl_;
};

// Every capture's host sync appends its pairs from the Service's one sync thread. Read lock-free from any thread.
class SteadyClock {
public:
    SteadyClock();
    ~SteadyClock();
    SteadyClock(const SteadyClock&) = delete;
    SteadyClock& operator=(const SteadyClock&) = delete;
    void append(int64_t tsc, int64_t mono_ns);
    std::optional<int64_t> ns(int64_t tsc) const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace tt::tt_metal::streaming_profiler
