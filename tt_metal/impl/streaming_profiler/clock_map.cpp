// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/streaming_profiler/clock_map.hpp"

#include <atomic>
#include <cmath>
#include <deque>
#include <limits>
#include <string>
#include <utility>

#include <tt-logger/tt-logger.hpp>
#include <tt_stl/assert.hpp>

#include "impl/streaming_profiler/host_sync.hpp"
#include "tt_metal/common/broadcast_ring.hpp"

namespace tt::tt_metal::streaming_profiler {

namespace {

template <typename Key>
struct Log {
    using Node = ClockNode<Key>;
    Log(uint32_t series_nodes, std::string name) : nodes(series_nodes), name(std::move(name)) {}
    BroadcastRing<Node, 4096> nodes;
    std::string name;
    Node last{.at = std::numeric_limits<Key>::lowest()};  // the writer's own copy
    alignas(64) std::atomic<Key> cover{std::numeric_limits<Key>::lowest()};
    bool full_warned = false;

    void append(const Node& node) {
        if (node.at <= last.at) {
            return;
        }
        if (!full_warned && nodes.size() == nodes.capacity()) {
            full_warned = true;
            log_warning(
                tt::LogMetal,
                "[streaming profiler] clock sync: {}'s series is full at {} points, so records older than its oldest "
                "point are placed less accurately",
                name,
                nodes.capacity());
        }
        last = node;
        nodes.writer().publish(last);
        extend(node.at);
    }
    void extend(Key c) {
        if (c > cover.load(std::memory_order_relaxed)) {
            cover.store(c, std::memory_order_release);
        }
    }
};

// A read that loses to the writer restarts from the new oldest node. The search ends with the last node at or before t
// in `below` and the next one in `above`.
template <typename Key>
std::optional<double> refill(const Log<Key>& log, SeriesCursor<Key>& c, Key t) noexcept {
    using Node = typename Log<Key>::Node;
    while (true) {
        const Key cover = log.cover.load(std::memory_order_acquire);
        const uint64_t f = log.nodes.oldest();
        const uint64_t n = log.nodes.published();
        if (n == f) {
            c = SeriesCursor<Key>{};
            return std::nullopt;
        }
        uint64_t lo = f, hi = n;
        bool overwritten = false;
        Node below{}, above{};
        const auto at_or_before = [&](uint64_t i) {
            Node x{};
            overwritten = overwritten || !log.nodes.read_at(i, x);
            (x.at <= t ? below : above) = x;
            return x.at <= t;
        };
        while (lo < hi && !overwritten) {
            const uint64_t mid = lo + (hi - lo) / 2;
            if (at_or_before(mid)) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        if (overwritten) {
            continue;
        }
        const Node& a = lo == f ? above : below;
        const Node& b = above;
        if (lo == f) {
            c = SeriesCursor<Key>{
                .a = std::numeric_limits<Key>::lowest(),
                .b = a.at,
                .origin = a.at,
                .value = a.value,
                .slope = a.tangent};
        } else if (lo < n) {
            c = SeriesCursor<Key>{
                .a = a.at,
                .b = b.at,
                .origin = a.at,
                .value = a.value,
                .slope = (b.value - a.value) / static_cast<double>(b.at - a.at)};
        } else if (t <= cover) {
            c = SeriesCursor<Key>{.a = a.at, .b = cover, .origin = a.at, .value = a.value, .slope = a.tangent};
        } else {
            // Only the sync engine's own reads of a still-growing series land past the cover.
            c = SeriesCursor<Key>{.origin = a.at, .value = a.value, .slope = a.tangent};
        }
        return c.at(t);
    }
}

template <typename Key>
std::optional<double> place(const Log<Key>& log, SeriesCursor<Key>& c, Key t) noexcept {
    return c.holds(t) ? c.at(t) : refill(log, c, t);
}

}  // namespace

struct ClockMap::Impl {
    Impl(size_t devices, uint32_t series_nodes, ClockBases bases) :
        host(series_nodes, "the host"), root_base(bases.root_refclk), tsc_base(bases.tsc) {
        for (size_t d = 0; d < devices; d++) {
            chips.emplace_back(series_nodes, "device " + std::to_string(d));
        }
    }
    std::deque<Log<int64_t>> chips;
    Log<double> host;
    const int64_t root_base, tsc_base;
};

ClockMap::ClockMap(size_t devices, uint32_t series_nodes, ClockBases bases) :
    impl_(std::make_unique<Impl>(devices, series_nodes, bases)) {}

ClockMap::Reader ClockMap::reader() const { return Reader(impl_->chips.size()); }

ClockMap::~ClockMap() = default;

void ClockMap::append(uint32_t dev, SyncNode node) {
    TT_FATAL(
        std::isfinite(node.value) && std::isfinite(node.tangent) && node.tangent > 0.0,
        "streaming profiler: placement node for device {} at wall {} is not a rate: root {} tangent {}",
        dev,
        node.at,
        node.value,
        node.tangent);
    impl_->chips[dev].append(node);
}

void ClockMap::finish(uint32_t dev) { impl_->chips[dev].extend(std::numeric_limits<int64_t>::max()); }

int64_t ClockMap::root_base() const noexcept { return impl_->root_base; }
int64_t ClockMap::tsc_base() const noexcept { return impl_->tsc_base; }

void ClockMap::append_host(HostNode node) {
    TT_FATAL(
        std::isfinite(node.at) && std::isfinite(node.value) && std::isfinite(node.tangent) && node.tangent > 0.0,
        "streaming profiler: host placement node at refclk {} is not a rate: tsc {} tangent {}",
        node.at,
        node.value,
        node.tangent);
    impl_->host.append(node);
}

// The cover stops at the host series' last node. Past it, two records at the same tick could take the tangent and the
// chord, and one chip's lanes would drift apart by the line's move between bursts.
int64_t ClockMap::cover_ticks(uint32_t dev) const noexcept {
    const Log<int64_t>& cl = impl_->chips[dev];
    const int64_t cover = cl.cover.load(std::memory_order_acquire);
    if (cover == std::numeric_limits<int64_t>::max()) {
        return cover;
    }
    const double host_cover = impl_->host.cover.load(std::memory_order_acquire);
    SyncNode last{};
    const uint64_t n = cl.nodes.published();
    if (n == cl.nodes.oldest() || !cl.nodes.read_at(n - 1, last)) {
        return cover;
    }
    const double wall = static_cast<double>(last.at) + (host_cover - last.value) / last.tangent;
    if (wall >= static_cast<double>(cover)) {
        return cover;
    }
    if (wall <= static_cast<double>(std::numeric_limits<int64_t>::min())) {
        return std::numeric_limits<int64_t>::min();
    }
    return static_cast<int64_t>(std::floor(wall));
}

bool ClockMap::has_host_nodes() const noexcept { return impl_->host.nodes.size() != 0; }

std::optional<double> ClockMap::root_offset(Reader& r, uint32_t dev, int64_t wall, double frac) const noexcept {
    SeriesCursor<int64_t>& c = r.chips_[dev];
    const std::optional<double> root = place(impl_->chips[dev], c, wall);
    return root ? std::optional(*root + c.slope * frac) : std::nullopt;
}

std::optional<double> ClockMap::host_tsc_offset(Reader& r, double root) const noexcept {
    return place(impl_->host, r.host_, root);
}

int64_t ClockMap::place_slow(Reader& r, uint32_t dev, int64_t wall) const {
    const std::optional<double> root = place(impl_->chips[dev], r.chips_[dev], wall);
    TT_FATAL(root, "streaming profiler: device {} has no placement node for its wall tick {}", dev, wall);
    const std::optional<double> tsc = place(impl_->host, r.host_, *root);
    TT_FATAL(tsc, "streaming profiler: device {}'s wall tick {} has no host placement node", dev, wall);
    r.tsc_base_ = impl_->tsc_base;
    return r.tsc_base_ + round_nearest(*tsc);
}

struct SteadyClock::Impl {
    Log<int64_t> steady{ClockMap::kSeriesNodes, "the host steady clock"};
    std::optional<int64_t> base;
};

SteadyClock::SteadyClock() : impl_(std::make_unique<Impl>()) {}
SteadyClock::~SteadyClock() = default;

void SteadyClock::append(int64_t tsc, int64_t mono_ns) {
    Impl& m = *impl_;
    const SyncNode& last = m.steady.last;
    if (tsc <= last.at) {
        return;
    }
    const int64_t base = m.base ? *m.base : *(m.base = mono_ns);
    const double value = static_cast<double>(mono_ns - base);
    const double ns_per_tick = last.at == std::numeric_limits<int64_t>::lowest()
                                   ? 1.0 / tsc_ticks_per_ns()
                                   : (value - last.value) / static_cast<double>(tsc - last.at);
    TT_FATAL(ns_per_tick > 0.0, "streaming profiler: steady clock node at tsc {} is not a rate: {}", tsc, ns_per_tick);
    m.steady.append(SyncNode{.at = tsc, .value = value, .tangent = ns_per_tick});
}

std::optional<int64_t> SteadyClock::ns(int64_t tsc) const noexcept {
    constinit thread_local SeriesCursor<int64_t> t_cursor{};
    const std::optional<double> rel = place(impl_->steady, t_cursor, tsc);
    return rel ? std::optional(*impl_->base + ClockMap::round_nearest(*rel)) : std::nullopt;
}

}  // namespace tt::tt_metal::streaming_profiler
