// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/streaming_profiler/host_sync.hpp"

#include <algorithm>
#include <chrono>
#include <ctime>
#include <limits>
#include <thread>
#include <vector>
#if defined(__x86_64__)
#include <x86intrin.h>
#endif

#include <numa.h>
#include <tt_stl/assert.hpp>
#include <umd/device/cluster.hpp>
#include <umd/device/io_window/io_window.hpp>
#include <umd/device/types/io_window_config.hpp>
#include <umd/device/types/core_coordinates.hpp>

#include "impl/streaming_profiler/clock_map.hpp"
#include "impl/streaming_profiler/service.hpp"
#include "impl/streaming_profiler/sync_engine.hpp"
#include "llrt/tt_cluster.hpp"

namespace tt::tt_metal::streaming_profiler {

namespace {

// The PCIe tile's SII register block in its own address space, and the count-from-reset timer inside it, per tables 4
// and 12 of the Blackhole PCIE_SS spec.
constexpr uint64_t kSiiBase = 0xFFFFFFFFF0000000ull;
constexpr uint32_t kCfrLo = 0xA8, kCfrHi = 0xAC;
constexpr uint32_t kBurstReads = 1000;
constexpr size_t kWindowBursts = 10;
constexpr size_t kMinLineBursts = 3;
constexpr int kSteadyBrackets = 16;

int64_t clock_ns(clockid_t id) {
    timespec ts{};
    clock_gettime(id, &ts);
    return static_cast<int64_t>(ts.tv_sec) * 1'000'000'000 + ts.tv_nsec;
}

int64_t tsc_now() noexcept {
#if defined(__x86_64__)
    _mm_lfence();
    const int64_t t = static_cast<int64_t>(__rdtsc());
    _mm_lfence();
    return t;
#else
    return clock_ns(CLOCK_MONOTONIC_RAW);
#endif
}

}  // namespace

double tsc_ticks_per_ns() {
    static const double rate = [] {
        const int64_t t0 = tsc_now(), r0 = clock_ns(CLOCK_MONOTONIC_RAW);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        const int64_t r1 = clock_ns(CLOCK_MONOTONIC_RAW), t1 = tsc_now();
        return static_cast<double>(t1 - t0) / static_cast<double>(r1 - r0);
    }();
    return rate;
}

int64_t steady_mono_ns(int64_t tsc) {
    const std::optional<int64_t> ns = service().steady().ns(tsc);
    TT_FATAL(ns, "streaming profiler: no steady_clock pair yet, before any capture");
    return *ns;
}

HostSync::HostSync(tt::Cluster& cluster, uint32_t chip_id, SteadyClock& steady) :
    numa_node_(static_cast<int>(cluster.get_numa_node_for_device(chip_id))), steady_(steady) {
    const auto pcie =
        cluster.get_driver()->get_soc_descriptor(chip_id).get_cores(CoreType::PCIE, CoordSystem::TRANSLATED);
    TT_FATAL(!pcie.empty(), "streaming profiler: host sync chip {} has no PCIe tile in its descriptor", chip_id);
    // Write-combined like other UMD windows. Reads are uncached either way, and tsc_now()'s fences order them.
    window_ = cluster.get_driver()->create_io_window(
        chip_id,
        pcie.front(),
        kSiiBase,
        tt::umd::HostIoWindowConfig{.mapping = tt::umd::HostMemoryCaching::WC, .size = kCfrHi + sizeof(uint32_t)});
    // Reading LO latches HI, and this is the only reader, so the pair is consistent.
    const uint32_t lo = window_->read32(kCfrLo);
    const uint32_t hi = window_->read32(kCfrHi);
    bases_ = {.root_refclk = static_cast<int64_t>((static_cast<uint64_t>(hi) << 32) | lo), .tsc = tsc_now()};
    // The first call sleeps 100 ms, so take it here rather than on the sync thread.
    static_cast<void>(tsc_ticks_per_ns());
}

HostSync::~HostSync() = default;

void HostSync::take(ClockMap& map) {
    struct Read {
        int64_t mid, rtt;
        uint64_t refclk;
    };
    std::vector<Read> reads;
    reads.reserve(kBurstReads);
    // LO wraps every 86 s and bursts can be further apart, so HI is re-read each burst.
    uint32_t hi = 0, lo_last = 0;
    for (uint32_t i = 0; i < kBurstReads; i++) {
        const int64_t t0 = tsc_now();
        const uint32_t lo = window_->read32(kCfrLo);
        const int64_t t1 = tsc_now();
        if (i == 0) {
            hi = window_->read32(kCfrHi);
        } else if (lo < lo_last) {
            hi++;
        }
        lo_last = lo;
        reads.push_back(
            Read{.mid = t0 + (t1 - t0) / 2, .rtt = t1 - t0, .refclk = (static_cast<uint64_t>(hi) << 32) | lo});
    }
    std::vector<int64_t> rtts(reads.size());
    std::ranges::transform(reads, rtts.begin(), &Read::rtt);
    std::ranges::nth_element(rtts, rtts.begin() + rtts.size() / 2);
    const int64_t cut = rtts[rtts.size() / 2];
    const Read& ref = reads.front();
    double st = 0.0, sr = 0.0;
    uint32_t n = 0;
    for (const Read& r : reads) {
        if (r.rtt <= cut) {
            st += static_cast<double>(r.mid - ref.mid);
            sr += static_cast<double>(r.refclk - ref.refclk);
            n++;
        }
    }
    points_.push_back(BurstPoint{
        .tsc = static_cast<double>(ref.mid - bases_.tsc) + st / n,
        .refclk = static_cast<double>(static_cast<int64_t>(ref.refclk) - bases_.root_refclk) + sr / n});
    if (points_.size() > kWindowBursts) {
        points_.pop_front();
    }
    bursts_++;
    if (points_.size() >= kMinLineBursts) {
        const LineFit f = fit_line(points_, &BurstPoint::refclk, &BurstPoint::tsc);
        const double a = f.y_mean - f.slope * f.x_mean;
        const double at = points_.back().refclk;
        map.append_host(HostNode{.at = at, .value = a + f.slope * at, .tangent = f.slope});
    }
    // CLOCK_MONOTONIC is slewed but never stepped, so the steady series is just these pairs, joined by straight lines.
    int64_t best_gap = std::numeric_limits<int64_t>::max(), best_tsc = 0, best_mono = 0;
    for (int i = 0; i < kSteadyBrackets; i++) {
        const int64_t t0 = tsc_now();
        const int64_t m = clock_ns(CLOCK_MONOTONIC);
        const int64_t t1 = tsc_now();
        if (t1 - t0 < best_gap) {
            best_gap = t1 - t0;
            best_tsc = t0 + (t1 - t0) / 2;
            best_mono = m;
        }
    }
    steady_.append(best_tsc, best_mono);
}

// At one burst per 100 ms, the refclk's ~0.05 ppm/s drift against the TSC keeps the line within tens of ns.
void HostSync::burst_if_due(ClockMap& map) {
    const auto now = std::chrono::steady_clock::now();
    if (bursts_ == 0) {
        first_due_ = now;
    } else if (now < first_due_ + std::chrono::milliseconds(100) * bursts_) {
        return;
    }
    // Reads from the other CPU socket take ~90 ns longer on one leg, which shifts a burst's midpoint by tens of ns.
    if (numa_node_ >= 0 && numa_available() != -1) {
        numa_run_on_node(numa_node_);
    }
    take(map);
}

// The closing burst brackets the capture's last records instead of leaving them on an extrapolated line.
void HostSync::finish(ClockMap& map) {
    do {
        take(map);
    } while (points_.size() < kMinLineBursts);
}

}  // namespace tt::tt_metal::streaming_profiler
