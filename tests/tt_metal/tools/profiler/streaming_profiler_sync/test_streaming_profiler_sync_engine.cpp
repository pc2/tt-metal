// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Host-only: the sync engine and clock map against a synthetic truth model, with no device. check_chain feeds three
// chips joined in a chain by two links, with chip 0 switching AICLK partway, chip 1 silent for longer than its refclk
// counter's 24-bit wrap, link streams with dropped and late stamps, and every counter a year past power-on. It checks
// every placement on the host timeline to within 0.5 ns and every steady_clock time to within 1 ns. check_retention
// overflows a series' capacity and checks where records before, between and after the kept nodes land.

#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <string_view>

#include <fmt/format.h>

#include "impl/streaming_profiler/service.hpp"
#include "impl/streaming_profiler/sync_engine.hpp"
#include "impl/streaming_profiler/host_sync.hpp"

using namespace tt::tt_metal;
using namespace tt::tt_metal::streaming_profiler;

constexpr uint32_t kLocal = kernel_profiler::kSyncKindLocal, kLink = kernel_profiler::kSyncKindLink;
constexpr uint32_t kT0 = kernel_profiler::kSyncRoleT0, kT1 = kernel_profiler::kSyncRoleT1;
constexpr uint32_t kT1B = kernel_profiler::kSyncRoleT1B, kT2 = kernel_profiler::kSyncRoleT2;

static int g_fail = 0;
static void check_near(const std::string& what, double got, double want, double tol) {
    if (std::fabs(got - want) > tol) {
        std::printf(
            "FAIL %s: got %.3f want %.3f (|err| %.3f > tol %.3f)\n",
            what.c_str(),
            got,
            want,
            std::fabs(got - want),
            tol);
        g_fail++;
    } else {
        std::printf("ok   %s: err %.3f ns (tol %.1f)\n", what.c_str(), got - want, tol);
    }
}
static void check_at(std::string_view label, double tau, double got, double want, double tol) {
    check_near(fmt::format("{} tau={:.3f}", label, tau), got, want, tol);
}
constexpr double kRefHz = kernel_profiler::kEthRefclkHz;
// The test's readings are exact, so a placement is off only by its rounding to a TSC tick (0.33 ns).
constexpr double kTol = 0.5;
constexpr double kF0 = 1.35e9;
constexpr double kSlow = 26.875 / 27.0;  // chip 0's AICLK after its DVFS switch: one 1/8 step of the PLL multiple
constexpr double kTauSwitch = 0.300;
constexpr double kOneWay = 1.0e-6;
constexpr double kTurn = 350e-9;
// Every clock reads as it would a year after power-on, so each count is far past 2^53 of its units.
constexpr int64_t kYear = int64_t{365} * 86400;
constexpr int64_t kRef0[3] = {kYear * 50'000'000, kYear * 50'000'000 + 1'000'000, kYear * 50'000'000 + 3'000'000};
constexpr int64_t kWall0[3] = {
    kYear * 1'350'000'000 + 1'000'000'000,
    kYear * 1'350'000'000 + 7'000'000'000,
    kYear * 1'350'000'000 + 4'000'000'000};
constexpr int64_t kTsc0 = kYear * 3'000'000'000;
constexpr int64_t kHost0 = kYear * 1'000'000'000;
constexpr double kTicksPerNs = 3.0;

double refclk(double tau) { return kRefHz * tau; }
double wall(int chip, double tau) {
    if (chip != 0 || tau <= kTauSwitch) {
        return kF0 * tau;
    }
    return kF0 * kTauSwitch + kSlow * kF0 * (tau - kTauSwitch);
}
double tsc(double tau) { return tau * 1e9 * kTicksPerNs; }
double host_ns(double tau) { return tau * 1e9; }
int64_t wall_tick(int chip, double tau) { return kWall0[chip] + std::llround(wall(chip, tau)); }
uint64_t hw_stamp(int chip, double tau) {
    return static_cast<uint64_t>(kRef0[chip] * kNsPerRefclk + std::llround(refclk(tau) * kNsPerRefclk));
}

void feed_local(SyncEngine& sync, uint32_t dev, uint64_t refclk, uint64_t wall8, uint32_t k8) {
    sync.on_record(
        dev,
        0,
        {.local = {
             .meta = kernel_profiler::word_of(kernel_profiler::SyncLocalMeta{.count = 1, .kind = kLocal}),
             .round = kernel_profiler::word_of(
                 kernel_profiler::SyncLocalRound{.k8 = {static_cast<uint8_t>(k8)}, .slope = static_cast<uint8_t>(k8)}),
             .refclk = refclk,
             .wall8 = wall8}});
}
void feed_link(SyncEngine& sync, uint32_t dev, uint32_t core, uint32_t round, uint32_t role, uint64_t stamp) {
    sync.on_record(
        dev,
        core,
        {.link = {
             .meta = kernel_profiler::word_of(kernel_profiler::SyncMeta{.role = role, .kind = kLink}),
             .round = round,
             .first = stamp,
             .count = 1}});
}

void check_chain() {
    CaptureContext ctx;
    for (uint32_t c = 0; c < 3; c++) {
        ctx.devices.push_back({.chip_id = c});
    }
    ctx.links.push_back(CaptureContext::Link{.dev_a = 0, .dev_b = 1, .core_a = 0, .core_b = 0});
    ctx.links.push_back(CaptureContext::Link{.dev_a = 1, .dev_b = 2, .core_a = 1, .core_b = 0});
    ClockMap map(3, ClockMap::kSeriesNodes, {.root_refclk = kRef0[0], .tsc = kTsc0});
    for (double tau : {0.0, 0.6}) {
        map.append_host(HostNode{.at = refclk(tau), .value = tsc(tau), .tangent = kTicksPerNs * 1e9 / kRefHz});
    }
    for (double tau : {0.0, 1.0}) {
        service().steady().append(kTsc0 + std::llround(tsc(tau)), kHost0 + std::llround(host_ns(tau)));
    }

    SyncEngine sync(ctx, map);
    // Chip 1 goes silent from 0.40 to 0.75 s, longer than the refclk's 24-bit period.
    constexpr uint32_t kK8Fast = 216, kK8Slow = 215;  // 27.0 and 26.875 wall ticks per refclk tick, in eighths
    const auto point = [&](int c, double tau, uint32_t k8) {
        feed_local(
            sync,
            static_cast<uint32_t>(c),
            static_cast<uint64_t>(kRef0[c] + std::llround(refclk(tau))),
            static_cast<uint64_t>(8 * kWall0[c] + std::llround(8.0 * wall(c, tau))),
            k8);
    };
    bool switched = false;
    for (int k = 0; k < 1000; k++) {
        const double tau = k * 1e-3;
        if (!switched && tau > kTauSwitch) {
            point(0, kTauSwitch - 1e-6, kK8Fast);
            point(0, kTauSwitch + 50e-6, kK8Slow);
            point(0, kTauSwitch + 150e-6, kK8Slow);
            switched = true;
        }
        for (int c = 0; c < 3; c++) {
            if (c == 1 && tau > 0.40 && tau < 0.75) {
                continue;
            }
            point(c, tau, c == 0 && tau > kTauSwitch ? kK8Slow : kK8Fast);
        }
    }
    // Two 300-round link sessions 1 ms apart, each spanning the solve's 250 ms window, with their streams damaged the
    // way a lapped consumer or a full ring damages them.
    const auto burst = [&](uint32_t snd_dev, uint32_t snd_core, uint32_t rcv_dev, uint32_t rcv_core) {
        const auto receiver = [&](uint32_t k) {
            const double t = 0.020 + k * 1e-3;
            feed_link(sync, rcv_dev, rcv_core, k, kT0, hw_stamp(snd_dev, t));
            feed_link(sync, rcv_dev, rcv_core, k, kT1, hw_stamp(rcv_dev, t + kOneWay));
        };
        for (uint32_t k = 0; k < 300; k++) {
            const double t = 0.020 + k * 1e-3;
            const double echo_out = t + kOneWay + kTurn, echo_in = t + 2 * kOneWay + kTurn;
            feed_link(sync, snd_dev, snd_core, k, kT1B, hw_stamp(rcv_dev, echo_out));
            if (k % 11 != 5) {
                feed_link(sync, snd_dev, snd_core, k, kT2, hw_stamp(snd_dev, echo_in));
            }
            if (k % 7 != 3 && k != 100) {
                receiver(k);
            }
            if (k == 105) {
                receiver(100);
            }
        }
    };
    burst(/*snd*/ 0, 0, /*rcv*/ 1, 0);
    burst(/*snd*/ 1, 1, /*rcv*/ 2, 0);
    sync.on_capture_end();

    ClockMap::Reader reader = map.reader();
    const auto placed_tsc = [&](int c, double tau) {
        return map.place_host(reader, static_cast<uint32_t>(c), wall_tick(c, tau));
    };
    const auto placed_ns = [&](int c, double tau) {
        return (static_cast<double>(placed_tsc(c, tau) - kTsc0) - tsc(tau)) / kTicksPerNs;
    };
    const auto steady_ns = [&](int c, double tau) {
        return static_cast<double>(steady_mono_ns(placed_tsc(c, tau)) - kHost0);
    };
    for (double tau : {0.050, 0.150, 0.280}) {
        check_at("(a) chip0 root pre-switch ", tau, placed_ns(0, tau), 0.0, kTol);
    }
    for (double tau : {0.400, 0.700, 0.950}) {
        check_at("(b) chip0 root post-switch", tau, placed_ns(0, tau), 0.0, kTol);
    }
    for (double tau : {0.050, 0.500, 0.950}) {
        check_at("(c) chip1 one hop ", tau, placed_ns(1, tau), 0.0, kTol);
        check_at("(d) chip2 two hops", tau, placed_ns(2, tau), 0.0, kTol);
    }
    for (int c : {0, 1, 2}) {
        for (double tau : {0.050, 0.500, 0.950}) {
            check_at(fmt::format("steady chip{}", c), tau, steady_ns(c, tau), host_ns(tau), 1.0);
        }
    }
}

void check_retention() {
    constexpr uint32_t kNodes = 1u << 14;
    constexpr uint32_t chip = 3;
    ClockMap map(chip + 1, kNodes, {});
    ClockMap::Reader map_reader = map.reader();
    constexpr int64_t step = 1000;
    // The first segment is steeper than the rest, which alternate in slope, and no node's tangent matches a chord, so
    // the retired first node, a kept node and a mid-series chord each place differently. Every value is a multiple of
    // 2^-5 below 2^43, where a double resolves 2^-10, so the map's arithmetic on them is exact.
    const auto value = [](uint32_t i) { return i == 0 ? 7.5e12 - 62.5 : 7.5e12 + 31.25 * i + 3.90625 * (i % 2); };
    const auto tangent = [](uint32_t i) { return 0.041015625 + 0.001953125 * (i % 3); };
    const uint32_t n = kNodes + 1;
    for (uint32_t i = 0; i < n; i++) {
        map.append(chip, SyncNode{.at = static_cast<int64_t>(i) * step, .value = value(i), .tangent = tangent(i)});
    }
    const auto root = [&](int64_t at) { return map.root_offset(map_reader, chip, at, 0.0).value_or(0.0); };
    const uint32_t mid = n / 2;
    check_near("retained: newest node", root(static_cast<int64_t>(n - 1) * step), value(n - 1), 1e-3);
    check_near(
        "retained: a node mid-series",
        root(static_cast<int64_t>(mid) * step + step / 2),
        (value(mid) + value(mid + 1)) / 2,
        1e-3);
    check_near(
        "retained: the retired first node (on the oldest kept tangent)", root(0), value(1) - tangent(1) * step, 1e-3);
}

int main() {
    check_chain();
    check_retention();
    if (g_fail != 0) {
        std::printf("FAILED (%d)\n", g_fail);
        return 1;
    }
    std::printf("PASSED\n");
    return 0;
}
