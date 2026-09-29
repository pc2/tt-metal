// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>

#include "hostdev/streaming_profiler_common.h"

namespace kernel_profiler {

template <typename T>
constexpr std::uint32_t word_of(const T& v) {
    static_assert(sizeof(T) == sizeof(std::uint32_t));
    return __builtin_bit_cast(std::uint32_t, v);
}
template <typename T>
constexpr T word_as(std::uint32_t w) {
    static_assert(sizeof(T) == sizeof(std::uint32_t));
    return __builtin_bit_cast(T, w);
}

constexpr std::uint32_t kEthRefclkHz = 50'000'000u;

constexpr std::uint32_t kSyncKindLocal = 0, kSyncKindLink = 1, kSyncKindRuler = 2;
// A LINK record's role. The receiver records the sender's egress average (T0) and its own ingress average (T1); the
// sender records the receiver's egress average (T1B) and its own ingress average (T2).
constexpr std::uint32_t kSyncRoleT0 = 0, kSyncRoleT1 = 1, kSyncRoleT1B = 2, kSyncRoleT2 = 3;

struct SyncMeta {
    std::uint32_t role : 8;
    std::uint32_t kind : 8;
    std::uint32_t rsvd : 16;
};

constexpr std::uint32_t kSyncLocalPoints = 3;
struct SyncLocalMeta {
    std::uint32_t count : 2;
    std::uint32_t dense : 1;  // RULER: every update around a clock change, not one in kSyncRulerKeepEvery
    std::uint32_t rsvd0 : 5;
    std::uint32_t kind : 8;
    std::uint32_t rsvd1 : 16;
};
struct SyncLocalRound {
    std::uint8_t k8[kSyncLocalPoints];  // each point's wall ticks per refclk tick, in eighths; 0 for a single sample
    std::uint8_t slope;
};
struct SyncLocalStep {
    std::uint32_t refclk : 16;
    std::int32_t wall_off : 16;  // eighths of a tick, from the first point's wall plus slope times the refclk step
};
static_assert(sizeof(SyncLocalRound) == sizeof(std::uint32_t) && sizeof(SyncLocalStep) == sizeof(std::uint32_t));
constexpr bool sync_local_step_fits(std::uint64_t dr, std::int32_t off) {
    return dr <= 0xFFFFu && off >= -32768 && off <= 32767;
}
struct SyncLocalRecord {
    std::uint32_t meta;
    std::uint32_t round;
    std::uint64_t refclk;
    std::uint64_t wall8;
    std::uint32_t steps[kSyncLocalPoints - 1];
};

// The average is first + sum_ns / count, in ns.
struct SyncLinkRecord {
    std::uint32_t meta;
    std::uint32_t round;
    std::uint64_t first;  // the first stamp less the stamping timer's PTP offset, ns
    std::uint64_t sum_ns;
    std::uint32_t count;
    std::uint32_t rsvd;
};

struct SyncHeader {
    std::uint32_t meta;
};
// One record of a sync frame, which is the SPSC frame prefix with the record count at SPSC_PREFIX_HEAD_0, then the
// records, padded to SPSC_SPAN_WIRE_CTRL_WORDS. The kind in header.meta says which member a record is.
union SyncRecord {
    SyncHeader header;
    SyncLocalRecord local;
    SyncLinkRecord link;
};
static_assert(sizeof(SyncRecord) == 32 && alignof(SyncRecord) == 8);
constexpr std::uint32_t kSyncRecordWords = sizeof(SyncRecord) / sizeof(std::uint32_t);

constexpr std::uint32_t kSyncRingRecords = 512;

// The sampler never gets more than a ring ahead of the model's head. A head published kSyncHeadStop past the model's
// position tells it to stop.
constexpr std::uint32_t kSyncSampleRingSamples = 32768;
constexpr std::uint32_t kSyncHeadStop = 1u << 31;
struct SyncSampleRing {
    std::uint32_t tail;
    std::uint32_t done;
    // An instant read before the first sample, the base the low words widen against: its refclk and its wall clock in
    // eighths.
    std::uint64_t refclk, wall8;
    std::uint32_t head;
    std::uint32_t rsvd[9];
    std::uint32_t samples[kSyncSampleRingSamples][2];
};
static_assert(offsetof(SyncSampleRing, refclk) == 8 && offsetof(SyncSampleRing, samples) == 64);
constexpr std::uint32_t kSyncFrameRecords = 32;

// Away from a clock change, the sync check's ruler (eth_clock_ruler.cpp) keeps one update in this many, and the host
// weights each by it.
constexpr std::uint32_t kSyncRulerKeepEvery = 8;

constexpr std::uint32_t kLinkSyncPaceTicks = 500'000;  // a round every 10 ms
// With the sync check a link runs a round every 1 ms, and every kLinkSyncCheckSolveEvery-th round feeds the link solve.
constexpr std::uint32_t kLinkSyncCheckPaceTicks = 50'000;
constexpr std::uint32_t kLinkSyncCheckSolveEvery = kLinkSyncPaceTicks / kLinkSyncCheckPaceTicks;
constexpr std::uint32_t kLinkSyncCtlRun = 1, kLinkSyncCtlStop = 2;
constexpr std::uint32_t kLinkSyncRoleNone = 0, kLinkSyncRoleSender = 1, kLinkSyncRoleReceiver = 2;
constexpr std::uint32_t kLinkSyncSlotWords = 96;  // eth_ptp_link.hpp's frames in flight
constexpr std::uint32_t kLinkSyncRingRecords = 8;
static_assert(
    (kSyncRingRecords & (kSyncRingRecords - 1)) == 0 && (kSyncSampleRingSamples & (kSyncSampleRingSamples - 1)) == 0 &&
    (kLinkSyncRingRecords & (kLinkSyncRingRecords - 1)) == 0);
static_assert(kLinkSyncRingRecords <= kSyncFrameRecords);

struct LinkSyncDiag {
    std::uint32_t rounds_lost;
    std::uint32_t bursts_mismatched;
    std::uint32_t frames_unstamped;
};

// A link end's L1, at the top of the active eth core's unreserved region, at the same address on both ends.
struct LinkSyncL1 {
    std::uint32_t slots[kLinkSyncSlotWords];
    std::uint32_t ctl;
    std::uint32_t done;
    LinkSyncDiag diag;
    // An end never waits for the eth relay. A round the ring has no room for isn't recorded, and counts in
    // LinkSyncDiag::rounds_lost.
    alignas(32) SyncRecord ring[kLinkSyncRingRecords];
};

constexpr std::uint32_t kEthSyncScratchBytes = sizeof(SyncSampleRing);
constexpr std::uint32_t kEthRelayMaxLinked = 16;

constexpr std::uint32_t kTileNetMaxPartners = 40;
constexpr std::uint32_t kTileNetBins = 128;
constexpr std::uint32_t kTileNetGoMeasure = 1, kTileNetGoExit = 2;

struct TileNetPartner {
    std::int32_t median2;  // the median of 2 * (partner wall - bracket midpoint), in the clocks' low words
    std::uint32_t coarse_lo, coarse_hi;  // the whole-clock difference, partner minus this tile
};
struct TileNetTable {
    std::uint32_t go;
    std::uint32_t ready;  // the host's nonce once the tile is up, its inverse once every partner is written
    TileNetPartner partner[kTileNetMaxPartners];
};
struct TileNetScratch {
    std::uint32_t landing[16];
    TileNetTable table;
    std::uint32_t hist[kTileNetBins];
};

}  // namespace kernel_profiler
