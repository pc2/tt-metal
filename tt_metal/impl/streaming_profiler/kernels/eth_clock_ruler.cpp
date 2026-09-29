// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Placement error peaks within a few microseconds of a clock change, so every update near one is kept. Elsewhere only
// one in kSyncRulerKeepEvery is, and the host weights those by that.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "hostdev/streaming_profiler_sync.h"
#include "tt_metal/impl/streaming_profiler/kernels/eth_clock_sampling.hpp"
#include "tt_metal/impl/streaming_profiler/kernels/eth_sync_ring.hpp"

constexpr uint32_t kCtrlAddr = get_named_compile_time_arg_val("ctrl_addr");
constexpr uint32_t kSyncRingAddr = get_named_compile_time_arg_val("sync_ring_addr");

namespace kp = kernel_profiler;
namespace eth_ptp = tt::tt_metal::eth_ptp;

constexpr uint32_t kAdjacentPair = 1;
constexpr uint32_t kHistory = 8;      // ~9 us of updates held back, so a change's run-up goes out dense
constexpr uint32_t kDenseAfter = 64;  // ~70 us of updates at kHistory's rate go out dense after a change
constexpr int64_t kOffLine8 = 16;     // two ticks: a steady update sits within a few eighths of the line

using Out = SyncRingWriter<kCtrlAddr, kSyncRingAddr>;
constexpr uint32_t kMetaDense = kp::word_of(kp::SyncLocalMeta{.dense = 1, .kind = kp::kSyncKindRuler});
constexpr uint32_t kMetaThin = kp::word_of(kp::SyncLocalMeta{.kind = kp::kSyncKindRuler});

void kernel_main() {
    volatile tt_l1_ptr kp::RelayCtrl* ctl = Out::ctrl();
    uint32_t cal_walk = eth_ptp::kWallClockLo.read() | 1u;
    sampler::Table table = sampler::calibrate(
        [&](uint32_t period, uint32_t pos8s, volatile tt_l1_ptr uint32_t* s) {
            return sampler::run(period, pos8s, s, cal_walk);
        },
        ctl);
    const uint32_t kept = 0xFFu << (8 * kAdjacentPair);
    table.pos8s = (table.pos8s & kept) | (static_cast<uint8_t>(sampler::kDropGap) * 0x010101u & ~kept);
    Out out;
    struct Held {
        uint64_t r, w8;
    };
    Held held[kHistory];
    uint32_t held_n = 0, held_next = 0, thin = 0, dense_left = 0;
    int64_t k8 = 0;
    const auto release = [&](const Held& h) {
        if (dense_left != 0) {
            out.add(h.r, h.w8, 0, static_cast<uint32_t>(k8), kMetaDense);
        } else if (++thin == kp::kSyncRulerKeepEvery) {
            thin = 0;
            out.add(h.r, h.w8, 0, static_cast<uint32_t>(k8), kMetaThin);
        }
    };
    const auto release_held = [&] {
        for (; held_n != 0; held_n--) {
            release(held[(held_next + kHistory - held_n) % kHistory]);
        }
    };
    if (ctl->stop == 0u) {
        const eth_ptp::Instant start = eth_ptp::read_instant();
        uint64_t r64 = start.refclk, w8 = start.wall() << 3;
        uint32_t r_lo = static_cast<uint32_t>(r64), w8_lo = static_cast<uint32_t>(w8);
        uint32_t iter = 0, walk = start.wall_lo | 1u;
        while (true) {
            uint32_t c[2];
            if (sampler::run(table.period, table.pos8s, c, walk) != 0) {
                r64 += c[0] - r_lo;
                w8 += c[1] - w8_lo;
                r_lo = c[0];
                w8_lo = c[1];
                if (held_n != 0) {
                    const Held& prev = held[(held_next + kHistory - 1) % kHistory];
                    const auto dr = static_cast<int64_t>(r64 - prev.r), dw = static_cast<int64_t>(w8 - prev.w8);
                    const int64_t off = dw - k8 * dr;
                    if (k8 == 0 || off > kOffLine8 || off < -kOffLine8) {
                        if (k8 != 0) {
                            dense_left = kDenseAfter;
                            release_held();
                        }
                        k8 = dr > 0 ? (dw + dr / 2) / dr : 0;
                    }
                }
                if (held_n == kHistory) {
                    release(held[held_next]);
                    held_n--;
                }
                held[held_next] = {r64, w8};
                held_next = (held_next + 1) % kHistory;
                held_n++;
                if (dense_left != 0) {
                    dense_left--;
                }
            }
            if ((++iter & 255u) != 0u) {
                continue;
            }
            invalidate_l1_cache();
            if (ctl->stop != 0u) {
                break;
            }
        }
        release_held();
    }
    out.close();
}
