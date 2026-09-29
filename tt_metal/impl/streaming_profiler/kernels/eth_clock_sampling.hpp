// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "hostdev/streaming_profiler_sync.h"
#include "internal/ethernet/eth_ptp_clock.hpp"

// The branch predictor is off for the pass, since a branch taken on the previous pass would otherwise mispredict on
// this one.
namespace sampler {
namespace eth_ptp = tt::tt_metal::eth_ptp;
FORCE_INLINE uint32_t xorshift(uint32_t& x) {
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return x;
}
// One 64-byte-aligned copy of the pass serves both calibration and sampling, so calibration measures exactly the reads
// that sample, whatever code surrounds them.
__attribute__((noinline, aligned(64))) uint32_t pass(uint32_t (&w)[3], uint32_t (&o)[4]) {
    uint32_t v[3][4];
    const auto read = [](uint32_t(&b)[4]) __attribute__((always_inline)) {
        b[1] = eth_ptp::kPtpCfrLo.read();
        b[2] = eth_ptp::kPtpCfrLo.read();
        b[0] = eth_ptp::kWallClockLo.read();
        b[3] = eth_ptp::kPtpCfrLo.read();
    };
    uint32_t st = 0;
    asm volatile("csrrsi zero, 0x7c0, 2" ::: "memory");
    read(v[2]);
    read(v[0]);
#pragma GCC unroll 20
    for (uint32_t k = 0; k < 20; k++) {
        uint32_t(&cur)[4] = v[(k + 1) % 3];
        const uint32_t(&prev)[4] = v[k % 3];
        const uint32_t(&prev2)[4] = v[(k + 2) % 3];
        read(cur);
        if (prev[3] != prev2[3]) {
            w[0] = prev2[0];
            w[1] = prev[0];
            w[2] = cur[0];
            o[0] = prev2[3];
            o[1] = prev[1];
            o[2] = prev[2];
            o[3] = prev[3];
            st = k + 1;
            break;
        }
    }
    asm volatile("csrrci zero, 0x7c0, 2" ::: "memory");
    return st;
}
// The tracker's range, 6, is its block length in cycles, so each pass starts at a uniform phase against the blocks. The
// ruler's range also covers the refclk's advance (20 ns, 16-27 cycles across the AICLK range); without that, the loop's
// own length phase-locks the passes to the refclk and each chip's updates settle on one AICLK cycle of the crossing.
template <uint32_t Range>
FORCE_INLINE uint32_t draw(uint32_t& walk) {
    static_assert(Range >= 1 && Range <= 64);
    return ((xorshift(walk) >> 16) * Range) >> 16;
}
template <uint32_t Range>
FORCE_INLINE void pad(uint32_t& walk) {
    const uint32_t k = draw<Range>(walk);
    asm volatile(
        ".option push\n\t.option norvc\n\t"
        "slli t1, %[k], 2\n\t"
        "auipc t0, 0\n\t"
        "addi t0, t0, 16 + %[n] * 4\n\t"
        "sub t0, t0, t1\n\t"
        "jr t0\n\t"
        ".rept %[n]\n\tnop\n\t.endr\n\t"
        ".option pop"
        :
        : [k] "r"(k), [n] "i"(Range)
        : "t0", "t1", "memory");
}
constexpr uint32_t kTrackerPadRange = 6;  // eth_clock_stream.hpp's .Lpass holds kTrackerPadRange - 1 nops
constexpr uint32_t kRulerPadRange = 64;
struct Table {
    uint32_t period;
    uint32_t pos8s;  // each gap's position in eighths, one signed byte per gap
};
constexpr int32_t kDropGap = -128;
// Runs one pass and returns 1 if it wrote a update to slot. With period 0 a update is the length of the block after its
// step, which never follows the pad. Otherwise a update needs blocks period apart and a gap whose byte in pos8s isn't
// kDropGap, and holds the new refclk and the step's wall time in eighths: the block's wall read plus that byte.
FORCE_INLINE uint32_t run(uint32_t period, uint32_t pos8s, volatile tt_l1_ptr uint32_t* slot, uint32_t& walk) {
    uint32_t w[3], o[4];
    pad<kRulerPadRange>(walk);
    if (pass(w, o) == 0) {
        return 0;
    }
    if (period == 0) {
        slot[0] = w[2] - w[1];
        return 1;
    }
    const uint32_t p = o[1] != o[0] ? 0u : o[2] != o[1] ? 1u : 2u;
    const int32_t pos8 = static_cast<int8_t>(pos8s >> (8 * p));
    if (w[1] - w[0] != period || w[2] - w[1] != period || pos8 == kDropGap) {
        return 0;
    }
    slot[0] = o[p + 1];
    slot[1] = (w[1] << 3) + static_cast<uint32_t>(pos8);
    return 1;
}
// `one(period, pos8s, slot)` runs one pass of the kernel's sampling reads and returns 1 if it wrote a update.
template <class One>
Table calibrate(One one, volatile tt_l1_ptr kernel_profiler::RelayCtrl* ctl) {
    Table t{};
    static uint32_t hist[64];
    uint32_t buf[2];
    volatile tt_l1_ptr uint32_t* slot = buf;
    for (uint32_t i = 0; i < 1024; i++) {
        if (one(0, 0, slot) != 0) {
            hist[slot[0] & 63u]++;
        }
    }
    for (uint32_t i = 1; i < 64; i++) {
        t.period = hist[i] > hist[t.period] ? i : t.period;
    }
    uint32_t pairs[3] = {}, total = 0;
    for (uint32_t i = 1; ctl->stop == 0u; i++) {
        // Gap p sits p eighths in, so a update's low three bits name its gap.
        if (one(t.period, 0x020100u, slot) != 0) {
            pairs[slot[1] & 7u]++;
            total++;
        }
        if ((i & 1023u) == 0u) {
            ctl->heartbeat++;
            invalidate_l1_cache();
            if ((total >= (1u << 16) && ctl->go != 0u) || total >= (1u << 26)) {
                break;
            }
        }
    }
    while (ctl->go == 0u && ctl->stop == 0u) {
        ctl->heartbeat++;
        invalidate_l1_cache();
    }
    uint32_t k = 0;
    while ((total >> k) >= (1u << 20)) {
        k++;
    }
    const uint32_t n = total >> k;
    if (n == 0) {
        return t;
    }
    int32_t g64[3];
    for (uint32_t p = 0; p < 3; p++) {
        g64[p] = static_cast<int32_t>((64u * t.period * (pairs[p] >> k)) / n);
    }
    const int32_t r1 = -64;  // the read before the wall read, in 64ths of a cycle from it
    const int32_t c64[3] = {r1 - g64[1] - g64[0] / 2, r1 - g64[1] / 2, r1 + g64[2] / 2};
    for (uint32_t p = 0; p < 3; p++) {
        t.pos8s |= (static_cast<uint32_t>((c64[p] + 4) >> 3) & 0xFFu) << (8 * p);
    }
    return t;
}
}  // namespace sampler
