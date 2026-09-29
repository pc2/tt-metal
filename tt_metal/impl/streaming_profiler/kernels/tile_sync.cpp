// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Partners share this tile's row or column. NoC 0 runs toward higher coordinates and NoC 1 toward lower, so a pair's
// two readings cross the same links in opposite directions.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "hostdev/streaming_profiler_sync.h"

// A remote tile's NIU samples its RISCV_DEBUG_REG_WALL_CLOCK_L when the read request arrives.
namespace tile_read {

inline uint32_t coord(uint32_t xy) {
    return static_cast<uint32_t>(get_noc_addr(xy & 0xFFFFu, xy >> 16, 0) >> NOC_ADDR_COORD_SHIFT) & NOC_COORDINATE_MASK;
}

// Keeps the firmware's count of issued reads, which the eth firmware checks at kernel exit.
inline uint32_t read(uint32_t noc, uint32_t coord, uint32_t addr, uint32_t scratch) {
    const uint32_t dst = scratch + (addr & 0x3Fu);
    NOC_CMD_BUF_WRITE_REG(noc, read_cmd_buf, NOC_RET_ADDR_LO, dst);
    NOC_CMD_BUF_WRITE_REG(noc, read_cmd_buf, NOC_TARG_ADDR_LO, addr);
    NOC_CMD_BUF_WRITE_REG(noc, read_cmd_buf, NOC_TARG_ADDR_MID, 0);
    NOC_CMD_BUF_WRITE_REG(noc, read_cmd_buf, NOC_TARG_ADDR_COORDINATE, coord);
    NOC_CMD_BUF_WRITE_REG(noc, read_cmd_buf, NOC_AT_LEN_BE, 4);
    const uint32_t before = NOC_STATUS_READ_REG(noc, NIU_MST_RD_RESP_RECEIVED);
    NOC_CMD_BUF_WRITE_REG(noc, read_cmd_buf, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);
    noc_reads_num_issued[noc] += 1;
    while (NOC_STATUS_READ_REG(noc, NIU_MST_RD_RESP_RECEIVED) == before) {
    }
    invalidate_l1_cache();
    return *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst);
}

template <typename Hi, typename Lo>
inline uint64_t wall64(Hi hi, Lo lo) {
    while (true) {
        const uint32_t hi0 = hi();
        const uint32_t l = lo();
        if (hi() == hi0) {
            return (static_cast<uint64_t>(hi0) << 32) | l;
        }
    }
}

inline uint64_t wall64(uint32_t noc, uint32_t coord, uint32_t scratch) {
    return wall64(
        [&] { return read(noc, coord, RISCV_DEBUG_REG_WALL_CLOCK_1, scratch); },
        [&] { return read(noc, coord, RISCV_DEBUG_REG_WALL_CLOCK_L, scratch); });
}

inline uint64_t own_wall64() {
    return wall64(
        [] { return *reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_1); },
        [] { return *reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L); });
}

// The request is programmed once and re-sent each rep (the NIU only clears NOC_CMD_CTRL on acceptance). The bracket
// holds only the send and the poll, since every instruction inside it adds to the far end's allowance.
template <typename F>
__attribute__((noinline, cold)) inline void bracket(
    uint32_t noc, uint32_t coord, uint32_t scratch, volatile uint32_t* wall, uint32_t reps, F f) {
    uint32_t prev = read(noc, coord, RISCV_DEBUG_REG_WALL_CLOCK_L, scratch);
    volatile tt_l1_ptr uint32_t* const land =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch + (RISCV_DEBUG_REG_WALL_CLOCK_L & 0x3Fu));
    volatile uint32_t* const ctrl =
        reinterpret_cast<volatile uint32_t*>(NOC_CMD_BUF_INSTANCE_OFFSET(noc, read_cmd_buf) + NOC_CMD_CTRL);
    for (uint32_t i = 0; i < reps; i++) {
        noc_reads_num_issued[noc] += 1;
        const uint32_t sentinel = prev ^ 0x80000000u;
        *land = sentinel;
        const uint32_t w0 = *wall;
        *ctrl = NOC_CTRL_SEND_REQ;
        do {
            invalidate_l1_cache();
        } while (*land == sentinel);
        const uint32_t w1 = *wall;
        const uint32_t v = *land;
        prev = v;
        f(i, 2 * static_cast<int64_t>(static_cast<int32_t>(v - w0)) - static_cast<int32_t>(w1 - w0));
    }
}

constexpr uint32_t kWarmup = 16;
constexpr uint32_t kBins = kernel_profiler::kTileNetBins;

inline int32_t quantile(const volatile tt_l1_ptr uint32_t* hist, uint32_t rank) {
    uint32_t seen = 0;
    for (uint32_t b = 0; b < kBins; b++) {
        seen += hist[b];
        if (seen >= rank) {
            return static_cast<int32_t>(b) - static_cast<int32_t>(kBins / 2);
        }
    }
    return static_cast<int32_t>(kBins / 2) - 1;
}

inline void measure(
    uint32_t noc,
    uint32_t coord,
    uint32_t scratch,
    uint32_t reps,
    volatile tt_l1_ptr uint32_t* hist,
    volatile tt_l1_ptr kernel_profiler::TileNetPartner& out) {
    volatile uint32_t* const wall = reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L);
    for (uint32_t b = 0; b < kBins; b++) {
        hist[b] = 0;
    }
    int64_t warm[kWarmup];
    bracket(noc, coord, scratch, wall, kWarmup, [&](uint32_t i, int64_t d) {
        uint32_t j = i;
        for (; j > 0 && warm[j - 1] > d; j--) {
            warm[j] = warm[j - 1];
        }
        warm[j] = d;
    });
    const int64_t centre = warm[kWarmup / 2];
    bracket(noc, coord, scratch, wall, reps, [&](uint32_t, int64_t d) {
        const int64_t b = d - centre + kBins / 2;
        hist[b < 0 ? 0 : b >= kBins ? kBins - 1 : b]++;
    });
    const uint64_t coarse = wall64(noc, coord, scratch) - own_wall64();
    out.median2 = static_cast<int32_t>(centre + quantile(hist, reps / 2 + 1));
    out.coarse_lo = static_cast<uint32_t>(coarse);
    out.coarse_hi = static_cast<uint32_t>(coarse >> 32);
}

}  // namespace tile_read

namespace kp = kernel_profiler;

void kernel_main() {
    const uint32_t scratch = get_arg_val<uint32_t>(0);
    const uint32_t reps = get_arg_val<uint32_t>(1);
    const uint32_t nonce = get_arg_val<uint32_t>(2);
    const uint32_t n = get_arg_val<uint32_t>(3);
    volatile tt_l1_ptr kp::TileNetScratch* s = reinterpret_cast<volatile tt_l1_ptr kp::TileNetScratch*>(scratch);
    volatile tt_l1_ptr kp::TileNetTable& tab = s->table;
    tab.go = 0;
    tab.ready = nonce;
    uint32_t go;
    do {
        invalidate_l1_cache();
        go = tab.go;
    } while (go == 0);
    if (go == kp::kTileNetGoMeasure) {
        for (uint32_t k = 0; k < n; k++) {
            const uint32_t w = get_arg_val<uint32_t>(4 + k);
            tile_read::measure(w >> 31, tile_read::coord(w & 0x7FFFFFFFu), scratch, reps, s->hist, tab.partner[k]);
        }
        tab.ready = ~nonce;
        do {
            invalidate_l1_cache();
        } while (tab.go != kp::kTileNetGoExit);
    }
    // On a dispatch core the scratch is the profiler's ring space, which its first frame expects to be zero.
    volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch);
    for (uint32_t i = 0; i < sizeof(kp::TileNetScratch) / 4; i++) {
        p[i] = 0;
    }
}
