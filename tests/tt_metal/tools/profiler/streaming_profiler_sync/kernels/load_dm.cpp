// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// NoC load for the sync workloads: bursts of writes to and reads from partner cores, each followed by an idle spin.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t scratch = get_arg_val<uint32_t>(0);
    const uint32_t bytes = get_arg_val<uint32_t>(1);
    const uint32_t bursts = get_arg_val<uint32_t>(2);
    const uint32_t iters = get_arg_val<uint32_t>(3);
    const uint32_t idle = get_arg_val<uint32_t>(4);
    const uint32_t n = get_arg_val<uint32_t>(5);
    const auto partner = [&](uint32_t i, uint32_t addr) {
        const uint32_t xy = get_arg_val<uint32_t>(6 + i % n);
        return get_noc_addr(xy & 0xFFFFu, xy >> 16, addr);
    };
    const uint32_t landing = scratch + (2 + noc_index) * bytes;
    volatile uint32_t* const wall = reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L);
    for (uint32_t b = 0; b < bursts; b++) {
        for (uint32_t i = 0; i < iters; i++) {
            for (uint32_t j = 0; j < 4; j++) {
                noc_async_write(scratch, partner(4 * i + j, landing), bytes);
                noc_async_read(partner(4 * i + j + 1, scratch), scratch + bytes, bytes);
            }
            noc_async_write_barrier();
            noc_async_read_barrier();
        }
        for (const uint32_t t0 = *wall; *wall - t0 < idle;) {
        }
    }
}
