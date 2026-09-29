// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "eth_l1_address_map.h"
#include "internal/ethernet/dataflow_api.h"
#include "tt_metal/impl/streaming_profiler/kernels/eth_ptp_link.hpp"

namespace eth_ptp = tt::tt_metal::eth_ptp;

constexpr bool kSender = get_compile_time_arg_val(0) != 0;
static eth_ptp::LinkEnd<kSender> g_link;
static constexpr uint32_t kHandshake = eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE;
// Base firmware shares ERISC0 and runs only when a kernel yields to it; the fabric router's default is every 10000
// idle loops.
static constexpr uint32_t kTurnsPerYield = 10000;

void kernel_main() {
    g_link.open(get_arg_val<uint32_t>(0));
    if constexpr (kSender) {
        eth_send_bytes(kHandshake, kHandshake, 16);
        eth_wait_for_receiver_done();
    } else {
        eth_wait_for_bytes(16);
        eth_receiver_channel_done(0);
    }
    g_link.start();
    for (uint32_t turn = 1; g_link.l1->ctl != kernel_profiler::kLinkSyncCtlStop; turn++) {
        g_link.step();
        if (turn % kTurnsPerYield == 0) {
            run_routing();
        }
    }
    g_link.stop();
    g_link.l1->done = 1;
}
