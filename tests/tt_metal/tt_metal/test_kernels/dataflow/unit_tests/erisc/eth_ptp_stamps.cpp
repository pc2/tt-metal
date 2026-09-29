// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// ActiveEthPtpStamps' kernel: one end of a link's stamped frame exchange, as initiator or echo.

#include <cstddef>
#include <cstdint>

#include "internal/ethernet/dataflow_api.h"
#include "internal/ethernet/eth_ptp.hpp"
#include "eth_ptp_stamps.hpp"

namespace eth_ptp = tt::tt_metal::eth_ptp;
using namespace eth_ptp_stamps;

constexpr bool kInitiator = get_compile_time_arg_val(0) != 0;
// A TX queue and header row that neither the firmware nor the fabric uses (eth_ptp.hpp), and any TCAM row and label.
constexpr uint32_t kTxq = 2, kHeaderRow = 3, kTcamRow = 63, kLabel = 0x15;
// A destination no firmware frame has (eth_ptp.hpp), and a rule that matches only it.
constexpr uint64_t kStampFrameDa = 0x02A5'A5A5'A5A5ull;
// A 1 in the mask means don't care, so the rule compares only the four 0xA5 bytes.
constexpr eth_ptp::RxTcamNonIpPattern kStampRowValues{.da = {0xA5A5'A500u, 0x0000'00A5u}};
constexpr eth_ptp::RxTcamNonIpPattern kStampRowMask{
    .sa = {~0u, ~0u, ~0u, ~0u},
    .da = {0x0000'00FFu, 0xFFFF'FF00u, ~0u, ~0u},
    .addr_flags = {.augmented_da = 0xF, .augmented_sa = 0xF},
    .ethertype = {.value = 0xFFFF, .augmented = 0xF},
    .priority = {.pcp = 7}};
// The MAC writes the egress stamp into `stamp`; `sync` carries the round's key and its echo.
struct Frame {
    uint32_t stamp[4];
    eth_channel_sync_t sync;
};
static_assert(eth_ptp::kFrameStampField + 10 <= offsetof(Frame, sync) && sizeof(Frame) <= kFrameBytes);
constexpr eth_ptp::TxQueue<kTxq> g_txq{};
static eth_ptp::PtpTimer g_timer;
static eth_ptp::TxHeaderRow<kTxq, kHeaderRow> g_header;
static eth_ptp::RxStampRule<kTcamRow, kLabel> g_rule;
static constexpr uint32_t kSpins = 1u << 20;

// Clear the frame's slot first: it still holds the last stamp that came in through it, and a frame the MAC didn't stamp
// has to arrive as zero.
inline void send(volatile tt_l1_ptr Frame* f) {
    f->stamp[eth_ptp::kFrameStampHiWord] = 0;
    f->stamp[eth_ptp::kFrameStampHiWord + 1] = 0;
    const uint32_t addr = reinterpret_cast<uint32_t>(f);
    internal_::eth_send_packet<false>(kTxq, addr >> 4, addr >> 4, kFrameBytes >> 4);
    while (internal_::eth_txq_is_busy(kTxq)) {
    }
}

bool take_ingress(uint64_t& ts, uint32_t& extra) {
    uint32_t n = 0;
    eth_ptp::RxStampFifo{}.drain<kLabel>([&](uint64_t t) {
        if (n++ == 0) {
            ts = t;
        }
    });
    extra += n > 1 ? n - 1 : 0;
    return n != 0;
}

template <typename Pred>
bool wait_for(Pred&& pred) {
    for (uint32_t s = 0; s < kSpins; s++) {
        invalidate_l1_cache();
        if (pred()) {
            return true;
        }
    }
    return false;
}

void kernel_main() {
    const uint32_t base = get_arg_val<uint32_t>(0);
    volatile tt_l1_ptr Result* res = reinterpret_cast<volatile tt_l1_ptr Result*>(base + kResultOffset);
    volatile tt_l1_ptr Frame* f = reinterpret_cast<volatile tt_l1_ptr Frame*>(base + kFrameOffset);
    volatile tt_l1_ptr eth_channel_sync_t* frame = &f->sync;
    frame->bytes_sent = 0;
    frame->receiver_ack = 0;
    res->done = 0;
    res->sel_before = eth_ptp::word_of(eth_ptp::txq_pkt_cfg_sel_sw(kTxq).read());
    res->no_match_before = eth_ptp::word_of(eth_ptp::kRxNoMatchActions.read());
    g_timer.start();
    g_rule.install(kStampRowValues, kStampRowMask);
    g_header.install(kStampFrameDa);
    g_txq.arm_in_frame();

    if constexpr (kInitiator) {
        eth_send_bytes(base, base, 16);
        eth_wait_for_receiver_done();
    } else {
        eth_wait_for_bytes(16);
        eth_receiver_channel_done(0);
    }

    uint32_t unstamped = 0, rx_missing = 0, rx_extra = 0, i = 0;
    for (; i < kRounds; i++) {
        const uint32_t key = 0x5A000000u | (i + 1);
        uint64_t ingress = 0;
        if constexpr (kInitiator) {
            frame->bytes_sent = key;
            frame->receiver_ack = 0;
            send(f);
            if (!wait_for([&] { return frame->receiver_ack == key; })) {
                break;
            }
        } else {
            if (!wait_for([&] { return frame->bytes_sent == key; })) {
                break;
            }
        }
        const uint64_t egress = eth_ptp::frame_stamp(f->stamp);
        rx_missing += !take_ingress(ingress, rx_extra);
        unstamped += egress == 0;
        res->stamps[i][0] = egress;
        res->stamps[i][1] = ingress;
        if constexpr (!kInitiator) {
            frame->receiver_ack = key;
            send(f);
        }
    }

    g_txq.disarm();
    g_header.restore();
    g_rule.remove();
    res->sel_after = eth_ptp::word_of(eth_ptp::txq_pkt_cfg_sel_sw(kTxq).read());
    res->no_match_after = eth_ptp::word_of(eth_ptp::kRxNoMatchActions.read());
    res->unstamped = unstamped;
    res->rx_missing = rx_missing;
    res->rx_extra = rx_extra;
    res->rounds = i;
    res->done = kDone;
}
