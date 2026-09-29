// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <atomic>
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "hostdev/streaming_profiler_sync.h"
#include "internal/ethernet/eth_ptp_clock.hpp"
#include "tt_metal/impl/streaming_profiler/kernels/eth_clock_sampling.hpp"
#include "tt_metal/impl/streaming_profiler/kernels/eth_clock_stream.hpp"

constexpr uint32_t kCtrlAddr = get_named_compile_time_arg_val("ctrl_addr");
constexpr uint32_t kSampleRingAddr = get_named_compile_time_arg_val("sample_ring_addr");

namespace kp = kernel_profiler;
namespace eth_ptp = tt::tt_metal::eth_ptp;

void kernel_main() {
    volatile tt_l1_ptr kp::RelayCtrl* ctl = reinterpret_cast<volatile tt_l1_ptr kp::RelayCtrl*>(kCtrlAddr);
    volatile tt_l1_ptr kp::SyncSampleRing* ring =
        reinterpret_cast<volatile tt_l1_ptr kp::SyncSampleRing*>(kSampleRingAddr);
    // The stream's pads, four uniform 0..5 draws each, drawn once so a update's pad is one load.
    static uint32_t pads[256];
    uint32_t walk = eth_ptp::kWallClockLo.read() | 1u;
    for (uint32_t& p : pads) {
        p = 4u * sampler::draw<sampler::kTrackerPadRange>(walk);
    }
    // Calibration's updates aren't samples, so they publish to a word the model never reads, and each call ends at its
    // first update, whose reload finds the head at kSyncHeadStop.
    uint32_t unpublished = 0;
    sampler::StreamArgs args{
        .refclk = reinterpret_cast<volatile uint32_t*>(eth_ptp::kPtpCfrLo.addr),
        .wall = reinterpret_cast<volatile uint32_t*>(eth_ptp::kWallClockLo.addr),
        .head = &ring->head,
        .pads = pads,
        .tail_word = &unpublished};
    ring->head = kp::kSyncHeadStop;
    const sampler::Table table = sampler::calibrate(
        [&](uint32_t period, uint32_t pos8s, volatile tt_l1_ptr uint32_t* slot) {
            const uint32_t from = args.tail;
            args.period = period;
            args.pos8s = pos8s;
            args.slots = slot;
            args.iters = 2;
            args.limit = from + 1;
            args.tail = sampler_stream(&args);
            return args.tail - from;
        },
        ctl);
    if (ctl->stop == 0u) {
        const eth_ptp::Instant start = eth_ptp::read_instant();
        ring->refclk = start.refclk;
        ring->wall8 = start.wall() << 3;
        ring->head = 0;
        args.period = table.period;
        args.pos8s = table.pos8s;
        args.slots = &ring->samples[0][0];
        args.mask = kp::kSyncSampleRingSamples - 1;
        args.iters = 0;
        args.tail_word = &ring->tail;
        args.tail = 0;
        args.limit = args.mask;
        const uint32_t tail = sampler_stream(&args);
        std::atomic_thread_fence(std::memory_order_release);
        ring->tail = tail;
    }
    std::atomic_thread_fence(std::memory_order_release);
    ring->done = 1;
}
