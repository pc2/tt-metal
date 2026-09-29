// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "hostdev/streaming_profiler_sync.h"

// The host reads every record's kind, local ones included, through SyncMeta.
static_assert(
    kernel_profiler::word_of(kernel_profiler::SyncLocalMeta{.kind = 0xFF}) ==
    kernel_profiler::word_of(kernel_profiler::SyncMeta{.kind = 0xFF}));

template <uint32_t CtrlAddr, uint32_t RingAddr>
struct SyncRingWriter {
    uint32_t tail = 0, dropped = 0;
    uint32_t n = 0, meta = 0, steps[kernel_profiler::kSyncLocalPoints - 1] = {};
    kernel_profiler::SyncLocalRound round{};
    uint64_t r0 = 0, w0 = 0;

    static volatile tt_l1_ptr kernel_profiler::RelayCtrl* ctrl() {
        return reinterpret_cast<volatile tt_l1_ptr kernel_profiler::RelayCtrl*>(CtrlAddr);
    }
    void emit() {
        invalidate_l1_cache();
        if (tail - ctrl()->sync_head >= kernel_profiler::kSyncRingRecords) {
            dropped++;
            return;
        }
        volatile tt_l1_ptr kernel_profiler::SyncLocalRecord* r =
            &reinterpret_cast<volatile tt_l1_ptr kernel_profiler::SyncRecord*>(
                 RingAddr)[tail % kernel_profiler::kSyncRingRecords]
                 .local;
        r->meta = meta | kernel_profiler::word_of(kernel_profiler::SyncLocalMeta{.count = n});
        r->round = kernel_profiler::word_of(round);
        r->refclk = r0;
        r->wall8 = w0;
        r->steps[0] = steps[0];
        r->steps[1] = steps[1];
        std::atomic_thread_fence(std::memory_order_release);
        ctrl()->sync_tail = ++tail;
    }
    void flush() {
        if (n != 0) {
            emit();
            n = 0;
        }
    }
    void close() {
        flush();
        ctrl()->dropped_sync = dropped;
        std::atomic_thread_fence(std::memory_order_release);
        ctrl()->done = kernel_profiler::kRelayDoneWord;
    }
    __attribute__((noinline)) void add(uint64_t r, uint64_t w8, uint32_t k8, uint32_t slope, uint32_t record_meta) {
        if (n != 0) {
            if (record_meta == meta) {
                const int32_t off = static_cast<int32_t>(
                    static_cast<uint32_t>(w8) - static_cast<uint32_t>(w0) -
                    round.slope * static_cast<uint32_t>(r - r0));
                if (kernel_profiler::sync_local_step_fits(r - r0, off)) {
                    steps[n - 1] = kernel_profiler::word_of(
                        kernel_profiler::SyncLocalStep{.refclk = static_cast<uint32_t>(r - r0), .wall_off = off});
                    round.k8[n] = static_cast<uint8_t>(k8);
                    if (++n == kernel_profiler::kSyncLocalPoints) {
                        flush();
                    }
                    return;
                }
            }
            flush();
        }
        r0 = r;
        w0 = w8;
        round.k8[0] = static_cast<uint8_t>(k8);
        round.slope = static_cast<uint8_t>(slope);
        meta = record_meta;
        n = 1;
    }
};
