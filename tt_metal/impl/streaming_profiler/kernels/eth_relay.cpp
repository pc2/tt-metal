// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Ships everything the chip's eth cores produce, so the clock tracker never leaves its sampling loop for the
// multi-microsecond PCIe flush each transfer ends in.

#include <array>
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/socket_api.h"
#include "hostdev/streaming_profiler_sync.h"
#include "tt_metal/impl/streaming_profiler/kernels/relay_common.hpp"

namespace kp = kernel_profiler;

constexpr uint32_t kFramesCfgAddr = get_named_compile_time_arg_val("frames_cfg");
constexpr uint32_t kSyncCfgAddr = get_named_compile_time_arg_val("sync_cfg");
constexpr uint32_t kStageAddr = get_named_compile_time_arg_val("stage");
constexpr uint32_t kCtrlAddr = get_named_compile_time_arg_val("ctrl");
constexpr uint32_t kCvScratch = get_named_compile_time_arg_val("scratch");
constexpr uint32_t kTrackerXy = get_named_compile_time_arg_val("tracker_xy");
constexpr uint32_t kTrackerProfL1 = get_named_compile_time_arg_val("tracker_prof_l1");
constexpr uint32_t kSyncRingAddr = get_named_compile_time_arg_val("sync_ring");
constexpr uint32_t kLinkRingAddr = get_named_compile_time_arg_val("link_ring");
#if defined(PROFILE_STREAMING_SYNC_CHECK)
constexpr uint32_t kRulerXy = get_named_compile_time_arg_val("ruler_xy");
#endif
// Only eth zones put profiler frames on the eth cores' rings; the link records and the sync rings ship either way.
#if defined(PROFILE_STREAMING_ETH)
constexpr bool kEthZones = true;
#else
constexpr bool kEthZones = false;
#endif

constexpr uint32_t kNumEthRisc = 2;  // DM0, DM1: the lanes that physically exist on an eth core
constexpr uint32_t kRecordBytes = sizeof(kp::SyncRecord);
constexpr uint32_t kLinkRecords = kp::kLinkSyncRingRecords;
constexpr uint32_t kBlockBytes = 64;
constexpr uint32_t kHeaderScratch = kCvScratch + kBlockBytes;
static_assert(kHeaderScratch + kBlockBytes <= kCvScratch + kp::kEthSyncScratchBytes);

FORCE_INLINE volatile tt_l1_ptr uint32_t* words(uint32_t addr) {
    return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr);
}

FORCE_INLINE uint64_t noc_at(uint32_t xy, uint32_t addr) { return get_noc_addr(xy & 0xFFFFu, xy >> 16, addr); }

FORCE_INLINE void write_word(uint32_t xy, uint32_t addr, uint32_t value) {
    noc_inline_dw_write(noc_at(xy, addr), value, 0xF);
}

inline volatile tt_l1_ptr uint32_t* read_in(uint32_t xy, uint32_t src, uint32_t dst, uint32_t bytes) {
    noc_async_read(noc_at(xy, src), dst, bytes);
    noc_async_read_barrier();
    return words(dst);
}

inline void copy_words(volatile tt_l1_ptr uint32_t* dst, const volatile tt_l1_ptr uint32_t* src, uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        dst[i] = src[i];
    }
}

inline void read_records(uint32_t xy, uint32_t ring, uint32_t ring_records, uint32_t first, uint32_t n) {
    constexpr uint32_t kDst = kStageAddr + kPrefix * 4u;
    const uint32_t h = first & (ring_records - 1);
    const uint32_t a = n < ring_records - h ? n : ring_records - h;
    noc_async_read(noc_at(xy, ring + h * kRecordBytes), kDst, a * kRecordBytes);
    if (a < n) {
        noc_async_read(noc_at(xy, ring), kDst + a * kRecordBytes, (n - a) * kRecordBytes);
    }
    noc_async_read_barrier();
}

__attribute__((noinline)) void ship(SocketSenderInterface& s, uint32_t xy, uint32_t off) {
    words(kStageAddr)[0] = kp::spsc_span_w0();
    words(kStageAddr)[kp::SPSC_PREFIX_XY] = xy;
    words(kStageAddr)[kLenWord] = off - kPrefix;
    const uint32_t bytes = page_round(off * 4u);
    const uint32_t pages = bytes / kPageBytes;
    socket_reserve_pages(s, pages);
    push_fifo(s, kStageAddr, s.write_ptr, bytes);
    socket_push_pages(s, pages);
    notify_bytes_sent(s);
}

struct EthRelay {
    // The relay is the only writer of the ring heads, so it keeps its own copy and reads only the tails' block of the
    // control vector.
    struct Core {
        uint32_t xy, l1;
        std::array<uint32_t, kp::PROFILER_SPSC_TENSIX_RISC> heads;
        uint32_t link_head = 0;
    };
    enum Socket : uint32_t { kSync, kFrames };
    static constexpr uint32_t kSockets = kEthZones ? 2 : 1;
    std::array<SocketSenderInterface, 2> sockets;
    uint32_t n_cores = 0;
    std::array<Core, 1 + kp::kEthRelayMaxLinked> cores;
    uint32_t tracker_head = 0;
#if defined(PROFILE_STREAMING_SYNC_CHECK)
    uint32_t ruler_head = 0;
#endif

    void start() {
        cores[0] = {.xy = kTrackerXy, .l1 = kTrackerProfL1};
        n_cores = 1 + get_arg_val<uint32_t>(0);
        for (uint32_t i = 1; i < n_cores; i++) {
            cores[i].xy = get_arg_val<uint32_t>(2 * i - 1);
            cores[i].l1 = get_arg_val<uint32_t>(2 * i);
        }
        sockets[kSync] = create_sender_socket_interface(kSyncCfgAddr);
        if constexpr (kEthZones) {
            for (uint32_t i = 0; i < n_cores; i++) {
                const volatile tt_l1_ptr uint32_t* cv = read_in(cores[i].xy, cores[i].l1, kCvScratch, kBlockBytes);
                for (uint32_t r = 0; r < kp::PROFILER_SPSC_TENSIX_RISC; r++) {
                    cores[i].heads[r] = cv[kp::SPSC_RING_HEAD_0 + r];
                }
            }
            sockets[kFrames] = create_sender_socket_interface(kFramesCfgAddr);
        }
        for (uint32_t i = 0; i < kSockets; i++) {
            set_sender_socket_page_size(sockets[i], kPageBytes);
        }
        noc_write_init_state<write_cmd_buf>(NOC_INDEX, NOC_UNICAST_WRITE_VC);
    }

    void ship_sync(uint32_t xy, uint32_t n) {
        volatile tt_l1_ptr uint32_t* frame = words(kStageAddr);
        frame[kp::SPSC_PREFIX_HEAD_0] = n;
        for (uint32_t r = 1; r < kp::PROFILER_SPSC_TENSIX_RISC; r++) {
            frame[kp::SPSC_PREFIX_HEAD_0 + r] = 0;
        }
        uint32_t off = kPrefix + n * kp::kSyncRecordWords;
        while (off < kPrefix + kWireCtrl) {
            frame[off++] = 0;
        }
        ship(sockets[kSync], xy, off);
    }

    __attribute__((noinline)) bool drain_ring(
        uint32_t xy,
        uint32_t frame_xy,
        uint32_t head_addr,
        uint32_t ring,
        uint32_t ring_records,
        uint32_t tail,
        uint32_t& consumed,
        bool all) {
        const uint32_t first = consumed;
        for (uint32_t left; (left = tail - consumed) != 0 && (all || left >= kp::kSyncFrameRecords);) {
            const uint32_t n = left < kp::kSyncFrameRecords ? left : kp::kSyncFrameRecords;
            read_records(xy, ring, ring_records, consumed, n);
            ship_sync(frame_xy, n);
            consumed += n;
            write_word(xy, head_addr, consumed);
        }
        return consumed != first;
    }

    bool drain_sync_ring(uint32_t xy, uint32_t& consumed, bool all) {
        constexpr uint32_t kTail = offsetof(kp::RelayCtrl, sync_tail);
        static_assert(kTail + 4u < kBlockBytes);
        const uint32_t tail = read_in(xy, kCtrlAddr, kHeaderScratch, kBlockBytes)[kTail / 4u];
        return drain_ring(
            xy,
            kTrackerXy,
            kCtrlAddr + offsetof(kp::RelayCtrl, sync_head),
            kSyncRingAddr,
            kp::kSyncRingRecords,
            tail,
            consumed,
            all);
    }

    // Without `all`, a ring ships only whole frames, since a frame pads to 24 words and a single record's frame would
    // be four times its size.
    bool drain(bool all) {
        bool shipped = drain_sync_ring(kTrackerXy, tracker_head, all);
#if defined(PROFILE_STREAMING_SYNC_CHECK)
        shipped = drain_sync_ring(kRulerXy, ruler_head, all) || shipped;
#endif
        return shipped;
    }

    // Read the tails first: a tail seen there bounds the ring words read after it.
    bool ship_core(Core& c) {
        constexpr uint32_t kBase = kp::SPSC_WIRE_CV_BASE;
        static_assert(kWireCtrl * 4u == kBlockBytes && kp::SPSC_LINK_SYNC_TAIL - kBase < kWireCtrl);
        volatile tt_l1_ptr uint32_t* blk = read_in(c.xy, c.l1 + kBase * 4u, kCvScratch, kBlockBytes);
        const bool linked = drain_ring(
            c.xy,
            c.xy,
            c.l1 + kp::SPSC_LINK_SYNC_HEAD * 4u,
            kLinkRingAddr,
            kLinkRecords,
            blk[kp::SPSC_LINK_SYNC_TAIL - kBase],
            c.link_head,
            true);
        if constexpr (!kEthZones) {
            return linked;
        }
        std::array<uint32_t, kNumEthRisc> takes;
        bool live = false;
        for (uint32_t r = 0; r < kNumEthRisc; r++) {
            takes[r] = blk[kp::SPSC_WIRE_TAIL_0 + r] - c.heads[r];
            live = live || takes[r] != 0;
        }
        if (!live) {
            return linked;
        }
        volatile tt_l1_ptr uint32_t* frame = words(kStageAddr);
        copy_words(frame + kPrefix, blk, kWireCtrl);
        for (uint32_t r = 0; r < kp::PROFILER_SPSC_TENSIX_RISC; r++) {
            frame[kp::SPSC_PREFIX_HEAD_0 + r] = c.heads[r];
        }
        uint32_t off = kPrefix + kWireCtrl;
        for (uint32_t r = 0; r < kNumEthRisc; r++) {
            if (takes[r] != 0) {
                const uint32_t ring_l1 = c.l1 + kp::PROFILER_L1_CONTROL_BUFFER_SIZE + r * kRingWords * 4u;
                off = place_run(
                    c.heads[r], takes[r], off, kStageAddr, [&](uint32_t src, uint32_t dst, uint32_t bytes, bool) {
                        noc_async_read(noc_at(c.xy, ring_l1 + src), dst, bytes);
                    });
            }
        }
        noc_async_read_barrier();
        for (uint32_t r = 0; r < kNumEthRisc; r++) {
            if (takes[r] != 0) {
                c.heads[r] += takes[r];
                write_word(c.xy, c.l1 + (kp::SPSC_RING_HEAD_0 + r) * 4u, c.heads[r]);
            }
        }
        ship(sockets[kFrames], c.xy, off);
        return true;
    }

    bool sweep() {
        bool live = false;
        for (uint32_t i = 0; i < n_cores; i++) {
            live = ship_core(cores[i]) || live;
        }
        return live;
    }
};

void kernel_main() {
    volatile tt_l1_ptr kp::RelayCtrl* ctl = reinterpret_cast<volatile tt_l1_ptr kp::RelayCtrl*>(kCtrlAddr);
    EthRelay d;
    d.start();
    while (ctl->go == 0u && ctl->stop == 0u) {
        ctl->heartbeat++;
        invalidate_l1_cache();
    }
    uint32_t gap = 0, deferred = 0;
    for (bool stop = false; !stop;) {
        invalidate_l1_cache();
        // The host writes stop after the ruler and tracker have stopped, so their tails are final by then.
        stop = ctl->stop != 0u;
        const bool busy = d.sweep();
        // A sync ring's partial frame waits the same way a DRISC lane below its ship gate does.
        const bool partial = stop || !busy || ++deferred >= kMaxDeferSweeps;
        deferred = partial ? 0u : deferred;
        idle_wait(gap, d.drain(partial) || busy, [] {});
    }
    ctl->done = kp::kRelayAwaitingAcksWord;
    for (uint32_t i = 0; i < EthRelay::kSockets; i++) {
        socket_barrier(d.sockets[i]);
    }
    noc_async_writes_flushed();
    for (uint32_t i = 0; i < EthRelay::kSockets; i++) {
        update_socket_config(d.sockets[i]);
    }
    ctl->done = kp::kRelayDoneWord;
}
