// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "hostdev/dev_msgs.h"
#include "hostdev/streaming_profiler_common.h"
#include "hostdev/streaming_profiler_sync.h"
#include "internal/ethernet/eth_ptp.hpp"

namespace tt::tt_metal::eth_ptp {

namespace kp = kernel_profiler;
static_assert(kp::kEthRefclkHz == kRefclkHz);

// A link's stamped frames take a TX queue and header row nothing else uses: the firmware holds header rows 0 to 2
// (eth_ptp.hpp), and the fabric routers send on TX queue 0 and, when two ERISCs run a router, its receiver on queue 1.
constexpr uint32_t kLinkTxq = 2;
constexpr uint32_t kLinkHeaderRow = 3;
constexpr uint32_t kLinkTcamRow = 63;
constexpr uint32_t kLinkLabel = 0x15;
using LinkQueue = TxQueue<kLinkTxq>;
// The stamp frames' destination. The RX rule matches only its middle four bytes, 0xA5, which no firmware frame has
// (eth_ptp.hpp) and which read the same in either byte order.
constexpr uint64_t kStampFrameDa = 0x02A5'A5A5'A5A5ull;
// A 1 in the mask means don't care, so the rule compares only the four 0xA5 bytes.
constexpr RxTcamNonIpPattern kStampRowValues{.da = {0xA5A5'A500u, 0x0000'00A5u}};
constexpr RxTcamNonIpPattern kStampRowMask{
    .sa = {~0u, ~0u, ~0u, ~0u},
    .da = {0x0000'00FFu, 0xFFFF'FF00u, ~0u, ~0u},
    .addr_flags = {.augmented_da = 0xF, .augmented_sa = 0xF},
    .ethertype = {.value = 0xFFFF, .augmented = 0xF},
    .priority = {.pcp = 7}};
#if defined(PROFILE_STREAMING_SYNC_CHECK)
constexpr bool kSyncCheck = true;
#else
constexpr bool kSyncCheck = false;
#endif

// The stamp FIFO doesn't say which frame a stamp is for, so the ends take turns: the sender sends a burst only once the
// last one is fully echoed, and the receiver takes a burst's stamps only once all its frames are in. Each take then
// finds exactly that burst's stamps, in order.
constexpr uint32_t kTripsPerRound = 96;
constexpr uint32_t kBurstFrames = 4;
constexpr uint32_t kBurstsPerRound = kTripsPerRound / kBurstFrames;
constexpr uint32_t kFrameBytes = 96;
constexpr uint32_t kCombPhases = 256;
constexpr uint32_t kFramePhaseStep = 157;  // odd, so the kCombPhases phases are a permutation
static_assert(kTripsPerRound % kBurstFrames == 0 && kBurstFrames >= 2 && (kFramePhaseStep & 1) == 1);

// A frame in L1. The MAC writes its egress stamp into `stamp` (kFrameStampField). In `sync`, bytes_sent is the frame's
// key, receiver_ack the key an echo answers and reserved_2 the round.
struct LinkFrame {
    uint32_t stamp[4];
    eth_channel_sync_t sync;
    uint32_t offset_lo, offset_hi;  // the sender's PTP offset
    uint32_t rsvd[(kFrameBytes - 40) / 4];
};
static_assert(sizeof(LinkFrame) == kFrameBytes && kFrameBytes % 16 == 0);
static_assert(kFrameStampField + 10 <= offsetof(LinkFrame, sync) && offsetof(LinkFrame, offset_lo) == 32);
static_assert(offsetof(kp::LinkSyncL1, slots) == 0 && kBurstFrames * kFrameBytes == sizeof(kp::LinkSyncL1::slots));

// Frame j uses slot j % kBurstFrames, at the same L1 address on both ends, so an echo lands on the frame it answers.
FORCE_INLINE constexpr uint32_t frame_at(uint32_t base, uint32_t j) { return base + (j % kBurstFrames) * kFrameBytes; }
FORCE_INLINE volatile LinkFrame& link_frame(uint32_t addr) { return *reinterpret_cast<volatile LinkFrame*>(addr); }
constexpr uint32_t kTripBits = 9;
constexpr uint32_t kTripMask = (1u << kTripBits) - 1u;
constexpr uint32_t frame_key(uint32_t round, uint32_t j) { return (round << kTripBits) | (j + 1); }
static_assert(kTripsPerRound <= kTripMask);
FORCE_INLINE constexpr uint32_t frame_phase_cycles(uint32_t k, uint32_t p16) {
    static_assert(kCombPhases * 16 == 1u << 12);
    return (((k * kFramePhaseStep) & (kCombPhases - 1)) * p16) >> 12;
}

// Polling alone exits late by an amount that depends on when the wait began, which would bend the grid by an
// AICLK-dependent amount.
struct Pacer {
    uint32_t iter16 = 0;
    // A single copy, because the loop's cycles per turn depend on where it's placed, and calibration has to time the
    // same instructions until() spins.
    __attribute__((noipa)) static void turns(uint32_t n) {
        for (uint32_t i = n; i != 0; i--) {
            std::atomic_signal_fence(std::memory_order_seq_cst);
        }
    }
    void calibrate() {
        const uint32_t a = kWallClockLo.read();
        turns(4096);
        const uint32_t b = kWallClockLo.read();
        iter16 = ((b - a) * 16u) / 4096u;
    }
    FORCE_INLINE void until(uint32_t target) const {
        const int32_t rem = static_cast<int32_t>(target - kWallClockLo.read()) - 16;
        if (rem > 0) {
            turns((static_cast<uint32_t>(rem) * 16u) / iter16);
        }
        while (static_cast<int32_t>(kWallClockLo.read() - target) < 0) {
        }
    }
};

// 64-bit because on a loaded router a round's stamps can spread over tens of milliseconds, and the offsets from the
// first stamp overflow 32 bits past ~89 ms.
struct StampSum {
    uint32_t n = 0;
    uint64_t base = 0;
    uint64_t rel = 0;
    FORCE_INLINE void reset() {
        n = 0;
        rel = 0;
    }
    FORCE_INLINE void add(uint64_t ts) {
        if (n == 0) {
            base = ts;
        }
        rel += ts - base;
        n++;
    }
};
// Clear the stamp field: the slot still holds the last stamp that came through it, and a frame the MAC didn't stamp has
// to read as zero.
FORCE_INLINE void carry_offset(volatile LinkFrame& f, const PtpTimer& timer) {
    f.offset_lo = static_cast<uint32_t>(timer.offset_ns);
    f.offset_hi = static_cast<uint32_t>(static_cast<uint64_t>(timer.offset_ns) >> 32);
    f.stamp[kFrameStampHiWord] = 0;
    f.stamp[kFrameStampHiWord + 1] = 0;
}

struct RoundStamps {
    StampSum tx, rx;
    int64_t peer_offset_ns = 0;
    __attribute__((noinline)) uint64_t peer_frame(const volatile LinkFrame& f) {
        peer_offset_ns = static_cast<int64_t>((static_cast<uint64_t>(f.offset_hi) << 32) | f.offset_lo);
        return frame_stamp(f.stamp);
    }
    // A frame's stamp is in the FIFO before the frame is visible, and nothing else of ours is in flight to this end, so
    // the FIFO holds one stamp per frame. The exceptions are a frame that came in twice (a Go-back-N resend is stamped,
    // then dropped as a duplicate) and a full FIFO.
    __attribute__((noinline)) void take_burst(const uint64_t* tx_ts, volatile kp::LinkSyncDiag& diag) {
        constexpr RxStampFifo fifo{};
        uint64_t rx_ts[kBurstFrames];
        bool ok = fifo.holds_exactly(kBurstFrames);
#pragma GCC unroll 1
        for (uint32_t i = 0; ok && i < kBurstFrames; i++) {
            RxStampFifo::Entry e{};
            ok = fifo.pop(e) && e.valid && e.label == kLinkLabel;
            rx_ts[i] = e.ts;
        }
        if (!ok) {
            fifo.flush();
            diag.bursts_mismatched++;
            return;
        }
#pragma GCC unroll 1
        for (uint32_t m = 0; m < kBurstFrames; m++) {
            if (tx_ts[m] == 0) {
                diag.frames_unstamped++;
                continue;
            }
            tx.add(tx_ts[m]);
            rx.add(rx_ts[m]);
        }
    }
};
// If the queue is busy, leave the frame for a later step. Waiting on a queue that a link-level resend keeps busy would
// stop the router serving the fabric.
FORCE_INLINE bool issue(uint32_t frame) {
    if (internal_::eth_txq_is_busy(kLinkTxq)) {
        return false;
    }
    internal_::eth_send_packet_unsafe(kLinkTxq, frame >> 4, frame >> 4, kFrameBytes >> 4);
    return true;
}

struct Grid {
    static constexpr uint32_t kLeadCycles = 64;
    uint32_t p16 = 0;
    uint32_t edge_wall = 0, edge_refclk = 0;
    Pacer pacer;
    void start() {
        pacer.calibrate();
        uint32_t w0 = 0, r0 = 0, w1 = 0, r1 = 0;
        next_refclk_update(w0, r0);
        while (kPtpCfrLo.read() - r0 < 1000u) {
        }
        next_refclk_update(w1, r1);
        p16 = ((w1 - w0) << 4) / ((r1 - r0) / 4u);
        edge_wall = w1;
        edge_refclk = r1;
    }
    __attribute__((noinline)) bool send(uint32_t base, uint32_t round, uint32_t j) {
        uint32_t w = 0, r = 0;
        next_refclk_update(w, r);
        const uint32_t cycles = w - edge_wall, updates = (r - edge_refclk) / 4u;
        if (updates != 0 && cycles < (1u << 27)) {
            p16 = (cycles << 4) / updates;
        }
        edge_wall = w;
        edge_refclk = r;
        pacer.until(w + kLeadCycles + frame_phase_cycles(round * kTripsPerRound + j, p16));
        return issue(frame_at(base, j));
    }
};

// Every member is zero-initialised, so an end has no .data for the firmware to copy; start() sets the rest.
struct EndBase {
    PtpTimer timer;
    TxHeaderRow<kLinkTxq, kLinkHeaderRow> header;
    RxStampRule<kLinkTcamRow, kLinkLabel> rule;
    volatile kp::LinkSyncL1* l1 = nullptr;
    uint32_t round = 0;
    bool started = false;
    Grid grid;
    RoundStamps stamps;
    uint32_t records = 0;

    void open(uint32_t link_l1) {
        l1 = reinterpret_cast<volatile kp::LinkSyncL1*>(link_l1);
        l1->diag.rounds_lost = 0;
        l1->diag.bursts_mismatched = 0;
        l1->diag.frames_unstamped = 0;
        // An end clears its slots before any peer frame can land, because a burst missing a frame would stall the link
        // for good. A resident end opens before the handshake, and a router sender only bursts after the host writes
        // Run once both routers are up, so RouterEnd::start clears any Run left by an earlier process.
#pragma GCC unroll 1
        for (uint32_t j = 0; j < kBurstFrames; j++) {
            link_frame(frame_at(link_l1, j)).sync.bytes_sent = 0;
        }
        timer.start();
        rule.install(kStampRowValues, kStampRowMask);
        header.install(kStampFrameDa);
        LinkQueue{}.arm_in_frame();
    }
    void start() {
        *records_tail() = 0;
        *records_head() = 0;
        grid.start();
    }
    void stop() {
        LinkQueue{}.disarm();
        header.restore();
        rule.remove();
    }

protected:
    FORCE_INLINE uint32_t slot_base() const { return reinterpret_cast<uintptr_t>(l1); }
    static FORCE_INLINE volatile uint32_t* control_vector() {
        return reinterpret_cast<volatile uint32_t*>(GET_MAILBOX_ADDRESS_DEV(profiler.control_vector));
    }
    static FORCE_INLINE volatile uint32_t* records_tail() { return control_vector() + kp::SPSC_LINK_SYNC_TAIL; }
    static FORCE_INLINE volatile uint32_t* records_head() { return control_vector() + kp::SPSC_LINK_SYNC_HEAD; }
    // The host does the average, since a 64-bit divide is a library routine in the ERISC's text.
    __attribute__((noinline)) void record(const StampSum& sum, int64_t ptp_offset_ns, uint32_t role) {
        volatile kp::SyncLinkRecord& r = l1->ring[records % kp::kLinkSyncRingRecords].link;
        r.meta = kp::word_of(kp::SyncMeta{.role = role, .kind = kp::kSyncKindLink});
        r.round = round;
        r.first = static_cast<uint64_t>(static_cast<int64_t>(sum.base) - ptp_offset_ns);
        r.sum_ns = sum.rel;
        r.count = sum.n;
        std::atomic_thread_fence(std::memory_order_release);
        *records_tail() = ++records;
    }
    __attribute__((noinline)) void close_round(uint32_t tx_role, uint32_t rx_role) {
        invalidate_l1_cache();
        const bool recorded = stamps.tx.n != 0 && records - *records_head() <= kp::kLinkSyncRingRecords - 2;
        if (recorded) {
            record(stamps.tx, stamps.peer_offset_ns, tx_role);
            record(stamps.rx, timer.offset_ns, rx_role);
        }
        l1->diag.rounds_lost += !recorded;
    }
    FORCE_INLINE void open_round(uint32_t next, uint32_t tx_role, uint32_t rx_role) {
        if (started) {
            close_round(tx_role, rx_role);
        }
        started = true;
        round = next;
        stamps.tx.reset();
        stamps.rx.reset();
    }
};

struct SenderLink : EndBase {
    static constexpr uint32_t kBurstTicks =
        (kSyncCheck ? kp::kLinkSyncCheckPaceTicks : kp::kLinkSyncPaceTicks) / kBurstsPerRound;
    uint64_t slot_cfr = 0;
    uint32_t slot_wall = 0;
    uint32_t out_j0 = 0, out_sent = 0, in_round = 0;

    void start() {
        EndBase::start();
        out_sent = kBurstFrames;
        resync();
    }
    FORCE_INLINE void resync() {
        const Instant now = read_instant();
        slot_cfr = now.refclk;
        schedule(now);
    }
    FORCE_INLINE void schedule(const Instant& now) {
        slot_wall = now.wall_lo + ((static_cast<uint32_t>(slot_cfr - now.refclk) * (grid.p16 >> 2)) >> 4);
    }
    FORCE_INLINE void step() {
        if (out_sent != kBurstFrames) {
            send_next();
            return;
        }
        const uint32_t w = kWallClockLo.read();
        if (static_cast<int32_t>(w - slot_wall) < 0) {
            return;
        }
        invalidate_l1_cache();
        if (l1->ctl != kp::kLinkSyncCtlRun) {
            in_round = 0;
            resync();
            return;
        }
        if (started && !echoed()) {
            return;
        }
        burst();
    }

private:
    FORCE_INLINE bool echoed() const {
        for (uint32_t i = 0; i < kBurstFrames; i++) {
            if (link_frame(frame_at(slot_base(), i)).sync.receiver_ack != frame_key(round, out_j0 + i)) {
                return false;
            }
        }
        return true;
    }
    __attribute__((noinline)) void send_next() { out_sent += grid.send(slot_base(), round, out_j0 + out_sent); }
    __attribute__((noinline)) void burst() {
        const Instant now = read_instant();
        if (started) {
            uint64_t tx[kBurstFrames];
#pragma GCC unroll 1
            for (uint32_t i = 0; i < kBurstFrames; i++) {
                tx[i] = stamps.peer_frame(link_frame(frame_at(slot_base(), i)));
            }
            stamps.take_burst(tx, l1->diag);
        }
        if (in_round == 0) {
            open_round(started ? round + 1 : round, kp::kSyncRoleT1B, kp::kSyncRoleT2);
        }
        out_j0 = in_round * kBurstFrames;
        if (++in_round == kBurstsPerRound) {
            in_round = 0;
        }
#pragma GCC unroll 1
        for (uint32_t i = 0; i < kBurstFrames; i++) {
            const uint32_t f = frame_at(slot_base(), i);
            carry_offset(link_frame(f), timer);
            volatile eth_channel_sync_t* s = &link_frame(f).sync;
            s->receiver_ack = 0;
            s->reserved_2 = round;
            s->bytes_sent = frame_key(round, out_j0 + i);
        }
        out_sent = 0;
        slot_cfr += kBurstTicks;
        if (static_cast<int64_t>(slot_cfr - now.refclk) < 0) {
            slot_cfr = now.refclk;
        }
        schedule(now);
    }
};

struct ReceiverLink : EndBase {
    uint32_t echo_j0 = 0, taken = 0, echoed = 0;
    uint64_t tx[kBurstFrames] = {};

    // Take a burst's ingress stamps when its last frame arrives, before that frame's echo; the sender sends nothing
    // more until every echo is in.
    FORCE_INLINE void step() {
        invalidate_l1_cache();
        if (echoed != taken) {
            echo();
        } else if (link_frame(frame_at(slot_base(), taken)).sync.bytes_sent != 0) {
            take();
            echo();
        }
    }

private:
    __attribute__((noinline)) void take() {
        const uint32_t f = frame_at(slot_base(), taken);
        volatile eth_channel_sync_t* s = &link_frame(f).sync;
        if (taken == 0) {
            if (!started || s->reserved_2 != round) {
                open_round(s->reserved_2, kp::kSyncRoleT0, kp::kSyncRoleT1);
            }
            echo_j0 = (s->bytes_sent & kTripMask) - 1;
        }
        tx[taken] = stamps.peer_frame(link_frame(f));
        if (taken == kBurstFrames - 1) {
            stamps.take_burst(tx, l1->diag);
        }
        carry_offset(link_frame(f), timer);
        s->receiver_ack = s->bytes_sent;
        s->bytes_sent = 0;
        taken++;
    }
    __attribute__((noinline)) void echo() {
        echoed += grid.send(slot_base(), round, echo_j0 + echoed);
        if (echoed == kBurstFrames) {
            taken = 0;
            echoed = 0;
        }
    }
};

template <bool Sender>
using LinkEnd = std::conditional_t<Sender, SenderLink, ReceiverLink>;

// The asm clobbers only ra and no memory, so F must take no arguments and must not touch memory its caller reads.
template <void (*F)()>
FORCE_INLINE void saved_call() {
    asm volatile(
        "addi sp, sp, -64\n\t"
        "sw t0, 0(sp)\n\t"
        "sw t1, 4(sp)\n\t"
        "sw t2, 8(sp)\n\t"
        "sw t3, 12(sp)\n\t"
        "sw t4, 16(sp)\n\t"
        "sw t5, 20(sp)\n\t"
        "sw t6, 24(sp)\n\t"
        "sw a0, 28(sp)\n\t"
        "sw a1, 32(sp)\n\t"
        "sw a2, 36(sp)\n\t"
        "sw a3, 40(sp)\n\t"
        "sw a4, 44(sp)\n\t"
        "sw a5, 48(sp)\n\t"
        "sw a6, 52(sp)\n\t"
        "sw a7, 56(sp)\n\t"
        "call %[fn]\n\t"
        "lw t0, 0(sp)\n\t"
        "lw t1, 4(sp)\n\t"
        "lw t2, 8(sp)\n\t"
        "lw t3, 12(sp)\n\t"
        "lw t4, 16(sp)\n\t"
        "lw t5, 20(sp)\n\t"
        "lw t6, 24(sp)\n\t"
        "lw a0, 28(sp)\n\t"
        "lw a1, 32(sp)\n\t"
        "lw a2, 36(sp)\n\t"
        "lw a3, 40(sp)\n\t"
        "lw a4, 44(sp)\n\t"
        "lw a5, 48(sp)\n\t"
        "lw a6, 52(sp)\n\t"
        "lw a7, 56(sp)\n\t"
        "addi sp, sp, 64"
        :
        : [fn] "s"(F)
        : "ra");
}

// The bodies are noipa and the router's main loop only reaches the step through saved_call, so the loop compiles
// exactly as it does without an end. A plain call would make the compiler spill around it.
template <bool Sender>
struct RouterEnd {
    static inline LinkEnd<Sender> end;
    __attribute__((noipa, cold)) static void start(uint32_t l1) {
        end.open(l1);
        if constexpr (Sender) {
            end.l1->ctl = 0;
        }
        end.start();
    }
    __attribute__((noipa)) static void step_body() { end.step(); }
    static constexpr uint32_t kRouterStepLoops = kSyncCheck ? 1 : 16;
    FORCE_INLINE void step(uint32_t iter) {
        if ((iter & (kRouterStepLoops - 1)) == 0) {
            saved_call<&step_body>();
        }
    }
    __attribute__((noipa, cold)) static void stop() { end.stop(); }
};

struct NoLinkEnd {
    void start(uint32_t) {}
    void step(uint32_t) {}
    void stop() {}
};

template <uint32_t Role>
using RouterHook =
    std::conditional_t<Role == kp::kLinkSyncRoleNone, NoLinkEnd, RouterEnd<Role == kp::kLinkSyncRoleSender>>;

}  // namespace tt::tt_metal::eth_ptp
