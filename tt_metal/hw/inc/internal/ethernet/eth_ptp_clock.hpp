// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#if !defined(ARCH_BLACKHOLE)
#error "eth_ptp_clock.hpp is Blackhole only"
#endif

#include <cstdint>

#include "internal/ethernet/tt_eth_ss_regs.h"
#include "internal/risc_attribs.h"

namespace tt::tt_metal::eth_ptp {

constexpr uint32_t kRefclkHz = 50'000'000u;
constexpr uint32_t kNsPerRefclkTick = 1'000'000'000u / kRefclkHz;
constexpr uint32_t kPtiRefclk = kNsPerRefclkTick << 16;  // the timer's per-tick increment, 8.16 fixed point

template <typename T>
constexpr uint32_t word_of(const T& v) {
    static_assert(sizeof(T) == sizeof(uint32_t));
    return __builtin_bit_cast(uint32_t, v);
}

// Every bit of a T struct has a name, reserved ones included, so that a value built with designated initializers has
// all its other bits zero.
template <typename T = uint32_t>
struct Reg {
    static_assert(sizeof(T) == sizeof(uint32_t));
    uint32_t addr;
    FORCE_INLINE T read() const { return __builtin_bit_cast(T, *reinterpret_cast<volatile uint32_t*>(addr)); }
    FORCE_INLINE void write(T v) const { *reinterpret_cast<volatile uint32_t*>(addr) = word_of(v); }
};

// The high word that goes with the low word is WALL_CLOCK_1_AT, latched by the low word's read; WALL_CLOCK_1 is live.
constexpr Reg<> kWallClockLo{ETH_RISC_REGS_START + ETH_RISC_WALL_CLOCK_0};
constexpr Reg<> kWallClockHi{ETH_RISC_REGS_START + ETH_RISC_WALL_CLOCK_1_AT};

struct PtpUpdateStat {
    uint32_t pti_pending : 1;
    uint32_t timestamp_pending : 1;
    uint32_t rsvd0 : 6;
    uint32_t pti_ack : 1;
    uint32_t timestamp_ack : 1;
    uint32_t rsvd1 : 6;
    uint32_t pti_late_err : 1;
    uint32_t timestamp_late_err : 1;
    uint32_t rsvd2 : 14;
};

constexpr Reg<> kPtpTimerCtrl{ETH_PTP_TIMER_REGS_START + ETH_PTP_TIMER_CTRL};
constexpr Reg<> kPtpFutureCfrLo{ETH_PTP_TIMER_REGS_START + ETH_PTP_TIMER_FUTURE_CFR_LO};
constexpr Reg<> kPtpFutureCfrHi{ETH_PTP_TIMER_REGS_START + ETH_PTP_TIMER_FUTURE_CFR_HI};
constexpr Reg<> kPtpFuturePti{ETH_PTP_TIMER_REGS_START + ETH_PTP_TIMER_FUTURE_PTI};
constexpr Reg<> kPtpFutureTimestampLo{ETH_PTP_TIMER_REGS_START + ETH_PTP_TIMER_FUTURE_TIMESTAMP_LO};
constexpr Reg<> kPtpFutureTimestampHi{ETH_PTP_TIMER_REGS_START + ETH_PTP_TIMER_FUTURE_TIMESTAMP_HI};
constexpr Reg<> kPtpUpdatePti{ETH_PTP_TIMER_REGS_START + ETH_PTP_TIMER_UPDATE_PTI};
constexpr Reg<> kPtpUpdateTimestamp{ETH_PTP_TIMER_REGS_START + ETH_PTP_TIMER_UPDATE_TIMESTAMP};
constexpr Reg<> kPtpSyncOffset1{ETH_PTP_TIMER_REGS_START + ETH_PTP_TIMER_SYNC_OFFSET1};
constexpr Reg<PtpUpdateStat> kPtpUpdateStat{ETH_PTP_TIMER_REGS_START + ETH_PTP_TIMER_UPDATE_STAT};
constexpr Reg<> kPtpPtiStat{ETH_PTP_TIMER_REGS_START + ETH_PTP_TIMER_PTI_STAT};
constexpr Reg<> kPtpCfrLo{ETH_PTP_TIMER_REGS_START + ETH_PTP_TIMER_CFR_LO};
constexpr Reg<> kPtpCfrHi{ETH_PTP_TIMER_REGS_START + ETH_PTP_TIMER_CFR_HI};
constexpr Reg<> kPtp32s32nsLo{ETH_PTP_TIMER_REGS_START + ETH_PTP_TIMER_32S_32NS_LO};
constexpr Reg<> kPtp32s32nsHi{ETH_PTP_TIMER_REGS_START + ETH_PTP_TIMER_32S_32NS_HI};
constexpr Reg<> kPtp64nsLo{ETH_PTP_TIMER_REGS_START + ETH_PTP_TIMER_64NS_LO};
constexpr Reg<> kPtp64nsHi{ETH_PTP_TIMER_REGS_START + ETH_PTP_TIMER_64NS_HI};
constexpr Reg<> kPtpSync32s32nsLo{ETH_PTP_TIMER_REGS_START + ETH_PTP_TIMER_SYNC_32S_32NS_LO};
constexpr Reg<> kPtpSync32s32nsHi{ETH_PTP_TIMER_REGS_START + ETH_PTP_TIMER_SYNC_32S_32NS_HI};
constexpr Reg<> kPtpSync64nsLo{ETH_PTP_TIMER_REGS_START + ETH_PTP_TIMER_SYNC_64NS_LO};
constexpr Reg<> kPtpSync64nsHi{ETH_PTP_TIMER_REGS_START + ETH_PTP_TIMER_SYNC_64NS_HI};

FORCE_INLINE uint64_t read64(Reg<> lo_reg, Reg<> hi_reg) {
    const uint32_t lo = lo_reg.read();
    const uint32_t hi = hi_reg.read();
    return (static_cast<uint64_t>(hi) << 32) | lo;
}
FORCE_INLINE uint64_t read_cfr() { return read64(kPtpCfrLo, kPtpCfrHi); }
FORCE_INLINE uint64_t read_ptp64ns() { return read64(kPtp64nsLo, kPtp64nsHi); }

struct Instant {
    uint32_t wall_lo, wall_hi;
    uint64_t refclk;
    uint64_t wall() const { return (static_cast<uint64_t>(wall_hi) << 32) | wall_lo; }
};
// The wall clock's low read latches its high word for only a few cycles, and the refclk read between the two takes
// longer, so a low word that wrapped before the high read tore the pair by 2^32.
FORCE_INLINE Instant read_instant() {
    Instant t;
    while (true) {
        const uint32_t hi0 = kWallClockHi.read();
        t.wall_lo = kWallClockLo.read();
        const uint32_t rlo = kPtpCfrLo.read();
        t.wall_hi = kWallClockHi.read();
        const uint32_t rhi = kPtpCfrHi.read();
        if (t.wall_hi == hi0) {
            t.refclk = (static_cast<uint64_t>(rhi) << 32) | rlo;
            return t;
        }
    }
}
// Spins until the refclk counter changes, then returns its new value and the wall clock read in the same loop pass.
FORCE_INLINE void next_refclk_update(uint32_t& wall, uint32_t& refclk) {
    uint32_t prev = kPtpCfrLo.read();
    while (true) {
        const uint32_t w = kWallClockLo.read();
        const uint32_t r = kPtpCfrLo.read();
        if (r != prev) {
            wall = w;
            refclk = r;
            return;
        }
        prev = r;
    }
}

}  // namespace tt::tt_metal::eth_ptp
