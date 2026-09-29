// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// AICLK only takes whole FBDIV values, each a multiple of the crystal the refclk counts, so wherever it holds, the wall
// clock gains exactly k8/8 ticks per refclk tick, k8 = FBDIV * 8 / (REFDIV * postdiv0), and its samples lie on one
// line of that slope to within a sample's width.

#include <atomic>
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "hostdev/streaming_profiler_sync.h"
#include "tt_metal/impl/streaming_profiler/kernels/eth_sync_ring.hpp"

constexpr uint32_t kCtrlAddr = get_named_compile_time_arg_val("ctrl_addr");
constexpr uint32_t kSyncRingAddr = get_named_compile_time_arg_val("sync_ring_addr");
constexpr uint32_t kSampleRingAddr = get_named_compile_time_arg_val("sample_ring_addr");

namespace kp = kernel_profiler;

// A window becomes a point at least once a millisecond, well within a SyncLocalStep's 16-bit refclk field.
constexpr uint32_t kPointTicks = kp::kEthRefclkHz / 1000;
// A sample is placed to within half its read pair, at most 1.5 cycles, so a sample twice that far from the line is off
// it.
constexpr int32_t kOffEighths = 24;
// The sampler sees nearly every refclk update, one per 80 ns (4 ticks). A step off the line adds an eighth of a tick
// per update, so after eight samples (32 ticks) it has left the line. A sample only joins a window once kHold samples
// after it are on the line too.
constexpr uint32_t kHold = 8;
// Off a line, every kGroup consecutive samples (~320 ns, over which a clock walk bends the wall ~0.06 cycles) become
// one point, their centroid. A new line opens once kSteady such points (~4 us) fit one k8; a PLL step takes ~1.3 us, so
// it can't fit a line in that span.
constexpr uint32_t kGroup = 4;
constexpr uint32_t kSteady = 12;

using Out = SyncRingWriter<kCtrlAddr, kSyncRingAddr>;
constexpr uint32_t kMetaLocal = kp::word_of(kp::SyncLocalMeta{.kind = kp::kSyncKindLocal});

struct Anchor {
    uint64_t R = 0, W = 0;
    uint32_t r = 0, w = 0;
    uint64_t full_r(uint32_t lo) const {
        return R + static_cast<uint64_t>(static_cast<int64_t>(static_cast<int32_t>(lo - r)));
    }
    uint64_t full_w(uint32_t lo) const {
        return W + static_cast<uint64_t>(static_cast<int64_t>(static_cast<int32_t>(lo - w)));
    }
    void move(uint32_t lr, uint32_t lw) {
        R = full_r(lr);
        W = full_w(lw);
        r = lr;
        w = lw;
    }
};

struct Model {
    uint32_t k8 = 0, slope = 0;
    uint32_t lr = 0, lw = 0;
    int32_t mean = 0;
    int64_t lsum = 0;
    uint32_t ln = 0;
    uint32_t r0 = 0, w0 = 0, sr = 0;
    // The window's residues minus the line's mean. Only a window close moves the mean, so this fits in 32 bits.
    int32_t ws = 0;
    uint32_t cnt = 0, size = 1;
    uint32_t last_r = 0;
    // A held sample's residue is recomputed when it joins the window, since a line change empties the hold.
    uint32_t hold[kHold][2];
    uint32_t hbeg = 0, hn = 0;
    uint32_t offr = 0, offw = 0;
    bool has_off = false;
    uint32_t gr0 = 0, gw0 = 0, gsr = 0, gsw = 0, gn = 0, glr = 0, glw = 0;
    uint32_t cr[kSteady], cw[kSteady];
    uint32_t nrecent = 0, head = 0;
    Anchor anchor;
    Out out;

    int32_t residue(uint32_t r, uint32_t w) const { return static_cast<int32_t>((w - lw) - k8 * (r - lr)); }
    void point(uint32_t r, uint32_t w, uint32_t k) {
        out.add(anchor.full_r(r), anchor.full_w(w), k, slope, kMetaLocal);
    }
    __attribute__((noinline)) void close_window() {
        const uint32_t dr = (sr + cnt / 2u) / cnt;
        const int32_t half = static_cast<int32_t>(cnt / 2u);
        // The window's error against its first sample: the sum of each sample's residue minus the first's.
        const int32_t first = static_cast<int32_t>((w0 - lw) - k8 * (r0 - lr));
        const int32_t se = ws - static_cast<int32_t>(cnt) * (first - mean);
        const int32_t e = (se + (se < 0 ? -half : half)) / static_cast<int32_t>(cnt);
        anchor.move(r0, w0);
        point(r0 + dr, w0 + k8 * dr + static_cast<uint32_t>(e), k8);
        lsum += static_cast<int64_t>(ws) + static_cast<int64_t>(cnt) * mean;
        ln += cnt;
        mean = static_cast<int32_t>(lsum / static_cast<int64_t>(ln));
        cnt = 0;
        size = size < (1u << 23) ? size * 2u : size;
    }
    FORCE_INLINE void window_add(uint32_t r, uint32_t w, int32_t res) {
        if (cnt == 0) {
            r0 = r;
            w0 = w;
            sr = 0;
            ws = 0;
        }
        const uint32_t dr = r - r0;
        sr += dr;
        ws += res - mean;
        last_r = r;
        if (++cnt == size || dr >= kPointTicks) {
            close_window();
        }
    }
    __attribute__((noinline)) void close_group() {
        const uint32_t span = glr - gr0;
        if (span != 0) {
            slope = ((glw - gw0) + span / 2u) / span;
        }
        // Every group but the capture's last closes full, so the division is nearly always by the constant.
        const auto div = [&](uint32_t x) { return gn == kGroup ? x / kGroup : x / gn; };
        const uint32_t dr = div(gsr), q = gsr - dr * gn;
        const uint32_t dw = div(gsw - slope * q + gn / 2u);
        anchor.move(gr0, gw0);
        const uint32_t r = gr0 + dr, w = gw0 + dw;
        point(r, w, 0);
        cr[(head + nrecent) % kSteady] = r;
        cw[(head + nrecent) % kSteady] = w;
        if (nrecent < kSteady) {
            nrecent++;
        } else {
            head = (head + 1) % kSteady;
        }
        gn = 0;
        if (nrecent == kSteady) {
            try_line();
        }
    }
    FORCE_INLINE void group_add(uint32_t r, uint32_t w) {
        if (gn == 0) {
            gr0 = r;
            gw0 = w;
            gsr = 0;
            gsw = 0;
        }
        gsr += r - gr0;
        gsw += w - gw0;
        glr = r;
        glw = w;
        if (++gn == kGroup) {
            close_group();
        }
    }
    __attribute__((noinline)) void try_line() {
        const uint32_t ar = cr[head], aw = cw[head];
        const uint32_t dr = cr[(head + kSteady - 1) % kSteady] - ar;
        if (dr == 0) {
            return;
        }
        const uint32_t k = ((cw[(head + kSteady - 1) % kSteady] - aw) + dr / 2u) / dr;
        int32_t res[kSteady];
        int32_t sum = 0;
        for (uint32_t i = 0; i < kSteady; i++) {
            const uint32_t j = (head + i) % kSteady;
            res[i] = static_cast<int32_t>((cw[j] - aw) - k * (cr[j] - ar));
            sum += res[i];
        }
        const int32_t m = sum / static_cast<int32_t>(kSteady);
        for (uint32_t i = 0; i < kSteady; i++) {
            if (static_cast<uint32_t>(res[i] - m + kOffEighths) > 2u * kOffEighths) {
                return;
            }
        }
        k8 = k;
        slope = k;
        lr = ar;
        lw = aw;
        lsum = sum;
        ln = kSteady;
        mean = m;
        cnt = 0;
        size = 1;
        hn = 0;
        has_off = false;
    }
    // End the line with a point where its samples last held it. Otherwise the chord from the last window's centroid, up
    // to half a window back, into the first group would leave the line well before the clock did.
    __attribute__((noinline)) void end_line(uint32_t r, uint32_t w) {
        if (cnt != 0) {
            close_window();
        }
        if (ln > kSteady) {
            point(last_r, lw + k8 * (last_r - lr) + static_cast<uint32_t>(mean), k8);
        }
        k8 = 0;
        nrecent = 0;
        head = 0;
        gn = 0;
        for (uint32_t i = 0; i < hn; i++) {
            group_add(hold[(hbeg + i) % kHold][0], hold[(hbeg + i) % kHold][1]);
        }
        hn = 0;
        group_add(offr, offw);
        group_add(r, w);
        has_off = false;
    }
    FORCE_INLINE void on_sample(uint32_t r, uint32_t w) {
        if (k8 == 0) {
            group_add(r, w);
            return;
        }
        const int32_t res = residue(r, w);
        if (static_cast<uint32_t>(res - mean + kOffEighths) > 2u * kOffEighths) {
            if (has_off) {
                end_line(r, w);
                return;
            }
            offr = r;
            offw = w;
            has_off = true;
            return;
        }
        has_off = false;
        if (hn == kHold) {
            window_add(hold[hbeg][0], hold[hbeg][1], residue(hold[hbeg][0], hold[hbeg][1]));
            hbeg = (hbeg + 1) % kHold;
            hn--;
        }
        const uint32_t j = (hbeg + hn) % kHold;
        hold[j][0] = r;
        hold[j][1] = w;
        hn++;
    }
    // Feeds the samples [p, end) in ring order. On a line, with a full hold and nothing pending off it, the state a
    // sample touches stays in registers; anything else goes through on_sample.
    void feed(const volatile tt_l1_ptr uint32_t (*p)[2], const volatile tt_l1_ptr uint32_t (*end)[2]) {
        while (p != end) {
            if (k8 == 0 || has_off || hn != kHold) {
                on_sample((*p)[0], (*p)[1]);
                p++;
                continue;
            }
            const uint32_t kk = k8, llr = lr, llw = lw;
            int32_t mn = mean, wsum = ws;
            uint32_t rr0 = r0, ssr = sr, n = cnt, sz = size, hb = hbeg, lr_last = last_r;
            bool off = false;
            // Load each sample's words and hold slot an iteration ahead, so their latency hides behind the math. The
            // one read past the end lands in L1 and is never used.
            uint32_t r = (*p)[0], w = (*p)[1], hr0 = hold[hb][0], hw0 = hold[hb][1];
            for (; p != end; p++) {
                const uint32_t rn = p[1][0], wn = p[1][1];
                const uint32_t hbn = (hb + 1) % kHold;
                const uint32_t hrn = hold[hbn][0], hwn = hold[hbn][1];
                const int32_t res = static_cast<int32_t>((w - llw) - kk * (r - llr));
                if (static_cast<uint32_t>(res - mn + kOffEighths) > 2u * kOffEighths) {
                    off = true;
                    break;
                }
                hold[hb][0] = r;
                hold[hb][1] = w;
                hb = hbn;
                if (n == 0) {
                    rr0 = hr0;
                    w0 = hw0;
                    ssr = 0;
                    wsum = 0;
                }
                const uint32_t dr = hr0 - rr0;
                ssr += dr;
                wsum += static_cast<int32_t>((hw0 - llw) - kk * (hr0 - llr)) - mn;
                lr_last = hr0;
                if (++n == sz || dr >= kPointTicks) {
                    r0 = rr0, sr = ssr, ws = wsum, cnt = n, last_r = lr_last;
                    close_window();
                    n = cnt, sz = size, mn = mean;
                }
                r = rn, w = wn, hr0 = hrn, hw0 = hwn;
            }
            r0 = rr0, sr = ssr, ws = wsum, cnt = n, hbeg = hb, last_r = lr_last;
            if (off) {
                on_sample((*p)[0], (*p)[1]);
                p++;
            }
        }
    }
    void finish() {
        if (k8 != 0) {
            for (uint32_t i = 0; i < hn; i++) {
                const uint32_t j = (hbeg + i) % kHold;
                window_add(hold[j][0], hold[j][1], residue(hold[j][0], hold[j][1]));
            }
            hn = 0;
            if (cnt != 0) {
                close_window();
            }
        } else if (gn != 0) {
            close_group();
        }
    }
};

void kernel_main() {
    volatile tt_l1_ptr kp::RelayCtrl* ctl = Out::ctrl();
    volatile tt_l1_ptr kp::SyncSampleRing* ring =
        reinterpret_cast<volatile tt_l1_ptr kp::SyncSampleRing*>(kSampleRingAddr);
    Model m;
    uint32_t next = 0;
    do {
        invalidate_l1_cache();
    } while (ring->tail == 0 && ring->done == 0);
    invalidate_l1_cache();
    const uint64_t r = ring->refclk, w = ring->wall8;
    m.anchor = Anchor{.R = r, .W = w, .r = static_cast<uint32_t>(r), .w = static_cast<uint32_t>(w)};
    constexpr uint32_t kChunk = 64, kRing = kp::kSyncSampleRingSamples;
    uint32_t stop = 0;
    while (true) {
        invalidate_l1_cache();
        const bool done = ring->done != 0;
        stop = ctl->stop != 0u ? kp::kSyncHeadStop : stop;
        const uint32_t tail = ring->tail;
        const uint32_t end = next + (tail - next < kChunk ? tail - next : kChunk);
        const uint32_t a = next % kRing;
        const uint32_t first = end - next < kRing - a ? end - next : kRing - a;
        m.feed(&ring->samples[a], &ring->samples[a + first]);
        m.feed(&ring->samples[0], &ring->samples[end - next - first]);
        next = end;
        std::atomic_thread_fence(std::memory_order_release);
        ring->head = next + stop;
        if (done && next == tail) {
            break;
        }
    }
    m.finish();
    m.out.close();
}
