// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/streaming_profiler/decode.hpp"

#include <algorithm>
#include <bit>
#include <cstring>
#include <limits>

#include <tt_stl/assert.hpp>

namespace tt::tt_metal::streaming_profiler {

namespace {

profiler::SpscRecConsts record_consts(const experimental::streaming_profiler::Core& core, int64_t offset) {
    TT_FATAL(
        core.logical.x < 256 && core.logical.y < 256 && core.physical.x < 256 && core.physical.y < 256 &&
            core.chip_id < 65536,
        "streaming profiler: core {} does not fit a record",
        core.physical.str());
    return profiler::SpscRecConsts{
        .coords =
            {static_cast<uint32_t>(core.logical.x) | (static_cast<uint32_t>(core.logical.y) << 8) |
                 (static_cast<uint32_t>(core.physical.x) << 16) | (static_cast<uint32_t>(core.physical.y) << 24),
             core.chip_id | (static_cast<uint32_t>(core.processor) << 16)},
        .offset = static_cast<uint64_t>(offset)};
}

constexpr uint64_t kEpoch = 1ull << 32;

// A wall-clock read is either right or exactly 2^32 high (kernel_profiler_streaming.hpp read_wall_clock), so the
// previous record borrowed the next epoch.
// noinline: inlined at its sites, the repair body raises the walk's register pressure.
__attribute__((noinline)) bool repair_prev_record(uint8_t* rec_start, bool zone, uint64_t prev, uint64_t ts) {
    if (prev < kEpoch || prev - ts > kEpoch || rec_start == nullptr) {
        return false;
    }
    uint64_t* rec = reinterpret_cast<uint64_t*>(rec_start);
    if (zone && rec[profiler::kSpscQwDuration] >= kEpoch) {
        rec[profiler::kSpscQwDuration] -= kEpoch;
    } else if (rec[profiler::kSpscQwTimestamp] >= kEpoch) {
        rec[profiler::kSpscQwTimestamp] -= kEpoch;
    } else {
        return false;
    }
    return true;
}

// A dur_hi zone's end may legitimately precede the previous record's when that record is the stall zone raised by
// this zone's own ring reservation.
template <profiler::PacketFormat F>
__attribute__((noinline)) uint64_t
run_repairs(const uint32_t* src, uint32_t n, uint8_t* first, uint64_t th_hi, uint64_t ts0, const SpscLane& lane) {
    constexpr uint32_t kRec = profiler::spsc_rec_bytes<F>;
    uint64_t order_regressions = 0;
    if constexpr (F.has_dur_hi()) {
        for (uint32_t k = 0; k < n; k++) {
            const uint32_t* r = src + F.words * k;
            if (r[F.dur_hi] != 0xFFFFFFFFu) {
                continue;
            }
            const uint64_t end = profiler::spsc_ts_at<F>(src, k, th_hi);
            const uint64_t dur = (static_cast<uint64_t>(r[F.dur_hi]) << 32) | r[F.dur_lo];
            if (end >= dur + kEpoch) {
                uint64_t* rec = reinterpret_cast<uint64_t*>(first + kRec * k);
                rec[profiler::kSpscQwDuration] = dur + kEpoch;
                rec[profiler::kSpscQwTimestamp] = end - rec[profiler::kSpscQwDuration];
            } else {
                order_regressions++;
            }
        }
    }
    const auto step = [&](uint64_t prev_ts, uint64_t ts, uint8_t* prev_rec, bool prev_zone) {
        if (!(ts < prev_ts)) {
            return;
        }
        if constexpr (F.has_dur_hi()) {
            const uint64_t* prev = reinterpret_cast<const uint64_t*>(prev_rec);
            if (prev != nullptr &&
                (static_cast<uint32_t>(prev[profiler::kSpscQwIds]) & PP_LOW27_MASK) == profiler::kSpscStallZoneId) {
                order_regressions++;
                return;
            }
        }
        const bool fixed = repair_prev_record(prev_rec, prev_zone, prev_ts, ts);
        order_regressions += !fixed;
    };
    step(lane.last_ts, ts0, lane.last_rec, lane.last_rec_zone != 0);
    if constexpr (!F.delta16) {
        for (uint32_t k = 1; k < n; k++) {
            step(
                profiler::spsc_ts_at<F>(src, k - 1, th_hi),
                profiler::spsc_ts_at<F>(src, k, th_hi),
                first + kRec * (k - 1),
                F.kind == profiler::Kind::Zone);
        }
    }
    return order_regressions;
}

template <profiler::PacketFormat F>
constexpr uint32_t sticky_value(const uint32_t* src) {
    static_assert(F.kind == profiler::Kind::Sticky);
    return F.value_word == 0 ? src[0] & PP_LOW27_MASK : src[F.value_word];
}

}  // namespace

void StreamDecoder::open(const CaptureContext::Device& dev) {
    const size_t num_cores = dev.lanes.size() / profiler::kSpscNRiscDecode;
    lanes_.assign(num_cores * profiler::kSpscNRiscDecode, {});
    consts_.assign(lanes_.size(), {});
    heads_.assign(num_cores, {});
    core_of_xy_.load(dev.core_xy);
    rec_consts_.reserve(dev.lanes.size());
    for (size_t li = 0; li < dev.lanes.size(); li++) {
        rec_consts_.push_back(record_consts(dev.lanes[li], dev.tile_offset[li / profiler::kSpscNRiscDecode]));
    }
}

// Decoding starts at whichever is later, the head mirror or the start of the extent. The mirror falls behind after an
// upstream loss, which the decoder accepts, and the extent falls behind after a late head write-back, whose overlap it
// skips.
StreamDecoder::Produced StreamDecoder::decode_frames(
    const uint32_t* frames, std::span<const uint32_t> frame_words, Out out) {
    namespace kp = kernel_profiler;
    using namespace profiler;
    // The kernels store through byte pointers, which would force a member reached through `this` to be reloaded after
    // every store.
    const SpscRecConsts* const lane_consts = rec_consts_.data();
    const uint64_t seq = batch_seq_;
    uint8_t* const zb = out.zones;
    uint8_t* const eb = out.events;
    uint8_t* const db = out.data;
    uint64_t zoff = 0;
    // Both point offsets share one register, data's in the high half. An array indexed by packet kind would live in
    // memory and put every update on a store-to-load chain.
    uint64_t pt_off = 0;
    uint64_t sz = 0;
    uint64_t unrepaired = 0;
    int64_t newest_ticks = std::numeric_limits<int64_t>::min();

    const uint32_t* frame = frames;
    for (const uint32_t frame_len : frame_words) {
        const uint32_t* ctrl = frame + kp::SPSC_SPAN_PREFIX_WORDS;
        const uint32_t core = core_of_xy_.find(frame[kp::SPSC_PREFIX_XY]);
        TT_FATAL(
            core != CoreTable::kNone,
            "streaming profiler: frame from NoC core {:#x}, which the capture did not seed",
            frame[kp::SPSC_PREFIX_XY]);
        uint32_t* const heads = heads_[core].data();
        SpscLane* const core_lanes = lanes_.data() + size_t{core} * kSpscNRiscDecode;
        SpscLaneConsts* const core_consts = consts_.data() + size_t{core} * kSpscNRiscDecode;
        const simde__m256i tail_v =
            simde_mm256_loadu_si256(reinterpret_cast<const simde__m256i*>(ctrl + kp::SPSC_WIRE_TAIL_0));
        const simde__m256i idle = simde_mm256_and_si256(
            simde_mm256_cmpeq_epi32(
                tail_v, simde_mm256_loadu_si256(reinterpret_cast<const simde__m256i*>(frame + kp::SPSC_PREFIX_HEAD_0))),
            simde_mm256_cmpeq_epi32(tail_v, simde_mm256_loadu_si256(reinterpret_cast<const simde__m256i*>(heads))));
        uint32_t busy = ~static_cast<uint32_t>(simde_mm256_movemask_ps(simde_mm256_castsi256_ps(idle))) &
                        ((1u << kSpscNRiscDecode) - 1u);
        uint64_t frame_ts = 0;
        uint32_t off = kp::SPSC_SPAN_PREFIX_WORDS + kp::SPSC_SPAN_WIRE_CTRL_WORDS;
        for (; busy != 0; busy &= busy - 1) {
            const uint32_t r = static_cast<uint32_t>(std::countr_zero(busy));
            const uint32_t lane = core * kSpscNRiscDecode + r;
            SpscLane& L = core_lanes[r];
            const uint32_t tail = ctrl[kp::SPSC_WIRE_TAIL_0 + r];
            const uint32_t start = frame[kp::SPSC_PREFIX_HEAD_0 + r];
            const uint32_t extent = tail - start;
            const uint32_t* p = nullptr;
            // A nearly full run that wraps arrives as the whole ring image, so its pad is phased for ring offset 0 and
            // its payload advances by the full ring.
            const bool ring_ordered = extent != 0 && kp::spsc_span_wrap_image(start, extent, kSpscRingCap);
            if (extent != 0) {
                off += kp::spsc_span_pack_pad(ring_ordered ? 0u : start, off);
                p = frame + off;
                off += ring_ordered ? kSpscRingCap : extent;
                TT_FATAL(
                    off <= frame_len,
                    "streaming profiler: frame control block places lane {} {} words past the frame's {}",
                    lane,
                    off - frame_len,
                    frame_len);
            }
            uint32_t head;
            if (L.seeded == 0) {
                L.seeded = 1;
                head = start;
            } else {
                head = heads[r];
                if (static_cast<int32_t>(start - head) > 0) {
                    head = start;
                    L.need_state = 1;
                }
            }
            heads[r] = tail;
            const uint32_t run = tail - head;
            if (run == 0) {
                continue;
            }
            SpscLaneConsts& lc = core_consts[r];
            uint32_t& th = L.timer_hi;
            uint32_t& pg = L.prog;
            uint64_t& cur = L.cursor;
            uint32_t lin[kSpscRingCap];
            if (ring_ordered) {
                const uint32_t hm = head & kSpscRingMask;
                const uint32_t first = kSpscRingCap - hm < run ? kSpscRingCap - hm : run;
                std::memcpy(lin, p + hm, first * sizeof(uint32_t));
                if (first < run) {
                    std::memcpy(lin + first, p, (run - first) * sizeof(uint32_t));
                }
                p = lin;
            } else {
                p += extent - run;
            }
            const uint32_t* const rd_end = ring_ordered ? p + run : frame + frame_len;
            uint32_t i = 0;
            uint32_t& na = L.need_anchor;
            if (L.need_state) {
                // The slots hold the state at the frame's tail. A sticky packet inside the run means the words before
                // it ran on an earlier value that isn't recorded here, so the run is picked up from its last sticky.
                th = ctrl[kp::SPSC_WIRE_TIMER_0 + r];
                pg = ctrl[kp::spsc_wire_prog_word(r)];
                uint32_t k = 0;
                while (k < run) {
                    const uint32_t t = pp_type(p[k]);
                    uint32_t w = kSpscWordsOfType[t];
                    if (w == 0) {
                        break;
                    }
                    if (kSpscDataMaskOfType[t] != 0) {
                        if (k + kSpscDataFormat.size_word >= run) {
                            break;
                        }
                        w += (p[k + kSpscDataFormat.size_word] >> kSpscDataFormat.size_shift) &
                             kSpscDataFormat.size_mask;
                    }
                    spsc_for_format<spsc_is_sticky>(t, [&]<PacketFormat F>() __attribute__((always_inline)) {
                        if (k + F.words <= run) {
                            (F.sets == PacketFormat::Sets::TimerHi ? th : pg) = sticky_value<F>(p + k);
                        }
                        i = k + w;
                    });
                    k += w;
                }
                L.need_state = 0;
                na = 1;
                spsc_lane_consts(lc, lane_consts[lane], th, pg);
            }
            uint64_t& lane_ts = L.last_ts;
            if (L.last_rec_seq != seq) {
                L.last_rec = nullptr;
                L.last_rec_seq = seq;
            }
            uint8_t*& lane_rec = L.last_rec;
            uint32_t& lane_rec_zone = L.last_rec_zone;
            const auto zones_emitted = [&](uint32_t n) {
                zoff += kSpscZoneBytes * n;
                lane_rec = zb + zoff - kSpscZoneBytes;
                lane_rec_zone = true;
            };
            while (i < run) {
                const uint32_t* const src = p + i;
                const uint32_t t = pp_type(src[0]);
                const uint32_t readable = static_cast<uint32_t>(rd_end - src);
                const uint32_t left = run - i;
                uint32_t got = 0;
                if ((kSpscPointTypes >> t) & 1u) {
                    // One branch for all of them, so random alternation between them doesn't mispredict.
                    bool blocked = false;
                    spsc_for_each_format<spsc_is_point>([&]<PacketFormat F>() __attribute__((always_inline)) {
                        if (!blocked && left >= 4u * F.words && readable >= 8u && spsc_run4<F>(src)) {
                            blocked = true;
                            uint8_t* const first = eb + static_cast<uint32_t>(pt_off);
                            const auto a = spsc_block<F>(src, readable, left / F.words, cur, lc, first);
                            pt_off += kSpscEventBytes * a.n;
                            if (a.n != 0) {
                                const uint64_t ts0 = spsc_ts_at<F>(src, 0, lc.th_hi);
                                if (__builtin_expect(ts0 < lane_ts || a.regress != 0, 0)) {
                                    unrepaired += run_repairs<F>(src, a.n, first, lc.th_hi, ts0, L);
                                }
                                lane_ts = a.ts_last;
                                lane_rec = eb + static_cast<uint32_t>(pt_off) - kSpscEventBytes;
                                lane_rec_zone = false;
                                got = F.words * a.n;
                            }
                        }
                    });
                    if (!blocked) {
                        const uint32_t dm = kSpscDataMaskOfType[t];
                        const uint32_t size_word = std::min<uint32_t>(kSpscDataFormat.size_word, readable - 1u);
                        const uint32_t n =
                            ((src[size_word] >> kSpscDataFormat.size_shift) & kSpscDataFormat.size_mask) & dm;
                        const uint32_t words = kSpscWordsOfType[t] + n;
                        if (left >= words) {
                            const uint64_t ts = lc.th_hi | src[kSpscDataFormat.ts_lo];
                            const uint32_t sh = dm & 32u;
                            uint8_t* const dst = (dm ? db : eb) + static_cast<uint32_t>(pt_off >> sh);
                            if (__builtin_expect(ts < lane_ts, 0)) {
                                unrepaired += run_repairs<kSpscDataFormat>(src, 1, dst, lc.th_hi, ts, L);
                            }
                            lane_ts = ts;
                            const uint32_t values = spsc_point(src, readable, n, lc, dst);
                            pt_off += static_cast<uint64_t>(
                                          kSpscEventBytes + ((kSpscDataBytes - kSpscEventBytes + values * 8u) & dm))
                                      << sh;
                            lane_rec = dst;
                            lane_rec_zone = false;
                            got = words;
                        }
                    }
                } else {
                    spsc_for_format<spsc_is_zone_or_sticky>(t, [&]<PacketFormat F>() __attribute__((always_inline)) {
                        if (left < F.words) {
                            return;
                        }
                        if constexpr (F.kind == Kind::Sticky) {
                            const uint32_t value = sticky_value<F>(src);
                            if constexpr (F.sets == PacketFormat::Sets::TimerHi) {
                                th = value;
                                spsc_lane_consts_th(lc, th);
                            } else {
                                pg = value;
                                spsc_lane_consts_prog(lc, pg);
                            }
                            got = F.words;
                        } else if constexpr (F.delta16) {
                            const bool long_run = left >= 4u * F.words && pp_type(src[3u * F.words]) == F.type;
                            const auto a = long_run ? spsc_block<F>(src, readable, left / F.words, cur, lc, zb + zoff)
                                                    : spsc_delta16_short<F>(src, left / F.words, cur, lc, zb + zoff);
                            if (a.n != 0) {
                                // Without a cursor the run decodes into the scratch space it would have used and isn't
                                // counted.
                                if (!na) {
                                    const uint64_t ts0 = cur + (src[1] >> 16);
                                    if (__builtin_expect(ts0 < lane_ts, 0)) {
                                        unrepaired += run_repairs<F>(src, a.n, zb + zoff, lc.th_hi, ts0, L);
                                    }
                                    lane_ts = a.ts_last;
                                    zones_emitted(a.n);
                                }
                                cur = a.ts_last;
                                got = F.words * a.n;
                            }
                        } else {
                            uint8_t* const first = zb + zoff;
                            const bool run = left > F.words && pp_type(src[F.words]) == F.type;
                            const SpscBlockResult a = run ? spsc_block<F>(src, readable, left / F.words, cur, lc, first)
                                                          : spsc_one<F>(src, readable, lc, first);
                            if (a.n == 0) {
                                return;
                            }
                            sz += a.stalls;
                            const uint64_t ts0 = spsc_ts_at<F>(src, 0, lc.th_hi);
                            if (__builtin_expect(
                                    (F.has_dur_hi() && a.wrapped != 0) || ts0 < lane_ts || a.regress != 0, 0)) {
                                unrepaired += run_repairs<F>(src, a.n, first, lc.th_hi, ts0, L);
                            }
                            lane_ts = a.ts_last;
                            zones_emitted(a.n);
                            if constexpr (F.reanchor) {
                                cur = a.ts_last;
                                na = 0;
                            }
                            got = F.words * a.n;
                        }
                    });
                }
                TT_FATAL(
                    got != 0,
                    "streaming profiler: undecodable word {:#010x} at offset {} of lane {}'s run of {}",
                    src[0],
                    i,
                    lane,
                    run);
                i += got;
            }
            frame_ts = std::max(frame_ts, lane_ts);
        }
        if (frame_ts != 0) {
            newest_ticks = std::max(
                newest_ticks,
                static_cast<int64_t>(frame_ts) + static_cast<int64_t>(lane_consts[core * kSpscNRiscDecode].offset));
        }
        frame += frame_len;
    }
    order_regressions += unrepaired;
    return Produced{
        static_cast<uint32_t>(zoff / kSpscZoneBytes),
        static_cast<uint32_t>(static_cast<uint32_t>(pt_off) / kSpscEventBytes),
        static_cast<uint32_t>(pt_off >> 32),
        newest_ticks,
        sz};
}

}  // namespace tt::tt_metal::streaming_profiler
