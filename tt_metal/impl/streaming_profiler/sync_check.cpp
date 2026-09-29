// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/streaming_profiler/sync_check.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <deque>
#include <limits>
#include <map>
#include <mutex>
#include <ranges>
#include <thread>

#include <fmt/format.h>
#include <tt-logger/tt-logger.hpp>
#include <tt_stl/assert.hpp>

#include "impl/streaming_profiler/service.hpp"

namespace tt::tt_metal::streaming_profiler {

namespace {

struct ErrorStats {
    double n = 0.0, sum = 0.0, worst = 0.0;
    double worst_t = 0.0;

    void add(double ns, double t, double weight) {
        n += weight;
        sum += weight * ns;
        if (std::abs(ns) > worst) {
            worst = std::abs(ns);
            worst_t = t;
        }
    }
    double mean() const { return n > 0.0 ? sum / n : 0.0; }
};

struct ErrorHistogram {
    static constexpr int kBinsPerNs = 16;
    static constexpr int kRangeNs = 256;
    static constexpr int kCentre = kRangeNs * kBinsPerNs;
    ErrorStats stats;
    std::vector<double> bins = std::vector<double>(2 * kCentre, 0.0);

    void add(double ns, double t, double weight) {
        const int b = static_cast<int>(std::floor(ns * kBinsPerNs)) + kCentre;
        if (b >= 0 && b < 2 * kCentre) {
            bins[b] += weight;
        }
        stats.add(ns, t, weight);
    }
    double abs_quantile(double q) const {
        if (stats.n <= 0.0) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        double acc = 0.0;
        for (int i = 0; i < kCentre; i++) {
            acc += bins[kCentre + i] + bins[kCentre - 1 - i];
            if (acc >= q * stats.n) {
                return (i + 0.5) / kBinsPerNs;
            }
        }
        return stats.worst;
    }
};

// AICLK weighted by how long each value held.
struct ClockStats {
    static constexpr double kBinMhz = 50.0;
    std::optional<std::pair<double, double>> last;
    double span = 0.0, sum = 0.0, sum2 = 0.0;
    double lo = std::numeric_limits<double>::infinity(), hi = 0.0;
    uint64_t changes = 0;
    std::map<int, double> by_bin;

    void add(double t, double mhz) {
        if (last) {
            const auto [t0, f] = *last;
            const double w = t - t0;
            if (w > 0.0) {
                span += w;
                sum += w * f;
                sum2 += w * f * f;
                lo = std::min(lo, f);
                hi = std::max(hi, f);
                by_bin[static_cast<int>(std::floor(f / kBinMhz))] += w;
            }
            changes += mhz != f ? 1 : 0;
        }
        last = {t, mhz};
    }
};

}  // namespace

// A link's fixed asymmetry hits the reference and the sync equally, so it cancels. It and the refclk read path shared
// with the tracker are all the check can't see.
struct SyncCheck::Impl {
    Impl(const CaptureContext& ctx, const ClockMap& map) :
        ctx_(ctx), map_(map), links_(ctx.links.size()), chips_(ctx.devices.size()) {
        for (uint32_t a = 0; a < chips_.size(); a++) {
            for (uint32_t b = a + 1; b < chips_.size(); b++) {
                pairs_.push_back(Pair{.a = a, .b = b});
            }
        }
        chips_[0].offset = 0.0;
        worker_ = std::thread([this] { run(); });
    }
    ~Impl() {
        if (worker_.joinable()) {
            stop();
        }
    }

    void submit(Batch& b) {
        {
            std::lock_guard lock(mu_);
            in_.rounds.insert(in_.rounds.end(), b.rounds.begin(), b.rounds.end());
            for (size_t d = 0; d < chips_.size(); d++) {
                in_.readings[d].insert(in_.readings[d].end(), b.readings[d].begin(), b.readings[d].end());
                in_.aiclk[d].insert(in_.aiclk[d].end(), b.aiclk[d].begin(), b.aiclk[d].end());
                if (b.offsets[d]) {
                    in_.offsets[d] = b.offsets[d];
                }
            }
        }
        b.rounds.clear();
        for (std::vector<Reading>& r : b.readings) {
            r.clear();
        }
        for (auto& a : b.aiclk) {
            a.clear();
        }
        std::ranges::fill(b.offsets, std::nullopt);
    }
    void finish() {
        stop();
        report();
    }
    void plots(const PlotFn& plot) const {
        TT_FATAL(!worker_.joinable(), "streaming profiler: the sync check's plots were read before it finished");
        const auto series = [&](const std::string& name, const WorstByMs& w) {
            std::vector<std::pair<double, double>> pts;
            for (size_t i = 0; i < w.ns.size(); i++) {
                if (w.ns[i] >= 0.0f) {
                    pts.emplace_back(
                        (static_cast<double>(w.first + static_cast<int64_t>(i)) + 0.5) * kStepTicks, w.ns[i]);
                }
            }
            plot(name, pts);
        };
        series("sync error bound (ns)", worst_);
        for (uint32_t d = 0; d < chips_.size(); d++) {
            series(fmt::format("sync error bound chip{} (ns)", ctx_.devices[d].chip_id), chips_[d].worst);
        }
    }

    // A line through 50 ms of held-out rounds (about 45 of them) is good to about 0.1 ns, and two crystals stay on a
    // line over that span to about 0.02 ns.
    static constexpr double kStepTicks = 50'000.0;         // 1 ms
    static constexpr double kHalfSpanTicks = 1'250'000.0;  // 25 ms
    // A span only gets a line if it has rounds within 5 ms of both edges and at least 20 of its ~45 rounds, so no line
    // extrapolates across a gap in its link's rounds.
    static constexpr double kEdgeTicks = 250'000.0;  // 5 ms
    static constexpr size_t kMinLineRounds = 20;
    static constexpr double kMaxGapTicks = 2'500.0;  // 50 us

    struct LinkRef {
        std::deque<RoundPoint> rounds;
        std::map<int64_t, LineFit> lines;
        std::optional<int64_t> next;
    };
    struct Sample {
        double t;
        float err;
        float weight;
    };
    struct WorstByMs {
        int64_t first = 0;
        std::deque<float> ns;
        void add(double t, double e) {
            const auto ms = static_cast<int64_t>(std::floor(t / kStepTicks));
            if (ns.empty()) {
                first = ms;
            }
            for (; ms < first; first--) {
                ns.push_front(-1.0f);
            }
            while (ms >= first + static_cast<int64_t>(ns.size())) {
                ns.push_back(-1.0f);
            }
            float& v = ns[ms - first];
            v = std::max(v, static_cast<float>(std::abs(e)));
        }
    };
    struct Chip {
        std::deque<Reading> waiting;
        std::optional<double> last_r;
        std::deque<Sample> placed;
        uint64_t popped = 0;
        uint64_t no_reference = 0, no_node = 0;
        std::optional<double> offset;
        ErrorStats err;
        WorstByMs worst;
        ClockStats clock;
        const Sample& at(uint64_t i) const { return placed[i - popped]; }
        uint64_t end() const { return popped + placed.size(); }
    };
    struct Pair {
        uint32_t a = 0, b = 0;
        uint64_t ia = 0, jb = 0;
        ErrorStats err;
    };
    enum class Ref { kReady, kWait, kNone };

    static int64_t step_of(double refclk) { return static_cast<int64_t>(std::llround(refclk / kStepTicks)); }
    int64_t sender_step(size_t li, double root) const { return step_of(root - *chips_[ctx_.links[li].dev_a].offset); }
    double root_step(uint32_t dev, double r) const { return (r + *chips_[dev].offset) / kStepTicks; }

    static void fit_steps(LinkRef& ref, bool all) {
        auto& rs = ref.rounds;
        if (rs.empty()) {
            return;
        }
        if (!ref.next) {
            ref.next = step_of(rs.front().mid);
        }
        const double newest = rs.back().mid;
        size_t lo = 0;
        for (;; ++*ref.next) {
            const double centre = static_cast<double>(*ref.next) * kStepTicks;
            if (all ? centre - kHalfSpanTicks > newest : newest < centre + kHalfSpanTicks) {
                break;
            }
            while (lo < rs.size() && rs[lo].mid < centre - kHalfSpanTicks) {
                lo++;
            }
            size_t hi = lo;
            while (hi < rs.size() && rs[hi].mid < centre + kHalfSpanTicks) {
                hi++;
            }
            const size_t n = hi - lo;
            if (n < kMinLineRounds || rs[lo].mid > centre - kHalfSpanTicks + kEdgeTicks ||
                rs[hi - 1].mid < centre + kHalfSpanTicks - kEdgeTicks) {
                continue;
            }
            ref.lines[*ref.next] =
                fit_line(rs | std::views::drop(lo) | std::views::take(n), &RoundPoint::mid, &RoundPoint::off);
        }
        rs.erase(rs.begin(), rs.begin() + static_cast<std::ptrdiff_t>(lo));
    }
    // A round completes when its last stamp record arrives, which can be a round or two after a later round.
    void absorb(const Round& in) {
        LinkRef& ref = links_[in.li];
        ref.rounds.insert(std::ranges::upper_bound(ref.rounds, in.p.mid, {}, &RoundPoint::mid), in.p);
    }
    Ref mesh_at(int64_t k, const std::vector<RootXf>*& out) {
        auto it = meshes_.find(k);
        if (it == meshes_.end()) {
            const double centre = static_cast<double>(k) * kStepTicks;
            std::vector<const LineFit*> lines(ctx_.links.size(), nullptr);
            bool complete = true;
            for (size_t li = 0; li < lines.size() && complete; li++) {
                const bool known = chips_[ctx_.links[li].dev_a].offset.has_value();
                const LinkRef& ref = links_[li];
                const int64_t ks = known ? sender_step(li, centre) : 0;
                if (!finishing_seen_ && (!known || !ref.next || ks >= *ref.next)) {
                    return Ref::kWait;
                }
                const auto line = known ? ref.lines.find(ks) : ref.lines.end();
                complete = line != ref.lines.end();
                lines[li] = complete ? &line->second : nullptr;
            }
            std::vector<RootXf> to_root;
            if (complete) {
                to_root = compose_on_root(ctx_, lines);
            }
            it = meshes_.emplace(k, std::move(to_root)).first;
        }
        out = &it->second;
        return out->empty() ? Ref::kNone : Ref::kReady;
    }
    Ref reference(uint32_t dev, double r, double& root) {
        if (dev == 0) {
            root = r;
            return Ref::kReady;
        }
        if (!chips_[dev].offset) {
            return finishing_seen_ ? Ref::kNone : Ref::kWait;
        }
        const double t = root_step(dev, r);
        const auto k = static_cast<int64_t>(std::floor(t));
        std::array<const std::vector<RootXf>*, 2> m{};
        for (int64_t i = 0; i < 2; i++) {
            if (const Ref s = mesh_at(k + i, m[i]); s != Ref::kReady) {
                return s;
            }
        }
        const double v0 = (*m[0])[dev](r);
        const double v1 = (*m[1])[dev](r);
        root = v0 + (t - static_cast<double>(k)) * (v1 - v0);
        return Ref::kReady;
    }
    bool place(uint32_t dev, ClockMap::Reader& reader) {
        Chip& c = chips_[dev];
        const int64_t cover = map_.cover_ticks(dev);
        bool moved = false;
        for (; !c.waiting.empty(); c.waiting.pop_front()) {
            const Reading& rd = c.waiting.front();
            const int64_t wall = rd.w8 >> 3;
            double root_ref = 0.0;
            const Ref ref = wall >= cover ? Ref::kWait : reference(dev, rd.r, root_ref);
            if (ref == Ref::kWait) {
                break;
            }
            moved = true;
            c.last_r = rd.r;
            if (ref == Ref::kNone) {
                c.no_reference++;
                continue;
            }
            const std::optional<double> placed =
                map_.root_offset(reader, dev, wall, static_cast<double>(rd.w8 & 7) / 8.0);
            if (!placed) {
                c.no_node++;
                continue;
            }
            const Sample x{root_ref, static_cast<float>((*placed - root_ref) * kNsPerRefclk), rd.weight};
            c.placed.push_back(x);
            c.err.add(x.err, x.t, x.weight);
            worst_.add(x.t, x.err);
            c.worst.add(x.t, x.err);
        }
        return moved;
    }
    bool pair_up(double until) {
        bool moved = false;
        for (Pair& p : pairs_) {
            Chip &A = chips_[p.a], &B = chips_[p.b];
            const uint64_t eb = B.end();
            if (eb == B.popped) {
                continue;
            }
            for (; p.ia < A.end() && A.at(p.ia).t < until; p.ia++) {
                const Sample& sa = A.at(p.ia);
                while (p.jb + 1 < eb && B.at(p.jb + 1).t <= sa.t) {
                    p.jb++;
                }
                const Sample* nb = &B.at(p.jb);
                if (p.jb + 1 < eb && std::abs(B.at(p.jb + 1).t - sa.t) < std::abs(nb->t - sa.t)) {
                    nb = &B.at(p.jb + 1);
                }
                const double gap = std::abs(nb->t - sa.t);
                if (gap > kMaxGapTicks) {
                    continue;
                }
                const double e = static_cast<double>(nb->err) - static_cast<double>(sa.err);
                p.err.add(e, sa.t, sa.weight);
                pooled_.add(e, sa.t, sa.weight);
                worst_.add(sa.t, e);
                A.worst.add(sa.t, e);
                B.worst.add(sa.t, e);
                moved = true;
            }
        }
        for (uint32_t d = 0; d < chips_.size(); d++) {
            Chip& c = chips_[d];
            uint64_t keep = c.end();
            for (const Pair& p : pairs_) {
                keep = std::min(keep, p.a == d ? p.ia : p.b == d ? p.jb : keep);
            }
            for (; c.popped < keep; c.popped++) {
                c.placed.pop_front();
            }
        }
        return moved;
    }
    void prune() {
        int64_t k = std::numeric_limits<int64_t>::max();
        for (uint32_t d = 0; d < chips_.size(); d++) {
            const Chip& c = chips_[d];
            if (!c.offset || (c.waiting.empty() && !c.last_r)) {
                return;
            }
            const double r = c.waiting.empty() ? *c.last_r : c.waiting.front().r;
            k = std::min(k, static_cast<int64_t>(std::floor(root_step(d, r))) - 1);
        }
        meshes_.erase(meshes_.begin(), meshes_.lower_bound(k));
        for (size_t li = 0; li < links_.size(); li++) {
            auto& lines = links_[li].lines;
            lines.erase(lines.begin(), lines.lower_bound(sender_step(li, static_cast<double>(k) * kStepTicks) - 1));
        }
    }
    void run() {
        set_thread_name("sp-check");
        ClockMap::Reader reader = map_.reader();
        Batch got(chips_.size());
        while (true) {
            {
                std::lock_guard lock(mu_);
                std::swap(got.rounds, in_.rounds);
                std::swap(got.readings, in_.readings);
                std::swap(got.aiclk, in_.aiclk);
                for (size_t d = 0; d < chips_.size(); d++) {
                    if (in_.offsets[d]) {
                        chips_[d].offset = in_.offsets[d];
                    }
                }
                finishing_seen_ = finishing_;
            }
            bool moved = !got.rounds.empty();
            for (const Round& r : got.rounds) {
                absorb(r);
            }
            got.rounds.clear();
            for (LinkRef& ref : links_) {
                fit_steps(ref, finishing_seen_);
            }
            double newest = std::numeric_limits<double>::infinity();
            for (uint32_t d = 0; d < chips_.size(); d++) {
                Chip& c = chips_[d];
                c.waiting.insert(c.waiting.end(), got.readings[d].begin(), got.readings[d].end());
                got.readings[d].clear();
                for (const auto& [t, ghz] : got.aiclk[d]) {
                    c.clock.add(t, 1e3 * ghz);
                }
                got.aiclk[d].clear();
                moved = place(d, reader) || moved;
                newest =
                    std::min(newest, c.placed.empty() ? -std::numeric_limits<double>::infinity() : c.placed.back().t);
            }
            moved = pair_up(finishing_seen_ ? std::numeric_limits<double>::infinity() : newest - kMaxGapTicks) || moved;
            if (finishing_seen_) {
                return;
            }
            prune();
            if (!moved) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
    }
    void stop() {
        {
            std::lock_guard lock(mu_);
            finishing_ = true;
        }
        worker_.join();
    }
    void report() const {
        double worst = 0.0, worst_t = 0.0, worst_mean = 0.0;
        std::string worst_of;
        size_t measured = 0;
        for (const Pair& p : pairs_) {
            if (p.err.n <= 0.0) {
                continue;
            }
            measured++;
            worst_mean = std::max(worst_mean, std::abs(p.err.mean()));
            if (p.err.worst > worst) {
                worst = p.err.worst;
                worst_t = p.err.worst_t;
                worst_of = fmt::format("chip {} - chip {}", ctx_.devices[p.a].chip_id, ctx_.devices[p.b].chip_id);
            }
        }
        uint64_t no_node = 0;
        for (uint32_t d = 0; d < chips_.size(); d++) {
            const Chip& c = chips_[d];
            no_node += c.no_node;
            if (c.err.n > 0.0 && c.err.worst > worst) {
                worst = c.err.worst;
                worst_t = c.err.worst_t;
                worst_of = fmt::format("chip {} against the reference", ctx_.devices[d].chip_id);
            }
        }
        if (no_node != 0) {
            log_warning(
                tt::LogMetal,
                "[streaming profiler] sync check: {} ruler readings the clock map could not place",
                no_node);
        }
        if (pooled_.stats.n <= 0.0) {
            log_warning(tt::LogMetal, "[streaming profiler] sync check: no chip pair measured");
            return;
        }
        log_info(
            tt::LogMetal,
            "[streaming profiler] sync check: chip-to-chip error of the global timeline, a bound, over {} of {} chip "
            "pairs and {:.0f} samples: |err| p50 {:.2f}, p99 {:.2f}, p99.9 {:.2f}, max {:.2f} ns ({}, {:.3f} s in); "
            "the largest pair's mean {:.2f} ns",
            measured,
            pairs_.size(),
            pooled_.stats.n,
            pooled_.abs_quantile(0.5),
            pooled_.abs_quantile(0.99),
            pooled_.abs_quantile(0.999),
            worst,
            worst_of,
            worst_t / kernel_profiler::kEthRefclkHz,
            worst_mean);
        for (uint32_t d = 0; d < chips_.size(); d++) {
            const Chip& c = chips_[d];
            if (c.err.n > 0.0) {
                log_info(
                    tt::LogMetal,
                    "[streaming profiler] sync check chip {}: {:.0f} readings against the reference, mean {:+.2f} ns, "
                    "max {:.2f} ns ({:.3f} s in); unplaced {} with no reference, {} with no node",
                    ctx_.devices[d].chip_id,
                    c.err.n,
                    c.err.mean(),
                    c.err.worst,
                    c.err.worst_t / kernel_profiler::kEthRefclkHz,
                    c.no_reference,
                    c.no_node);
            }
            const ClockStats& k = c.clock;
            if (k.span > 0.0) {
                std::string bins;
                for (const auto& [b, w] : k.by_bin) {
                    if (w >= 1e-3 * k.span) {
                        bins += fmt::format(
                            "{}{:.0f} {:.1f}%", bins.empty() ? "" : ", ", b * ClockStats::kBinMhz, 100.0 * w / k.span);
                    }
                }
                const double mean = k.sum / k.span;
                log_info(
                    tt::LogMetal,
                    "[streaming profiler] sync check chip {} AICLK: mean {:.0f} MHz, sd {:.0f}, {:.0f}-{:.0f} MHz, "
                    "{:.1f} changes/s; time by {:.0f} MHz bin: {}",
                    ctx_.devices[d].chip_id,
                    mean,
                    std::sqrt(std::max(0.0, k.sum2 / k.span - mean * mean)),
                    k.lo,
                    k.hi,
                    static_cast<double>(k.changes) / (k.span / kernel_profiler::kEthRefclkHz),
                    ClockStats::kBinMhz,
                    bins);
            }
        }
    }

    const CaptureContext ctx_;
    const ClockMap& map_;
    std::vector<LinkRef> links_;
    std::vector<Chip> chips_;
    std::vector<Pair> pairs_;
    std::map<int64_t, std::vector<RootXf>> meshes_;
    ErrorHistogram pooled_;
    WorstByMs worst_;
    bool finishing_seen_ = false;
    std::mutex mu_;
    Batch in_{chips_.size()};
    bool finishing_ = false;
    std::thread worker_;
};

SyncCheck::SyncCheck(const CaptureContext& ctx, const ClockMap& map) : impl_(std::make_unique<Impl>(ctx, map)) {}
SyncCheck::~SyncCheck() = default;
void SyncCheck::submit(Batch& b) { impl_->submit(b); }
void SyncCheck::finish() { impl_->finish(); }
void SyncCheck::plots(const PlotFn& plot) const { impl_->plots(plot); }

}  // namespace tt::tt_metal::streaming_profiler
