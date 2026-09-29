// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/streaming_profiler/sync_engine.hpp"

#include "impl/streaming_profiler/service.hpp"
#include "impl/streaming_profiler/sync_check.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <deque>
#include <limits>
#include <map>
#include <ranges>
#include <span>
#include <string>
#include <utility>

#include <tt_stl/assert.hpp>

#include <fmt/format.h>
#include <client/TracyProfiler.hpp>

namespace tt::tt_metal::streaming_profiler {

Potential solve_potential(
    std::span<const std::array<std::optional<size_t>, 2>> ends, std::span<const double> value, size_t unknowns) {
    const size_t n = unknowns;
    Potential p{.x = std::vector<double>(n, 0.0)};
    std::vector<double>& x = p.x;
    std::vector<double> N(n * n, 0.0);
    const auto at = [&](size_t i, size_t j) -> double& { return N[i * n + j]; };
    for (size_t i = 0; i < ends.size(); i++) {
        const auto [a, b] = ends[i];
        if (a) {
            at(*a, *a) += 1.0;
            x[*a] += value[i];
        }
        if (b) {
            at(*b, *b) += 1.0;
            x[*b] -= value[i];
        }
        if (a && b) {
            at(*a, *b) -= 1.0;
            at(*b, *a) -= 1.0;
        }
    }
    for (size_t j = 0; j < n; j++) {
        double d = at(j, j);
        for (size_t k = 0; k < j; k++) {
            d -= at(j, k) * at(j, k);
        }
        if (d <= 1e-9) {
            p.unreached = j;
            return p;
        }
        at(j, j) = std::sqrt(d);
        for (size_t i = j + 1; i < n; i++) {
            double v = at(i, j);
            for (size_t k = 0; k < j; k++) {
                v -= at(i, k) * at(j, k);
            }
            at(i, j) = v / at(j, j);
        }
    }
    for (size_t i = 0; i < n; i++) {
        for (size_t k = 0; k < i; k++) {
            x[i] -= at(i, k) * x[k];
        }
        x[i] /= at(i, i);
    }
    for (size_t i = n; i-- > 0;) {
        for (size_t k = i + 1; k < n; k++) {
            x[i] -= at(k, i) * x[k];
        }
        x[i] /= at(i, i);
    }
    return p;
}

// Every link weighs the same: its error is the fixed asymmetry each link training draws, which stamp precision doesn't
// reflect.
std::vector<RootXf> compose_on_root(const CaptureContext& ctx, std::span<const LineFit* const> lines) {
    const size_t devices = ctx.devices.size();
    std::vector<RootXf> to_root(devices);
    to_root[0] = RootXf{.scale = 1.0, .shift = 0.0, .ok = true};
    const std::vector<bool> reached =
        reached_from_root(ctx.links, devices, [&](size_t li) { return lines[li] != nullptr; });
    std::vector<std::optional<size_t>> idx(devices);
    size_t n = 0;
    for (size_t dev = 1; dev < devices; dev++) {
        if (reached[dev]) {
            idx[dev] = n++;
        }
    }
    if (n == 0) {
        return to_root;
    }
    std::vector<std::array<std::optional<size_t>, 2>> ends;
    std::vector<const LineFit*> used;
    std::vector<double> log_rate;
    for (size_t li = 0; li < lines.size(); li++) {
        const CaptureContext::Link& L = ctx.links[li];
        if (lines[li] != nullptr && reached[L.dev_a] && reached[L.dev_b]) {
            ends.push_back({idx[L.dev_a], idx[L.dev_b]});
            used.push_back(lines[li]);
            log_rate.push_back(std::log1p(lines[li]->slope));  // A_snd = A_rcv * (1 + rate)
        }
    }
    const std::vector<double> x = solve_potential(ends, log_rate, n).x;
    const auto A_of = [&](std::optional<size_t> i) { return i ? std::exp(x[*i]) : 1.0; };
    // At the link's midpoint the sender reads mid and the receiver reads mid + offset, and both are the same instant on
    // the root:
    // A_snd * mid + B_snd = A_rcv * (mid + offset) + B_rcv.
    std::vector<double> value(ends.size());
    for (size_t i = 0; i < ends.size(); i++) {
        value[i] = A_of(ends[i][1]) * (used[i]->x_mean + used[i]->y_mean) - A_of(ends[i][0]) * used[i]->x_mean;
    }
    const std::vector<double> B = solve_potential(ends, value, n).x;
    for (size_t dev = 1; dev < devices; dev++) {
        if (idx[dev]) {
            to_root[dev] = RootXf{.scale = A_of(idx[dev]), .shift = B[*idx[dev]], .ok = true};
        }
    }
    return to_root;
}

namespace kp = kernel_profiler;

namespace {

struct LocalPoint {
    uint64_t r, w8;
    uint32_t k8;
};
struct LocalPoints {
    std::array<LocalPoint, kp::kSyncLocalPoints> points{};
    uint32_t count = 0;
    auto begin() const { return points.begin(); }
    auto end() const { return points.begin() + count; }
};

LocalPoints unpack_local(const kp::SyncLocalRecord& rec) {
    const auto round = std::bit_cast<kp::SyncLocalRound>(rec.round);
    LocalPoints out{.count = kp::word_as<kp::SyncLocalMeta>(rec.meta).count};
    for (uint32_t i = 0; i < out.count; i++) {
        uint64_t r = rec.refclk, w = rec.wall8;
        if (i != 0) {
            const auto s = kp::word_as<kp::SyncLocalStep>(rec.steps[i - 1]);
            r += s.refclk;
            w += uint64_t{round.slope} * s.refclk + static_cast<uint64_t>(static_cast<int64_t>(s.wall_off));
        }
        out.points[i] = {r, w, round.k8[i]};
    }
    return out;
}

}  // namespace

struct SyncEngine::Impl {
    Impl(const CaptureContext& ctx, ClockMap& map) :
        map_(map), reader_(map.reader()), ctx_(ctx), chips_(ctx.devices.size()), links_(ctx.links.size()) {
        chips_[0].refclk_base = map_.root_base();
        for (size_t dev = 0; dev < chips_.size(); dev++) {
            chips_[dev].aiclk_plot = service().plot_name(fmt::format("AICLK chip{} (GHz)", ctx.devices[dev].chip_id));
        }
        for (size_t li = 0; li < ctx.links.size(); li++) {
            const CaptureContext::Link& L = ctx.links[li];
            side_of_[{L.dev_a, L.core_a}] = {li, true};
            side_of_[{L.dev_b, L.core_b}] = {li, false};
        }
        if (ctx.sync_check) {
            check_ = std::make_unique<SyncCheck>(ctx, map_);
            check_batch_ = SyncCheck::Batch(ctx.devices.size());
        }
    }

    struct Instant {
        int64_t r = 0;
        int64_t w8 = 0;
        uint32_t k8 = 0;
        double wall() const { return static_cast<double>(w8) / 8.0; }
        int64_t wall_tick() const { return (w8 + 4) >> 3; }
    };
    // Values are offsets from the chip's bases, so no double holds a count since power-on.
    struct Chip {
        std::optional<int64_t> refclk_base, wall_base;
        std::deque<Instant> pts;
        size_t published = 0;
        const char* aiclk_plot = nullptr;
    };
    using Round = std::array<std::optional<double>, 4>;
    struct Link {
        std::map<uint32_t, Round> pending;
        std::vector<RoundPoint> rounds;
        std::optional<LineFit> line;
        double solved_at = -std::numeric_limits<double>::infinity();
    };
    // Over 250 ms two chips' crystals stay on a line to about 0.4 ns, and averaging the ~25 rounds in that window
    // brings a round's ~0.6 ns of stamp noise down to about 0.12 ns.
    static constexpr double kLinkWindowTicks = 12'500'000.0;  // 250 ms
    // Rounds waiting for the other end's record: about 41 s of them at one round per 10 ms, or 4 s under the sync
    // check.
    static constexpr size_t kPendingMax = 4096;

    static int64_t base_of(std::optional<int64_t>& base, int64_t v) { return base ? *base : *(base = v); }

    void on_record(uint32_t dev, uint32_t core, const kp::SyncRecord& rec) {
        const uint32_t kind = kp::word_as<kp::SyncMeta>(rec.header.meta).kind;
        if (kind == kp::kSyncKindLink) {
            on_stamp(dev, core, rec.link);
            return;
        }
        TT_FATAL(
            kind == kp::kSyncKindLocal || kind == kp::kSyncKindRuler,
            "streaming profiler: device {} sent a sync record of kind {}",
            dev,
            kind);
        const LocalPoints points = unpack_local(rec.local);
        Chip& c = chips_[dev];
        if (kind == kp::kSyncKindRuler) {
            TT_FATAL(check_ != nullptr, "streaming profiler: device {} sent ruler readings with no sync check", dev);
            const float weight = kp::word_as<kp::SyncLocalMeta>(rec.header.meta).dense ? 1.0f : kp::kSyncRulerKeepEvery;
            const int64_t offset8 = 8 * ctx_.devices[dev].ruler_offset;
            for (const LocalPoint& s : points) {
                const auto r = static_cast<int64_t>(s.r);
                check_batch_.readings[dev].push_back(
                    {static_cast<double>(r - base_of(c.refclk_base, r)), static_cast<int64_t>(s.w8) + offset8, weight});
            }
            return;
        }
        for (const LocalPoint& s : points) {
            const auto r = static_cast<int64_t>(s.r), w8 = static_cast<int64_t>(s.w8);
            const Instant p{r - base_of(c.refclk_base, r), w8 - 8 * base_of(c.wall_base, w8 >> 3), s.k8};
            if (c.pts.empty() || (p.r > c.pts.back().r && p.w8 > c.pts.back().w8)) {
                c.pts.push_back(p);
            }
        }
    }

    bool on_batch_end() {
        bool moved = false;
        for (uint32_t dev = 0; dev < chips_.size(); dev++) {
            moved |= publish_dev(dev);
        }
        if (check_) {
            check_->submit(check_batch_);
        }
        return moved;
    }

    void on_stamp(uint32_t dev, uint32_t core, const kp::SyncLinkRecord& s) {
        const uint32_t role = kp::word_as<kp::SyncMeta>(s.meta).role;
        const auto side = side_of_.find({dev, core});
        TT_FATAL(
            side != side_of_.end(), "streaming profiler: device {} core {} sent a link stamp for no link", dev, core);
        const auto [li, sender] = side->second;
        TT_FATAL(
            role <= kp::kSyncRoleT2 && (role >= kp::kSyncRoleT1B) == sender,
            "streaming profiler: device {} core {} sent link stamp role {}, which its end does not stamp",
            dev,
            core,
            role);
        const uint32_t n = s.count;
        TT_FATAL(n != 0, "streaming profiler: device {} core {} sent a link stamp average of no stamps", dev, core);
        // T0 and T2 are on the sender's refclk and T1 and T1B on the receiver's, whichever end recorded them.
        const CaptureContext::Link& L = ctx_.links[li];
        const uint32_t domain = role == kp::kSyncRoleT0 || role == kp::kSyncRoleT2 ? L.dev_a : L.dev_b;
        const uint64_t sum = s.sum_ns;
        const int64_t ns = static_cast<int64_t>(s.first) + static_cast<int64_t>(sum / n);
        Link& link = links_[li];
        Round& r = link.pending[s.round];
        const int64_t base = kNsPerRefclk * base_of(chips_[domain].refclk_base, ns / kNsPerRefclk);
        r[role] = static_cast<double>(ns - base) + static_cast<double>(sum % n) / n;
        if (std::ranges::all_of(r, [](const std::optional<double>& t) { return t.has_value(); })) {
            // The ends may average different frames. The midpoints stay unbiased because the refclk relation is a line.
            const double mid_a = 0.5 * (*r[kp::kSyncRoleT0] + *r[kp::kSyncRoleT2]) / kNsPerRefclk;
            const double mid_b = 0.5 * (*r[kp::kSyncRoleT1] + *r[kp::kSyncRoleT1B]) / kNsPerRefclk;
            const RoundPoint p{mid_a, mid_b - mid_a};
            link.pending.erase(s.round);
            if (check_ && s.round % kp::kLinkSyncCheckSolveEvery != 0) {
                check_batch_.rounds.push_back({li, p});
            } else {
                link.rounds.push_back(p);
                solve(link, /*whole=*/false);
            }
        }
        while (link.pending.size() > kPendingMax) {
            link.pending.erase(link.pending.begin());
        }
    }

    void solve(Link& link, bool whole) {
        const std::vector<RoundPoint>& rounds = link.rounds;
        const double newest = rounds.back().mid;
        if (newest <= link.solved_at) {
            return;
        }
        const double start = newest - kLinkWindowTicks;
        size_t begin = rounds.size() - 1;
        while (begin > 0 && rounds[begin].mid > start) {
            begin--;
        }
        if (!whole && rounds[begin].mid > start) {
            return;
        }
        link.solved_at = newest;
        link.line = fit_line(std::span(rounds).subspan(begin), &RoundPoint::mid, &RoundPoint::off);
        link.rounds.erase(link.rounds.begin(), link.rounds.begin() + static_cast<std::ptrdiff_t>(begin));
        links_moved_ = true;
    }

    const std::vector<RootXf>& transforms() {
        if (links_moved_) {
            std::vector<const LineFit*> lines(links_.size());
            std::ranges::transform(links_, lines.begin(), [](const Link& l) { return l.line ? &*l.line : nullptr; });
            to_root_ = compose_on_root(ctx_, lines);
            links_moved_ = false;
        }
        return to_root_;
    }

    // Frozen nodes never move, since records were placed against them. Shifting fresh nodes to meet them and fading the
    // shift out is worse: every mismatch at a join becomes an offset carried for the whole fade.
    bool publish_dev(uint32_t dev) {
        Chip& c = chips_[dev];
        if (c.published == c.pts.size() || !map_.has_host_nodes()) {
            return false;
        }
        const RootXf& xf = transforms()[dev];
        if (!xf.ok) {
            return false;
        }
        const std::deque<Instant>& pts = c.pts;
        const size_t before = c.published;
        std::vector<std::pair<double, double>>& aiclk = aiclk_;
        aiclk.clear();
        const auto root_at = [&](const Instant& p) { return xf(static_cast<double>(p.r)); };
        for (; c.published < pts.size(); c.published++) {
            const Instant& p = pts[c.published];
            const double root = root_at(p);
            double tangent = 0.0;
            if (p.k8 != 0) {
                tangent = xf.scale * 8.0 / p.k8;
            } else if (pts.size() > 1) {
                const Instant& o = pts[c.published == 0 ? 1 : c.published - 1];
                tangent = (root_at(o) - root) / (o.wall() - p.wall());
            } else {
                break;
            }
            map_.append(
                dev,
                SyncNode{
                    .at = *c.wall_base + p.wall_tick(),
                    .value = root + tangent * (static_cast<double>(p.wall_tick()) - p.wall()),
                    .tangent = tangent});
            if (p.k8 != 0) {
                aiclk.emplace_back(root, p.k8 / 8.0 * kp::kEthRefclkHz * 1e-9);
            }
        }
        if (c.published == before) {
            return false;
        }
        if (check_) {
            const Instant& p = pts[c.published - 1];
            check_batch_.offsets[dev] = root_at(p) - static_cast<double>(p.r);
            check_batch_.aiclk[dev].insert(check_batch_.aiclk[dev].end(), aiclk.begin(), aiclk.end());
        }
        plot(c.aiclk_plot, aiclk);
        c.pts.erase(c.pts.begin(), c.pts.begin() + static_cast<std::ptrdiff_t>(c.published - 1));
        c.published = 1;
        return true;
    }

    void plot([[maybe_unused]] const char* name, [[maybe_unused]] std::span<const std::pair<double, double>> series) {
#if defined(TRACY_ENABLE)
        for (const auto& [root, value] : series) {
            if (const std::optional<double> tsc = map_.host_tsc_offset(reader_, root)) {
                tracy::Profiler::PlotDataAt(name, value, map_.tsc_base() + std::llround(*tsc));
            }
        }
#endif
    }

    void on_capture_end() {
        for (size_t li = 0; li < links_.size(); li++) {
            Link& link = links_[li];
            if (!link.line && link.rounds.size() >= 2) {
                solve(link, /*whole=*/true);
            }
            const CaptureContext::Link& L = ctx_.links[li];
            TT_FATAL(
                link.line,
                "streaming profiler: d2d sync link chip {} eth({},{}) -> chip {} eth({},{}) not solved: {} complete "
                "rounds and {} waiting for their other end",
                ctx_.devices[L.dev_a].chip_id,
                L.eth_a.x,
                L.eth_a.y,
                ctx_.devices[L.dev_b].chip_id,
                L.eth_b.x,
                L.eth_b.y,
                link.rounds.size(),
                link.pending.size());
        }
        for (uint32_t dev = 0; dev < chips_.size(); dev++) {
            publish_dev(dev);
        }
        for (uint32_t dev = 0; dev < chips_.size(); dev++) {
            map_.finish(dev);
        }
        if (check_) {
            check_->submit(check_batch_);
            check_->finish();
            check_->plots([this](const std::string& name, std::span<const std::pair<double, double>> series) {
                plot(service().plot_name(name), series);
            });
            check_.reset();
        }
    }

    ClockMap& map_;
    ClockMap::Reader reader_;
    CaptureContext ctx_;
    std::vector<Chip> chips_;
    std::vector<Link> links_;
    std::map<std::pair<uint32_t, uint32_t>, std::pair<size_t, bool>> side_of_;
    std::vector<RootXf> to_root_;
    bool links_moved_ = true;
    std::vector<std::pair<double, double>> aiclk_;
    std::unique_ptr<SyncCheck> check_;
    SyncCheck::Batch check_batch_;
};

SyncEngine::SyncEngine(const CaptureContext& ctx, ClockMap& map) : impl_(std::make_unique<Impl>(ctx, map)) {}
SyncEngine::~SyncEngine() = default;
void SyncEngine::on_record(uint32_t dev, uint32_t core, const kp::SyncRecord& rec) { impl_->on_record(dev, core, rec); }
bool SyncEngine::on_batch_end() { return impl_->on_batch_end(); }
void SyncEngine::on_capture_end() { impl_->on_capture_end(); }

}  // namespace tt::tt_metal::streaming_profiler
