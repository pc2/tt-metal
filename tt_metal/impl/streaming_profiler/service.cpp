// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/streaming_profiler/service.hpp"
#include "impl/streaming_profiler/receiver.hpp"
#include "impl/streaming_profiler/host_sync.hpp"
#include "impl/streaming_profiler/sync_engine.hpp"

#include <algorithm>
#include <array>
#include <cstdio>
#include <deque>
#include <limits>
#include <memory>
#include <thread>
#include <utility>
#include <pthread.h>

#include <tracy/Tracy.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt_stl/assert.hpp>
#include <tt_stl/indestructible.hpp>
#include <tt_stl/tt_pause.hpp>

#include "llrt/rtoptions.hpp"
#include "impl/streaming_profiler/csv_consumers.hpp"
#include "impl/streaming_profiler/decode.hpp"
#include "impl/streaming_profiler/tracy_consumer.hpp"

namespace tt::tt_metal::streaming_profiler {

namespace api = experimental::streaming_profiler;

namespace {

thread_local bool t_in_consumer = false;
thread_local api::detail::CallbackId t_consumer_id{};

}  // namespace

void set_thread_name(const std::string& name) {
    tracy::SetThreadName(name.c_str());
    char buf[16];
    std::snprintf(buf, sizeof(buf), "%s", name.c_str());
    pthread_setname_np(pthread_self(), buf);
}

struct Service::AttachedStream {
    StreamDecoder dec;
    uint64_t cursor = 0;
    uint64_t dropped = 0;
    uint32_t chip = 0;
    uint32_t dev = 0;
    uint32_t stream = 0;
    std::deque<Parked*> pending;
    int64_t cover_seen = std::numeric_limits<int64_t>::min();
};
struct Service::Attached {
    Receiver* producer = nullptr;
    std::vector<std::unique_ptr<AttachedStream>> streams;
    ClockMap::Reader reader;
};
struct Service::Parked {
    Attached* a;
    uint32_t dev;
    bool delivered;
    StreamDecoder::Out out;
    StreamDecoder::Produced n;
    uint64_t dropped;
};

constexpr uint32_t kBatchFrames = 64;

namespace {

// Many times one batch's worst case: 64 full frames is 4.1 MB of records per kind and 4.8 MB of data.
constexpr size_t kZonesArenaBytes = size_t{128} << 20;
constexpr size_t kEventsArenaBytes = size_t{64} << 20;
constexpr size_t kDataArenaBytes = size_t{64} << 20;

struct Ring {
    std::unique_ptr<uint8_t[]> buf;
    size_t cap = 0;
    size_t head = 0;
    size_t tail = 0;
    size_t wrap = 0;
    size_t live = 0;
    bool wrapped = false;
    explicit Ring(size_t bytes) : buf(std::make_unique_for_overwrite<uint8_t[]>(bytes)), cap(bytes) {}
    bool holds(const uint8_t* p) const { return p >= buf.get() && p < buf.get() + cap; }
    uint8_t* reserve(size_t n) {
        if (live == 0) {
            head = tail = 0;
            wrapped = false;
        }
        if (!wrapped) {
            if (head + n <= cap) {
                return buf.get() + head;
            }
            if (n <= tail) {
                wrap = head;
                head = 0;
                wrapped = true;
                return buf.get();
            }
            return nullptr;
        }
        return head + n <= tail ? buf.get() + head : nullptr;
    }
    // An empty range isn't live, so the range before it still ends where the next one starts.
    void commit(uint8_t* p, size_t used) {
        head = static_cast<size_t>(p - buf.get()) + used;
        live += used != 0;
    }
    void release(uint8_t* p, size_t used) {
        tail = static_cast<size_t>(p - buf.get()) + used;
        live--;
        if (wrapped && tail == wrap) {
            tail = 0;
            wrapped = false;
        }
    }
};

struct Arena {
    std::vector<Ring> rings;
    explicit Arena(size_t bytes) { rings.emplace_back(bytes); }
    uint8_t* reserve(size_t n) {
        if (uint8_t* p = rings.back().reserve(n)) {
            return p;
        }
        rings.emplace_back(std::max(rings.back().cap * 2, n));
        return rings.back().reserve(n);
    }
    void commit(uint8_t* p, size_t used) { rings.back().commit(p, used); }
    void release(uint8_t* p, size_t used) {
        if (used == 0) {
            return;
        }
        for (auto r = rings.begin(); r != rings.end(); ++r) {
            if (r->holds(p)) {
                r->release(p, used);
                if (r->live == 0 && r + 1 != rings.end()) {
                    rings.erase(r);
                }
                return;
            }
        }
    }
};

struct Arenas {
    Arena zones{kZonesArenaBytes}, events{kEventsArenaBytes}, data{kDataArenaBytes};

    StreamDecoder::Out reserve(const StreamDecoder::Capacity& cap) {
        return {zones.reserve(cap.zones), events.reserve(cap.events), data.reserve(cap.data)};
    }
    void commit(const StreamDecoder::Out& out, const StreamDecoder::Produced& n) {
        each(out, n, [](Arena& a, uint8_t* p, size_t used) { a.commit(p, used); });
    }
    void release(const StreamDecoder::Out& out, const StreamDecoder::Produced& n) {
        each(out, n, [](Arena& a, uint8_t* p, size_t used) { a.release(p, used); });
    }

private:
    template <typename Op>
    void each(const StreamDecoder::Out& out, const StreamDecoder::Produced& n, Op op) {
        op(zones, out.zones, size_t{n.zones} * profiler::kSpscZoneBytes);
        op(events, out.events, size_t{n.events} * profiler::kSpscEventBytes);
        op(data, out.data, n.data_bytes);
    }
};

}  // namespace

struct Service::Consumer {
    std::string name;
    BatchCallback cb;
    api::detail::CallbackId id{};
    std::thread thread;
    std::atomic<bool> stop{false};
    ControlQueue control;
};

Service::Service() : steady_(std::make_unique<SteadyClock>()) { init_site_registry(); }

SteadyClock& Service::steady() { return *steady_; }

const char* Service::plot_name(const std::string& name) {
    std::lock_guard lock(plot_names_mu_);
    return plot_names_.insert(name).first->c_str();
}

void Service::set_tile_clocks(ContextId context_id, uint32_t chip, TileClocks clocks) {
    std::lock_guard<std::mutex> lk(tile_clocks_mu_);
    const bool inserted = tile_clocks_.try_emplace({context_id, chip}, std::move(clocks)).second;
    TT_FATAL(inserted, "streaming profiler: device {} already has tile clocks", chip);
}

const TileClocks* Service::tile_clocks(ContextId context_id, uint32_t chip) const {
    std::lock_guard<std::mutex> lk(tile_clocks_mu_);
    const auto it = tile_clocks_.find({context_id, chip});
    return it == tile_clocks_.end() ? nullptr : &it->second;
}

Service& service() {
    static ttsl::Indestructible<Service> instance;
    return instance.get();
}

void Service::post_control(ControlQueue& q, Receiver* producer, bool attach) {
    {
        std::lock_guard<std::mutex> lk(q.mu);
        q.items.emplace_back(producer, attach);
    }
    q.pending.store(true, std::memory_order_release);
    pending_acks_++;
    wake_consumers();
}

void Service::wait_acks(std::unique_lock<std::mutex>& lk) {
    ack_cv_.wait(lk, [&] { return pending_acks_ == 0; });
}

api::detail::CallbackId Service::add_consumer(std::string name, BatchCallback cb) {
    TT_FATAL(!t_in_consumer, "streaming profiler: add_consumer must not be called from a consumer callback");
    std::lock_guard<std::mutex> topo(topology_mu_);
    auto c = std::make_unique<Consumer>();
    c->name = std::move(name);
    c->cb = std::move(cb);
    Consumer& ref = *c;
    std::unique_lock<std::mutex> lk(mu_);
    ref.id = api::detail::CallbackId{next_id_++};
    for (Receiver* p : producers_) {
        post_control(ref.control, p, true);
    }
    consumers_.push_back(std::move(c));
    ref.thread = std::thread(&Service::consumer_thread, this, std::ref(ref));
    wait_acks(lk);
    return ref.id;
}

void Service::remove_consumer(api::detail::CallbackId id) {
    const bool self = id == t_consumer_id;
    TT_FATAL(self || !t_in_consumer, "streaming profiler: a consumer callback may unregister only itself");
    std::unique_lock<std::mutex> topo(topology_mu_, std::defer_lock);
    if (!self) {
        topo.lock();
    }
    std::unique_ptr<Consumer> victim;
    {
        std::lock_guard<std::mutex> lk(mu_);
        auto it = std::find_if(consumers_.begin(), consumers_.end(), [&](const auto& c) { return c->id == id; });
        TT_FATAL(it != consumers_.end(), "streaming profiler: unknown consumer {}", static_cast<uint64_t>(id));
        if (self) {
            (*it)->stop.store(true, std::memory_order_release);
            return;
        }
        victim = std::move(*it);
        consumers_.erase(it);
    }
    victim->stop.store(true, std::memory_order_release);
    wake_consumers();
    victim->thread.join();
}

void Service::attach_producer(Receiver& producer) {
    std::lock_guard<std::mutex> topo(topology_mu_);
    std::unique_lock<std::mutex> lk(mu_);
    TT_FATAL(
        std::find(producers_.begin(), producers_.end(), &producer) == producers_.end(),
        "streaming profiler: producer attached twice");
    producers_.push_back(&producer);
    if (!sync_thread_.joinable()) {
        sync_thread_ = std::thread(&Service::sync_thread, this);
    }
    post_control(sync_control_, &producer, true);
    wait_acks(lk);
    for (auto& c : consumers_) {
        post_control(c->control, &producer, true);
    }
    wait_acks(lk);
}

void Service::detach_producer(Receiver& producer) {
    std::lock_guard<std::mutex> topo(topology_mu_);
    bool last = false;
    {
        std::unique_lock<std::mutex> lk(mu_);
        auto it = std::find(producers_.begin(), producers_.end(), &producer);
        TT_FATAL(it != producers_.end(), "streaming profiler: detaching a producer that is not attached");
        post_control(sync_control_, &producer, false);
        wait_acks(lk);
        for (auto& c : consumers_) {
            post_control(c->control, &producer, false);
        }
        wait_acks(lk);
        producers_.erase(it);
        last = producers_.empty();
    }
    if (last) {
        for (const auto& write : file_sinks_) {
            write();
        }
    }
}

bool Service::is_active() const {
    std::lock_guard<std::mutex> lk(mu_);
    return !producers_.empty();
}

void Service::register_builtin_consumers(const tt::llrt::RunTimeOptions& rtoptions) {
    std::call_once(builtins_once_, [&] {
        auto add_sink = [&]<typename C>(const char* name, const std::shared_ptr<C>& c) {
            builtin_callbacks_.push_back(api::RegisterCallback([c](const typename C::Batch& b) { (*c)(b); }, name));
        };
        auto add_file_sink = [&]<typename C>(const char* name, const std::shared_ptr<C>& c) {
            add_sink(name, c);
            file_sinks_.push_back([c] { c->write_csv(); });
        };
#if defined(TRACY_ENABLE)
        if (rtoptions.get_streaming_profiler_tracy_enabled()) {
            add_sink("tracy", std::make_shared<TracyConsumer>());
        }
#endif
        if (const std::string& path = rtoptions.get_streaming_profiler_zone_csv_path(); !path.empty()) {
            add_file_sink("zone-csv", std::make_shared<ZoneCsvConsumer>(path));
        }
        if (const std::string& path = rtoptions.get_streaming_profiler_ops_csv_path(); !path.empty()) {
            add_file_sink("ops-csv", std::make_shared<OpsCsvConsumer>(path));
        }
    });
}

class Service::StreamWalker {
public:
    virtual ~StreamWalker() = default;

    void run() {
        set_thread_name(name_);
        t_in_consumer = true;
        // A parked reader costs every publish a futex wake.
        constexpr uint32_t kSpinsBeforePark = 1000;
        uint32_t empty_passes = 0;
        while (true) {
            const uint32_t seen = service_.wake_token();
            if (control_.pending.load(std::memory_order_acquire)) {
                control_.pending.store(false, std::memory_order_release);
                handle_control(true);
            }
            if (stop_ != nullptr && stop_->load(std::memory_order_acquire)) {
                break;
            }
            bool any = false;
            for (auto& a : attached_) {
                any |= pass(*a);
            }
            any |= idle_pass();
            if (any) {
                empty_passes = 0;
                continue;
            }
            if (++empty_passes < kSpinsBeforePark) {
                ttsl::pause();
                continue;
            }
            service_.wait_wake(seen);
        }
        stopping();
        attached_.clear();
        handle_control(false);
    }

protected:
    StreamWalker(
        Service& service, ControlQueue& control, std::string name, const std::atomic<bool>* stop, bool sync_streams) :
        service_(service),
        frames_buf_(std::make_unique_for_overwrite<std::byte[]>(kFramesBytes)),
        control_(control),
        name_(std::move(name)),
        stop_(stop),
        sync_streams_(sync_streams) {}

    static constexpr size_t kFramesBytes = size_t{kBatchFrames} * profiler::kSpscMaxFrameWords * 4;

    virtual bool read(Attached& a, AttachedStream& s) = 0;
    virtual void closed(Attached& a) = 0;
    virtual bool idle_pass() { return false; }
    virtual void stopping() {}

    bool walk(Attached& a, AttachedStream& s, Walked& w) {
        const ProducerStream& ps = a.producer->streams()[s.stream];
        w = walk_frames(
            ps.fifo,
            s.cursor,
            ps.walked->load(std::memory_order_acquire),
            ps.marks,
            std::span<std::byte>(frames_buf_.get(), kFramesBytes),
            frame_words_,
            [&] { return a.producer->live_head(s.stream); });
        if (w.cursor == s.cursor) {
            return false;
        }
        s.cursor = w.cursor;
        s.dropped += w.dropped;
        return true;
    }

    Service& service_;
    std::vector<std::unique_ptr<Attached>> attached_;
    std::array<uint32_t, kBatchFrames> frame_words_{};
    std::unique_ptr<std::byte[]> frames_buf_;

private:
    void handle_control(bool run) {
        std::vector<std::pair<Receiver*, bool>> control;
        {
            std::lock_guard<std::mutex> lk(control_.mu);
            control.swap(control_.items);
        }
        if (run) {
            for (const auto& [p, is_attach] : control) {
                is_attach ? attach(p) : detach(p);
            }
        }
        std::lock_guard<std::mutex> lk(service_.mu_);
        service_.pending_acks_ -= control.size();
        service_.ack_cv_.notify_all();
    }

    void attach(Receiver* p) {
        auto a = std::make_unique<Attached>();
        a->producer = p;
        a->reader = p->clock_map().reader();
        const CaptureContext& ctx = p->capture_context();
        const auto streams = p->streams();
        for (uint32_t si = 0; si < streams.size(); si++) {
            const ProducerStream& ps = streams[si];
            if (ps.sync != sync_streams_) {
                continue;
            }
            auto s = std::make_unique<AttachedStream>();
            s->stream = si;
            s->cursor = ps.walked->load(std::memory_order_acquire);
            const CaptureContext::Device& dev = ctx.devices[ps.dev];
            s->chip = dev.chip_id;
            s->dev = ps.dev;
            s->dec.open(dev);
            a->streams.push_back(std::move(s));
        }
        attached_.push_back(std::move(a));
    }

    void detach(Receiver* p) {
        auto it = std::find_if(attached_.begin(), attached_.end(), [&](const auto& a) { return a->producer == p; });
        if (it == attached_.end()) {
            return;
        }
        Attached& a = **it;
        while (pass(a)) {
        }
        closed(a);
        attached_.erase(it);
    }

    // One batch per stream per pass, so no stream laps while another drains.
    bool pass(Attached& a) {
        bool any = false;
        for (const auto& s : a.streams) {
            any |= read(a, *s);
        }
        return any;
    }

    ControlQueue& control_;
    const std::string name_;
    const std::atomic<bool>* stop_;
    const bool sync_streams_;
};

class Service::SyncLoop : public StreamWalker {
public:
    explicit SyncLoop(Service& service) : StreamWalker(service, service.sync_control_, "sp-sync", nullptr, true) {}

private:
    bool read(Attached& a, AttachedStream& s) override {
        Walked w;
        if (!walk(a, s, w)) {
            return false;
        }
        namespace kp = kernel_profiler;
        const uint32_t dev = s.dev;
        const std::byte* p = frames_buf_.get();
        for (uint32_t i = 0; i < w.frames; i++) {
            const uint32_t* f = reinterpret_cast<const uint32_t*>(p);
            const uint32_t n = f[kp::SPSC_PREFIX_HEAD_0];
            const uint32_t core = s.dec.core_of_xy().find(f[kp::SPSC_PREFIX_XY]);
            TT_FATAL(
                core != CoreTable::kNone && kp::SPSC_SPAN_PREFIX_WORDS + n * kp::kSyncRecordWords <= frame_words_[i],
                "streaming profiler: a sync frame of chip {} names core {:#x} with {} records in {} words",
                s.chip,
                f[kp::SPSC_PREFIX_XY],
                n,
                frame_words_[i]);
            for (uint32_t r = 0; r < n; r++) {
                a.producer->sync().on_record(
                    dev, core, reinterpret_cast<const kp::SyncRecord*>(f + kp::SPSC_SPAN_PREFIX_WORDS)[r]);
            }
            p += size_t{frame_words_[i]} * 4;
        }
        if (a.producer->sync().on_batch_end()) {
            service_.wake_consumers();
        }
        return true;
    }

    bool idle_pass() override {
        for (const auto& a : attached_) {
            a->producer->host_sync().burst_if_due(a->producer->clock_map());
        }
        return false;
    }

    void closed(Attached& a) override {
        a.producer->host_sync().finish(a.producer->clock_map());
        a.producer->sync().on_capture_end();
        for (const auto& s : a.streams) {
            if (s->dropped != 0) {
                log_warning(
                    tt::LogMetal,
                    "[streaming profiler] the sync engine missed {} bytes of chip {}'s sync stream",
                    s->dropped,
                    s->chip);
            }
        }
    }
};

void Service::sync_thread() { SyncLoop(*this).run(); }

class Service::ConsumerLoop : public StreamWalker {
public:
    ConsumerLoop(Service& service, Consumer& c) :
        StreamWalker(service, c.control, "sp-con:" + c.name, &c.stop, false), c_(c) {}

private:
    bool read(Attached& a, AttachedStream& s) override {
        Walked w;
        if (!walk(a, s, w)) {
            return false;
        }
        size_t words = 0;
        for (uint32_t i = 0; i < w.frames; i++) {
            words += frame_words_[i];
        }
        const StreamDecoder::Out out = arenas_.reserve(StreamDecoder::out_capacity(words, w.frames));
        const StreamDecoder::Produced n = s.dec.decode_frames(
            reinterpret_cast<const uint32_t*>(frames_buf_.get()),
            std::span<const uint32_t>(frame_words_.data(), w.frames),
            out);
        s.dec.commit();
        arenas_.commit(out, n);
        parked_.push_back(Parked{.a = &a, .dev = s.dev, .delivered = false, .out = out, .n = n, .dropped = w.dropped});
        s.pending.push_back(&parked_.back());
        return true;
    }

    // The sync engine detached first, so the covers are final.
    void closed(Attached& a) override {
        for (auto& sp : a.streams) {
            for (Parked* pk : sp->pending) {
                deliver(*pk);
            }
            sp->pending.clear();
        }
        release_delivered();
        report(a);
    }

    bool idle_pass() override { return drain(); }

    void stopping() override {
        for (const auto& a : attached_) {
            report(*a);
        }
    }

    void report(const Attached& a) const {
        uint64_t dropped = 0, order_regressions = 0;
        for (const auto& s : a.streams) {
            order_regressions += s->dec.order_regressions;
            dropped += s->dropped;
        }
        if (dropped != 0) {
            log_warning(
                tt::LogMetal, "[streaming profiler] consumer \"{}\" missed {} bytes of frames", c_.name, dropped);
        }
        if (order_regressions != 0) {
            log_warning(
                tt::LogMetal, "[streaming profiler] consumer \"{}\": {} order regressions", c_.name, order_regressions);
        }
    }

    bool drain() {
        bool any = false;
        for (auto& a : attached_) {
            const ClockMap& map = a->producer->clock_map();
            for (auto& sp : a->streams) {
                AttachedStream& s = *sp;
                while (!s.pending.empty()) {
                    Parked& pk = *s.pending.front();
                    if (pk.n.newest_ticks > s.cover_seen) {
                        s.cover_seen = map.cover_ticks(s.dev);
                        if (pk.n.newest_ticks > s.cover_seen) {
                            break;
                        }
                    }
                    deliver(pk);
                    s.pending.pop_front();
                    any = true;
                }
            }
        }
        release_delivered();
        return any;
    }

    void deliver(Parked& pk) {
        place(pk);
        const api::detail::BatchData b{
            .zones = std::span<const api::Zone>(reinterpret_cast<const api::Zone*>(pk.out.zones), pk.n.zones),
            .timestamped_data = std::ranges::subrange(
                api::TimestampedData::iterator(reinterpret_cast<const std::byte*>(pk.out.data)),
                api::TimestampedData::iterator(reinterpret_cast<const std::byte*>(pk.out.data) + pk.n.data_bytes)),
            .events = std::span<const api::Event>(reinterpret_cast<const api::Event*>(pk.out.events), pk.n.events),
            .dropped_bytes = pk.dropped,
            .stall_count = pk.n.stalls};
        try {
            if (!c_.stop.load(std::memory_order_relaxed)) {
                c_.cb(b);
            }
        } catch (const std::exception& ex) {
            log_warning(tt::LogMetal, "[streaming profiler] consumer \"{}\" threw: {}", c_.name, ex.what());
        }
        pk.delivered = true;
    }

    // The decoder leaves each record's tile offset in its tsc_ slot, and placement overwrites it with the host time.
    void place(Parked& pk) {
        const ClockMap& map = pk.a->producer->clock_map();
        ClockMap::Reader& reader = pk.a->reader;
        const uint32_t dev = pk.dev;
        using namespace profiler;
        const auto place_point = [&](uint64_t* q) {
            q[kSpscQwTsc] = static_cast<uint64_t>(
                map.place_host(reader, dev, static_cast<int64_t>(q[kSpscQwTimestamp] + q[kSpscQwTsc])));
        };
        for (uint32_t i = 0; i < pk.n.zones; i++) {
            uint64_t* const q = reinterpret_cast<uint64_t*>(pk.out.zones + size_t{i} * kSpscZoneBytes);
            const int64_t wall = static_cast<int64_t>(q[kSpscQwTimestamp] + q[kSpscQwTsc]);
            const int64_t start = map.place_host(reader, dev, wall);
            const int64_t end = map.place_host(reader, dev, wall + static_cast<int64_t>(q[kSpscQwDuration]));
            q[kSpscQwTsc] = static_cast<uint64_t>(start);
            q[kSpscQwEndTsc] = static_cast<uint64_t>(end);
        }
        for (uint32_t i = 0; i < pk.n.events; i++) {
            place_point(reinterpret_cast<uint64_t*>(pk.out.events + size_t{i} * kSpscEventBytes));
        }
        for (uint8_t* p = pk.out.data; p < pk.out.data + pk.n.data_bytes;) {
            place_point(reinterpret_cast<uint64_t*>(p));
            p += reinterpret_cast<const api::TimestampedData*>(p)->size_bytes();
        }
    }

    void release_delivered() {
        while (!parked_.empty() && parked_.front().delivered) {
            arenas_.release(parked_.front().out, parked_.front().n);
            parked_.pop_front();
        }
    }

    Consumer& c_;
    Arenas arenas_;
    std::deque<Parked> parked_;
};

void Service::consumer_thread(Consumer& c) {
    t_consumer_id = c.id;
    ConsumerLoop(*this, c).run();
    // A self-unregistered consumer stays listed until here, so detach still waits on its acks.
    std::unique_ptr<Consumer> self;
    std::lock_guard<std::mutex> lk(mu_);
    const auto it = std::find_if(consumers_.begin(), consumers_.end(), [&](const auto& p) { return p.get() == &c; });
    if (it == consumers_.end()) {
        return;
    }
    {
        std::lock_guard<std::mutex> q(c.control.mu);
        pending_acks_ -= c.control.items.size();
        c.control.items.clear();
    }
    ack_cv_.notify_all();
    self = std::move(*it);
    consumers_.erase(it);
    self->thread.detach();
}

}  // namespace tt::tt_metal::streaming_profiler
