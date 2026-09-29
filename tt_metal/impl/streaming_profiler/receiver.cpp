// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/streaming_profiler/receiver.hpp"

#include "distributed/mesh_device_impl.hpp"
#include "impl/streaming_profiler/host_sync.hpp"
#include "impl/streaming_profiler/sync_engine.hpp"
#include <tt-metalium/mesh_device.hpp>

#include <algorithm>
#include <bit>
#include <chrono>
#include <thread>
#include <utility>
#include <vector>
#include <sys/prctl.h>
#include <numa.h>

#include <tt-logger/tt-logger.hpp>
#include <tt_stl/assert.hpp>

#include <tt-metalium/experimental/sockets/d2h_socket.hpp>

#include "context/metal_context.hpp"
#include "llrt/zone_meta.hpp"
#include "impl/streaming_profiler/spsc_packet.h"

namespace tt::tt_metal::streaming_profiler {

namespace {

// Return credits about once per relay push, not every poll.
constexpr uint32_t kAckBatchPages = 8 * profiler::kSpscMaxFramePages;
constexpr uint32_t kPageWords = kernel_profiler::SPSC_SPAN_PAGE_WORDS;
constexpr uint32_t kPageBytes = kPageWords * 4;
// Idle sleep. It only delays frames, never credits, and the FIFO holds over 1 ms of traffic.
constexpr uint32_t kProbeSleepCapUs = 100;

}  // namespace

Receiver::Receiver(
    std::unique_ptr<ClockMap> map,
    std::unique_ptr<SyncEngine> sync,
    std::unique_ptr<HostSync> host_sync,
    std::unique_ptr<DevicePrograms> programs) :
    map_(std::move(map)), sync_(std::move(sync)), host_sync_(std::move(host_sync)), programs_(std::move(programs)) {
    for (const CapturedSocket& cs : programs_->sockets()) {
        auto s = std::make_unique<Stream>();
        s->sock = cs;
        s->fifo = cs.socket->host_fifo();
        TT_FATAL(
            cs.socket->get_fifo_curr_size() == s->fifo.size() && s->fifo.size() % kPageBytes == 0 &&
                std::has_single_bit(s->fifo.size()),
            "streaming profiler: the host FIFO must be a power-of-two number of pages");
        s->capacity = s->fifo.size() / kPageBytes;
        s->marks = std::vector<std::atomic<uint64_t>>(s->fifo.size() / kMarkBytes);
        for (auto& m : s->marks) {
            m.store(UINT64_MAX, std::memory_order_relaxed);
        }
        streams_view_.push_back(
            {s->fifo, &s->walked_bytes, cs.dev, std::span<const std::atomic<uint64_t>>(s->marks), cs.sync});
        streams_.push_back(std::move(s));
    }
}

std::unique_ptr<Receiver> Receiver::create(const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
    auto& mc = MetalContext::instance(mesh_device->impl().get_context_id());
    service().register_builtin_consumers(mc.rtoptions());
    auto programs = std::make_unique<DevicePrograms>();
    if (!programs->boot(mesh_device)) {
        return nullptr;
    }
    const CaptureContext& ctx = programs->capture_context();
    auto host_sync = std::make_unique<HostSync>(mc.get_cluster(), ctx.devices.front().chip_id, service().steady());
    auto map = std::make_unique<ClockMap>(ctx.devices.size(), ClockMap::kSeriesNodes, host_sync->bases());
    auto sync = std::make_unique<SyncEngine>(ctx, *map);
    std::unique_ptr<Receiver> receiver(
        new Receiver(std::move(map), std::move(sync), std::move(host_sync), std::move(programs)));
    service().attach_producer(*receiver);
    receiver->attached_ = true;
    for (uint32_t d = 0; d < receiver->capture_context().devices.size(); d++) {
        std::vector<Stream*> owned;
        for (auto& s : receiver->streams_) {
            if (s->sock.dev == d) {
                owned.push_back(s.get());
            }
        }
        if (!owned.empty()) {
            receiver->ingest_threads_.emplace_back(&Receiver::ingest_thread, receiver.get(), std::move(owned));
        }
    }
    receiver->programs_->start();
    return receiver;
}

Receiver::~Receiver() {
    programs_->quiesce([this](uint32_t device_index, uint32_t socket_index, RelayState state) {
        stream(device_index, socket_index).relay.store(state, std::memory_order_release);
    });
    for (auto& t : ingest_threads_) {
        t.join();
    }
    if (attached_) {
        service().detach_producer(*this);
    }
    programs_->verify_completeness();
    log_report();
    const uint64_t foreign = llrt::ZoneMetaRegistry::instance().foreign_sections();
    const uint64_t collisions = llrt::ZoneMetaRegistry::instance().collisions();
    if (collisions != 0 || foreign != 0) {
        log_warning(
            tt::LogMetal,
            "[streaming profiler] zone names: {} id collisions, {} foreign metadata sections ignored (the JIT "
            "cache holds ELFs from a different .tt_zone_meta layout)",
            collisions,
            foreign);
    }
}

bool Receiver::poll(Stream& s) {
    // pages_available() counts from the last ack, so it includes pages already seen.
    const uint64_t arrived = s.acked + s.sock.socket->pages_available();
    TT_FATAL(
        arrived >= s.arrived,
        "streaming profiler: device {} socket {} bytes_sent went backwards ({} to {} pages)",
        s.sock.dev,
        s.sock.index,
        s.arrived,
        arrived);
    if (arrived == s.arrived) {
        return false;
    }
    s.arrived = arrived;
    s.arrived_bytes.store(arrived * kPageBytes, std::memory_order_release);
    s.fullest = std::max(s.fullest, arrived - s.acked);
    if (s.consumed < arrived) {
        __builtin_prefetch(s.page(s.consumed));
    }
    return true;
}

// The relay only reports bytes the PCIe tile acked, so every page below `arrived` is a header or inside the frame
// before it.
bool Receiver::walk_frame(Stream& s) {
    namespace kp = kernel_profiler;
    if (s.consumed >= s.arrived) {
        return false;
    }
    const uint32_t* page = reinterpret_cast<const uint32_t*>(s.page(s.consumed));
    const uint32_t w0 = page[0];
    const uint32_t w1 = page[1];
    TT_FATAL(
        pp_is_bulkspan(w0) && w1 >= kp::SPSC_SPAN_WIRE_CTRL_WORDS && w1 <= profiler::kSpscMaxPayloadWords,
        "streaming profiler: device {} socket {} page {} is not a frame header ({:#010x} {:#010x}); {} of {} pages "
        "landed",
        s.sock.dev,
        s.sock.index,
        s.consumed,
        w0,
        w1,
        s.arrived - s.acked,
        s.capacity);
    const uint32_t fw = kp::spsc_span_frame_words(w1);
    const uint64_t frame_pages = fw / kPageWords;
    if (s.arrived - s.consumed < frame_pages) {
        return false;
    }
    s.frames++;
    const uint64_t start = s.consumed * kPageBytes;
    if (start / kMarkBytes != s.mark_block) {
        s.mark_block = start / kMarkBytes;
        s.marks[s.mark_block % s.marks.size()].store(start, std::memory_order_release);
    }
    s.consumed += frame_pages;
    // Never read past `arrived`: the device may still be rewriting those pages.
    for (uint64_t p = s.consumed; p < s.arrived && p <= s.consumed + 3 * frame_pages; p += frame_pages) {
        __builtin_prefetch(s.page(p));
    }
    // Return credits as the walk earns them, so one slow socket doesn't let the others fill.
    if (s.consumed - s.acked >= kAckBatchPages) {
        s.sock.socket->pop(static_cast<uint32_t>(s.consumed - s.acked), true);
        s.acked = s.consumed;
    }
    return true;
}

// Credits only follow the walk, and a drained relay reports Done only once every byte it sent is credited, so Done
// means the walk has consumed every landed page.
bool Receiver::settle(Stream& s) {
    const uint64_t walked = s.consumed * kPageBytes;
    const bool published = walked != s.walked_bytes.load(std::memory_order_relaxed);
    if (published) {
        s.walked_bytes.store(walked, std::memory_order_release);
    }
    const RelayState relay = s.relay.load(std::memory_order_acquire);
    if (relay == RelayState::Running) {
        return published;
    }
    if (s.acked < s.consumed) {
        s.sock.socket->pop(static_cast<uint32_t>(s.consumed - s.acked), true);
        s.acked = s.consumed;
    }
    s.retired = relay == RelayState::Done;
    return published;
}

uint64_t Receiver::live_head(uint32_t stream) const {
    const Stream& s = *streams_[stream];
    return widen_head(s.arrived_bytes.load(std::memory_order_acquire), s.sock.socket->bytes_sent());
}

void Receiver::ingest_thread(std::vector<Stream*> streams) {
    std::string name = "sp-ingest:";
    for (Stream* s : streams) {
        name += std::to_string(s->sock.dev) + "." + std::to_string(s->sock.index) + ",";
    }
    name.pop_back();
    set_thread_name(name);
    prctl(PR_SET_TIMERSLACK, 1000);  // default 50 us slack would round every probe sleep up to it
    // Walked from the other NUMA node, the header reads' cache misses run at half rate and the FIFOs fill.
    if (const int node = streams.front()->sock.numa_node; node >= 0 && numa_available() != -1) {
        numa_run_on_node(node);
    }
    uint32_t sleep_us = 1;
    while (true) {
        bool any = false;
        for (Stream* s : streams) {
            any |= poll(*s);
        }
        for (bool progress = true; progress;) {
            progress = false;
            for (Stream* s : streams) {
                progress |= walk_frame(*s);
            }
        }
        bool published = false;
        for (Stream* s : streams) {
            published |= settle(*s);
        }
        if (published) {
            service().wake_consumers();
        }
        std::erase_if(streams, [](const Stream* s) { return s->retired; });
        if (streams.empty()) {
            break;
        }
        if (any) {
            sleep_us = 1;
            continue;
        }
        std::this_thread::sleep_for(std::chrono::microseconds(sleep_us));
        sleep_us = std::min(sleep_us + sleep_us / 4 + 1, kProbeSleepCapUs);
    }
}

Receiver::Stream& Receiver::stream(uint32_t device_index, uint32_t socket_index) {
    for (auto& s : streams_) {
        if (s->sock.dev == device_index && s->sock.index == socket_index) {
            return *s;
        }
    }
    TT_THROW("streaming profiler: no stream for device {} socket {}", device_index, socket_index);
}

void Receiver::log_report() const {
    uint64_t pages = 0, frames = 0;
    std::string fill;
    for (const auto& s : streams_) {
        fill += fmt::format("{}{}%", fill.empty() ? "" : " ", s->fullest * 100 / s->capacity);
        pages += s->arrived;
        frames += s->frames;
    }
    log_info(
        tt::LogMetal,
        "[streaming profiler] capture: {} frames, {:.1f} MB from {} device(s); FIFO high-water marks {}",
        frames,
        pages * static_cast<double>(kPageBytes) / 1e6,
        capture_context().devices.size(),
        fill);
}

}  // namespace tt::tt_metal::streaming_profiler
