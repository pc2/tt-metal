// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <thread>
#include <vector>

#include "impl/streaming_profiler/device_programs.hpp"
#include "impl/streaming_profiler/service.hpp"

namespace tt::tt_metal {

namespace distributed {
class MeshDevice;
}  // namespace distributed

namespace streaming_profiler {

class ClockMap;
class HostSync;
class SyncEngine;

class Receiver {
public:
    static std::unique_ptr<Receiver> create(const std::shared_ptr<distributed::MeshDevice>& mesh_device);
    ~Receiver();

    Receiver(const Receiver&) = delete;
    Receiver& operator=(const Receiver&) = delete;

    std::span<const ProducerStream> streams() const { return streams_view_; }
    const CaptureContext& capture_context() const { return programs_->capture_context(); }
    // Callable from any consumer thread.
    uint64_t live_head(uint32_t stream) const;
    const ClockMap& clock_map() const { return *map_; }
    ClockMap& clock_map() { return *map_; }
    SyncEngine& sync() { return *sync_; }
    HostSync& host_sync() { return *host_sync_; }

private:
    Receiver(
        std::unique_ptr<ClockMap> map,
        std::unique_ptr<SyncEngine> sync,
        std::unique_ptr<HostSync> host_sync,
        std::unique_ptr<DevicePrograms> programs);

    struct Stream {
        CapturedSocket sock;
        std::span<const std::byte> fifo;
        uint64_t capacity = 0;            // a power of two
        std::atomic<RelayState> relay{RelayState::Running};
        bool retired = false;

        uint64_t arrived = 0, consumed = 0, acked = 0;
        uint64_t fullest = 0;
        std::atomic<uint64_t> arrived_bytes{0};
        std::atomic<uint64_t> walked_bytes{0};
        uint64_t frames = 0;
        std::vector<std::atomic<uint64_t>> marks;  // UINT64_MAX until written
        uint64_t mark_block = UINT64_MAX;

        const std::byte* page(uint64_t p) const {
            return fifo.data() + (p & (capacity - 1)) * kernel_profiler::SPSC_SPAN_PAGE_WORDS * 4;
        }
    };

    void ingest_thread(std::vector<Stream*> streams);
    bool poll(Stream& s);
    bool walk_frame(Stream& s);
    bool settle(Stream& s);
    Stream& stream(uint32_t device_index, uint32_t socket_index);
    void log_report() const;

    std::unique_ptr<ClockMap> map_;
    std::unique_ptr<SyncEngine> sync_;
    std::unique_ptr<HostSync> host_sync_;  // released before the devices close
    std::unique_ptr<DevicePrograms> programs_;
    std::vector<std::unique_ptr<Stream>> streams_;
    std::vector<ProducerStream> streams_view_;
    std::vector<std::thread> ingest_threads_;
    bool attached_ = false;
};

}  // namespace streaming_profiler
}  // namespace tt::tt_metal
