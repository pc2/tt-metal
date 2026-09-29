// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/streaming_profiler/csv_consumers.hpp"

#include <unistd.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <span>
#include <string>
#include <string_view>

#include <tt-logger/tt-logger.hpp>
#include <tt_stl/assert.hpp>

namespace tt::tt_metal::streaming_profiler {

namespace api = experimental::streaming_profiler;

CsvFile::CsvFile(const std::string& path, std::string_view what) : path_(path), f_(std::fopen(path.c_str(), "w")) {
    TT_FATAL(f_ != nullptr, "streaming profiler: cannot open {} for the {}", path, what);
}

namespace {

// The ids of the old wait halves stay reserved, so a reader still keyed on them can't pick up something else.
struct SyncName {
    const char* name;
    uint32_t legacy_id;
};
constexpr SyncName kSyncNames[] = {
    {"SYNC-CB-PUSH", 1000},
    // 1001, 1002 reserved: the SYNC-CB-WAIT halves are one zone
    {"SYNC-SEM-SET", 1003},
    {"SYNC-SEM-SET-REMOTE", 1004},
    // 1005, 1006 reserved: the SYNC-SEM-WAIT halves are one zone
    {"SYNC-SEM-WAIT-KEY", 1007},
    {"SYNC-CB-WAIT-KEY", 1008},
    {"SYNC-CB-RESERVE-KEY", 1009},
    {"SYNC-CB-POP", 1010},
};

uint32_t sync_legacy_id(std::string_view name) {
    for (const SyncName& s : kSyncNames) {
        if (name == s.name) {
            return s.legacy_id;
        }
    }
    return 0;
}

// The reader pairs START and END rows by order and type, so the id only has to be stable per name and never 0.
uint32_t name_hash(std::string_view name) {
    uint32_t h = 2166136261u;
    for (unsigned char ch : name) {
        h = (h ^ ch) * 16777619u;
    }
    return (h & 0x7FFFFFFu) | 0x8000000u;  // outside the legacy sync-id range
}

// Named the way the classic CSV prints tracy::RiscType, which has a single ERISC for every eth RISC.
constexpr std::array<const char*, kProcessorCount> kCsvRiscNames = {
    "BRISC", "NCRISC", "TRISC_0", "TRISC_1", "TRISC_2", "ERISC", "ERISC"};

}  // namespace

ZoneCsvConsumer::ZoneCsvConsumer(const std::string& path) : file_(path, "zone CSV") {}

ZoneCsvConsumer::Row ZoneCsvConsumer::row_for(const api::Core& core) {
    Row r;
    r.chip = core.chip_id;
    r.core_x = static_cast<uint16_t>(core.physical.x);
    r.core_y = static_cast<uint16_t>(core.physical.y);
    r.logical_x = static_cast<uint16_t>(core.logical.x);
    r.logical_y = static_cast<uint16_t>(core.logical.y);
    r.processor = static_cast<uint8_t>(core.processor);
    return r;
}

void ZoneCsvConsumer::operator()(const Batch& batch) {
    dropped_ += batch.dropped_bytes();
    for (const api::Zone& z : batch.zones()) {
        zone_cycles_ += z.end_device_cycles() - z.start_device_cycles();
        zone_tsc_ += z.end_tsc() - z.start_tsc();
        const uint32_t id = name_hash(z.site().name);
        for (int end = 0; end < 2; end++) {
            Row& r = rows_.emplace_back(row_for(z.core()));
            r.timer_id = id;
            r.timestamp = end ? z.end_device_cycles() : z.start_device_cycles();
            r.prog = z.runtime_id();
            r.zone_name = z.site().name;
            r.type = end ? "ZONE_END" : "ZONE_START";
        }
    }
    for (const api::TimestampedData& d : batch.timestamped_data()) {
        // Only for sync events, where the reader treats `data` as a CB id or semaphore address.
        const uint32_t legacy = sync_legacy_id(d.site().name);
        if (legacy == 0) {
            continue;
        }
        const std::span<const uint64_t> payload = d.payload();
        if (payload.empty()) {
            continue;  // a semaphore event at address 0 would invent a dependency
        }
        Row& r = rows_.emplace_back(row_for(d.core()));
        r.timer_id = legacy;
        r.timestamp = d.device_cycles();
        r.data = payload.front();
        r.type = "TS_DATA";
    }
}

void ZoneCsvConsumer::write_csv() {
    if (zone_tsc_ <= 0) {
        return;
    }
    FILE* const f = file_.begin([&](FILE* f) {
        const double freq_mhz =
            1000.0 * static_cast<double>(zone_cycles_) / (static_cast<double>(zone_tsc_) * api::NsPerTscTick());
        std::fprintf(f, "ARCH: blackhole, CHIP_FREQ[MHz]: %.0f, Max Compute Cores: 0\n", freq_mhz);
        std::fprintf(
            f,
            "PCIe slot, core_x, core_y, RISC processor type, timer_id, "
            "time[cycles since reset], data, run host ID, trace id, trace id counter, "
            "zone name, type, source line, source file, meta data, logical_x, logical_y\n");
    });
    // Use the PID rather than a constant, so two hand-concatenated captures get different ids and the reader's
    // multi-run warning still fires.
    const uint32_t run_id = static_cast<uint32_t>(::getpid());
    for (const Row& r : rows_) {
        std::fprintf(
            f,
            "%u, %u, %u, %s, %u, %llu, %llu, %u, %u, 0, %.*s, %s, 0, streaming, , %u, %u\n",
            r.chip,
            r.core_x,
            r.core_y,
            kCsvRiscNames[r.processor],
            r.timer_id,
            static_cast<unsigned long long>(r.timestamp),
            static_cast<unsigned long long>(r.data),
            run_id,
            r.prog,
            static_cast<int>(r.zone_name.size()),
            r.zone_name.empty() ? "" : r.zone_name.data(),
            r.type,
            r.logical_x,
            r.logical_y);
    }
    std::fflush(f);
    log_info(
        tt::LogMetal,
        "[streaming profiler] zone CSV: wrote {} rows to {} ({} dropped bytes)",
        rows_.size(),
        file_.path(),
        dropped_);
    rows_.clear();
    dropped_ = 0;
}

OpsCsvConsumer::OpsCsvConsumer(const std::string& path) : file_(path, "ops CSV") {}

void OpsCsvConsumer::operator()(const Batch& batch) {
    for (const auto& z : batch.zones()) {
        if (z.runtime_id() == 0 || !z.site().name.ends_with("-KERNEL")) {
            continue;
        }
        const api::Core c = z.core();
        const bool eth = c.processor >= api::Processor::ERISC0;
        const uint32_t risc = static_cast<uint32_t>(eth ? api::Processor::ERISC0 : c.processor);
        // An eth core's logical coordinate can equal a Tensix core's.
        const uint32_t core_key =
            (eth ? 1u << 31 : 0u) | (static_cast<uint32_t>(c.logical.y) << 16) | static_cast<uint32_t>(c.logical.x);
        // The wrapper zone never nests, so the k-th one on a lane for a program is its k-th execution.
        uint32_t& completed = pair_count_[{c.chip_id, core_key, static_cast<uint32_t>(c.processor), z.runtime_id()}];
        OpAgg& op = ops_[{c.chip_id, z.runtime_id(), completed}];
        completed++;
        op.k_start = std::min(op.k_start, z.start_device_cycles());
        op.k_end = std::max(op.k_end, z.end_device_cycles());
        const int64_t start = z.start_tsc();
        const int64_t end = z.end_tsc();
        auto& core = op.cores.try_emplace(core_key, INT64_MAX, INT64_MIN).first->second;
        op.h_start_last = std::max(op.h_start_last, start);
        op.h_risc_start[risc] = std::min(op.h_risc_start[risc], start);
        core.first = std::min(core.first, start);
        op.h_risc_end[risc] = std::max(op.h_risc_end[risc], end);
        core.second = std::max(core.second, end);
    }
}

void OpsCsvConsumer::write_csv() {
    FILE* const f = file_.begin([](FILE* f) {
        std::fputs(
            "DEVICE ID,GLOBAL CALL COUNT,EXECUTION,CORE COUNT,DEVICE KERNEL START CYCLE,DEVICE KERNEL END CYCLE,"
            "DEVICE KERNEL DURATION [ns],DEVICE KERNEL DURATION DM START [ns],"
            "DEVICE KERNEL DURATION PER CORE MIN [ns],DEVICE KERNEL DURATION PER CORE MAX [ns],"
            "DEVICE KERNEL DURATION PER CORE AVG [ns],DEVICE KERNEL FIRST TO LAST START [ns],"
            "DEVICE BRISC KERNEL DURATION [ns],DEVICE NCRISC KERNEL DURATION [ns],"
            "DEVICE TRISC0 KERNEL DURATION [ns],DEVICE TRISC1 KERNEL DURATION [ns],"
            "DEVICE TRISC2 KERNEL DURATION [ns],DEVICE ERISC KERNEL DURATION [ns]\n",
            f);
    });
    for (const auto& [key, op] : ops_) {
        const auto& [chip, prog, exec] = key;
        auto ns = [](int64_t start, int64_t end) {
            return end > start ? static_cast<double>(end - start) * api::NsPerTscTick() : 0.0;
        };
        const auto risc_start = [&](api::Processor p) { return op.h_risc_start[static_cast<uint32_t>(p)]; };
        const auto risc_ns = [&](api::Processor p) {
            return ns(risc_start(p), op.h_risc_end[static_cast<uint32_t>(p)]);
        };
        const int64_t h_start = std::ranges::min(op.h_risc_start);
        const int64_t h_end = std::ranges::max(op.h_risc_end);
        const int64_t h_dm_start = std::min(
            {risc_start(api::Processor::BRISC),
             risc_start(api::Processor::NCRISC),
             risc_start(api::Processor::ERISC0)});
        double core_min = 0.0, core_max = 0.0, core_sum = 0.0;
        uint32_t core_n = 0;
        for (const auto& [c, se] : op.cores) {
            if (se.second <= se.first) {
                continue;
            }
            const double d = ns(se.first, se.second);
            core_min = core_n == 0 ? d : std::min(core_min, d);
            core_max = std::max(core_max, d);
            core_sum += d;
            core_n++;
        }
        std::fprintf(
            f,
            "%u,%u,%u,%u,%llu,%llu,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,",
            chip,
            prog,
            exec,
            core_n,
            static_cast<unsigned long long>(op.k_start),
            static_cast<unsigned long long>(op.k_end),
            ns(h_start, h_end),
            ns(h_dm_start, h_end),
            core_min,
            core_max,
            core_n != 0 ? core_sum / core_n : 0.0,
            ns(h_start, op.h_start_last),
            risc_ns(api::Processor::BRISC),
            risc_ns(api::Processor::NCRISC),
            risc_ns(api::Processor::TRISC0),
            risc_ns(api::Processor::TRISC1),
            risc_ns(api::Processor::TRISC2));
        // The classic report leaves the column empty for an op with no eth kernel zone.
        if (op.h_risc_end[static_cast<uint32_t>(api::Processor::ERISC0)] != INT64_MIN) {
            std::fprintf(f, "%.0f", risc_ns(api::Processor::ERISC0));
        }
        std::fputc('\n', f);
    }
    std::fflush(f);
    ops_.clear();
    pair_count_.clear();
}

}  // namespace tt::tt_metal::streaming_profiler
