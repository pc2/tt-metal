// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>
#include <cstdio>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

#include "impl/streaming_profiler/capture_context.hpp"

namespace tt::tt_metal::streaming_profiler {

class CsvFile {
public:
    CsvFile(const std::string& path, std::string_view what);
    const std::string& path() const { return path_; }
    template <typename WriteHeader>
    std::FILE* begin(WriteHeader&& write_header) {
        if (!header_written_) {
            write_header(f_.get());
            header_written_ = true;
        }
        return f_.get();
    }

private:
    struct Close {
        void operator()(std::FILE* f) const { std::fclose(f); }
    };
    std::string path_;
    std::unique_ptr<std::FILE, Close> f_;
    bool header_written_ = false;
};

// Writes the classic per-zone device profiler CSV (profile_log_device.csv) in that file's exact format: the DRAM
// device profiler stands down under the streaming profiler and every downstream tool reads that file.
class ZoneCsvConsumer {
public:
    using Batch = experimental::streaming_profiler::Batch<
        experimental::streaming_profiler::RecordType::Zones |
        experimental::streaming_profiler::RecordType::TimestampedData>;
    explicit ZoneCsvConsumer(const std::string& path);
    void operator()(const Batch& batch);
    // Call only between captures.
    void write_csv();

private:
    struct Row {
        uint32_t chip = 0;
        uint16_t core_x = 0, core_y = 0;
        uint16_t logical_x = 0, logical_y = 0;
        uint8_t processor = 0;
        uint32_t timer_id = 0;
        uint64_t timestamp = 0;
        uint64_t data = 0;
        // runtime_id goes in `trace id`, not `run host ID`: that column means which host run produced the row, and
        // an op id there trips the reader's concatenated-capture warning.
        uint32_t prog = 0;
        std::string_view zone_name;  // in the zone-name registry, alive for the process
        const char* type = "";
    };

    static Row row_for(const experimental::streaming_profiler::Core& core);

    CsvFile file_;
    std::vector<Row> rows_;
    uint64_t zone_cycles_ = 0;
    int64_t zone_tsc_ = 0;
    uint64_t dropped_ = 0;
};

// Rows join against a classic ops_perf_results CSV on GLOBAL CALL COUNT. Trace replays reuse a host id, so an op's
// executions are split by ordinal.
class OpsCsvConsumer {
public:
    using Batch = experimental::streaming_profiler::Batch<experimental::streaming_profiler::RecordType::Zones>;
    explicit OpsCsvConsumer(const std::string& path);
    void operator()(const Batch& batch);
    // Call only between captures.
    void write_csv();

private:
    // The Tensix RISCs' columns, then ERISC0's, which holds both ERISCs.
    static constexpr uint32_t kRiscColumns =
        static_cast<uint32_t>(experimental::streaming_profiler::Processor::ERISC0) + 1;

    // Kernel start and end in device cycles for the CSV's CYCLE columns and on the host TSC for every [ns] column,
    // since under DVFS a difference of cycles has no single rate to convert with.
    struct OpAgg {
        uint64_t k_start = UINT64_MAX, k_end = 0;
        int64_t h_start_last = INT64_MIN;
        std::array<int64_t, kRiscColumns> h_risc_start{};
        std::array<int64_t, kRiscColumns> h_risc_end{};
        std::map<uint32_t, std::pair<int64_t, int64_t>> cores;
        OpAgg() {
            h_risc_start.fill(INT64_MAX);
            h_risc_end.fill(INT64_MIN);
        }
    };

    CsvFile file_;
    std::map<std::tuple<uint32_t, uint32_t, uint32_t>, OpAgg> ops_;
    std::map<std::tuple<uint32_t, uint32_t, uint32_t, uint32_t>, uint32_t> pair_count_;
};

}  // namespace tt::tt_metal::streaming_profiler
