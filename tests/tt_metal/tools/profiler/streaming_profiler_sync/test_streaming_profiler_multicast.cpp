// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Checks each chip's timeline core to core. One Tensix core per chip broadcasts rounds over each NoC, and every worker
// records the arrival as an MC_RX zone. A multicast takes a fixed 9 cycles per router hop, so once each arrival is on
// the host timeline and its hops are subtracted, every core should report the same instant. The test fails if any
// core's mean differs from the others' by a cycle or more, a round goes missing or records are dropped. The broadcasts
// run after each phase of idle, low load, high load and di/dt bursts, so the check covers the clock under DVFS; --scale
// shortens the phases. Needs a Blackhole system; about 4 minutes at the default scale.

// The load never runs during the broadcasts: its NoC traffic and L1 contention would delay arrivals by amounts the hop
// count doesn't account for, which the check can't tell apart from clock error.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include <fmt/format.h>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/streaming_profiler.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/system_mesh.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt_stl/assert.hpp>

#include "workload_common.hpp"

using namespace tt;
using namespace tt::tt_metal;
namespace sp = tt::tt_metal::experimental::streaming_profiler;
using namespace streaming_profiler_workload;

namespace {
constexpr uint32_t kValuesAddr = 0x148000;
constexpr uint32_t kRounds = 4000;
static_assert(kRounds % 2 == 0);
// 9 cycles per router hop (BlackholeA0/NoC README.md; measured 9.00).
constexpr double kCyclesPerHop = 9.0;

struct Arrival {
    int64_t tsc;
    int64_t ticks;
    uint16_t chip;
    uint8_t x, y;
    uint8_t px, py;
};

distributed::MeshWorkload make_multicast(distributed::MeshDevice& mesh, uint32_t runtime_id) {
    const CoreRange cores = worker_grid(mesh);
    const CoreCoord &lo = cores.start_coord, &hi = cores.end_coord;
    const CoreCoord vlo = mesh.worker_core_from_logical_core(lo);
    const CoreCoord vhi = mesh.worker_core_from_logical_core(hi);
    const uint32_t num_dests = static_cast<uint32_t>(cores.size() - 1);
    Program program = CreateProgram();
    program.set_runtime_id(runtime_id);
    const auto kid = CreateKernel(
        program,
        std::string(kKernelDir) + "multicast_dm.cpp",
        CoreRangeSet(cores),
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    for (const CoreCoord& c : cores) {
        const uint32_t role = c == lo ? 0u : (c == hi ? 1u : 2u);
        // A NoC 1 rectangle runs from its start corner at the high coordinates down to the low ones.
        const CoreCoord& s = role == 1 ? vhi : vlo;
        const CoreCoord& e = role == 1 ? vlo : vhi;
        SetRuntimeArgs(
            program,
            kid,
            c,
            {role,
             static_cast<uint32_t>(s.x),
             static_cast<uint32_t>(s.y),
             static_cast<uint32_t>(e.x),
             static_cast<uint32_t>(e.y),
             num_dests,
             kFlagAddr,
             kValuesAddr,
             kRounds});
    }
    distributed::MeshWorkload w;
    w.add_program(distributed::MeshCoordinateRange(mesh.shape()), std::move(program));
    return w;
}

uint32_t multicast(distributed::MeshDevice& mesh, uint32_t runtime_id) {
    const CoreRange grid = worker_grid(mesh);
    std::vector<FlagCore> cores;
    for (IDevice* d : mesh.get_devices()) {
        for (const CoreCoord& c : grid) {
            cores.push_back({d, c});
        }
    }
    clear_flags(cores);
    distributed::MeshWorkload w = make_multicast(mesh, runtime_id);
    run_once(mesh.mesh_command_queue(), w);
    return count_gave_up(cores, "multicast");
}

struct Verdict {
    double worst_span_ticks = 0.0;
    bool incomplete = false;
};

using CoreKey = std::pair<size_t, size_t>;
using Round = std::map<CoreKey, const Arrival*>;
struct Lane {
    CoreCoord src;
    std::vector<Round> rounds;
};
struct Lanes {
    std::map<std::pair<uint32_t, uint32_t>, Lane> by_chip_noc;
    bool incomplete = false;
};

Lanes group_lanes(const std::vector<Arrival>& arrivals, const CoreCoord& grid, size_t num_chips) {
    std::map<uint32_t, std::vector<const Arrival*>> by_chip;
    for (const Arrival& a : arrivals) {
        by_chip[a.chip].push_back(&a);
    }
    const CoreKey lo_key{0, 0}, hi_key{grid.x - 1, grid.y - 1};
    Lanes lanes;
    lanes.incomplete = by_chip.size() != num_chips;
    for (auto& [chip, v] : by_chip) {
        std::map<CoreKey, std::vector<const Arrival*>> by_core;
        for (const Arrival* a : v) {
            by_core[CoreKey{a->x, a->y}].push_back(a);
        }
        bool counts_ok = by_core.size() == grid.x * grid.y;
        for (auto& [c, list] : by_core) {
            std::sort(list.begin(), list.end(), [](const Arrival* a, const Arrival* b) { return a->ticks < b->ticks; });
            const size_t want = c == lo_key || c == hi_key ? kRounds / 2 : kRounds;
            if (list.size() != want) {
                std::printf(
                    "[multicast] chip %u core (%zu,%zu): %zu arrivals, expected %zu\n",
                    chip,
                    c.first,
                    c.second,
                    list.size(),
                    want);
                counts_ok = false;
            }
        }
        if (!counts_ok) {
            lanes.incomplete = true;
            continue;
        }
        Lane& odd = lanes.by_chip_noc[{chip, 0u}];
        Lane& even = lanes.by_chip_noc[{chip, 1u}];
        const Arrival* lo_a = by_core.at(lo_key).front();
        const Arrival* hi_a = by_core.at(hi_key).front();
        odd.src = CoreCoord(lo_a->px, lo_a->py);
        even.src = CoreCoord(hi_a->px, hi_a->py);
        odd.rounds.resize(kRounds / 2);
        even.rounds.resize(kRounds / 2);
        for (const auto& [c, list] : by_core) {
            for (size_t k = 0; k < list.size(); k++) {
                const size_t r = c == lo_key ? 2 * k + 2 : (c == hi_key ? 2 * k + 1 : k + 1);
                ((r & 1u) ? odd : even).rounds[(r - 1) / 2][c] = list[k];
            }
        }
    }
    lanes.incomplete = lanes.incomplete || lanes.by_chip_noc.empty();
    return lanes;
}

struct LaneSpan {
    size_t cores = 0, rounds = 0;
    double ns_per_tick = 0.0, span_ns = 0.0, noise_ns = 0.0;
};

LaneSpan solve_lane(uint32_t chip, uint32_t noc, const Lane& lane) {
    const double ns_per_tsc = sp::NsPerTscTick();
    const std::vector<Round>& rounds = lane.rounds;
    const CoreCoord src = lane.src;
    std::vector<CoreKey> cores;
    std::vector<double> hops;
    for (const auto& [c, a] : rounds.front()) {
        const CoreCoord p(a->px, a->py);
        const int dx = noc == 0 ? static_cast<int>(p.x) - static_cast<int>(src.x)
                                : static_cast<int>(src.x) - static_cast<int>(p.x);
        const int dy = noc == 0 ? static_cast<int>(p.y) - static_cast<int>(src.y)
                                : static_cast<int>(src.y) - static_cast<int>(p.y);
        TT_FATAL(dx >= 0 && dy >= 0, "chip {} NoC {}: core {} is behind the source {}", chip, noc, p.str(), src.str());
        cores.push_back(c);
        hops.push_back(static_cast<double>(dx + dy));
    }
    const size_t m = cores.size(), n = rounds.size();
    std::vector<std::vector<double>> e(m, std::vector<double>(n));
    double tau_sum = 0.0;
    for (size_t k = 0; k < n; k++) {
        const size_t k2 = k + 1 < n ? k + 1 : k - 1;
        const Arrival* a = rounds[k].at(cores[0]);
        const Arrival* b = rounds[k2].at(cores[0]);
        const double tau = static_cast<double>(b->tsc - a->tsc) * ns_per_tsc / static_cast<double>(b->ticks - a->ticks);
        tau_sum += tau;
        double mh = 0.0;
        for (size_t i = 0; i < m; i++) {
            const Arrival* x = rounds[k].at(cores[i]);
            e[i][k] = static_cast<double>(x->tsc - a->tsc) * ns_per_tsc - kCyclesPerHop * hops[i] * tau;
            mh += e[i][k];
        }
        for (size_t i = 0; i < m; i++) {
            e[i][k] -= mh / static_cast<double>(m);
        }
    }
    const double tau = tau_sum / static_cast<double>(n);
    double lo_e = 1e30, hi_e = -1e30, noise = 0.0;
    for (size_t i = 0; i < m; i++) {
        double sum = 0.0, sum2 = 0.0;
        for (const double x : e[i]) {
            sum += x;
            sum2 += x * x;
        }
        const double mean = sum / static_cast<double>(n);
        noise = std::max(noise, std::sqrt(std::max(sum2 / static_cast<double>(n) - mean * mean, 0.0) / n));
        lo_e = std::min(lo_e, mean);
        hi_e = std::max(hi_e, mean);
    }
    return LaneSpan{.cores = m, .rounds = n, .ns_per_tick = tau, .span_ns = hi_e - lo_e, .noise_ns = noise};
}

Verdict check(const std::vector<Arrival>& arrivals, const CoreCoord& grid, size_t num_chips) {
    const Lanes lanes = group_lanes(arrivals, grid, num_chips);
    Verdict verdict{.incomplete = lanes.incomplete};
    std::printf("chip noc cores rounds  ns/tick   span ns (ticks)  noise ns\n");
    for (const auto& [key, lane] : lanes.by_chip_noc) {
        const auto [chip, noc] = key;
        const LaneSpan span = solve_lane(chip, noc, lane);
        const double span_ticks = span.span_ns / span.ns_per_tick;
        verdict.worst_span_ticks = std::max(verdict.worst_span_ticks, span_ticks);
        std::printf(
            "%4u %3u %5zu  %5zu  %.4f   %6.3f (%5.2f)   %6.3f\n",
            chip,
            noc,
            span.cores,
            span.rounds,
            span.ns_per_tick,
            span.span_ns,
            span_ticks,
            span.noise_ns);
    }
    return verdict;
}

enum class Kind { Idle, Low, High, Didt };
struct Phase {
    const char* name;
    double seconds;
    Kind kind;
};

struct PhaseResult {
    std::string name;
    uint32_t gave_up;
};
}  // namespace

int main(int argc, char** argv) {
    double scale = 1.0;
    for (int i = 1; i < argc; i++) {
        const std::string_view a = argv[i];
        if (a == "--scale" && i + 1 < argc) {
            scale = std::strtod(argv[++i], nullptr);
        } else {
            std::printf("usage: %s [--scale F]\n", argv[0]);
            return 1;
        }
    }
    if (!(scale > 0.0)) {
        std::printf("[multicast] --scale must be positive\n");
        return 1;
    }

    std::map<uint32_t, std::vector<Arrival>> by_phase;
    uint64_t dropped_bytes = 0;
    auto sub = sp::RegisterCallback(
        [&](const sp::Batch<sp::RecordType::Zones>& b) {
            dropped_bytes += b.dropped_bytes();
            for (const sp::Zone& z : b.zones()) {
                if (std::string_view(z.site().name) != "MC_RX") {
                    continue;
                }
                const sp::Core c = z.core();
                by_phase[z.runtime_id()].push_back(Arrival{
                    z.start_tsc(),
                    static_cast<int64_t>(z.start_device_cycles()),
                    static_cast<uint16_t>(c.chip_id),
                    static_cast<uint8_t>(c.logical.x),
                    static_cast<uint8_t>(c.logical.y),
                    static_cast<uint8_t>(c.physical.x),
                    static_cast<uint8_t>(c.physical.y)});
            }
        },
        "multicast");

    auto mesh_device = open_system_mesh("multicast");
    if (!mesh_device) {
        return 1;
    }
    const CoreCoord grid = mesh_device->compute_with_storage_grid_size();
    const size_t num_chips = mesh_device->get_devices().size();

    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    const LoadRate rate = calibrate_load(*mesh_device);
    const uint32_t idle_1ms = static_cast<uint32_t>(mesh_device->get_devices().front()->get_clock_rate_mhz()) * 1000u;
    const CoreRange workers = worker_grid(*mesh_device);
    auto w_low = make_load(*mesh_device, CoreRange(CoreCoord(0, 0)), rate.lasting(1e-4));
    auto w_high = make_load(*mesh_device, workers, rate.lasting(0.05));
    auto w_didt = make_load(*mesh_device, workers, rate.lasting(1e-3, LoadSpec{.bursts = 25, .idle_cycles = idle_1ms}));
    const double high_load_s = run_once(cq, w_high);
    std::printf(
        "[multicast] %zux%zu Tensix cores x %u rounds (%u a NoC) on %zu chips; load calibrated at %.2f us per "
        "matmul block, %.2f us per NoC round; high-load program %.1f ms\n",
        grid.x,
        grid.y,
        kRounds,
        kRounds / 2,
        num_chips,
        rate.mm_s * 1e6,
        rate.dm_s * 1e6,
        high_load_s * 1e3);
    std::fflush(stdout);

    const std::vector<Phase> phases = {
        {"start", 0, Kind::Idle},
        {"idle", 30, Kind::Idle},
        {"low load", 30, Kind::Low},
        {"high load", 30, Kind::High},
        {"di/dt", 30, Kind::Didt},
        {"idle", 60, Kind::Idle},
    };
    std::vector<PhaseResult> results;
    for (size_t k = 0; k < phases.size(); k++) {
        const Phase& p = phases[k];
        const double seconds = p.seconds * scale;
        const auto t0 = Clock::now();
        do {
            switch (p.kind) {
                case Kind::Idle: std::this_thread::sleep_for(std::chrono::duration<double>(seconds)); break;
                case Kind::Low:
                    run_once(cq, w_low);
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                    break;
                case Kind::High: run_once(cq, w_high); break;
                case Kind::Didt: run_once(cq, w_didt); break;
            }
        } while (seconds_since(t0) < seconds);
        results.push_back(
            {fmt::format("{} {:g} s", p.name, seconds), multicast(*mesh_device, static_cast<uint32_t>(k + 1))});
        std::printf("[multicast] phase %zu/%zu %s done\n", k + 1, phases.size(), results.back().name.c_str());
        std::fflush(stdout);
    }
    mesh_device->close();
    sub = {};

    double worst_span_ticks = 0.0;
    uint32_t total_gave_up = 0;
    size_t failed = 0;
    for (size_t k = 0; k < results.size(); k++) {
        const PhaseResult& r = results[k];
        std::printf("[multicast] phase %zu/%zu %s\n", k + 1, results.size(), r.name.c_str());
        const auto id = static_cast<uint32_t>(k + 1);
        const Verdict v = check(by_phase[id], grid, num_chips);
        by_phase.erase(id);
        const bool pass = v.worst_span_ticks < 1.0 && r.gave_up == 0 && !v.incomplete;
        std::printf(
            "[multicast] %s: %s, worst span %.2f ticks; %u cores gave up%s\n",
            r.name.c_str(),
            pass ? "PASS" : "FAIL",
            v.worst_span_ticks,
            r.gave_up,
            v.incomplete ? "; SOME CHIPS MISSING OR WITH WRONG ARRIVAL COUNTS" : "");
        worst_span_ticks = std::max(worst_span_ticks, v.worst_span_ticks);
        total_gave_up += r.gave_up;
        failed += pass ? 0 : 1;
    }
    const bool pass = failed == 0 && dropped_bytes == 0;
    std::printf(
        "[multicast] %s: worst core-to-core span of the device-local timeline %.2f ticks over %zu phases, %zu failed; "
        "%u cores gave up, %llu bytes dropped\n",
        pass ? "PASS" : "FAIL",
        worst_span_ticks,
        phases.size(),
        failed,
        total_gave_up,
        static_cast<unsigned long long>(dropped_bytes));
    return pass ? 0 : 1;
}
