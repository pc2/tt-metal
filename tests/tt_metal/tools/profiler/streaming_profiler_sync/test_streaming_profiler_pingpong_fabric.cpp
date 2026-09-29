// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Fabric traffic under the profiler. Over 2D fabric, one worker on each side of every linked chip pair ping-pongs
// atomic increments for --seconds, recording a PP_TX zone per send and a PP_RX zone per arrival, then again with --load
// while the other workers run matmul and NoC load. It fails if a kernel gives up waiting for its peer or any round's
// zones don't all reach the host. As a sanity check of the synced timeline, it also fails if a pair's first rounds
// break causality or either direction's one-way time is far from half the round trip. --idle just opens the mesh and
// sleeps, for the sync check's idle arm.
// The sync gate runs it with the sync check on, so its traffic shares the routers with the link sync.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
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
constexpr uint32_t kRounds = 20000;  // a pass long enough (~25 ms) that the host work between passes is the minority

struct Pair {
    uint32_t chip_a, chip_b;
    CoreCoord core_a, core_b;
    uint32_t link_a, link_b;
};

constexpr uint32_t kTimedRounds = 1000;
struct Stamp {
    uint64_t cycles;
    int64_t tsc;
    bool tx;
};
// Every zone is counted; the first kTimedRounds rounds' zones are also kept for the timeline check.
struct Stamps {
    size_t tx = 0, rx = 0;
    std::vector<Stamp> first;
};
using StampsByCore = std::map<std::pair<uint32_t, CoreCoord>, Stamps>;
struct Chip {
    distributed::MeshCoordinate coord;
    IDevice* dev;
    tt::tt_fabric::FabricNodeId node;
};
using Chips = std::map<uint32_t, Chip>;

std::vector<Pair> find_pairs(distributed::MeshDevice& mesh_device, const Chips& chips) {
    const CoreCoord grid = mesh_device.compute_with_storage_grid_size();
    std::map<uint32_t, uint32_t> next_worker;
    const auto take_worker = [&](uint32_t chip) {
        const uint32_t i = next_worker[chip]++;
        TT_FATAL(i < grid.x * grid.y, "chip {} has more fabric neighbours than worker cores", chip);
        return CoreCoord(i % grid.x, i / grid.x);
    };
    std::vector<Pair> pairs;
    for (const auto& [a, ca] : chips) {
        for (const auto& [b, cb] : chips) {
            if (a >= b || tt::tt_fabric::get_neighbor_eth_directions(ca.node, cb.node).empty()) {
                continue;
            }
            const auto links_ab = tt::tt_fabric::get_forwarding_link_indices(ca.node, cb.node);
            const auto links_ba = tt::tt_fabric::get_forwarding_link_indices(cb.node, ca.node);
            TT_FATAL(!links_ab.empty() && !links_ba.empty(), "no fabric link between chips {} and {}", a, b);
            pairs.push_back(Pair{a, b, take_worker(a), take_worker(b), links_ab.front(), links_ba.front()});
        }
    }
    return pairs;
}

void arm(
    Program& program,
    distributed::MeshDevice& mesh_device,
    const CoreCoord& core,
    uint32_t role,
    const CoreCoord& peer,
    const tt::tt_fabric::FabricNodeId& src,
    const tt::tt_fabric::FabricNodeId& dst,
    uint32_t link_idx) {
    const auto kid = CreateKernel(
        program,
        std::string(kKernelDir) + "pingpong_fabric_dm.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    const CoreCoord vpeer = mesh_device.worker_core_from_logical_core(peer);
    std::vector<uint32_t> args = {
        role,
        static_cast<uint32_t>(vpeer.x),
        static_cast<uint32_t>(vpeer.y),
        kFlagAddr,
        kRounds,
        static_cast<uint32_t>(dst.chip_id),
        static_cast<uint32_t>(dst.mesh_id.get())};
    tt::tt_fabric::append_fabric_connection_rt_args(src, dst, link_idx, program, core, args);
    SetRuntimeArgs(program, kid, core, args);
}

distributed::MeshWorkload build_workload(
    distributed::MeshDevice& mesh_device,
    const std::vector<Pair>& pairs,
    const Chips& chips,
    const std::optional<LoadSpec>& load) {
    std::map<uint32_t, Program> programs;
    std::map<uint32_t, std::set<CoreCoord>> ends;
    for (const auto& [chip, _] : chips) {
        programs.emplace(chip, CreateProgram());
    }
    for (const Pair& p : pairs) {
        const auto &na = chips.at(p.chip_a).node, &nb = chips.at(p.chip_b).node;
        arm(programs.at(p.chip_a), mesh_device, p.core_a, 0, p.core_b, na, nb, p.link_a);
        arm(programs.at(p.chip_b), mesh_device, p.core_b, 1, p.core_a, nb, na, p.link_b);
        ends[p.chip_a].insert(p.core_a);
        ends[p.chip_b].insert(p.core_b);
    }
    distributed::MeshWorkload workload;
    const CoreRange workers = worker_grid(mesh_device);
    for (auto& [chip, program] : programs) {
        if (load) {
            std::vector<CoreRange> others;
            for (const CoreCoord& c : workers) {
                if (!ends[chip].contains(c)) {
                    others.emplace_back(c);
                }
            }
            add_load(program, mesh_device, CoreRangeSet(others), *load, ends[chip]);
        }
        const auto& c = chips.at(chip).coord;
        workload.add_program(distributed::MeshCoordinateRange(c, c), std::move(program));
    }
    return workload;
}

uint32_t count_missing(const std::vector<Pair>& pairs, const StampsByCore& by_core, size_t expected) {
    uint32_t failures = 0;
    for (const Pair& p : pairs) {
        auto ia = by_core.find({p.chip_a, p.core_a}), ib = by_core.find({p.chip_b, p.core_b});
        const size_t atx = ia == by_core.end() ? 0 : ia->second.tx, arx = ia == by_core.end() ? 0 : ia->second.rx;
        const size_t btx = ib == by_core.end() ? 0 : ib->second.tx, brx = ib == by_core.end() ? 0 : ib->second.rx;
        if (atx != expected || brx != expected || btx != expected || arx != expected) {
            std::printf(
                "chip %u - chip %u: stamps %zu/%zu/%zu/%zu of %zu\n", p.chip_a, p.chip_b, atx, brx, btx, arx, expected);
            failures++;
        }
    }
    return failures;
}
// Each direction's ping and reply times on the synced timeline, against the round trip minus the turnaround, which each
// end measures on its own clock. A sync error between two chips pushes one direction's one-way times up and the other's
// down. On the 8-chip LoudBox the ping takes 54% of that span and the reply 46% on every pair, whichever chip sends, so
// a 35-65% band leaves the fabric room and still catches any sync error above about 0.2 us.
constexpr double kLegBand = 0.15;

uint32_t check_timeline(const std::vector<Pair>& pairs, StampsByCore& by_core) {
    const double ns = sp::NsPerTscTick();
    const auto median = [](std::vector<double> v) {
        std::ranges::nth_element(v, v.begin() + static_cast<std::ptrdiff_t>(v.size() / 2));
        return v[v.size() / 2];
    };
    uint32_t failures = 0;
    for (const Pair& p : pairs) {
        std::vector<Stamp>& a = by_core[{p.chip_a, p.core_a}].first;
        std::vector<Stamp>& b = by_core[{p.chip_b, p.core_b}].first;
        for (std::vector<Stamp>* v : {&a, &b}) {
            std::ranges::sort(*v, {}, &Stamp::cycles);
        }
        enum Leg : size_t { kPingAB, kPingBA, kReplyAB, kReplyBA, kLegs };
        std::array<std::vector<double>, kLegs> legs;
        std::vector<double> base;
        uint32_t acausal = 0, misread = 0;
        for (size_t k = 0; k < std::min(a.size(), b.size()) / 2; k++) {
            // Round k + 1: core_b (role 1) sends first on odd rounds, core_a on even ones.
            const bool b_first = (k & 1u) == 0;
            const Stamp* s = &(b_first ? b : a)[2 * k];
            const Stamp* r = &(b_first ? a : b)[2 * k];
            if (!s[0].tx || s[1].tx || r[0].tx || !r[1].tx) {
                misread++;
                continue;
            }
            const double t1 = s[0].tsc * ns, t4 = s[1].tsc * ns, t2 = r[0].tsc * ns, t3 = r[1].tsc * ns;
            acausal += (t2 <= t1 ? 1u : 0u) + (t4 <= t3 ? 1u : 0u);
            legs[b_first ? kPingBA : kPingAB].push_back(t2 - t1);
            legs[b_first ? kReplyAB : kReplyBA].push_back(t4 - t3);
            base.push_back((t4 - t1) - (t3 - t2));
        }
        if (std::ranges::any_of(legs, [](const std::vector<double>& v) { return v.empty(); })) {
            std::printf("chip %u - chip %u: FAIL, no timed rounds in some direction\n", p.chip_a, p.chip_b);
            failures++;
            continue;
        }
        const double full = median(base);
        std::array<double, kLegs> m{};
        bool ok = acausal == 0 && misread == 0;
        for (size_t i = 0; i < legs.size(); i++) {
            m[i] = median(legs[i]);
            ok = ok && std::abs(m[i] / full - 0.5) <= kLegBand;
        }
        std::printf(
            "chip %u - chip %u: %s, of %.0f ns round trip less turnaround: ping %.0f / %.0f ns, reply %.0f / %.0f ns "
            "(a to b / b to a); %u acausal, %u misread of %zu rounds\n",
            p.chip_a,
            p.chip_b,
            ok ? "ok" : "FAIL",
            full,
            m[kPingAB],
            m[kPingBA],
            m[kReplyAB],
            m[kReplyBA],
            acausal,
            misread,
            base.size() + misread);
        failures += ok ? 0 : 1;
    }
    return failures;
}
}  // namespace

int main(int argc, char** argv) {
    double seconds = 10.0;
    bool with_load = false, idle = false;
    for (int i = 1; i < argc; i++) {
        const std::string_view a = argv[i];
        if (a == "--seconds" && i + 1 < argc) {
            seconds = std::strtod(argv[++i], nullptr);
        } else if (a == "--load" && !idle) {
            with_load = true;
        } else if (a == "--idle" && !with_load) {
            idle = true;
        } else {
            std::fprintf(stderr, "usage: %s [--seconds S] [--load | --idle]\n", argv[0]);
            return 2;
        }
    }
    StampsByCore by_core;
    sp::Callback sub;
    if (!idle) {
        sub = sp::RegisterCallback(
            [&](const sp::Batch<sp::RecordType::Zones>& b) {
                for (const sp::Zone& z : b.zones()) {
                    const std::string_view name = z.site().name;
                    if (name == "PP_TX" || name == "PP_RX") {
                        Stamps& e = by_core[{z.core().chip_id, z.core().logical}];
                        (name == "PP_TX" ? e.tx : e.rx)++;
                        if (e.first.size() < 2 * kTimedRounds) {
                            e.first.push_back({z.start_device_cycles(), z.start_tsc(), name == "PP_TX"});
                        }
                    }
                }
            },
            "pingpong-fabric");
        tt::tt_fabric::SetFabricConfig(tt::tt_fabric::FabricConfig::FABRIC_2D);
    }
    auto mesh_device = open_system_mesh("pingpong-fabric");
    if (!mesh_device) {
        return 1;
    }

    Chips chips;
    if (!idle) {
        for (const auto& c : distributed::MeshCoordinateRange(mesh_device->shape())) {
            IDevice* d = mesh_device->get_device(c);
            chips.emplace(d->id(), Chip{c, d, tt::tt_fabric::get_fabric_node_id_from_physical_chip_id(d->id())});
        }
    }
    const std::vector<Pair> pairs = find_pairs(*mesh_device, chips);
    std::vector<FlagCore> flag_cores;
    for (const Pair& p : pairs) {
        flag_cores.push_back({chips.at(p.chip_a).dev, p.core_a});
        flag_cores.push_back({chips.at(p.chip_b).dev, p.core_b});
    }
    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    uint32_t failures = 0, passes = 0;
    if (idle) {
        std::printf("[pingpong-fabric] idle, %zu chips, %.1f s\n", mesh_device->num_devices(), seconds);
        std::fflush(stdout);
        std::this_thread::sleep_for(std::chrono::duration<double>(seconds));
    } else {
        TT_FATAL(!pairs.empty(), "no fabric-linked chip pairs");
        std::printf(
            "[pingpong-fabric] %zu linked pairs x %u rounds on %zu chips\n", pairs.size(), kRounds, chips.size());
        std::fflush(stdout);
        const auto run_for = [&](distributed::MeshWorkload& w, double seconds) {
            const auto t0 = Clock::now();
            double one = 0.0;
            do {
                clear_flags(flag_cores);
                one = run_once(cq, w);
                failures += count_gave_up(flag_cores, "pingpong-fabric");
                passes++;
            } while (seconds_since(t0) < seconds);
            return one;
        };
        distributed::MeshWorkload unloaded = build_workload(*mesh_device, pairs, chips, std::nullopt);
        const double unloaded_s = run_for(unloaded, seconds);
        std::printf("[pingpong-fabric] %u unloaded passes, %.1f ms each\n", passes, unloaded_s * 1e3);
        if (with_load) {
            distributed::MeshWorkload loaded =
                build_workload(*mesh_device, pairs, chips, calibrate_load(*mesh_device).lasting(unloaded_s));
            const uint32_t before = passes;
            const double loaded_s = run_for(loaded, seconds);
            std::printf("[pingpong-fabric] %u loaded passes, %.1f ms each\n", passes - before, loaded_s * 1e3);
        }
    }
    std::fflush(stdout);
    mesh_device->close();
    sub = {};
    failures += count_missing(pairs, by_core, static_cast<size_t>(passes) * kRounds);
    failures += check_timeline(pairs, by_core);
    return failures == 0 ? 0 : 1;
}
