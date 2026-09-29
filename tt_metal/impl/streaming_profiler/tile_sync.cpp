// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/streaming_profiler/tile_sync.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <map>
#include <optional>
#include <set>
#include <string_view>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/tt_metal.hpp>

#include "context/metal_context.hpp"
#include "hostdev/streaming_profiler_sync.h"
#include "impl/kernels/kernel.hpp"
#include "impl/streaming_profiler/device_programs.hpp"
#include "impl/streaming_profiler/link_sync.hpp"
#include "impl/streaming_profiler/service.hpp"
#include "impl/streaming_profiler/sync_engine.hpp"
#include "llrt/hal.hpp"
#include "llrt/metal_soc_descriptor.hpp"
#include "llrt/tt_cluster.hpp"

namespace tt::tt_metal::streaming_profiler {

int64_t TileClocks::offset(CoreType type, const CoreCoord& logical) const {
    const auto it =
        std::ranges::find_if(tiles, [&](const TileClock& t) { return t.type == type && t.logical == logical; });
    TT_FATAL(
        it != tiles.end(), "streaming profiler: tile ({},{}) is not in its chip's tile clocks", logical.x, logical.y);
    return it->offset;
}

namespace {

constexpr uint32_t kReps = 256;

// median2 is the median, over a pair's reads, of 2 * (partner wall - bracket midpoint), in the clocks' low words.
struct Reading {
    uint32_t t, noc;
    std::optional<uint32_t> twin;
    int32_t median2 = 0;
    int64_t coarse = 0;
    int64_t whole2() const {
        return 2 * coarse + static_cast<int32_t>(static_cast<uint32_t>(median2) - static_cast<uint32_t>(2 * coarse));
    }
};

struct Tile : CoreCoords {
    CoreType type;
    HalProgrammableCoreType core;
    uint32_t scratch = 0;
    uint64_t host_scratch = 0;
    std::vector<Reading> reads;
};

// NoC 0 runs towards higher raw coordinates, NoC 1 towards lower.
bool aligned(const CoreCoord& a, const CoreCoord& b) { return a != b && (a.x == b.x || a.y == b.y); }
bool upward(const CoreCoord& a, const CoreCoord& b) { return a.x == b.x ? b.y > a.y : b.x > a.x; }

std::string_view kind_name(CoreType t) {
    return t == CoreType::WORKER ? "Tensix" : t == CoreType::DRAM ? "DRAM" : "eth";
}

// A profiler ring can't be scratch, even zeroed afterwards: the first frames depend on its contents.
std::vector<Tile> plan_tiles(IDevice* device, ContextId ctx) {
    auto& mc = MetalContext::instance(ctx);
    auto& cluster = mc.get_cluster();
    const auto& hal = mc.hal();
    const uint32_t chip = static_cast<uint32_t>(device->id());
    const auto& soc = cluster.get_soc_desc(chip);
    std::vector<Tile> tiles;
    const auto add = [&](CoreType type, HalProgrammableCoreType core, const CoreCoord& logical, uint32_t scratch) {
        const uint64_t host = core == HalProgrammableCoreType::DRAM
                                  ? hal.get_dev_noc_addr(core, HalL1MemAddrType::UNRESERVED) +
                                        (scratch - hal.get_dev_addr(core, HalL1MemAddrType::UNRESERVED))
                                  : scratch;
        tiles.push_back(Tile{{locate(cluster, chip, logical, type)}, type, core, scratch, host});
    };
    const auto add_unreserved = [&](CoreType type, HalProgrammableCoreType core, const CoreCoord& logical) {
        add(type, core, logical, hal.get_dev_addr(core, HalL1MemAddrType::UNRESERVED));
    };
    const uint32_t user_l1 = static_cast<uint32_t>(device->allocator()->get_base_allocator_addr(HalMemType::L1));
    TT_FATAL(
        hal.get_dev_size(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::PROFILER) >=
            kernel_profiler::PROFILER_L1_CONTROL_BUFFER_SIZE + sizeof(kernel_profiler::TileNetScratch),
        "streaming profiler: a profiler L1 region cannot hold the tile clock scratch");
    const uint32_t dispatch_scratch =
        static_cast<uint32_t>(hal.get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::PROFILER)) +
        kernel_profiler::PROFILER_L1_CONTROL_BUFFER_SIZE;
    const CoreCoord compute = device->compute_with_storage_grid_size();
    const CoreCoord grid = soc.get_grid_size(CoreType::TENSIX);
    for (uint32_t y = 0; y < grid.y; y++) {
        for (uint32_t x = 0; x < grid.x; x++) {
            const bool is_compute = x < compute.x && y < compute.y;
            add(CoreType::WORKER, HalProgrammableCoreType::TENSIX, {x, y}, is_compute ? user_l1 : dispatch_scratch);
        }
    }
    if (hal.has_programmable_core_type(HalProgrammableCoreType::IDLE_ETH)) {
        for (const CoreCoord& l : sorted_yx(device->get_inactive_ethernet_cores())) {
            add_unreserved(CoreType::ETH, HalProgrammableCoreType::IDLE_ETH, l);
        }
    }
    if (hal.has_programmable_core_type(HalProgrammableCoreType::ACTIVE_ETH)) {
        for (const CoreCoord& l : sorted_yx(device->get_active_ethernet_cores(/*skip_reserved_tunnel_cores=*/false))) {
            add_unreserved(CoreType::ETH, HalProgrammableCoreType::ACTIVE_ETH, l);
        }
    }
    if (hal.has_programmable_core_type(HalProgrammableCoreType::DRAM)) {
        // The cluster gives a DRAM core's translated coordinate as its physical one. The NoC 0 grid position, which
        // says which Tensix row it shares, comes from the SoC descriptor in the same order.
        const std::vector<CoreCoord> logical = soc.get_metal_dram_cores(CoordSystem::LOGICAL);
        const std::vector<CoreCoord> noc0 = soc.get_metal_dram_cores(CoordSystem::NOC0);
        for (size_t i = 0; i < logical.size(); i++) {
            if (dram_view_endpoint_noc_mask(soc, logical[i]) == 0) {
                add_unreserved(CoreType::DRAM, HalProgrammableCoreType::DRAM, logical[i]);
                tiles.back().phys = noc0[i];
            }
        }
    }
    // One idle eth tile also reads every eth tile in its row over the other NoC. Around the ring the two NoCs' hops
    // cancel, and the initiator's own latency is the same for every target.
    const auto qi =
        static_cast<uint32_t>(std::ranges::find(tiles, HalProgrammableCoreType::IDLE_ETH, &Tile::core) - tiles.begin());
    for (uint32_t s = 0; s < tiles.size(); s++) {
        Tile& a = tiles[s];
        for (uint32_t t = 0; t < tiles.size(); t++) {
            if (aligned(a.phys, tiles[t].phys)) {
                a.reads.push_back(Reading{t, upward(a.phys, tiles[t].phys) ? 0u : 1u});
            }
        }
        if (s == qi) {
            for (uint32_t i = 0, direct = static_cast<uint32_t>(a.reads.size()); i < direct; i++) {
                if (tiles[a.reads[i].t].type == CoreType::ETH) {
                    a.reads.push_back(Reading{a.reads[i].t, a.reads[i].noc ^ 1u, i});
                }
            }
        }
        TT_FATAL(
            a.reads.size() <= kernel_profiler::kTileNetMaxPartners,
            "streaming profiler: tile ({},{}) has {} row and column partners, the table holds {}",
            a.logical.x,
            a.logical.y,
            a.reads.size(),
            kernel_profiler::kTileNetMaxPartners);
    }
    return tiles;
}

KernelHandle create_tile_kernel(Program& p, HalProgrammableCoreType core, const CoreRangeSet& cores) {
    const char* src = "tt_metal/impl/streaming_profiler/kernels/tile_sync.cpp";
    switch (core) {
        case HalProgrammableCoreType::TENSIX:
            return CreateKernel(
                p,
                src,
                cores,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
        case HalProgrammableCoreType::IDLE_ETH:
            return CreateKernel(p, src, cores, EthernetConfig{.eth_mode = Eth::IDLE, .noc = NOC::RISCV_0_default});
        case HalProgrammableCoreType::ACTIVE_ETH:
            return CreateKernel(p, src, cores, EthernetConfig{.noc = NOC::RISCV_0_default});
        case HalProgrammableCoreType::DRAM: return CreateKernel(p, src, cores, DramConfig{.noc = NOC::NOC_0});
        case HalProgrammableCoreType::DISPATCH:
        case HalProgrammableCoreType::COUNT: break;
    }
    TT_THROW("Unreachable");
}

// If a later step throws, the kernels stay up waiting on their go words.
void read_network(IDevice* device, ContextId ctx, std::vector<Tile>& tiles) {
    auto& mc = MetalContext::instance(ctx);
    auto& cluster = mc.get_cluster();
    const auto& hal = mc.hal();
    const uint32_t chip = static_cast<uint32_t>(device->id());
    // Setting bit 4 and clearing bit 5 keeps both the nonce and ~nonce away from 0 and from the go words, which share
    // the table's first word.
    const uint32_t nonce =
        (static_cast<uint32_t>(std::chrono::steady_clock::now().time_since_epoch().count()) | 0x10u) & ~0x20u;
    struct Kind {
        std::set<CoreRange> cores;
        std::optional<Program> program;
        KernelHandle kernel = 0;
    };
    std::map<HalProgrammableCoreType, Kind> kinds;
    for (const Tile& a : tiles) {
        kinds[a.core].cores.insert(CoreRange(a.logical, a.logical));
    }
    for (auto& [core, kind] : kinds) {
        kind.program = CreateProgram();
        kind.kernel = create_tile_kernel(*kind.program, core, CoreRangeSet(kind.cores));
    }
    const auto table_of = [](const Tile& a) {
        return a.host_scratch + offsetof(kernel_profiler::TileNetScratch, table);
    };
    // Firmware takes its ring position from the control vector at the session's first launch (this one), so a stale
    // tail would put it thousands of words ahead.
    for (const Tile& a : tiles) {
        std::vector<uint32_t> args{a.scratch, kReps, nonce, static_cast<uint32_t>(a.reads.size())};
        for (const Reading& r : a.reads) {
            args.push_back((r.noc << 31) | packed_xy(tiles[r.t].virt));
        }
        const Kind& kind = kinds[a.core];
        SetRuntimeArgs(*kind.program, kind.kernel, a.logical, args);
        zero_l1(cluster, chip, a.virt, table_of(a), 2 * sizeof(uint32_t));
        zero_profiler_control(cluster, chip, a.virt, host_l1_addr(hal, a.core, HalL1MemAddrType::PROFILER));
    }
    for (auto& [core, kind] : kinds) {
        launch_resident(device, *kind.program);
    }
    for (Tile& a : tiles) {
        const tt_cxy_pair core(chip, a.virt);
        kernel_profiler::TileNetTable t{};
        const auto table_bytes = static_cast<uint32_t>(
            offsetof(kernel_profiler::TileNetTable, partner) +
            a.reads.size() * sizeof(kernel_profiler::TileNetPartner));
        const auto await = [&](uint32_t ready) {
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
            do {
                cluster.read_core(&t, table_bytes, core, table_of(a));
                TT_FATAL(
                    t.ready == ready || std::chrono::steady_clock::now() < deadline,
                    "streaming profiler: device {} tile {} did not post its tile clock table",
                    chip,
                    a.virt.str());
            } while (t.ready != ready);
        };
        await(nonce);
        const uint32_t go = kernel_profiler::kTileNetGoMeasure;
        cluster.write_core(&go, sizeof(go), core, table_of(a));
        await(~nonce);
        for (size_t m = 0; m < a.reads.size(); m++) {
            const kernel_profiler::TileNetPartner& p = t.partner[m];
            a.reads[m].median2 = p.median2;
            a.reads[m].coarse = static_cast<int64_t>((uint64_t{p.coarse_hi} << 32) | p.coarse_lo);
        }
    }
    for (const Tile& a : tiles) {
        const uint32_t go = kernel_profiler::kTileNetGoExit;
        cluster.write_core(&go, sizeof(go), tt_cxy_pair(chip, a.virt), table_of(a));
    }
    for (auto& [core, kind] : kinds) {
        detail::WaitProgramDone(device, *kind.program, false);
    }
    // Fast dispatch's go signal reaches every core and a host-launched kernel leaves its slot valid, so restore the
    // firmware's initial launch message.
    for (const Tile& a : tiles) {
        auto msg = hal.get_dev_msgs_factory(a.core).create<dev_msgs::launch_msg_t>();
        cluster.write_core(
            msg.data(),
            static_cast<uint32_t>(msg.size()),
            tt_cxy_pair(chip, a.virt),
            host_l1_addr(hal, a.core, HalL1MemAddrType::LAUNCH));
    }
}

}  // namespace

void measure_tile_clocks(IDevice* device, ContextId ctx) {
    const auto& mc = MetalContext::instance(ctx);
    const uint32_t chip = static_cast<uint32_t>(device->id());
    if (!can_capture(mc) || service().tile_clocks(ctx, chip) != nullptr) {
        return;
    }
    std::vector<Tile> tiles = plan_tiles(device, ctx);
    read_network(device, ctx, tiles);
    const auto is_active_eth = [&](uint32_t i) { return tiles[i].core == HalProgrammableCoreType::ACTIVE_ETH; };
    const uint32_t n = static_cast<uint32_t>(tiles.size());

    std::vector<std::optional<size_t>> col(n);
    std::vector<uint32_t> tile_of;
    for (uint32_t i = 1; i < n; i++) {
        if (!is_active_eth(i)) {
            col[i] = tile_of.size();
            tile_of.push_back(i);
        }
    }
    // An active eth tile's reads leave its NIU about 6 cycles later than an idle eth tile's, which would bias a
    // mirrored pair with one by a tick and a half. So those tiles are placed from the both-NoC reads instead.
    std::vector<std::array<std::optional<size_t>, 2>> ends;
    std::vector<double> offsets;
    for (uint32_t s = 0; s < n; s++) {
        for (const Reading& r : tiles[s].reads) {
            if (r.twin || r.noc != 0 || is_active_eth(s) || is_active_eth(r.t)) {
                continue;
            }
            const Reading& back = *std::ranges::find(tiles[r.t].reads, s, &Reading::t);
            ends.push_back({col[r.t], col[s]});
            offsets.push_back(static_cast<double>(r.whole2() - back.whole2()) / 4.0);
        }
    }
    const Potential potential = solve_potential(ends, offsets, tile_of.size());
    TT_FATAL(
        !potential.unreached,
        "streaming profiler: device {} {} tile ({},{}) has no chain of mirrored pairs to the first Tensix tile",
        chip,
        kind_name(tiles[tile_of[*potential.unreached]].type),
        tiles[tile_of[*potential.unreached]].logical.x,
        tiles[tile_of[*potential.unreached]].logical.y);
    const std::vector<double>& solved = potential.x;
    std::vector<double> x(n, 0.0);
    for (size_t c = 0; c < tile_of.size(); c++) {
        x[tile_of[c]] = solved[c];
    }

    std::vector<std::optional<double>> est(n);
    double shift = 0.0;
    size_t refs = 0;
    for (const Tile& a : tiles) {
        for (const Reading& r : a.reads) {
            if (!r.twin) {
                continue;
            }
            est[r.t] = static_cast<double>(a.reads[*r.twin].whole2() + r.whole2()) / 4.0;
            if (!is_active_eth(r.t)) {
                shift += x[r.t] - *est[r.t];
                refs++;
            }
        }
    }
    for (uint32_t t = 0; t < n; t++) {
        if (!is_active_eth(t)) {
            continue;
        }
        TT_FATAL(
            refs != 0 && est[t],
            "streaming profiler: device {} active eth tile ({},{}) has no both-NoC reading to place it from",
            chip,
            tiles[t].logical.x,
            tiles[t].logical.y);
        x[t] = shift / static_cast<double>(refs) + *est[t];
    }

    TileClocks clocks;
    for (uint32_t i = 0; i < n; i++) {
        clocks.tiles.push_back(TileClock{tiles[i].type, tiles[i].logical, std::llround(x[i])});
    }
    service().set_tile_clocks(ctx, chip, std::move(clocks));
}

}  // namespace tt::tt_metal::streaming_profiler
