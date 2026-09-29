// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/streaming_profiler/device_programs.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <concepts>
#include <cstdlib>
#include <iterator>
#include <numeric>
#include <ranges>
#include <span>
#include <string>
#include <fmt/format.h>
#include <fmt/ranges.h>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <utility>

#include <tt-logger/tt-logger.hpp>

#include <tt-metalium/device.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/kernel_types.hpp>

#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include <tt-metalium/experimental/fabric/fabric_types.hpp>
#include <tt-metalium/experimental/sockets/d2h_socket.hpp>
#include <tt-metalium/experimental/sockets/mesh_socket.hpp>
#include <umd/device/cluster.hpp>
#include <umd/device/types/core_coordinates.hpp>

#include "context/metal_context.hpp"
#include "distributed/mesh_device_impl.hpp"
#include "impl/kernels/kernel.hpp"
#include "llrt/metal_soc_descriptor.hpp"
#include "llrt/tt_cluster.hpp"
#include "hostdev/streaming_profiler_common.h"
#include "hostdev/streaming_profiler_sync.h"
#include "impl/streaming_profiler/link_sync.hpp"
#include "impl/streaming_profiler/service.hpp"
#include "impl/streaming_profiler/sync_engine.hpp"

namespace tt::tt_metal::streaming_profiler {

namespace {

// Views 7 and 2 are the least reliable at bring-up, so they're dropped first.
constexpr std::array<uint32_t, 8> kRelayBankRoster = {5u, 6u, 4u, 1u, 0u, 3u, 7u, 2u};
// Two generations of two slots, plus three the spool splits into its two bounce buffers.
constexpr uint32_t kStageSlots = 7;
// A core's record is 128 B, a power of two so the kernel indexes it with a shift. 72 covers two relays on a 140-core
// grid.
constexpr uint32_t kMaxRelayCores = 72;
constexpr uint32_t kScratchBytes = kMaxRelayCores * 128;
static_assert(kScratchBytes % 64 == 0);
constexpr uint32_t kCfgReserve = 8 * 1024;
constexpr uint32_t kPageSize = kernel_profiler::SPSC_SPAN_PAGE_WORDS * 4;
constexpr uint32_t kNRisc = kernel_profiler::PROFILER_SPSC_TENSIX_RISC;
// Rides out multi-second page-fault stalls on the sync thread.
constexpr uint32_t kEthMinFifoBytes = 128u << 20;
constexpr uint32_t kEthSyncRingBytes = kernel_profiler::kSyncRingRecords * sizeof(kernel_profiler::SyncRecord);
constexpr uint32_t kCtrlBytes = 2 * kernel_profiler::kRelayCtrlWordStride;
static_assert(sizeof(kernel_profiler::RelayCtrl) <= kCtrlBytes);
constexpr uint32_t kRingHeaderBytes = 64;
constexpr uint32_t kArmedOffset = kernel_profiler::PROFILER_ARMED * sizeof(uint32_t);
constexpr uint32_t kLinkCtl = offsetof(kernel_profiler::LinkSyncL1, ctl);
constexpr uint32_t kLinkDone = offsetof(kernel_profiler::LinkSyncL1, done);
constexpr uint32_t kLinkDiag = offsetof(kernel_profiler::LinkSyncL1, diag);

struct SocketSpec {
    uint32_t cfg = 0;
    uint32_t fifo_bytes = 0;
    bool sync = false;
};

void write_u32(tt::Cluster& cluster, uint32_t chip, const CoreCoord& virt, uint64_t addr, uint32_t value) {
    cluster.write_core(&value, sizeof(value), tt_cxy_pair(chip, virt), addr);
}

template <std::predicate<uint32_t> Pred>
[[nodiscard]] bool poll_word(
    tt::Cluster& cluster, const tt_cxy_pair& core, uint64_t addr, std::chrono::milliseconds timeout, Pred pred) {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (true) {
        uint32_t word = 0;
        cluster.read_core(&word, sizeof(word), core, addr);
        if (pred(word)) {
            return true;
        }
        if (std::chrono::steady_clock::now() >= deadline) {
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

void stop_and_await(
    tt::Cluster& cluster,
    const tt_cxy_pair& core,
    uint64_t stop_addr,
    uint32_t stop_val,
    uint64_t done_addr,
    std::string_view what) {
    cluster.write_core(&stop_val, sizeof(stop_val), core, stop_addr);
    const bool stopped =
        poll_word(cluster, core, done_addr, std::chrono::seconds(10), [](uint32_t w) { return w != 0; });
    TT_FATAL(stopped, "streaming profiler: device {} {} did not stop within 10 s", core.chip, what);
}

KernelHandle idle_eth_kernel(
    Program& program,
    const std::string& src,
    const CoreCoord& core,
    DataMovementProcessor processor,
    std::unordered_map<std::string, uint32_t> named_args) {
    return CreateKernel(
        program,
        src,
        core,
        EthernetConfig{
            .eth_mode = Eth::IDLE,
            .noc = processor == DataMovementProcessor::RISCV_0 ? NOC::RISCV_0_default : NOC::RISCV_1_default,
            .processor = processor,
            .named_compile_args = std::move(named_args)});
}

void compile_resident(IDevice* device, Program& program) {
    detail::CompileProgram(device, program, /*force_slow_dispatch=*/true);
    detail::WriteRuntimeArgsToDevice(device, program, /*force_slow_dispatch=*/true);
}

void launch_compiled(IDevice* device, Program& program) {
    detail::LaunchProgram(device, program, /*wait_until_cores_done=*/false, /*force_slow_dispatch=*/true);
}

}  // namespace

CoreCoords locate(tt::Cluster& cluster, uint32_t chip, const CoreCoord& logical, CoreType type) {
    return CoreCoords{
        .logical = logical,
        .virt = cluster.get_virtual_coordinate_from_logical_coordinates(chip, logical, type),
        .phys = cluster.get_physical_coordinate_from_logical_coordinates(chip, logical, type, /*no_warn=*/true)};
}

uint64_t host_l1_addr(const Hal& hal, HalProgrammableCoreType core, HalL1MemAddrType type) {
    return core == HalProgrammableCoreType::DRAM ? hal.get_dev_noc_addr(core, type) : hal.get_dev_addr(core, type);
}

void zero_l1(tt::Cluster& cluster, uint32_t chip, const CoreCoord& virt, uint64_t addr, uint32_t bytes) {
    static constexpr std::array<uint8_t, kernel_profiler::PROFILER_L1_CONTROL_BUFFER_SIZE> kZero{};
    TT_FATAL(bytes <= kZero.size(), "streaming profiler: zero_l1 of {} B exceeds its {} B buffer", bytes, kZero.size());
    cluster.write_core(kZero.data(), bytes, tt_cxy_pair(chip, virt), addr);
}

void zero_profiler_control(tt::Cluster& cluster, uint32_t chip, const CoreCoord& virt, uint64_t addr) {
    zero_l1(cluster, chip, virt, addr, kernel_profiler::PROFILER_L1_CONTROL_BUFFER_SIZE);
}

uint8_t dram_view_endpoint_noc_mask(const metal_SocDescriptor& soc, const CoreCoord& logical) {
    return soc.get_dram_endpoint_noc_mask(soc.get_physical_dram_core_from_logical(logical));
}

void launch_resident(IDevice* device, Program& program) {
    compile_resident(device, program);
    launch_compiled(device, program);
}

std::vector<CoreCoord> sorted_yx(const std::unordered_set<CoreCoord>& cores) {
    std::vector<CoreCoord> out(cores.begin(), cores.end());
    std::ranges::sort(out, {}, [](const CoreCoord& c) { return std::pair(c.y, c.x); });
    return out;
}

DevicePrograms::DeviceCtx::DeviceCtx() = default;
DevicePrograms::DeviceCtx::~DeviceCtx() = default;
DevicePrograms::DeviceCtx::DeviceCtx(DeviceCtx&&) noexcept = default;

DevicePrograms::DevicePrograms() = default;

DevicePrograms::~DevicePrograms() {
    if (!quiesced_) {
        quiesce({});
    }
}

tt::Cluster& DevicePrograms::get_cluster() const { return MetalContext::instance(context_id_).get_cluster(); }

void DevicePrograms::carve_l1(const Hal& hal) {
    prof_l1_ = hal.get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::PROFILER);
    const uint32_t slot_bytes = kernel_profiler::spsc_span_slot_words(kNRisc) * sizeof(uint32_t);
    const uint32_t base = hal.get_dev_addr(HalProgrammableCoreType::DRAM, HalL1MemAddrType::UNRESERVED);
    const uint32_t region = hal.get_dev_size(HalProgrammableCoreType::DRAM, HalL1MemAddrType::UNRESERVED);
    l1_.stage_base = base;
    l1_.core_records = l1_.stage_base + kStageSlots * slot_bytes;
    l1_.ctrl_local = l1_.core_records + kScratchBytes;
    l1_.cfg = base + region - kCfgReserve;
    l1_.ctrl =
        hal.get_dev_noc_addr(HalProgrammableCoreType::DRAM, HalL1MemAddrType::UNRESERVED) + (l1_.ctrl_local - base);
    TT_FATAL(
        l1_.ctrl_local + kCtrlBytes <= l1_.cfg,
        "streaming profiler: DRISC L1 ({} B unreserved) cannot hold a relay's {} staging slots, core records and "
        "socket config",
        region,
        kStageSlots);

    const uint32_t ebase = hal.get_dev_addr(HalProgrammableCoreType::IDLE_ETH, HalL1MemAddrType::UNRESERVED);
    const uint32_t esize = hal.get_dev_size(HalProgrammableCoreType::IDLE_ETH, HalL1MemAddrType::UNRESERVED);
    const uint32_t need =
        kCfgReserve + kCtrlBytes + slot_bytes + kernel_profiler::kEthSyncScratchBytes + kEthSyncRingBytes + kPageSize;
    eth_l1_.cfg = ebase + esize - kCfgReserve;
    eth_l1_.sync_cfg = eth_l1_.cfg + kCfgReserve / 2;
    eth_l1_.ctrl = eth_l1_.cfg - kCtrlBytes;
    eth_l1_.stage = (eth_l1_.ctrl - slot_bytes) & ~(kPageSize - 1u);  // the pack pads assume a page-aligned slot
    eth_l1_.scratch = (eth_l1_.stage - kernel_profiler::kEthSyncScratchBytes) & ~(kPageSize - 1u);
    eth_l1_.sync_ring = eth_l1_.scratch - kEthSyncRingBytes;
    eth_l1_.link = link_sync::l1_addr(hal);
    eth_l1_.link_ring = eth_l1_.link + offsetof(kernel_profiler::LinkSyncL1, ring);
    TT_FATAL(
        esize >= need && eth_l1_.sync_ring >= ebase,
        "streaming profiler: idle-eth L1 too small for the clock tracker ({} B unreserved, {} needed)",
        esize,
        need);

    const uint32_t bytes = MetalContext::instance(context_id_).rtoptions().get_streaming_profiler_spool_mb() << 20;
    if (bytes != 0) {
        TT_FATAL(
            hal.get_dev_size(HalDramMemAddrType::PROFILER) >= bytes,
            "streaming profiler: the HAL's profiler DRAM region ({} B) is smaller than the {} B GDDR spool",
            hal.get_dev_size(HalDramMemAddrType::PROFILER),
            bytes);
        l1_.spool_addr = static_cast<uint32_t>(hal.get_dev_addr(HalDramMemAddrType::PROFILER));
        l1_.spool_bytes = bytes;
    }
}

bool DevicePrograms::boot(const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
    context_id_ = mesh_device->impl().get_context_id();
    auto& cluster = get_cluster();
    const auto& hal = MetalContext::instance(context_id_).hal();
    const auto& rtopts = MetalContext::instance(context_id_).rtoptions();
    fabric_link_sync_ = MetalContext::instance(context_id_).get_fabric_config() != tt_fabric::FabricConfig::DISABLED;
    capture_.sync_check = rtopts.get_streaming_profiler_sync_check_enabled();

    if (!can_capture(MetalContext::instance(context_id_))) {
        log_warning(
            tt::LogMetal, "[streaming profiler] not capturing: it needs Blackhole with DRAM programmable cores");
        return false;
    }
    carve_l1(hal);
    const uint32_t relay_fifo_bytes = (rtopts.get_streaming_profiler_fifo_mb() << 20) / kPageSize * kPageSize;
    const uint32_t eth_fifo_bytes = std::max(relay_fifo_bytes, kEthMinFifoBytes);

    for (const auto& coord : distributed::MeshCoordinateRange(mesh_device->shape())) {
        if (!mesh_device->is_local(coord)) {
            continue;
        }
        DeviceCtx& ctx = devices_.emplace_back();
        ctx.device = mesh_device->get_device(coord);
        ctx.chip_id = static_cast<uint32_t>(ctx.device->id());
        ctx.numa_node = static_cast<int>(cluster.get_numa_node_for_device(ctx.chip_id));
        ctx.cap.chip_id = ctx.chip_id;
        enumerate_worker_grid(mesh_device, ctx);
        enumerate_eth_cores(ctx);
        choose_relay_cores(mesh_device, ctx);

        const TileClocks* clocks = service().tile_clocks(context_id_, ctx.chip_id);
        TT_FATAL(clocks != nullptr, "streaming profiler: device {} has no tile clocks", ctx.chip_id);
        const int64_t wall = clocks->offset(CoreType::ETH, ctx.tracker.core.logical);
        for (size_t i = 0; i < ctx.producers.size(); i++) {
            const CoreType type = i < ctx.n_workers ? CoreType::WORKER : CoreType::ETH;
            ctx.cap.tile_offset.push_back(wall - clocks->offset(type, ctx.producers[i].logical));
        }
        if (ctx.ruler) {
            ctx.cap.ruler_offset = wall - clocks->offset(CoreType::ETH, ctx.ruler->core.logical);
        }

        const auto start = [&](ResidentCore& r,
                               HalProgrammableCoreType core_type,
                               const std::vector<SocketSpec>& specs,
                               std::unique_ptr<Program> program) {
            std::vector<std::unique_ptr<distributed::D2HSocket>> made;
            for (const SocketSpec& spec : specs) {
                made.push_back(std::make_unique<distributed::D2HSocket>(
                    mesh_device,
                    distributed::MeshCoreCoord{coord, r.core.phys},
                    spec.fifo_bytes,
                    distributed::D2HSocket::ExternalConfigBuffer{.address = spec.cfg, .sender_core_type = core_type},
                    distributed::D2HSocket::ProcessScope::InProcess));
                made.back()->set_page_size(kPageSize);
            }
            // Teardown leaves stop at 1 or 2, which the eth relay would take as an immediate stop.
            zero_l1(cluster, ctx.chip_id, r.core.virt, r.ctrl, kCtrlBytes);
            try {
                launch_resident(ctx.device, *program);
            } catch (const std::exception& e) {
                TT_THROW("streaming profiler: device {} {} failed to load: {}", ctx.chip_id, r.name, e.what());
            }
            // A relay launches fire-and-forget, so a core stuck in reset reports nothing. Two heartbeats prove its loop
            // runs.
            uint32_t hb = 0;
            const bool started = poll_word(
                cluster,
                tt_cxy_pair(ctx.chip_id, r.core.virt),
                r.ctrl + offsetof(kernel_profiler::RelayCtrl, heartbeat),
                std::chrono::milliseconds(500),
                [&](uint32_t w) {
                    hb = w;
                    return w >= 2;
                });
            TT_FATAL(
                started,
                "streaming profiler: device {} {} did not start (heartbeat {} after launch)",
                ctx.chip_id,
                r.name,
                hb);
            r.sock_idx = static_cast<uint32_t>(ctx.sockets.size());
            r.n_sockets = static_cast<uint32_t>(made.size());
            for (size_t k = 0; k < made.size(); k++) {
                sockets_.push_back(CapturedSocket{
                    .socket = made[k].get(),
                    .dev = static_cast<uint32_t>(devices_.size() - 1),
                    .index = static_cast<uint32_t>(ctx.sockets.size()),
                    .numa_node = ctx.numa_node,
                    .sync = specs[k].sync});
                ctx.sockets.push_back(std::move(made[k]));
            }
            r.program = std::move(program);
        };
        for (uint32_t d = 0; d < ctx.relays.size(); d++) {
            if (auto program = relay_program(ctx, d)) {
                start(
                    ctx.relays[d],
                    HalProgrammableCoreType::DRAM,
                    {{.cfg = l1_.cfg, .fifo_bytes = relay_fifo_bytes}},
                    std::move(program));
            }
        }
        start(ctx.tracker, HalProgrammableCoreType::IDLE_ETH, {}, tracker_program(ctx));
        std::vector<SocketSpec> eth_sockets = {{.cfg = eth_l1_.sync_cfg, .fifo_bytes = eth_fifo_bytes, .sync = true}};
        if (rtopts.get_streaming_profiler_eth_enabled()) {
            eth_sockets.push_back({.cfg = eth_l1_.cfg, .fifo_bytes = relay_fifo_bytes});
        }
        start(ctx.eth_relay, HalProgrammableCoreType::IDLE_ETH, eth_sockets, eth_relay_program(ctx));
        set_producers_armed(ctx, true);
    }
    plan_links();
    if (devices_.empty()) {
        return false;
    }
    for (DeviceCtx& ctx : devices_) {
        capture_.devices.push_back(std::move(ctx.cap));
    }
    log_info(
        tt::LogMetal,
        "[streaming profiler] active on {} device(s){}",
        devices_.size(),
        rtopts.get_streaming_profiler_tracy_enabled() ? " with the Tracy sink" : "");
    return true;
}

DevicePrograms::Producer& DevicePrograms::enroll(
    DeviceCtx& ctx, const CoreCoords& core, CoreType type, uint64_t prof_l1, bool blocking) {
    using experimental::streaming_profiler::Processor;
    // An eth core has two RISCs; its other lanes never carry a record.
    constexpr std::array<Processor, kNRisc> kTensix = {
        Processor::BRISC, Processor::NCRISC, Processor::TRISC0, Processor::TRISC1, Processor::TRISC2};
    constexpr std::array<Processor, kNRisc> kEth = {
        Processor::ERISC0, Processor::ERISC1, Processor::ERISC1, Processor::ERISC1, Processor::ERISC1};
    ctx.cap.core_xy.push_back(packed_xy(core.virt));
    for (const Processor processor : type == CoreType::ETH ? kEth : kTensix) {
        ctx.cap.lanes.push_back(experimental::streaming_profiler::Core{
            .logical = core.logical,
            .physical = core.phys,
            .chip_id = static_cast<ChipId>(ctx.chip_id),
            .processor = processor});
    }
    return ctx.producers.emplace_back(Producer{core, prof_l1, blocking});
}

void DevicePrograms::enumerate_worker_grid(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device, DeviceCtx& ctx) {
    auto& cluster = get_cluster();
    const uint32_t chip = ctx.chip_id;
    // A producer that isn't drained fills its ring and hangs the host at close.
    const CoreCoord grid = mesh_device->compute_with_storage_grid_size();
    for (uint32_t ly = 0; ly < grid.y; ly++) {
        for (uint32_t lx = 0; lx < grid.x; lx++) {
            const Producer& p = enroll(
                ctx, locate(cluster, chip, CoreCoord{lx, ly}, CoreType::WORKER), CoreType::WORKER, prof_l1_, true);
            zero_profiler_control(cluster, chip, p.virt, p.prof_l1);
        }
    }
    ctx.n_workers = static_cast<uint32_t>(ctx.producers.size());

    // All Tensix cores: a dispatch core's ring is never drained and its L1 survives re-init, so a stale armed flag
    // there hangs close.
    const CoreCoord tensix_grid = cluster.get_soc_desc(chip).get_grid_size(CoreType::TENSIX);
    for (uint32_t ly = 0; ly < tensix_grid.y; ly++) {
        for (uint32_t lx = 0; lx < tensix_grid.x; lx++) {
            const CoreCoord v =
                cluster.get_virtual_coordinate_from_logical_coordinates(chip, CoreCoord{lx, ly}, CoreType::WORKER);
            write_u32(cluster, chip, v, prof_l1_ + kArmedOffset, 0);
        }
    }
}

void DevicePrograms::choose_relay_cores(const std::shared_ptr<distributed::MeshDevice>& mesh_device, DeviceCtx& ctx) {
    const uint32_t chip = ctx.chip_id;
    const auto& soc = get_cluster().get_soc_desc(chip);
    const uint32_t nbanks = static_cast<uint32_t>(soc.get_num_dram_views());
    TT_FATAL(nbanks != 0, "streaming profiler: device {} has no DRAM views to host a relay", chip);
    ctx.relays.resize(std::min<uint32_t>(kRelayBankRoster.size(), nbanks));
    std::vector<uint32_t> banks;
    std::ranges::copy_if(kRelayBankRoster, std::back_inserter(banks), [&](uint32_t b) { return b < nbanks; });
    for (uint32_t d = 0; d < ctx.relays.size(); d++) {
        ctx.relays[d].core.logical = mesh_device->impl().pick_unused_dram_logical_core(ctx.device, banks[d]);
    }
    // pick_unused_dram_logical_core() can't see two views resolving to one physical port, and two relays on one L1
    // would overlap.
    for (uint32_t a = 0; a < ctx.relays.size(); a++) {
        for (uint32_t b = a + 1; b < ctx.relays.size(); b++) {
            TT_FATAL(
                ctx.relays[a].core.logical != ctx.relays[b].core.logical,
                "streaming profiler: DRISC {} (DRAM view {}) and DRISC {} (DRAM view {}) both resolve to logical "
                "DRAM core ({},{}). Two resident relay kernels cannot share a core.",
                a,
                banks[a],
                b,
                banks[b],
                ctx.relays[a].core.logical.x,
                ctx.relays[a].core.logical.y);
        }
    }
    // A channel carved into several views has an endpoint set per view, so one view's free subchannel can be another's
    // endpoint.
    for (uint32_t d = 0; d < ctx.relays.size(); d++) {
        ResidentCore& relay = ctx.relays[d];
        const CoreCoord translated = soc.get_physical_dram_core_from_logical(relay.core.logical);
        const uint8_t noc2axi_mask = dram_view_endpoint_noc_mask(soc, relay.core.logical);
        TT_FATAL(
            noc2axi_mask == 0,
            "streaming profiler: device {} relay {}'s DRISC ({},{}) is a DRAM view's preferred endpoint on NOC mask "
            "{:#x}, so firmware holds those NIUs in NOC2AXI mode and the relay cannot initiate NoC traffic on them",
            chip,
            d,
            translated.x,
            translated.y,
            noc2axi_mask);
        const tt::umd::CoreCoord phys = soc.translate_coord_to(
            tt::umd::CoreCoord(translated.x, translated.y, CoreType::DRAM, CoordSystem::TRANSLATED), CoordSystem::NOC0);
        relay.core.phys = CoreCoord(phys.x, phys.y);
        relay.core.virt = ctx.device->virtual_core_from_logical_core(relay.core.logical, CoreType::DRAM);
        relay.name = fmt::format("relay {}", d);
        relay.ctrl = l1_.ctrl;
    }
}

std::unique_ptr<Program> DevicePrograms::relay_program(const DeviceCtx& ctx, uint32_t d) {
    const ResidentCore& relay = ctx.relays[d];
    const uint32_t num_cores = ctx.n_workers;
    const uint32_t lo = static_cast<uint32_t>((static_cast<uint64_t>(num_cores) * d) / ctx.relays.size());
    const uint32_t hi = static_cast<uint32_t>((static_cast<uint64_t>(num_cores) * (d + 1)) / ctx.relays.size());
    const uint32_t my_cores = hi - lo;
    TT_FATAL(
        my_cores <= kMaxRelayCores,
        "streaming profiler: device {} relay {} would own {} cores but a relay holds at most {}; this {}-core grid "
        "needs {} relays and the part has {}",
        ctx.chip_id,
        d,
        my_cores,
        kMaxRelayCores,
        num_cores,
        (num_cores + kMaxRelayCores - 1) / kMaxRelayCores,
        ctx.relays.size());
    if (my_cores == 0) {
        return nullptr;
    }
    // The relay is built with PROFILE_KERNEL and nothing drains its own profiler ring, which fills after about 74
    // launches in one reset window and hangs the RISC in firmware init.
    const auto& hal = MetalContext::instance(context_id_).hal();
    zero_profiler_control(
        get_cluster(),
        ctx.chip_id,
        relay.core.virt,
        host_l1_addr(hal, HalProgrammableCoreType::DRAM, HalL1MemAddrType::PROFILER));
    auto program = std::make_unique<Program>(CreateProgram());
    const KernelHandle kid = CreateKernel(
        *program,
        "tt_metal/impl/streaming_profiler/kernels/drisc_relay.cpp",
        relay.core.logical,
        // NOC 1 writes take about twice as long, so a relay there would take nearly every stall.
        DramConfig{
            .noc = NOC::NOC_0,
            .defines = {{"STREAMING_PROFILER_RELAY_KERNEL", "1"}},
            .named_compile_args = {
                {"stage_base", l1_.stage_base},
                {"n_stage", kStageSlots},
                {"core_records", l1_.core_records},
                {"ctrl", l1_.ctrl_local},
                {"socket_config_addr", l1_.cfg},
                {"max_cores", kMaxRelayCores},
                // Spreads the relays over two request VCs.
                {"write_vc", (d & 2u) ? 0u : 1u},
                {"spool_base", l1_.spool_addr},
                {"spool_bytes", l1_.spool_bytes}}});
    std::vector<uint32_t> rt = {my_cores, static_cast<uint32_t>(prof_l1_)};
    // Reversed so the last-launched cores get the first-serviced slots.
    std::ranges::copy(std::span(ctx.cap.core_xy).subspan(lo, my_cores) | std::views::reverse, std::back_inserter(rt));
    SetRuntimeArgs(*program, kid, relay.core.logical, rt);
    return program;
}

void DevicePrograms::enumerate_eth_cores(DeviceCtx& ctx) {
    auto& cluster = get_cluster();
    const auto& hal = MetalContext::instance(context_id_).hal();
    const uint32_t chip = ctx.chip_id;
    const std::vector<CoreCoord> idle = sorted_yx(ctx.device->get_inactive_ethernet_cores());
    TT_FATAL(
        idle.size() >= 2,
        "streaming profiler: device {} has {} idle ethernet cores; the clock tracker and its eth relay need two",
        chip,
        idle.size());
    std::vector<CoreCoords> idle_eth;
    std::ranges::transform(idle, std::back_inserter(idle_eth), [&](const CoreCoord& l) {
        return locate(cluster, chip, l, CoreType::ETH);
    });
    ctx.tracker = {.core = idle_eth.front(), .name = "clock tracker", .ctrl = eth_l1_.ctrl};
    const uint64_t ieth_prof_l1 = hal.get_dev_addr(HalProgrammableCoreType::IDLE_ETH, HalL1MemAddrType::PROFILER);
    const Producer& pp = enroll(ctx, ctx.tracker.core, CoreType::ETH, ieth_prof_l1, true);
    zero_profiler_control(cluster, chip, pp.virt, pp.prof_l1);
    const auto nearest = [&](const CoreCoord& to, const std::optional<CoreCoord>& skip) -> std::optional<CoreCoords> {
        auto others =
            idle_eth | std::views::drop(1) | std::views::filter([&](const CoreCoords& c) { return c.phys != skip; });
        const auto it = std::ranges::min_element(others, {}, [&](const CoreCoords& c) {
            return static_cast<uint32_t>(std::abs(static_cast<int>(c.phys.x) - static_cast<int>(to.x))) +
                   static_cast<uint32_t>(std::abs(static_cast<int>(c.phys.y) - static_cast<int>(to.y)));
        });
        return it == others.end() ? std::nullopt : std::optional(*it);
    };
    ctx.eth_relay = {
        .core = *nearest(ctx.tracker.core.phys, std::nullopt), .name = "idle-eth relay", .ctrl = eth_l1_.ctrl};
    if (capture_.sync_check) {
        if (const std::optional<CoreCoords> ruler = nearest(ctx.eth_relay.core.phys, ctx.eth_relay.core.phys)) {
            ctx.ruler = ResidentCore{.core = *ruler, .name = "the sync check's ruler", .ctrl = eth_l1_.ctrl};
        }
        TT_FATAL(
            ctx.ruler.has_value(),
            "streaming profiler: device {} has {} idle ethernet cores; the sync check's ruler needs a third",
            chip,
            idle_eth.size());
    }
    const uint64_t aeth_prof_l1 = hal.get_dev_addr(HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::PROFILER);
    for (const CoreCoord& al : sorted_yx(ctx.device->get_active_ethernet_cores(/*skip_reserved_tunnel_cores=*/true))) {
        const Producer& p = enroll(ctx, locate(cluster, chip, al, CoreType::ETH), CoreType::ETH, aeth_prof_l1, false);
        if (!fabric_link_sync_) {
            zero_profiler_control(cluster, chip, p.virt, p.prof_l1);
            continue;
        }
        // A fabric router already started this ring from its tail, so zeroing it would desync the tracker.
        std::array<uint32_t, kNRisc> tails{};
        cluster.read_core(
            tails.data(),
            sizeof(tails),
            tt_cxy_pair(chip, p.virt),
            p.prof_l1 + kernel_profiler::SPSC_RING_TAIL_0 * sizeof(uint32_t));
        cluster.write_core(
            tails.data(),
            sizeof(tails),
            tt_cxy_pair(chip, p.virt),
            p.prof_l1 + kernel_profiler::SPSC_RING_HEAD_0 * sizeof(uint32_t));
        // Safe to clear: nothing records before Run, and a core without a link end would keep an old tail.
        for (const uint32_t word : {kernel_profiler::SPSC_LINK_SYNC_TAIL, kernel_profiler::SPSC_LINK_SYNC_HEAD}) {
            write_u32(cluster, chip, p.virt, p.prof_l1 + word * sizeof(uint32_t), 0);
        }
    }
    for (const CoreCoords& c : idle_eth | std::views::drop(1)) {
        if (c.phys != ctx.eth_relay.core.phys && (!ctx.ruler || c.phys != ctx.ruler->core.phys)) {
            const Producer& p = enroll(ctx, c, CoreType::ETH, ieth_prof_l1, false);
            zero_profiler_control(cluster, chip, p.virt, p.prof_l1);
        }
    }
}

std::unique_ptr<Program> DevicePrograms::tracker_program(const DeviceCtx& ctx) {
    const CoreCoord& core = ctx.tracker.core.logical;
    zero_l1(get_cluster(), ctx.chip_id, ctx.tracker.core.virt, eth_l1_.scratch, kRingHeaderBytes);
    auto program = std::make_unique<Program>(CreateProgram());
    idle_eth_kernel(
        *program,
        "tt_metal/impl/streaming_profiler/kernels/eth_clock_sampler.cpp",
        core,
        DataMovementProcessor::RISCV_0,
        {{"ctrl_addr", eth_l1_.ctrl}, {"sample_ring_addr", eth_l1_.scratch}});
    idle_eth_kernel(
        *program,
        "tt_metal/impl/streaming_profiler/kernels/eth_clock_model.cpp",
        core,
        DataMovementProcessor::RISCV_1,
        {{"ctrl_addr", eth_l1_.ctrl}, {"sync_ring_addr", eth_l1_.sync_ring}, {"sample_ring_addr", eth_l1_.scratch}});
    return program;
}

std::unique_ptr<Program> DevicePrograms::eth_relay_program(const DeviceCtx& ctx) {
    auto program = std::make_unique<Program>(CreateProgram());
    const Producer& tracker = ctx.producers[ctx.n_workers];
    std::unordered_map<std::string, uint32_t> relay_args = {
        {"frames_cfg", eth_l1_.cfg},
        {"sync_cfg", eth_l1_.sync_cfg},
        {"stage", eth_l1_.stage},
        {"ctrl", eth_l1_.ctrl},
        {"scratch", eth_l1_.scratch},
        {"tracker_xy", packed_xy(tracker.virt)},
        {"tracker_prof_l1", static_cast<uint32_t>(tracker.prof_l1)},
        {"sync_ring", eth_l1_.sync_ring},
        {"link_ring", eth_l1_.link_ring}};
    if (ctx.ruler) {
        const CoreCoords& rc = ctx.ruler->core;
        relay_args["ruler_xy"] = packed_xy(rc.virt);
        zero_l1(get_cluster(), ctx.chip_id, rc.virt, ctx.ruler->ctrl, kCtrlBytes);
        idle_eth_kernel(
            *program,
            "tt_metal/impl/streaming_profiler/kernels/eth_clock_ruler.cpp",
            rc.logical,
            DataMovementProcessor::RISCV_0,
            {{"ctrl_addr", eth_l1_.ctrl}, {"sync_ring_addr", eth_l1_.sync_ring}});
    }
    const KernelHandle kid = idle_eth_kernel(
        *program,
        "tt_metal/impl/streaming_profiler/kernels/eth_relay.cpp",
        ctx.eth_relay.core.logical,
        DataMovementProcessor::RISCV_0,
        std::move(relay_args));
    const auto linked = std::span(ctx.producers).subspan(ctx.n_workers + 1);
    const auto n_linked = static_cast<uint32_t>(linked.size());
    TT_FATAL(
        n_linked <= kernel_profiler::kEthRelayMaxLinked,
        "streaming profiler: device {} has {} eth cores to drain; the eth relay holds at most {}",
        ctx.chip_id,
        n_linked,
        kernel_profiler::kEthRelayMaxLinked);
    std::vector<uint32_t> rt = {n_linked};
    for (const Producer& p : linked) {
        rt.push_back(packed_xy(p.virt));
        rt.push_back(static_cast<uint32_t>(p.prof_l1));
    }
    SetRuntimeArgs(*program, kid, ctx.eth_relay.core.logical, rt);
    return program;
}

void DevicePrograms::set_producers_armed(const DeviceCtx& ctx, bool armed) {
    auto& cluster = get_cluster();
    for (const Producer& p : ctx.producers | std::views::filter(&Producer::blocking)) {
        write_u32(cluster, ctx.chip_id, p.virt, p.prof_l1 + kArmedOffset, armed ? 1u : 0u);
    }
}

void DevicePrograms::plan_links() {
    auto& mc = MetalContext::instance(context_id_);
    const auto core_of = [&](size_t di, const CoreCoord& eth) {
        const DeviceCtx& d = devices_[di];
        const auto linked = std::ranges::subrange(d.producers.begin() + d.n_workers + 1, d.producers.end());
        const auto it = std::ranges::find(linked, eth, &Producer::logical);
        TT_FATAL(
            it != d.producers.end(),
            "streaming profiler: device {} link sync end eth({},{}) is not among the tracker's linked cores",
            d.chip_id,
            eth.x,
            eth.y);
        return static_cast<uint32_t>(it - d.producers.begin());
    };
    for (size_t a = 0; a < devices_.size(); a++) {
        const uint32_t chip_a = devices_[a].chip_id;
        for (size_t b = a + 1; b < devices_.size(); b++) {
            for (const link_sync::Link& link : link_sync::links_between(mc, chip_a, devices_[b].chip_id)) {
                const bool flip = link.chip_a != chip_a;
                const size_t dev_a = flip ? b : a, dev_b = flip ? a : b;
                capture_.links.push_back(CaptureContext::Link{
                    .dev_a = static_cast<uint32_t>(dev_a),
                    .dev_b = static_cast<uint32_t>(dev_b),
                    .core_a = core_of(dev_a, link.eth_a),
                    .core_b = core_of(dev_b, link.eth_b),
                    .eth_a = link.eth_a,
                    .eth_b = link.eth_b});
            }
        }
    }
    if (devices_.empty()) {
        return;
    }
    const std::vector<bool> reached = reached_from_root(capture_.links, devices_.size(), [](size_t) { return true; });
    std::vector<uint32_t> unreachable;
    for (size_t di = 0; di < devices_.size(); di++) {
        if (!reached[di]) {
            unreachable.push_back(devices_[di].chip_id);
        }
    }
    TT_FATAL(
        unreachable.empty(),
        "streaming profiler: no link path to the root chip {} from chips {}",
        devices_[0].chip_id,
        fmt::format("{}", fmt::join(unreachable, ", ")));
}

std::array<DevicePrograms::LinkEnd, 2> DevicePrograms::link_ends(const CaptureContext::Link& L) const {
    const auto end = [&](uint32_t dev, uint32_t core, const CoreCoord& eth, bool sender) {
        const DeviceCtx& d = devices_[dev];
        const Producer& p = d.producers[core];
        return LinkEnd{d.device, d.chip_id, eth, p.virt, p.prof_l1, sender};
    };
    return {end(L.dev_a, L.core_a, L.eth_a, true), end(L.dev_b, L.core_b, L.eth_b, false)};
}

void DevicePrograms::launch_links() {
    auto& cluster = get_cluster();
    links_running_ = true;
    // A router's sender waits for Run like a resident one.
    if (!fabric_link_sync_) {
        for (const CaptureContext::Link& L : capture_.links) {
            const std::array ends = link_ends(L);
            static_assert(kLinkDone == kLinkCtl + sizeof(uint32_t));
            for (const LinkEnd& e : ends) {
                zero_l1(cluster, e.chip, e.virt, eth_l1_.link + kLinkCtl, 2 * sizeof(uint32_t));
            }
            // Compile both first: the sender starts its handshake as soon as it runs.
            std::array<std::unique_ptr<Program>, 2> programs;
            for (size_t i = 0; i < ends.size(); i++) {
                programs[i] = std::make_unique<Program>(CreateProgram());
                const KernelHandle kid = CreateKernel(
                    *programs[i],
                    "tt_metal/impl/streaming_profiler/kernels/eth_ptp_link_end.cpp",
                    ends[i].eth,
                    EthernetConfig{.noc = NOC::RISCV_0_default, .compile_args = {ends[i].sender ? 1u : 0u}});
                SetRuntimeArgs(*programs[i], kid, ends[i].eth, {eth_l1_.link});
                compile_resident(ends[i].device, *programs[i]);
            }
            for (size_t i = 0; i < ends.size(); i++) {
                launch_compiled(ends[i].device, *programs[i]);
            }
            resident_links_.push_back(std::move(programs));
        }
    }
    for (const CaptureContext::Link& L : capture_.links) {
        const LinkEnd sender = link_ends(L)[0];
        write_u32(cluster, sender.chip, sender.virt, eth_l1_.link + kLinkCtl, kernel_profiler::kLinkSyncCtlRun);
    }
}

void DevicePrograms::stop_links() {
    if (!links_running_) {
        return;
    }
    auto& cluster = get_cluster();
    const uint32_t link_l1 = eth_l1_.link;
    // Each end must record two solved rounds before any stops: the engine solves a link from two rounds, and an end
    // records a round when the next one starts. Under the sync check only every kLinkSyncCheckSolveEvery-th round is
    // solved.
    constexpr uint32_t kRecordsPerRound = 2;
    const uint32_t rounds_needed = capture_.sync_check ? kernel_profiler::kLinkSyncCheckSolveEvery + 1 : 2;
    for (const CaptureContext::Link& L : capture_.links) {
        for (const LinkEnd& e : link_ends(L)) {
            if (!poll_word(
                    cluster,
                    tt_cxy_pair(e.chip, e.virt),
                    e.prof_l1 + kernel_profiler::SPSC_LINK_SYNC_TAIL * sizeof(uint32_t),
                    std::chrono::seconds(1),
                    [&](uint32_t tail) { return tail >= kRecordsPerRound * rounds_needed; })) {
                log_warning(
                    tt::LogMetal,
                    "[streaming profiler] link sync chip {} {}: fewer than {} rounds recorded",
                    e.chip,
                    e.virt.str(),
                    rounds_needed);
            }
        }
    }
    for (const CaptureContext::Link& L : capture_.links) {
        for (const LinkEnd& e : link_ends(L)) {
            const std::string_view role = e.sender ? "sender" : "receiver";
            if (!fabric_link_sync_) {
                stop_and_await(
                    cluster,
                    tt_cxy_pair(e.chip, e.virt),
                    link_l1 + kLinkCtl,
                    kernel_profiler::kLinkSyncCtlStop,
                    link_l1 + kLinkDone,
                    fmt::format("link sync {} {}", role, e.virt.str()));
            } else if (e.sender) {
                write_u32(cluster, e.chip, e.virt, link_l1 + kLinkCtl, kernel_profiler::kLinkSyncCtlStop);
            }
            kernel_profiler::LinkSyncDiag d{};
            cluster.read_core(&d, sizeof(d), tt_cxy_pair(e.chip, e.virt), link_l1 + kLinkDiag);
            if (d.rounds_lost != 0 || d.bursts_mismatched != 0 || d.frames_unstamped != 0) {
                log_warning(
                    tt::LogMetal,
                    "[streaming profiler] link sync chip {} {}: {} rounds not recorded, {} bursts whose ingress stamps "
                    "did not match their frames, {} frames without an egress stamp",
                    e.chip,
                    role,
                    d.rounds_lost,
                    d.bursts_mismatched,
                    d.frames_unstamped);
            }
        }
    }
    resident_links_.clear();
    links_running_ = false;
}

void DevicePrograms::start() {
    auto& cluster = get_cluster();
    const uint64_t go = eth_l1_.ctrl + offsetof(kernel_profiler::RelayCtrl, go);
    for (const DeviceCtx& ctx : devices_) {
        write_u32(cluster, ctx.chip_id, ctx.tracker.core.virt, go, 1);
        write_u32(cluster, ctx.chip_id, ctx.eth_relay.core.virt, go, 1);
    }
    // The ruler starts after the tracker's first instant, so its readings always have one before them.
    for (const DeviceCtx& ctx : devices_ | std::views::filter([](const DeviceCtx& c) { return c.ruler.has_value(); })) {
        const bool instant = poll_word(
            cluster,
            tt_cxy_pair(ctx.chip_id, ctx.tracker.core.virt),
            eth_l1_.ctrl + offsetof(kernel_profiler::RelayCtrl, sync_tail),
            std::chrono::seconds(10),
            [](uint32_t w) { return w != 0; });
        TT_FATAL(
            instant, "streaming profiler: device {} the clock tracker published no instant within 10 s", ctx.chip_id);
        write_u32(cluster, ctx.chip_id, ctx.ruler->core.virt, go, 1);
    }
    launch_links();
}

void DevicePrograms::request_stop(const DeviceCtx& ctx, const ResidentCore& r) {
    write_u32(
        get_cluster(),
        ctx.chip_id,
        r.core.virt,
        r.ctrl + offsetof(kernel_profiler::RelayCtrl, stop),
        kernel_profiler::kRelayStopQuiesce);
}

void DevicePrograms::await_stop(
    uint32_t device_index, const DeviceCtx& ctx, const ResidentCore& r, const RelayStateFn& on_state) {
    const auto sockets = std::span(ctx.sockets).subspan(r.sock_idx, r.n_sockets);
    const auto report = [&](RelayState state) {
        for (uint32_t k = 0; k < r.n_sockets; k++) {
            on_state(device_index, r.sock_idx + k, state);
        }
    };
    auto& cluster = get_cluster();
    bool awaiting_acks = false;
    uint32_t state = 0;
    const bool done =
        poll_word(cluster, tt_cxy_pair(ctx.chip_id, r.core.virt), r.ctrl, std::chrono::seconds(10), [&](uint32_t w) {
            // With no consumer, discard the pages so the relay's barriers still complete.
            for (const auto& sock : sockets) {
                if (!on_state && sock->pages_available() != 0) {
                    sock->discard_pending_pages();
                }
            }
            state = w & kernel_profiler::kRelayDoneMask;
            if (!awaiting_acks && on_state && state == kernel_profiler::kRelayAwaitingAcksWord) {
                report(RelayState::AwaitingAcks);
                awaiting_acks = true;
            }
            return state == kernel_profiler::kRelayDoneWord;
        });
    TT_FATAL(
        done,
        "streaming profiler: device {} {} did not finish within 10 s of its stop (state {:#x})",
        ctx.chip_id,
        r.name,
        state);
    // Done comes after the relay's socket barriers, so every byte is acked by then.
    if (on_state) {
        report(RelayState::Done);
    }
}

void DevicePrograms::quiesce(const RelayStateFn& on_state) {
    if (quiesced_) {
        return;
    }
    quiesced_ = true;
    stop_links();
    auto& cluster = get_cluster();
    // All devices are asked to stop a stage before any is waited on, so they stop together.
    const auto stage = [&](auto&& cores_of) {
        for (const DeviceCtx& ctx : devices_) {
            for (const ResidentCore* r : cores_of(ctx)) {
                request_stop(ctx, *r);
            }
        }
        for (uint32_t device_index = 0; device_index < devices_.size(); device_index++) {
            for (const ResidentCore* r : cores_of(devices_[device_index])) {
                await_stop(device_index, devices_[device_index], *r, on_state);
            }
        }
    };
    const auto running = [](const ResidentCore& r) {
        return r.program ? std::vector<const ResidentCore*>{&r} : std::vector<const ResidentCore*>{};
    };
    // The ruler rides in the eth relay's program.
    const auto ruler_running = [](const DeviceCtx& ctx) { return ctx.ruler && ctx.eth_relay.program; };
    stage([](const DeviceCtx& ctx) {
        std::vector<const ResidentCore*> relays;
        for (const ResidentCore& relay : ctx.relays) {
            if (relay.program) {
                relays.push_back(&relay);
            }
        }
        return relays;
    });
    // The ruler stops first, so its readings always have an instant after them.
    stage([&](const DeviceCtx& ctx) {
        return ruler_running(ctx) ? std::vector<const ResidentCore*>{&*ctx.ruler} : std::vector<const ResidentCore*>{};
    });
    // The tracker stops first, so its ring's tail is final when the eth relay drains it.
    stage([&](const DeviceCtx& ctx) { return running(ctx.tracker); });
    stage([&](const DeviceCtx& ctx) { return running(ctx.eth_relay); });
    for (const DeviceCtx& ctx : devices_) {
        if (ruler_running(ctx)) {
            kernel_profiler::RelayCtrl c{};
            cluster.read_core(&c, sizeof(c), tt_cxy_pair(ctx.chip_id, ctx.ruler->core.virt), ctx.ruler->ctrl);
            if (c.dropped_sync != 0) {
                log_warning(
                    tt::LogMetal,
                    "[streaming profiler] Device {}: the sync check's ruler dropped {} of its {} readings on a full "
                    "ring",
                    ctx.chip_id,
                    c.dropped_sync,
                    c.sync_tail + c.dropped_sync);
            }
        }
        if (ctx.tracker.program) {
            kernel_profiler::RelayCtrl c{};
            cluster.read_core(&c, sizeof(c), tt_cxy_pair(ctx.chip_id, ctx.tracker.core.virt), ctx.tracker.ctrl);
            if (c.dropped_sync != 0) {
                log_warning(
                    tt::LogMetal,
                    "[streaming profiler] Device {}: the clock tracker dropped {} clock instants on a full ring",
                    ctx.chip_id,
                    c.dropped_sync);
            }
        }
        set_producers_armed(ctx, false);
    }
}

void DevicePrograms::verify_completeness() const {
    auto& cluster = get_cluster();
    for (const DeviceCtx& ctx : devices_) {
        std::array<uint32_t, kernel_profiler::SPSC_CONTROL_END> cv{};
        uint64_t stranded_words = 0, stranded_lanes = 0;
        std::array<std::vector<std::pair<uint32_t, uint32_t>>, 2> counted;
        for (size_t ci = 0; ci < ctx.producers.size(); ci++) {
            const Producer& p = ctx.producers[ci];
            cluster.read_core(cv.data(), sizeof(cv), tt_cxy_pair(ctx.chip_id, p.virt), p.prof_l1);
            const auto stalls =
                std::span(cv).subspan(kernel_profiler::SPSC_STALL_COUNT_0, kernel_profiler::SPSC_STALL_COUNT_MAX);
            if (const uint32_t total = std::accumulate(stalls.begin(), stalls.end(), 0u); total != 0) {
                counted[p.blocking].emplace_back(total, static_cast<uint32_t>(ci));
            }
            for (uint32_t r = 0; r < kNRisc; r++) {
                const int32_t left = static_cast<int32_t>(
                    cv[kernel_profiler::SPSC_RING_TAIL_0 + r] - cv[kernel_profiler::SPSC_RING_HEAD_0 + r]);
                if (left > 0) {
                    stranded_lanes++;
                    stranded_words += static_cast<uint32_t>(left);
                }
            }
        }
        for (const bool blocking : {true, false}) {
            auto& list = counted[blocking];
            if (list.empty()) {
                continue;
            }
            std::ranges::sort(list, std::greater<>());
            uint64_t total = 0;
            std::string top;
            for (const auto& [count, ci] : list) {
                const CoreCoord& v = ctx.producers[ci].virt;
                top += fmt::format("{}({},{})#{}={}", top.empty() ? "" : " ", v.x, v.y, ci, count);
                total += count;
            }
            log_warning(
                tt::LogMetal,
                "[streaming profiler] Device {}: {} {} on {} of {} cores; (virt x,y)#index=count: {}",
                ctx.chip_id,
                total,
                blocking ? "profiler stalls" : "records dropped on a full ring nothing may stall for",
                list.size(),
                ctx.producers.size(),
                top);
        }
        if (stranded_lanes != 0) {
            log_warning(
                tt::LogMetal,
                "[streaming profiler] Device {}: {} words on {} lanes were published after their relay's last sweep "
                "and are not in the capture",
                ctx.chip_id,
                stranded_words,
                stranded_lanes);
        }
    }
}

}  // namespace tt::tt_metal::streaming_profiler
