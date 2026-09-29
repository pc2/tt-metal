// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// The Blackhole Ethernet 1588 layer on real links. On each active eth link between two chips, an initiator and an echo
// kernel exchange 256 frames with one-step egress stamps and an RX stamp rule. It checks that every frame was stamped
// on egress and ingress with no extra stamps, that the stamps are causal round by round, that each round's one-way time
// is within 40 ns of the median, and that the offsets' rms residual around a fitted line is under 20 ns (both bounds
// follow from the stamps' 20 ns grid). It also checks that both kernels restore the TX header row and RX settings they
// changed. Needs slow dispatch (TT_METAL_SLOW_DISPATCH_MODE=1) on a multi-chip Blackhole system.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <map>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/distributed.hpp>

#include "device_fixture.hpp"
#include "eth_test_common.hpp"
#include "impl/context/metal_context.hpp"
#include "tt_metal/tt_metal/test_kernels/dataflow/unit_tests/erisc/eth_ptp_stamps.hpp"

namespace tt::tt_metal {
namespace {

using eth_ptp_stamps::Result;

constexpr const char* kKernel = "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/erisc/eth_ptp_stamps.cpp";

struct LinkFit {
    std::vector<double> one_way;
    double one_way_p10 = 0, one_way_p50 = 0, one_way_p90 = 0;
    double slope = 0;
    double resid_ns = 0;
};

LinkFit fit(const Result& a, const Result& b, const std::string& link) {
    // Round i: t0 initiator egress, t1 echo ingress, t1b echo egress, t2 initiator ingress, each end in its own PTP ns.
    LinkFit f;
    std::vector<double> offset, at;
    for (uint32_t i = 0; i < eth_ptp_stamps::kRounds; i++) {
        const auto t0 = static_cast<double>(b.stamps[i][0]), t1 = static_cast<double>(b.stamps[i][1]);
        const auto t1b = static_cast<double>(a.stamps[i][0]), t2 = static_cast<double>(a.stamps[i][1]);
        EXPECT_LT(t0, t2) << link << " round " << i;
        EXPECT_LT(t1, t1b) << link << " round " << i;
        if (i > 0) {
            EXPECT_GT(t0, static_cast<double>(a.stamps[i - 1][1])) << link << " round " << i;
            EXPECT_GT(t1, static_cast<double>(a.stamps[i - 1][0])) << link << " round " << i;
        }
        f.one_way.push_back(0.5 * ((t2 - t0) - (t1b - t1)));
        offset.push_back(0.5 * ((t1 + t1b) - (t0 + t2)));
        at.push_back(0.5 * (t0 + t2));
    }
    const double n = static_cast<double>(at.size());
    double sx = 0, sy = 0, sxx = 0, sxy = 0;
    for (size_t i = 0; i < at.size(); i++) {
        const double x = at[i] - at.front();
        sx += x, sy += offset[i], sxx += x * x, sxy += x * offset[i];
    }
    f.slope = (n * sxy - sx * sy) / (n * sxx - sx * sx);
    const double inter = (sy - f.slope * sx) / n;
    double ss = 0;
    for (size_t i = 0; i < at.size(); i++) {
        const double r = offset[i] - (inter + f.slope * (at[i] - at.front()));
        ss += r * r;
    }
    f.resid_ns = std::sqrt(ss / n);
    std::sort(f.one_way.begin(), f.one_way.end());
    const auto pct = [&](size_t p) { return f.one_way[std::min(f.one_way.size() - 1, f.one_way.size() * p / 100)]; };
    f.one_way_p10 = pct(10);
    f.one_way_p50 = pct(50);
    f.one_way_p90 = pct(90);
    return f;
}

void run_link(
    MeshDispatchFixture* fixture,
    const std::shared_ptr<distributed::MeshDevice>& mesh_a,
    const std::shared_ptr<distributed::MeshDevice>& mesh_b,
    const CoreCoord& eth_a,
    const CoreCoord& eth_b) {
    IDevice* dev_a = mesh_a->get_devices()[0];
    IDevice* dev_b = mesh_b->get_devices()[0];
    const uint32_t base =
        MetalContext::instance().hal().get_dev_addr(HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::UNRESERVED);
    const uint32_t res_addr = base + eth_ptp_stamps::kResultOffset;
    std::vector<uint32_t> zero(sizeof(Result) / sizeof(uint32_t), 0);
    if (!detail::WriteToDeviceL1(dev_a, eth_a, res_addr, zero, CoreType::ETH) ||
        !detail::WriteToDeviceL1(dev_b, eth_b, res_addr, zero, CoreType::ETH)) {
        ADD_FAILURE() << "could not clear the results on chip " << dev_a->id() << " eth " << eth_a.str() << " and chip "
                      << dev_b->id() << " eth " << eth_b.str();
        return;
    }

    const auto workload = [&](const CoreCoord& core, bool initiator) {
        Program program;
        EthernetConfig config{.compile_args = {initiator ? 1u : 0u}};
        eth_test_common::set_arch_specific_eth_config(config);
        const KernelHandle k = CreateKernel(program, kKernel, core, config);
        SetRuntimeArgs(program, k, core, {base});
        distributed::MeshWorkload w;
        const distributed::MeshCoordinate zero_coord(0, 0);
        w.add_program(distributed::MeshCoordinateRange(zero_coord, zero_coord), std::move(program));
        return w;
    };
    distributed::MeshWorkload wa = workload(eth_a, true);
    distributed::MeshWorkload wb = workload(eth_b, false);
    if (fixture->IsSlowDispatch()) {
        std::thread ta([&] { fixture->RunProgram(mesh_a, wa); });
        std::thread tb([&] { fixture->RunProgram(mesh_b, wb); });
        ta.join();
        tb.join();
    } else {
        fixture->RunProgram(mesh_b, wb, true);
        fixture->RunProgram(mesh_a, wa, true);
        fixture->FinishCommands(mesh_a);
        fixture->FinishCommands(mesh_b);
    }

    const auto read = [&](IDevice* dev, const CoreCoord& core) {
        std::vector<uint32_t> words;
        EXPECT_TRUE(detail::ReadFromDeviceL1(dev, core, res_addr, sizeof(Result), words, CoreType::ETH));
        Result r{};
        std::memcpy(&r, words.data(), sizeof(r));
        return r;
    };
    const Result a = read(dev_a, eth_a);
    const Result b = read(dev_b, eth_b);
    const std::string link =
        fmt::format("chip {} eth {} -> chip {} eth {}", dev_a->id(), eth_a.str(), dev_b->id(), eth_b.str());

    for (const auto& [r, end] : {std::pair{&a, "initiator"}, std::pair{&b, "echo"}}) {
        ASSERT_EQ(r->done, eth_ptp_stamps::kDone) << link << ": the " << end << " kernel did not finish";
        EXPECT_EQ(r->sel_after, r->sel_before)
            << link << ": the " << end << "'s TX header row selection was not restored";
        EXPECT_EQ(r->no_match_after, r->no_match_before)
            << link << ": the " << end << "'s RX no-match actions were not restored";
        ASSERT_EQ(r->rounds, eth_ptp_stamps::kRounds) << link << ": the " << end << " stopped waiting for its peer";
        EXPECT_EQ(r->unstamped, 0u) << link << ": frames to the " << end << " without an egress stamp";
        EXPECT_EQ(r->rx_missing, 0u) << link << ": " << end << " frames without an ingress stamp";
        EXPECT_EQ(r->rx_extra, 0u) << link << ": " << end << " ingress stamps beyond one per frame";
    }

    const LinkFit f = fit(a, b, link);
    const double ow = f.one_way_p50;
    log_info(
        tt::LogTest,
        "{}: one way inside the stamps {:.1f} ns (p10-p90 {:.1f}), offset residual {:.2f} ns rms, rate {:+.3f} ppm",
        link,
        ow,
        f.one_way_p90 - f.one_way_p10,
        f.resid_ns,
        f.slope * 1e6);
    EXPECT_GT(ow, 0.0) << link;
    // A round's one-way time is half the difference of two spans, and all four stamps sit on the timer's 20 ns grid, so
    // a correctly paired round is within 20 ns of the true delay and within two ticks of the median.
    for (const double v : f.one_way) {
        EXPECT_LT(std::abs(v - ow), 40.0) << link;
    }
    // An offset is half of one pair of stamps' sum minus another's, and each stamp is up to one 20 ns tick below its
    // true time, so every offset is within 20 ns of the true line and the residual around the fitted one is under 20
    // ns.
    EXPECT_LT(f.resid_ns, 20.0) << link;
}

}  // namespace

TEST_F(MeshDeviceFixture, ActiveEthPtpStamps) {
    if (arch_ != tt::ARCH::BLACKHOLE) {
        GTEST_SKIP() << "1588 stamping is Blackhole's";
    }
    auto& cluster = MetalContext::instance().get_cluster();
    std::map<ChipId, std::shared_ptr<distributed::MeshDevice>> mesh_of;
    for (const auto& mesh : devices_) {
        mesh_of.emplace(mesh->get_device_ids()[0], mesh);
    }
    size_t links = 0;
    for (const auto& [chip_a, mesh_a] : mesh_of) {
        for (const auto& [chip_b, cores] : cluster.get_ethernet_cores_grouped_by_connected_chips(chip_a)) {
            const auto mesh_b = mesh_of.find(chip_b);
            if (chip_b <= chip_a || mesh_b == mesh_of.end()) {
                continue;
            }
            for (const CoreCoord& eth_a : cores) {
                const CoreCoord eth_b =
                    std::get<1>(cluster.get_connected_ethernet_core(std::make_tuple(chip_a, eth_a)));
                run_link(this, mesh_a, mesh_b->second, eth_a, eth_b);
                links++;
            }
        }
    }
    if (links == 0) {
        GTEST_SKIP() << "no ethernet link between local chips";
    }
}

}  // namespace tt::tt_metal
