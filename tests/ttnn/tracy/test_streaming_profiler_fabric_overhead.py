# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""The streaming profiler's cost to the fabric. Runs test_tt_fabric's unicast microbench (latency and bandwidth over
linear, mesh and ring routes) with the profiler off as the baseline, then on, then with the sync check, and bounds the
latency and bandwidth each adds per test. The sync-check arm also gates the sync's accuracy. A last arm turns on
Ethernet-core zones (TT_METAL_STREAMING_PROFILER_ETH=1) and checks that every router still builds and runs and that
eth zones reach the zone CSV. Needs four or more Blackhole chips; about 2 minutes warm, 5 cold.
"""

from __future__ import annotations

import csv
import re
import statistics
from pathlib import Path

import pytest

from tests.ttnn.tracy.streaming_profiler_sync_gate import (
    PROBE_TIMEOUT_S,
    built,
    check_sync_accuracy,
    run,
    skip_unless_blackhole,
)
from tools.tracy.common import TT_METAL_HOME


FABRIC_BIN = built(Path("test/tt_metal/tt_fabric/test_infra/test_tt_fabric"))
CONFIG = (
    Path(TT_METAL_HOME) / "tests/tt_metal/tt_fabric/test_infra/test_yamls/test_fabric_ubench_at_least_2x2_mesh.yaml"
)
TIMEOUT_S = 180
ARMS = {
    "off": {},
    "on": {"TT_METAL_STREAMING_PROFILER": "1"},
    "sync_check": {"TT_METAL_STREAMING_PROFILER": "1", "TT_METAL_STREAMING_PROFILER_SYNC_CHECK": "1"},
    "eth_zones": {"TT_METAL_STREAMING_PROFILER": "1", "TT_METAL_STREAMING_PROFILER_ETH": "1"},
}
# Latency added, in ns: the median over tests of p50 and p99, and any single test's p99 and max. A test's p99 and max
# move 100+ ns run to run, so only the median p99 gets a tight bound. A test's p50 moves ~10 ns, so the median p50 stays
# within ~2 ns of zero with the profiler off in both runs; the profiler adds about 1 ns and the sync check about 9. Over
# four runs on an 8-chip LoudBox the profiler added at most +85 ns to the median p99, +264 ns to a test's p99 and +680
# ns to a max, and the sync check at most +366, +608 and +680 ns.
LATENCY_BOUNDS = {"on": (15.0, 120.0, 400.0, 900.0), "sync_check": (30.0, 700.0, 1000.0, 1200.0)}
# Over the same runs the median bandwidth ratio never fell below 1.024 and no test below 0.997 with the profiler on
# (1.010 and 0.9665 with the sync check). The bounds sit about twice the run-to-run noise below that; calling the
# router's step directly instead of through saved_call gave a worst of 0.981.
BANDWIDTH_BOUNDS = {"on": (0.996, 0.989), "sync_check": (0.95, 0.93)}

RESULTS = re.compile(r"=== Latency Test Results for (\S+) ===")
RUNNING = re.compile(r"Running Test: (\S+)")
BANDWIDTH = re.compile(r"BW \(GB/s\)=([0-9.]+)")
# A row of the latency table tt_fabric_test_latency_results.cpp:462 logs; the third column is Net Latency.
ROW = re.compile(r"\b(Max|P99|P50)\s+[0-9.]+\s+[0-9.]+\s+([0-9.]+)\s+[0-9.]+")

pytestmark = pytest.mark.timeout(PROBE_TIMEOUT_S + 2 * TIMEOUT_S + 60)


def _run(env_extra: dict) -> tuple[dict, dict, str]:
    out = run([FABRIC_BIN, "--test_config", CONFIG, "--filter", "name.*Unicast*"], env_extra, TIMEOUT_S)
    latency, bandwidth, name, running = {}, {}, None, None
    for line in out.splitlines():
        if m := RESULTS.search(line):
            name = m.group(1)
            latency[name] = {}
        elif name and (m := ROW.search(line)):
            latency[name][m.group(1)] = float(m.group(2))
        if m := RUNNING.search(line):
            running = m.group(1)
        elif running and not running.startswith("Latency") and (m := BANDWIDTH.search(line)):
            bandwidth.setdefault(running, []).append(float(m.group(1)))
    latency = {k: v for k, v in latency.items() if len(v) == 3}
    bandwidth = {k: statistics.mean(v) for k, v in bandwidth.items()}
    assert latency and bandwidth, f"no latency or bandwidth results:\n{out[-4000:]}"
    return latency, bandwidth, out


@pytest.fixture(scope="module")
def baseline():
    skip_unless_blackhole(min_chips=4)
    assert FABRIC_BIN.exists(), f"{FABRIC_BIN} not built (build with --build-tests)"
    return _run(ARMS["off"])


@pytest.mark.parametrize("arm", ["on", "sync_check"])
def test_streaming_profiler_fabric_overhead(baseline, arm):
    base_lat, base_bw, _ = baseline
    lat, bw, out = _run(ARMS[arm])
    assert set(lat) == set(base_lat), f"latency tests differ: {sorted(set(lat) ^ set(base_lat))}"
    assert set(bw) == set(base_bw), f"bandwidth tests differ: {sorted(set(bw) ^ set(base_bw))}"
    p50 = {t: lat[t]["P50"] - base_lat[t]["P50"] for t in base_lat}
    p99 = {t: lat[t]["P99"] - base_lat[t]["P99"] for t in base_lat}
    mx = {t: lat[t]["Max"] - base_lat[t]["Max"] for t in base_lat}
    ratio = {t: bw[t] / base_bw[t] for t in base_bw}
    p50_bound, median_bound, p99_bound, max_bound = LATENCY_BOUNDS[arm]
    median_ratio, worst_ratio = BANDWIDTH_BOUNDS[arm]
    table = "\n".join(
        f"  {t}: p50 {p50[t]:+.0f} ns, p99 {p99[t]:+.0f} ns, max {mx[t]:+.0f} ns" for t in sorted(base_lat)
    )
    worst = "\n".join(f"  {t}: {r:.3f}" for r, t in sorted((r, t) for t, r in ratio.items())[:8])
    print(f"\n[fabric-overhead {arm}] latency added:\n{table}")
    print(
        f"[fabric-overhead {arm}] bandwidth ratio: median {statistics.median(ratio.values()):.4f}, "
        f"worst {min(ratio.values()):.4f}; worst tests:\n{worst}"
    )
    assert statistics.median(p50.values()) <= p50_bound, f"median p50 added > {p50_bound} ns:\n{table}"
    assert statistics.median(p99.values()) <= median_bound, f"median p99 added > {median_bound} ns:\n{table}"
    assert max(p99.values()) <= p99_bound, f"a test's p99 added > {p99_bound} ns:\n{table}"
    assert max(mx.values()) <= max_bound, f"a test's max added > {max_bound} ns:\n{table}"
    assert statistics.median(ratio.values()) >= median_ratio, f"median bandwidth ratio < {median_ratio}:\n{worst}"
    assert min(ratio.values()) >= worst_ratio, f"a test's bandwidth ratio < {worst_ratio}:\n{worst}"
    if arm == "sync_check":
        check_sync_accuracy(out)


def test_streaming_profiler_fabric_eth_zones(baseline, tmp_path):
    """With Ethernet-core zones on, every router still fits its kernel config buffer, every test runs and the routers'
    zones reach the zone CSV."""
    base_lat, base_bw, _ = baseline
    zone_csv = tmp_path / "zones.csv"
    lat, bw, _ = _run(ARMS["eth_zones"] | {"TT_METAL_STREAMING_PROFILER_ZONE_CSV": str(zone_csv)})
    assert set(lat) == set(base_lat), f"latency tests differ: {sorted(set(lat) ^ set(base_lat))}"
    assert set(bw) == set(base_bw), f"bandwidth tests differ: {sorted(set(bw) ^ set(base_bw))}"
    with zone_csv.open() as f:
        f.readline()
        riscs = {row["RISC processor type"] for row in csv.DictReader(f, skipinitialspace=True)}
    assert "ERISC" in riscs, f"no ERISC zone in the zone CSV, only {sorted(riscs)}"
