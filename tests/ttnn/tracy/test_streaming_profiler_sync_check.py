# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""The clock sync's accuracy gate. Each workload runs as a subprocess with the streaming profiler and its sync check on
(TT_METAL_STREAMING_PROFILER_SYNC_CHECK=1), which puts a ruler on one more idle eth core per chip and logs, at each
capture's end, the chip-to-chip error of the global timeline measured against held-out link rounds. The test asserts
that every chip pair was measured, p99.9 <= 3.5 ns and max <= 6 ns with every reading counted, no sync warnings in
the log, and, for the three di/dt workloads, that AICLK swung at least 100 MHz so DVFS was exercised. The workloads:
an idle mesh, fabric ping-pong with and without load, load with di/dt bursts, the FF1 matmul and SDPA di/dt tests,
and two ttnn CCL ops. Needs a multi-chip Blackhole system (an 8-chip LoudBox in CI); about 4 minutes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from tests.ttnn.tracy.streaming_profiler_sync_gate import (
    PROBE_TIMEOUT_S,
    aiclk_swing_mhz,
    built,
    check_sync_accuracy,
    run,
    skip_unless_blackhole,
)

PINGPONG = built(Path("test/tt_metal/tools/profiler/test_streaming_profiler_pingpong_fabric"))
MULTICAST = built(Path("test/tt_metal/tools/profiler/test_streaming_profiler_multicast"))
CCL = [Path(sys.executable), Path(__file__).with_name("streaming_profiler_ccl_workload.py")]
DIDT = [Path(sys.executable), "-m", "pytest", "-s"]
DIDT_TESTS = Path(__file__).parents[2] / "didt"
SECONDS = 20
WORKLOADS = {
    "idle": [PINGPONG, "--idle", "--seconds", str(SECONDS)],
    "fabric_traffic_load": [PINGPONG, "--load"],
    "load_didt": [MULTICAST, "--scale", "0.3"],
    "didt_ff1_matmul": [
        *DIDT,
        f"{DIDT_TESTS}/test_ff1_matmul.py::test_ff1_matmul",
        "-k",
        "all and without_gelu",
        "--didt-workload-iterations",
        "3000",
    ],
    "didt_sdpa": [
        *DIDT,
        f"{DIDT_TESTS}/test_sdpa_op.py::test_sdpa_op",
        "-k",
        "all and bf16_HiFi2",
        "--didt-workload-iterations",
        "500",
    ],
    "ccl_all_gather_ring": [*CCL, "--op", "all_gather", "--fabric", "ring", "--seconds", str(SECONDS)],
    "ccl_all_reduce_2d_load": [*CCL, "--op", "all_reduce", "--fabric", "2d", "--load", "4", "--seconds", str(SECONDS)],
}
TIMEOUT_S = 300
# The di/dt workloads are here for the DVFS their throttling causes: some chip's AICLK swings 125-456 MHz in each of
# them, against the 6 MHz an idle chip jitters by.
DVFS_WORKLOADS = {"load_didt", "didt_ff1_matmul", "didt_sdpa"}
MIN_DVFS_SWING_MHZ = 100
ENV = {"TT_METAL_STREAMING_PROFILER": "1", "TT_METAL_STREAMING_PROFILER_SYNC_CHECK": "1"}


@pytest.mark.timeout(PROBE_TIMEOUT_S + TIMEOUT_S + 60)
@pytest.mark.parametrize("workload", list(WORKLOADS))
def test_streaming_profiler_sync_check(workload):
    skip_unless_blackhole(min_chips=2)
    exe = WORKLOADS[workload][0]
    assert exe.exists(), f"{exe} not built (build with --build-tests)"
    out = run(WORKLOADS[workload], ENV, TIMEOUT_S)
    check_sync_accuracy(out)
    if workload in DVFS_WORKLOADS:
        swing = aiclk_swing_mhz(out)
        assert swing >= MIN_DVFS_SWING_MHZ, f"AICLK moved at most {swing} MHz on any chip, so no DVFS was exercised"
