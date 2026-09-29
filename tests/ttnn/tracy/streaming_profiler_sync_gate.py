# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for the streaming profiler's clock sync gates: the bounds on the chip-to-chip error that a run with
TT_METAL_STREAMING_PROFILER_SYNC_CHECK=1 logs at each capture's end, the check of that log, the system probe and the
workload runner."""

from __future__ import annotations

import functools
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from tools.tracy.common import TT_METAL_HOME

# ~1.25x and ~1.4x the worst an 8-chip LoudBox has shown over three runs of the suite: p99.9 2.78 ns, max 4.38 ns with
# every reading counted.
P999_BOUND_NS = 3.5
MAX_BOUND_NS = 6.0

HEADLINE = re.compile(
    r"sync check: chip-to-chip error of the global timeline, a bound, over (\d+) of \d+ chip pairs and \d+ samples: "
    r"\|err\| p50 [0-9.]+, p99 [0-9.]+, p99\.9 ([0-9.]+), max ([0-9.]+) ns \(([^)]*)\)"
)
CHIP_LINE = re.compile(r"sync check chip (\d+): \d+ readings against the reference, .*?max ([0-9.]+) ns")
AICLK_LINE = re.compile(r"sync check chip (\d+) AICLK: (.*?)(?: \(\w+\.cpp:\d+\))?$", re.MULTILINE)
AICLK_RANGE = re.compile(r"sync check chip \d+ AICLK: mean \d+ MHz, sd \d+, (\d+)-(\d+) MHz")
FORBIDDEN = [
    "no chip pair measured",  # sync_check.cpp
    "the clock map could not place",  # sync_check.cpp
    "sync stream",  # service.cpp
    "not solved",  # sync_engine.cpp
    "overflows region",  # tt_elffile.cpp
    "the clock tracker's sync ring",  # device_programs.cpp
    "the sync check's ruler",  # device_programs.cpp
    "TT_FATAL",  # tt_stl/assert.hpp
]
PROBE_TIMEOUT_S = 300
# CI's logger colors every message, which puts an escape code right before a line's first word.
ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")


def built(rel: Path) -> Path:
    """The build output `rel`: under `build_Release` when only that tree has it, else under `build`. CI builds into
    `build` (build-artifact.yaml passes `--build-dir build`), and a local build_metal.sh run leaves `build` as a symlink
    to `build_Release`, so trying both finds the binary in either layout: naming only `build_Release` misses CI's."""
    for d in ("build", "build_Release"):
        cand = Path(TT_METAL_HOME) / d / rel
        if cand.exists():
            return cand
    return Path(TT_METAL_HOME) / "build" / rel


def run(args: list[str | Path], env_extra: dict[str, str], timeout: float) -> str:
    """Run `args` in the caller's environment without its TT_METAL_STREAMING_PROFILER* variables and with `env_extra`
    added, and return the run's stdout followed by its stderr. A run with the streaming profiler on that never logged
    it active skips the calling test, as test_streaming_profiler.py does: the system is not Blackhole or its firmware
    has no DRAM programmable cores. A nonzero exit, or a run still going after `timeout` seconds, fails the calling
    test with the tail of the run's output."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("TT_METAL_STREAMING_PROFILER")}
    env.update(env_extra)
    args = [str(a) for a in args]
    try:
        p = subprocess.run(args, env=env, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        # On POSIX the output captured before the timeout comes back as bytes even with text=True.
        out = "".join(s.decode(errors="replace") if isinstance(s, bytes) else s or "" for s in (e.stdout, e.stderr))
        out = ANSI_ESCAPE.sub("", out)
        pytest.fail(f"{args[0]} timed out after {timeout} s:\n{out[-4000:]}")
    out = ANSI_ESCAPE.sub("", p.stdout + p.stderr)
    if "TT_METAL_STREAMING_PROFILER" in env_extra and "[streaming profiler] active on" not in out:
        pytest.skip("streaming profiler did not start (not Blackhole / no DRAM programmable cores)")
    assert p.returncode == 0, f"{args[0]} exited {p.returncode}:\n{out[-4000:]}"
    return out


@functools.cache
def system() -> tuple[str, int]:
    """Read in a subprocess so the pytest parent never takes the PCIe lock."""
    code = "import ttnn; print('SYSTEM', ttnn.get_arch_name(), ttnn.get_num_devices())"
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=PROBE_TIMEOUT_S)
    m = re.search(r"^SYSTEM (\S+) (\d+)$", p.stdout, re.MULTILINE)
    assert m, f"no system probe output:\n{p.stdout[-2000:]}\n{p.stderr[-2000:]}"
    return m.group(1).lower(), int(m.group(2))


def skip_unless_blackhole(min_chips: int) -> None:
    arch, chips = system()
    if arch != "blackhole":
        pytest.skip(f"the streaming profiler runs on Blackhole, not {arch}")
    if chips < min_chips:
        pytest.skip(f"needs at least {min_chips} chips, this system has {chips}")


def aiclk_swing_mhz(out: str) -> int:
    """The widest range, in MHz, that any chip's AICLK covered in any of the run's captures."""
    return max((int(hi) - int(lo) for lo, hi in AICLK_RANGE.findall(out)), default=0)


def check_sync_accuracy(out: str) -> None:
    heads = list(HEADLINE.finditer(out))
    assert heads, f"no sync check report:\n{out[-4000:]}"
    for k, m in enumerate(heads):
        block = out[m.start() : heads[k + 1].start() if k + 1 < len(heads) else len(out)]
        chips = {int(c): float(v) for c, v in CHIP_LINE.findall(block)}
        assert len(chips) == system()[1], f"the report covers chips {sorted(chips)} of {system()[1]}"
        measured, p999, worst, where = m.groups()
        print(f"\n[sync-check] {m.group(0)}")
        print("[sync-check] per chip max ns: " + " ".join(f"c{c} {v:.2f}" for c, v in sorted(chips.items())))
        for c, clock in AICLK_LINE.findall(block):
            print(f"[sync-check] chip {c} AICLK: {clock}")
        n = len(chips)
        assert int(measured) == n * (n - 1) // 2, f"{measured} chip pairs measured for {n} chips"
        assert float(p999) <= P999_BOUND_NS, f"p99.9 {p999} ns > {P999_BOUND_NS} ns ({m.group(0)})"
        assert float(worst) <= MAX_BOUND_NS, f"max {worst} ns > {MAX_BOUND_NS} ns ({where})"
    for bad in FORBIDDEN:
        assert bad not in out, f"'{bad}' in the log:\n" + "\n".join(l for l in out.splitlines() if bad in l)[:4000]
