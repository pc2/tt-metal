# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Per-cell op reports for test_prefill_layer_perf_chunk_n run under Tracy.

The test writes a manifest of the cells it measured (layer type, chunk, signpost pair, host time)
to PREFILL_SUMMARIES/layer_perf. After `python -m tracy -r` has written ops_perf_results_*.csv, this
script slices that CSV once per cell with tt-perf-report and writes a markdown summary to
PREFILL_SUMMARIES/perf, which the Blaze CI job publishes to its job summary.

This only parses files; it never opens a device.

    python models/demos/gemma4_d_p/demo/layer_perf_report.py --profiler-dir generated/profiler
"""

import argparse
import csv
import getpass
import json
import os
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

MANIFEST_DIR = "layer_perf"
SUMMARY_NAME = "gemma4_layer_perf.md"


def summaries_root():
    """PREFILL_SUMMARIES, or a per-user default so a shared fixed dir cannot block other users."""
    return Path(os.getenv("PREFILL_SUMMARIES", f"/tmp/prefill_summaries_{getpass.getuser()}"))


def is_primary_rank():
    """True on MPI rank 0 or outside MPI, so `mpirun --pernode` ranks do not race on one file."""
    for var in ("OMPI_COMM_WORLD_RANK", "PMIX_RANK", "PMI_RANK"):
        rank = os.environ.get(var)
        if rank is not None:
            return rank == "0"
    return True


def _safe_name(text):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


def write_manifest(run_id, cells, **meta):
    """On rank 0, persist the measured cells as PREFILL_SUMMARIES/layer_perf/manifest_<run_id>.json."""
    if not is_primary_rank():
        return None
    out_dir = summaries_root() / MANIFEST_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"manifest_{_safe_name(run_id)}.json"
    path.write_text(json.dumps({"run_id": run_id, **meta, "cells": cells}, indent=2))
    return path


def find_ops_csv(profiler_dir):
    """Newest ops_perf_results_*.csv under a Tracy artifacts folder."""
    found = sorted(Path(profiler_dir).glob("reports/**/ops_perf_results_*.csv"), key=lambda p: p.stat().st_mtime)
    return found[-1] if found else None


def signposts_in(ops_csv):
    """Names of the signpost rows in a raw Tracy ops CSV."""
    with open(ops_csv, newline="") as f:
        return {r["OP CODE"] for r in csv.DictReader(f) if r.get("OP TYPE") == "signpost"}


def _tt_perf_report_cmd():
    exe = shutil.which("tt-perf-report")
    return [exe] if exe else [sys.executable, "-m", "tt_perf_report.perf_report"]


def run_tt_perf_report(ops_csv, start_signpost, stop_signpost, out_csv):
    """Slice ops_csv between two signposts; writes out_csv, its _stacked summary, and a .log."""
    cmd = [
        *_tt_perf_report_cmd(),
        "--start-signpost",
        start_signpost,
        "--end-signpost",
        stop_signpost,
        "--no-color",
        "--csv",
        str(out_csv),
        str(ops_csv),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    Path(out_csv).with_suffix(".log").write_text(f"$ {' '.join(cmd)}\n{proc.stdout}{proc.stderr}")
    return proc.returncode == 0 and Path(out_csv).exists()


def _float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def summarize_cell_csv(path, top_n=5):
    """Totals and the top ops by device time from one tt-perf-report CSV (times in microseconds)."""
    with open(path, newline="") as f:
        # Skip signposts, host ops, and rows with an invalid device duration.
        rows = [r for r in csv.DictReader(f) if _float(r.get("Device Time")) is not None]
    # The first op the replay executes carries a gap that reaches back before the start signpost
    # (host staging, ring-cache init), outside the measurement. Rows are not in execution order, so
    # find it by Global Call Count.
    call_counts = [_float(r.get("Global Call Count")) for r in rows]
    if rows and None not in call_counts:
        leading = call_counts.index(min(call_counts))
    else:
        leading = 0
    kernel_us = gap_us = 0.0
    by_op = defaultdict(lambda: [0.0, 0])
    for i, row in enumerate(rows):
        device_us = _float(row["Device Time"])
        kernel_us += device_us
        if i != leading:
            gap_us += _float(row.get("Op-to-Op Gap")) or 0.0
        op = row.get("OP Code", "").strip().lstrip("'")
        by_op[op][0] += device_us
        by_op[op][1] += 1
    n_ops = len(rows)
    top = sorted(by_op.items(), key=lambda kv: kv[1][0], reverse=True)[:top_n]
    return {
        "n_ops": n_ops,
        "kernel_us": kernel_us,
        "span_us": kernel_us + gap_us,
        "top_ops": [{"op": op, "device_us": us, "count": count} for op, (us, count) in top],
    }


def _cell_text(cell):
    host = f"host {cell['measured_ms']:.2f}"
    if cell.get("report") is None:
        return f"report failed<br>{host}"
    r = cell["report"]
    return f"**{r['kernel_us'] / 1000:.2f}**<br>span {r['span_us'] / 1000:.2f}<br>{host}"


def render_markdown(manifests):
    """One grid (layer type x chunk) and per-cell top-op tables per manifest."""
    lines = []
    for m in manifests:
        shape = "x".join(str(d) for d in m.get("mesh_shape", ()))
        lines += [
            f"### Gemma4 layer perf: {m['run_id']}",
            "",
            f"Context {m.get('context_len')}, chunk {m.get('chunk_size')}, mesh {shape}. Each cell: "
            "**device-kernel time**, span (kernel time plus op-to-op gaps), and host-measured replay, in ms. "
            "Device times are tt-perf-report's device-merged values.",
            "",
        ]
        chunks = sorted({c["chunk_idx"] for c in m["cells"]})
        types = list(dict.fromkeys(c["layer_type"] for c in m["cells"]))
        grid = {(c["layer_type"], c["chunk_idx"]): c for c in m["cells"]}
        lines.append("| Layer | " + " | ".join(f"Chunk {i}" for i in chunks) + " |")
        lines.append("|---" * (len(chunks) + 1) + "|")
        for lt in types:
            layer_idx = next(c["layer_idx"] for c in m["cells"] if c["layer_type"] == lt)
            row = [_cell_text(grid[(lt, i)]) if (lt, i) in grid else "–" for i in chunks]
            lines.append(f"| {lt} (layer {layer_idx}) | " + " | ".join(row) + " |")
        lines.append("")
        for c in m["cells"]:
            r = c.get("report")
            if r is None:
                continue
            lines += [
                f"<details><summary>{c['layer_type']} chunk {c['chunk_idx']}: top ops of {r['n_ops']}</summary>",
                "",
                "| Op | Device ms | % of kernel | Count |",
                "|---|---:|---:|---:|",
            ]
            for op in r["top_ops"]:
                pct = 100 * op["device_us"] / r["kernel_us"] if r["kernel_us"] else 0.0
                lines.append(f"| `{op['op']}` | {op['device_us'] / 1000:.3f} | {pct:.1f} | {op['count']} |")
            lines += ["", "</details>", ""]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--profiler-dir", default="generated/profiler", help="Tracy artifacts folder (-o)")
    parser.add_argument("--root", type=Path, default=None, help="Summaries root (default: PREFILL_SUMMARIES)")
    parser.add_argument("--top", type=int, default=5, help="Ops listed per cell")
    args = parser.parse_args(argv)

    root = args.root or summaries_root()
    out_dir = root / MANIFEST_DIR
    manifests = [json.loads(p.read_text()) for p in sorted(out_dir.glob("manifest_*.json"))]
    if not manifests:
        print(f"error: no manifest_*.json under {out_dir}; did the layer-perf test run?", file=sys.stderr)
        return 1
    ops_csv = find_ops_csv(args.profiler_dir)
    if ops_csv is None:
        print(f"error: no reports/**/ops_perf_results_*.csv under {args.profiler_dir}", file=sys.stderr)
        return 1
    # Keep the raw CSV next to the per-cell slices so the uploaded artifact can be re-sliced offline.
    shutil.copy2(ops_csv, out_dir / ops_csv.name)
    print(f"ops CSV: {ops_csv}")
    # tt-perf-report only warns on a missing signpost and then leaves that side of the range open,
    # so a manifest from another run (a stale local one) would be sliced against the wrong ops.
    signposts = signposts_in(ops_csv)

    failed = []
    for m in manifests:
        cell_dir = out_dir / _safe_name(m["run_id"])
        cell_dir.mkdir(parents=True, exist_ok=True)
        for c in m["cells"]:
            out_csv = cell_dir / f"{c['layer_type']}_chunk{c['chunk_idx']}.csv"
            label = f"{m['run_id']} {c['layer_type']} chunk {c['chunk_idx']}"
            missing = [s for s in (c["start_signpost"], c["stop_signpost"]) if s not in signposts]
            if missing:
                c["report"] = None
                failed.append(label)
                print(f"{label}: signposts {missing} are not in {ops_csv.name}")
                continue
            ok = run_tt_perf_report(ops_csv, c["start_signpost"], c["stop_signpost"], out_csv)
            c["report"] = summarize_cell_csv(out_csv, args.top) if ok else None
            if c["report"] is None or c["report"]["n_ops"] == 0:
                failed.append(label)
                print(f"{label}: no ops reported, see {out_csv.with_suffix('.log')}")
            else:
                print(f"{label}: {c['report']['kernel_us'] / 1000:.2f}ms device kernel")

    (out_dir / "summary.json").write_text(json.dumps(manifests, indent=2))
    summary_dir = root / "perf"
    summary_dir.mkdir(parents=True, exist_ok=True)
    (summary_dir / SUMMARY_NAME).write_text(render_markdown(manifests))
    print(f"summary: {summary_dir / SUMMARY_NAME}")
    if failed:
        print(f"error: no valid report for {', '.join(failed)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
