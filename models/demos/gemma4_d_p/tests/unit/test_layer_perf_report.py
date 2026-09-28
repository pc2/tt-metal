# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host checks for the layer-perf report: manifest in, per-cell slices and markdown out."""

import csv
import json

from models.demos.gemma4_d_p.demo import layer_perf_report as lpr

PERF_REPORT_HEADERS = [
    "ID",
    "Total %",
    "Bound",
    "OP Code",
    "Device",
    "Device Time",
    "Op-to-Op Gap",
    "Global Call Count",
]


def _write_cell_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PERF_REPORT_HEADERS)
        writer.writeheader()
        for op, device_us, gap_us, *call_count in rows:
            row = {"OP Code": op, "Device Time": device_us, "Op-to-Op Gap": gap_us}
            if call_count:
                row["Global Call Count"] = call_count[0]
            writer.writerow(row)


def _cell(layer_type, chunk_idx, layer_idx, measured_ms):
    start, stop = (
        f"gemma4-layer-{layer_type}-chunk{chunk_idx}-start",
        f"gemma4-layer-{layer_type}-chunk{chunk_idx}-stop",
    )
    return {
        "chunk_idx": chunk_idx,
        "layer_type": layer_type,
        "layer_idx": layer_idx,
        "chunk_start": chunk_idx * 8192,
        "measured_ms": measured_ms,
        "start_signpost": start,
        "stop_signpost": stop,
    }


def _write_ops_csv(profiler_dir, cells):
    """A raw Tracy ops CSV holding each cell's signpost pair around one device op."""
    reports = profiler_dir / "reports" / "r"
    reports.mkdir(parents=True)
    path = reports / "ops_perf_results_r.csv"
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["OP CODE", "OP TYPE"])
        for c in cells:
            writer.writerows([[c["start_signpost"], "signpost"], ["MatmulDeviceOperation", "tt_dnn_device"]])
            writer.writerow([c["stop_signpost"], "signpost"])
    return path


def test_summarize_skips_rows_without_device_time(tmp_path):
    path = tmp_path / "cell.csv"
    _write_cell_csv(
        path,
        [
            ("gemma4-layer-global-chunk0-start (signpost)", "", "", ""),
            ("MatmulDeviceOperation", "300.0", "5.0", "11"),
            ("RingJointSDPADeviceOperation", "500.0", "10.0", "12"),
            # Executed first (lowest call count) though not listed first: its gap reaches back
            # before the start signpost, so span excludes it.
            ("MatmulDeviceOperation", "200.0", "1000000.0", "10"),
            ("to_torch (torch)", "", "", ""),
        ],
    )
    summary = lpr.summarize_cell_csv(path, top_n=1)
    assert summary["n_ops"] == 3
    assert summary["kernel_us"] == 1000.0
    assert summary["span_us"] == 1015.0
    assert summary["top_ops"] == [{"op": "MatmulDeviceOperation", "device_us": 500.0, "count": 2}]


def test_main_slices_every_manifest_cell(tmp_path, monkeypatch):
    root = tmp_path / "summaries"
    monkeypatch.setenv("PREFILL_SUMMARIES", str(root))
    monkeypatch.delenv("OMPI_COMM_WORLD_RANK", raising=False)
    cells = [_cell("global", 0, 5, 9.0), _cell("local", 0, 0, 4.0), _cell("global", 31, 5, 12.0)]
    lpr.write_manifest(
        "blackhole-chunkci-both-sz8192-ctx_256k-8x4", cells, context_len=262144, chunk_size=8192, mesh_shape=(8, 4)
    )

    _write_ops_csv(tmp_path / "profiler", cells)

    sliced = []

    def fake_report(ops_csv, start, stop, out_csv):
        sliced.append((start, stop))
        _write_cell_csv(out_csv, [("MatmulDeviceOperation", "1500.0", "100.0")])
        return True

    monkeypatch.setattr(lpr, "run_tt_perf_report", fake_report)
    assert lpr.main(["--profiler-dir", str(tmp_path / "profiler")]) == 0

    assert sliced == [(c["start_signpost"], c["stop_signpost"]) for c in cells]
    out_dir = root / lpr.MANIFEST_DIR
    assert (out_dir / "ops_perf_results_r.csv").exists()
    summary = json.loads((out_dir / "summary.json").read_text())
    assert all(c["report"]["kernel_us"] == 1500.0 for c in summary[0]["cells"])

    md = (root / "perf" / lpr.SUMMARY_NAME).read_text()
    assert "| Layer | Chunk 0 | Chunk 31 |" in md
    assert "| local (layer 0) | **1.50**<br>span 1.50<br>host 4.00 | – |" in md


def test_main_fails_when_a_cell_has_no_ops(tmp_path, monkeypatch):
    monkeypatch.setenv("PREFILL_SUMMARIES", str(tmp_path))
    monkeypatch.delenv("OMPI_COMM_WORLD_RANK", raising=False)
    cells = [_cell("global", 0, 5, 9.0)]
    lpr.write_manifest("run", cells, context_len=262144, chunk_size=8192, mesh_shape=(8, 4))
    _write_ops_csv(tmp_path / "profiler", cells)
    monkeypatch.setattr(lpr, "run_tt_perf_report", lambda *_: False)
    assert lpr.main(["--profiler-dir", str(tmp_path / "profiler")]) == 1
    assert "report failed" in (tmp_path / "perf" / lpr.SUMMARY_NAME).read_text()


def test_main_refuses_a_cell_whose_signposts_are_not_in_the_csv(tmp_path, monkeypatch):
    # A stale manifest from another run: tt-perf-report would slice an open range, so never call it.
    monkeypatch.setenv("PREFILL_SUMMARIES", str(tmp_path))
    monkeypatch.delenv("OMPI_COMM_WORLD_RANK", raising=False)
    measured, stale = _cell("global", 0, 5, 9.0), _cell("global", 7, 5, 11.0)
    lpr.write_manifest("current", [measured], context_len=262144, chunk_size=8192, mesh_shape=(8, 4))
    lpr.write_manifest("stale", [stale], context_len=262144, chunk_size=8192, mesh_shape=(8, 4))
    _write_ops_csv(tmp_path / "profiler", [measured])

    sliced = []

    def fake_report(ops_csv, start, stop, out_csv):
        sliced.append(start)
        _write_cell_csv(out_csv, [("MatmulDeviceOperation", "1500.0", "100.0")])
        return True

    monkeypatch.setattr(lpr, "run_tt_perf_report", fake_report)
    assert lpr.main(["--profiler-dir", str(tmp_path / "profiler")]) == 1
    assert sliced == [measured["start_signpost"]]
    summary = {m["run_id"]: m for m in json.loads((tmp_path / lpr.MANIFEST_DIR / "summary.json").read_text())}
    assert summary["current"]["cells"][0]["report"]["kernel_us"] == 1500.0
    assert summary["stale"]["cells"][0]["report"] is None


def test_manifest_is_written_only_on_rank_zero(tmp_path, monkeypatch):
    monkeypatch.setenv("PREFILL_SUMMARIES", str(tmp_path))
    monkeypatch.setenv("OMPI_COMM_WORLD_RANK", "1")
    assert lpr.write_manifest("run", []) is None
    assert not (tmp_path / lpr.MANIFEST_DIR).exists()
