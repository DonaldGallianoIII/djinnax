"""Fail-closed benchmark orchestration (audit S3): an official sweep
must never silently run on CPU, lose an engine, or aggregate a partial
matrix. CPU-runnable; CI core."""

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_strict_mode_refuses_cpu_backend(tmp_path):
    r = subprocess.run(
        [sys.executable, str(REPO / "benchmarks" / "bench_head_to_head.py"),
         "--batches", "64", "--steps", "2", "--reps", "1"],
        env={"PATH": "/usr/bin:/bin", "JAX_PLATFORMS": "cpu",
             "HOME": str(tmp_path)},
        capture_output=True, text=True, timeout=300,
    )
    assert r.returncode != 0
    assert "strict mode" in (r.stdout + r.stderr)


def test_aggregate_refuses_partial_matrix(tmp_path):
    partial = tmp_path / "partial.jsonl"
    rows = [
        {"engine": "ttt/djinn", "B": 64, "steps_per_sec": 100.0, "run": 0},
        {"engine": "ttt/djinn", "B": 64, "steps_per_sec": 105.0, "run": 1},
        {"engine": "ttt/pgx", "B": 64, "steps_per_sec": 90.0, "run": 0},
    ]
    partial.write_text("".join(json.dumps(r) + "\n" for r in rows))
    r = subprocess.run(
        [sys.executable, str(REPO / "benchmarks" / "sweep_stats.py"),
         "--aggregate-only", str(partial)],
        capture_output=True, text=True, timeout=120,
    )
    assert r.returncode != 0
    assert "incomplete sweep matrix" in (r.stdout + r.stderr)


def test_aggregate_accepts_complete_matrix(tmp_path):
    full = tmp_path / "full.jsonl"
    rows = [
        {"engine": e, "B": 64, "steps_per_sec": s + 10 * run, "run": run}
        for run in (0, 1)
        for e, s in (("ttt/djinn", 100.0), ("ttt/pgx", 90.0))
    ]
    full.write_text("".join(json.dumps(r) + "\n" for r in rows))
    r = subprocess.run(
        [sys.executable, str(REPO / "benchmarks" / "sweep_stats.py"),
         "--aggregate-only", str(full)],
        capture_output=True, text=True, timeout=120,
    )
    assert r.returncode == 0, r.stdout + r.stderr
    assert "ttt" in r.stdout
