"""
Phase 5 - Step 3: MLflow integration.

Logs benchmark experiments to MLflow:
- Hyperparameters
- Metrics
- Model/result artifacts
- Benchmark plots

Usage:
    python benchmarks/mlflow_integration.py
    python benchmarks/mlflow_integration.py --experiment llm-inference-lab --run-name phase5_full
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import mlflow


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_DIR = ROOT / "results"


def _safe_load_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _latest_file(directory: Path, pattern: str) -> Path | None:
    files = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


def _latest_dir(directory: Path, pattern: str) -> Path | None:
    dirs = [p for p in directory.glob(pattern) if p.is_dir()]
    dirs = sorted(dirs, key=lambda p: p.stat().st_mtime)
    return dirs[-1] if dirs else None


def _mode_file_name(mode_key: str) -> str:
    mapping = {"fp16": "fp16_results.json", "8bit": "8bit_results.json", "4bit": "4bit_results.json"}
    return mapping[mode_key]


def _normalize_tracking_uri(uri: str) -> str:
    """
    Ensure local filesystem paths are expressed as valid MLflow file URIs.
    This is required on Windows where raw paths like C:\\... are parsed as scheme 'c'.
    """
    parsed = urlparse(uri)
    if parsed.scheme in {"file", "http", "https", "sqlite", "postgresql", "mysql", "mssql", "databricks", "databricks-uc", "uc"}:
        return uri

    as_path = Path(uri)
    if not as_path.is_absolute():
        as_path = (ROOT / as_path).resolve()
    return as_path.as_uri()


def _log_quantization_section(results_dir: Path, modes: list[str]) -> None:
    for mode in modes:
        file_path = results_dir / _mode_file_name(mode)
        records = _safe_load_json(file_path)
        if not records:
            continue

        mode_label = str(records[0].get("mode", mode)).lower().replace("-", "").replace(" ", "")
        ttft_vals = [r.get("ttft_ms") for r in records if isinstance(r.get("ttft_ms"), (int, float))]
        tps_vals = [r.get("tok_per_sec") for r in records if isinstance(r.get("tok_per_sec"), (int, float))]
        mem_vals = [r.get("mem_gb") for r in records if isinstance(r.get("mem_gb"), (int, float))]
        qual_vals = [r.get("quality_score") for r in records if isinstance(r.get("quality_score"), (int, float))]

        mlflow.log_metric(f"{mode_label}_samples", float(len(records)))
        if ttft_vals:
            mlflow.log_metric(f"{mode_label}_avg_ttft_ms", statistics.mean(ttft_vals))
        if tps_vals:
            mlflow.log_metric(f"{mode_label}_avg_tok_per_sec", statistics.mean(tps_vals))
        if mem_vals:
            mlflow.log_metric(f"{mode_label}_avg_mem_gb", statistics.mean(mem_vals))
        if qual_vals:
            mlflow.log_metric(f"{mode_label}_avg_quality_score", statistics.mean(qual_vals))

        mlflow.log_artifact(str(file_path), artifact_path="model_artifacts/quantization")


def _log_kv_cache_section(results_dir: Path) -> None:
    kv_dir = results_dir / "kv_cache_experiment-results"
    kv_json = kv_dir / "kv_cache_results.json"
    rows = _safe_load_json(kv_json)
    if not rows:
        return

    cache_on = [r for r in rows if r.get("label") == "cache_on"]
    cache_off = [r for r in rows if r.get("label") == "cache_off"]
    if cache_on and cache_off:
        on_tps = statistics.mean(r["tok_per_sec"] for r in cache_on if isinstance(r.get("tok_per_sec"), (int, float)))
        off_tps = statistics.mean(r["tok_per_sec"] for r in cache_off if isinstance(r.get("tok_per_sec"), (int, float)))
        if off_tps > 0:
            mlflow.log_metric("kv_cache_avg_speedup_ratio", on_tps / off_tps)

    mlflow.log_artifact(str(kv_json), artifact_path="benchmark_data/kv_cache")
    for plot_name in ("kv_cache_speedup.png", "kv_cache_tokpersec.png", "kv_cache_ttft.png"):
        plot_path = kv_dir / plot_name
        if plot_path.exists():
            mlflow.log_artifact(str(plot_path), artifact_path="benchmark_plots/kv_cache")


def _log_batching_section(results_dir: Path) -> None:
    batch_dir = results_dir / "batching-results"
    batch_json = batch_dir / "batching_results.json"
    rows = _safe_load_json(batch_json)
    if not rows:
        return

    valid = [r for r in rows if isinstance(r, dict) and r.get("status") != "oom_skipped"]
    if valid:
        tps = [r["total_tok_per_sec"] for r in valid if isinstance(r.get("total_tok_per_sec"), (int, float))]
        latency = [r["per_prompt_ms"] for r in valid if isinstance(r.get("per_prompt_ms"), (int, float))]
        if tps:
            mlflow.log_metric("batching_peak_tok_per_sec", max(tps))
            mlflow.log_metric("batching_avg_tok_per_sec", statistics.mean(tps))
        if latency:
            mlflow.log_metric("batching_avg_per_prompt_ms", statistics.mean(latency))

    mlflow.log_artifact(str(batch_json), artifact_path="benchmark_data/batching")
    for plot_name in ("batching_throughput.png", "batching_latency.png", "batching_efficiency.png"):
        plot_path = batch_dir / plot_name
        if plot_path.exists():
            mlflow.log_artifact(str(plot_path), artifact_path="benchmark_plots/batching")


def _log_vllm_section(results_dir: Path) -> None:
    vllm_dir = results_dir / "vllm"
    vllm_json = vllm_dir / "vllm_results.json"
    rows = _safe_load_json(vllm_json)
    if rows:
        tps = [r["total_tok_per_sec"] for r in rows if isinstance(r.get("total_tok_per_sec"), (int, float))]
        if tps:
            mlflow.log_metric("vllm_avg_tok_per_sec", statistics.mean(tps))
            mlflow.log_metric("vllm_peak_tok_per_sec", max(tps))
        mlflow.log_artifact(str(vllm_json), artifact_path="benchmark_data/vllm")

    for file_name in ("vllm_comparison.png", "vllm_comparison_table.txt"):
        p = vllm_dir / file_name
        if p.exists():
            artifact_root = "benchmark_plots/vllm" if p.suffix == ".png" else "benchmark_data/vllm"
            mlflow.log_artifact(str(p), artifact_path=artifact_root)


def _log_load_test_section(results_dir: Path) -> None:
    load_root = results_dir / "load-testing"
    latest_run = _latest_dir(load_root, "run_*")
    if not latest_run:
        return

    summary_json = latest_run / "load_test_summary.json"
    summary = _safe_load_json(summary_json)
    if summary:
        mlflow.log_param("loadtest_host", str(summary.get("host")))
        mlflow.log_param("loadtest_duration", str(summary.get("duration")))
        mlflow.log_param("loadtest_spawn_rate", str(summary.get("spawn_rate")))
        mlflow.log_param("loadtest_users", ",".join(str(u) for u in summary.get("users", [])))

        rows = summary.get("results", [])
        valid = [r for r in rows if isinstance(r.get("generate_rps"), (int, float))]
        if valid:
            mlflow.log_metric("loadtest_peak_generate_rps", max(r["generate_rps"] for r in valid))
            mlflow.log_metric("loadtest_avg_generate_rps", statistics.mean(r["generate_rps"] for r in valid))
            p95_vals = [r["generate_p95_ms"] for r in valid if isinstance(r.get("generate_p95_ms"), (int, float))]
            if p95_vals:
                mlflow.log_metric("loadtest_avg_generate_p95_ms", statistics.mean(p95_vals))

        mlflow.log_artifact(str(summary_json), artifact_path="benchmark_data/load_testing")

    trend_csv = latest_run / "throughput_trend.csv"
    if trend_csv.exists():
        mlflow.log_artifact(str(trend_csv), artifact_path="benchmark_data/load_testing")

    for csv_file in latest_run.glob("users_*_stats.csv"):
        mlflow.log_artifact(str(csv_file), artifact_path="benchmark_data/load_testing/raw")
    for csv_file in latest_run.glob("users_*_failures.csv"):
        mlflow.log_artifact(str(csv_file), artifact_path="benchmark_data/load_testing/raw")
    for csv_file in latest_run.glob("users_*_exceptions.csv"):
        mlflow.log_artifact(str(csv_file), artifact_path="benchmark_data/load_testing/raw")


def _log_benchmark_manifest(results_dir: Path) -> dict[str, Any]:
    manifest = _latest_file(results_dir, "benchmark_run_*.json")
    if not manifest:
        return {}
    data = _safe_load_json(manifest) or {}
    mlflow.log_artifact(str(manifest), artifact_path="benchmark_data/manifests")
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description="Log benchmark results to MLflow")
    parser.add_argument("--tracking-uri", default=str(ROOT / "mlruns"), help="MLflow tracking URI")
    parser.add_argument("--experiment", default="llm-inference-lab", help="MLflow experiment name")
    parser.add_argument("--run-name", default=None, help="MLflow run name")
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR), help="Results directory")
    args = parser.parse_args()

    results_dir = Path(args.results_dir).resolve()
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    tracking_uri = _normalize_tracking_uri(args.tracking_uri)
    os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(args.experiment)

    run_name = args.run_name or f"phase5_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("project", "llm-inference-lab")
        mlflow.log_param("phase", "5")
        mlflow.log_param("step", "3_mlflow_integration")
        mlflow.log_param("results_dir", str(results_dir))
        mlflow.log_param("logged_at_utc", datetime.now(timezone.utc).isoformat())

        manifest = _log_benchmark_manifest(results_dir)
        modes = manifest.get("modes", ["fp16", "8bit", "4bit"])
        model_id = manifest.get("model")
        platform = manifest.get("platform", {})
        if model_id:
            mlflow.log_param("model_id", str(model_id))
        if modes:
            mlflow.log_param("modes", ",".join(modes))
        if platform:
            for key in ("system", "release", "python", "cuda_available", "gpu_name"):
                if key in platform:
                    mlflow.log_param(f"platform_{key}", str(platform[key]))

        _log_quantization_section(results_dir, modes)
        _log_kv_cache_section(results_dir)
        _log_batching_section(results_dir)
        _log_vllm_section(results_dir)
        _log_load_test_section(results_dir)

        print(f"[mlflow] Run logged successfully: {run_name}")
        print(f"[mlflow] Tracking URI: {tracking_uri}")
        print(f"[mlflow] Experiment: {args.experiment}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

