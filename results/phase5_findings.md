# Phase 5 Findings - Benchmark Suite and Tracking

## Scope completed

Phase 5 Steps 1-3 are implemented and validated:
- Step 1: benchmark orchestration runner (`benchmarks/runner.py`)
- Step 2: load testing with Locust (`benchmarks/locustfile.py`, `benchmarks/load_test_runner.py`)
- Step 3: MLflow experiment logging (`benchmarks/mlflow_integration.py`)

## Step 1 - Benchmark runner status

Run manifest confirms successful end-to-end execution on GPU:
- Manifest: `results/benchmark_run_20260327_162857.json`
- `success: true`
- Model: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- Platform: Linux, CUDA enabled, Tesla T4

Stage outcomes:
- `quantization_fp16`: ok (20 prompts)
- `quantization_8bit`: ok (20 prompts)
- `quantization_4bit`: ok (20 prompts)
- `kv_cache`: ok
- `batching`: ok
- `vllm`: skipped (not installed in that Colab session)

Artifacts produced:
- `results/fp16_results.json`
- `results/8bit_results.json`
- `results/4bit_results.json`
- `results/kv_cache_experiment-results/*`
- `results/batching-results/*`
- `results/vllm/*` (where available)

## Step 2 - Load testing status

Latest load-test run:
- Summary: `results/load-testing/run_20260327_164725/load_test_summary.json`
- Trend CSV: `results/load-testing/run_20260327_164725/throughput_trend.csv`
- Duration per sweep: 180s
- Spawn rate: 1 user/s
- User levels: 1, 5, 10, 20

Observed results (`POST /generate`):
- 1 user: `generate_rps=1.16`, `avg=272.8ms`, `p95=330ms`
- 5 users: `generate_rps=4.27`, `avg=495.3ms`, `p95=910ms`
- 10 users: `generate_rps=4.61`, `avg=1258.0ms`, `p95=2000ms`
- 20 users: `generate_rps=4.61`, `avg=2931.8ms`, `p95=3800ms`
- Failure ratio: `0.0` at all tested concurrency levels

Interpretation:
- Throughput scales quickly from 1 to ~5 users.
- System saturates around ~4.6 `generate` req/s.
- Latency increases with concurrency after saturation (expected behavior).

## Step 3 - MLflow integration status

Execution succeeded:
- Command used: `python benchmarks/mlflow_integration.py --experiment llm-inference-lab --run-name phase5_step3`
- Run created in experiment: `llm-inference-lab`
- Tracking URI: local file store under `mlruns`

Logged to MLflow:
- Hyperparameters (model, modes, platform, load-test config)
- Metrics (quantization, KV-cache, batching, vLLM when present, load testing)
- Artifacts:
  - Benchmark JSON outputs
  - Load-test summaries + raw CSV
  - Benchmark plots

Note:
- MLflow shows a deprecation warning for filesystem backend.
- Current setup is still valid for this capstone; production migration can move to SQLite/Postgres backend later.

## What is next

Remaining in Phase 5:
- Step 4: React dashboard (`dashboard/`) for benchmark visualization:
  - live comparison table
  - memory vs quality plot
  - latency distribution chart
  - routing decision log view

At this point, benchmarking + load testing + experiment tracking are complete and in a strong state.

