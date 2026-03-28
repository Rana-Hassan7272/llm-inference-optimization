# LLM Inference Lab

<div align="center">

**A production-grade LLM serving system built from scratch — tiered precision routing, SSE streaming, vLLM dynamic batching, automated load testing, and a live observability dashboard.**

[![Live Dashboard](https://img.shields.io/badge/Dashboard-Live%20on%20Vercel-black?style=for-the-badge&logo=vercel)](https://llm-inference-lab.vercel.app/)
[![API](https://img.shields.io/badge/API-Live%20on%20Render-46E3B7?style=for-the-badge&logo=render)](https://llm-inference-lab.onrender.com/)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)

</div>

---

## What This Project Proves

Most LLM projects stop at "call the API and log the latency." This project goes further — it builds the serving stack itself. Starting from a single Colab T4 GPU, it implements every layer of a production inference system: quantization tiers, an adaptive router that selects precision based on prompt complexity, a streaming server, batch throughput benchmarks against vLLM, automated load testing across concurrency levels, MLflow experiment tracking, and a live React dashboard wired to real metrics.

**The result:** a deployed, fully observable LLM inference stack with reproducible benchmarks from `docker compose up`.

---

## Live Deployments

| Service | URL | Stack |
|---|---|---|
| 🖥️ Dashboard | [llm-inference-lab.vercel.app](https://llm-inference-lab.vercel.app/) | React + Vite → Vercel |
| ⚡ API | [llm-inference-lab.onrender.com](https://llm-inference-lab.onrender.com/) | FastAPI → Docker → Render |

---

## Dashboard

[![Dashboard Overview](https://github.com/Rana-Hassan7272/llm-inference-lab/raw/main/results/Screenshot%20(1907).png)](https://llm-inference-lab.vercel.app/)

[![Dashboard Charts](https://github.com/Rana-Hassan7272/llm-inference-lab/raw/main/results/Screenshot%20(1908).png)](https://llm-inference-lab.vercel.app/)

The dashboard renders four live panels from committed JSON artifacts:
- **Comparison table** — all precision tiers side-by-side: latency, TPS, VRAM, quality score
- **Memory vs quality scatter** — visualises the quantization tradeoff curve
- **Latency/throughput dual-axis** — shows how the system behaves under increasing concurrency
- **Routing log** — live stream of adaptive router decisions with reasoning

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Client Layer                             │
│          React Dashboard (Vercel)  ·  CLI  ·  curl             │
└────────────────────────┬────────────────────────────────────────┘
                         │  HTTP / SSE
┌────────────────────────▼────────────────────────────────────────┐
│                    API Gateway                                  │
│              FastAPI + Uvicorn  (Docker → Render)               │
│                                                                 │
│  POST /generate          blocking generation + routing          │
│  POST /generate/stream   SSE token streaming                    │
│  GET  /benchmark/{tier}  3-prompt benchmark per tier            │
│  GET  /router/explain    routing decision explanation           │
│  GET  /routing-log       full JSONL routing history             │
│  GET  /health  /status   system liveness + model status         │
└──────────┬──────────────────────────────┬───────────────────────┘
           │                              │
┌──────────▼──────────┐      ┌────────────▼──────────────────────┐
│   Adaptive Router   │      │         ModelManager              │
│                     │      │                                   │
│  prompt length  →   │      │  4-bit  (fast tier)   lazy load  │
│  keyword intent →   │      │  8-bit  (balanced)    lazy load  │
│  force_tier flag    │      │  FP16   (quality)     lazy load  │
│                     │      │                                   │
│  routes to tier ──► │      │  per-tier CUDA locks              │
│                     │      │  VRAM tracking                    │
└─────────────────────┘      └────────────────────────────────────┘
                                         │
┌────────────────────────────────────────▼───────────────────────┐
│                    Generation Paths                            │
│                                                                │
│  Blocking:   full sequence → TTFT + TPS + total latency        │
│  Streaming:  TextIteratorStreamer thread → SSE events          │
└────────────────────────────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│              Observability & Experiment Pipeline                │
│                                                                 │
│  MLflow (file store)   ·   routing_log.jsonl                   │
│  Locust load tests     ·   results/ artifacts                  │
│  dashboard_bundle.json →   Vercel public/data/                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Results

### KV Cache — Speedup vs Context Length

Measured on TinyLlama-1.1B, Tesla T4, 64 fixed new tokens.

| Context Length | Cache ON (tok/s) | Cache OFF (tok/s) | Speedup |
|:-:|:-:|:-:|:-:|
| 128 tokens | 32.36 | 30.57 | **1.06×** |
| 256 tokens | 31.34 | 17.33 | **1.81×** |
| 512 tokens | 29.78 | 7.75 | **3.84×** |
| 1024 tokens | 29.12 | 2.84 | **10.25×** |

The speedup is superlinear. At 1024-token context, KV cache is not a nice-to-have — it is a 10× throughput requirement. Without it, the model re-computes the full attention matrix from scratch at every generation step: O(T²) work per token. With cache: O(1) per step for the new token.

![KV Cache Speedup](https://github.com/Rana-Hassan7272/llm-inference-lab/raw/main/results/kv_cache_experiment-results/kv_cache_speedup.png)

---

### Static Batching — Near-Linear Throughput Scaling

| Batch Size | Total Throughput (tok/s) | Scaling Factor |
|:-:|:-:|:-:|
| 1 | 30.31 | baseline |
| 2 | 60.58 | **2.00×** |
| 4 | 127.77 | **4.21×** |
| 8 | 262.59 | **8.66×** |

Static batching achieves near-linear scaling up to batch 8 on a T4. GPU parallelism means batching 8 requests costs nearly the same VRAM bandwidth as batching 1 — you get 8× the throughput for ~1× the memory cost.

![Batching Throughput](https://github.com/Rana-Hassan7272/llm-inference-lab/raw/main/results/batching-results/batching_throughput.png)

---

### vLLM Dynamic Batching vs Manual Static Batching

| Batch Size | vLLM (tok/s) | Manual (tok/s) | vLLM Advantage |
|:-:|:-:|:-:|:-:|
| 1 | 106.26 | 30.31 | **3.51×** |
| 2 | 195.90 | 60.58 | **3.23×** |
| 4 | 385.91 | 127.77 | **3.02×** |
| 8 | 750.57 | 262.59 | **2.86×** |

vLLM's PagedAttention removes the fixed KV cache allocation overhead that limits manual batching. At batch 1 the gap is 3.5× — vLLM is running continuous batching under the hood even for a "single" request. At batch 8 the gap narrows slightly to 2.86× as manual batching becomes more GPU-efficient, but vLLM still wins at every point.

![vLLM Comparison](https://github.com/Rana-Hassan7272/llm-inference-lab/raw/main/results/vllm/vllm_comparison.png)

---

### Load Testing — Concurrency Behaviour

Measured with Locust against the API running locally (Colab → localhost). Zero failed requests across all concurrency levels.

| Concurrent Users | Requests/s | Avg Latency | P95 Latency | Failure Rate |
|:-:|:-:|:-:|:-:|:-:|
| 1 | 1.16 | 273 ms | 330 ms | 0.0% |
| 5 | 4.27 | 495 ms | 910 ms | 0.0% |
| 10 | 4.61 | 1,258 ms | 2,000 ms | 0.0% |
| 20 | 4.61 | 2,932 ms | 3,800 ms | 0.0% |

Throughput plateaus at ~4.6 req/s between users=10 and users=20 — the system hits its generation-time ceiling (single GPU, one inference thread). The API queue handles the overflow: latency climbs linearly but the **failure rate stays at 0%** even at 20 concurrent users. This is the production-reliability result that matters.

---

### Representative GPU Inference — Tesla T4

```
Model     : TinyLlama/TinyLlama-1.1B-Chat-v1.0
Prompt    : "To be, or not to be,"
Params    : max_new_tokens=64, temperature=0.8, top_p=0.9, top_k=40
─────────────────────────────────────────────────────────────────
Total time   : 4.134 s
Tokens       : 64
TPS          : 15.48 tok/s
VRAM used    : 1.94 GB
Device       : GPU (Tesla T4)
Tier         : router-selected (adaptive)
```

---

## Adaptive Router

The router selects a precision tier per request based on prompt characteristics — no GPU wasted on a 4-bit model for complex reasoning, no FP16 overhead on a short factual query.

```python
# Routing heuristics (inference/adaptive_router.py)
short prompt (<50 tokens), no reasoning keywords  →  fast    (4-bit BitsAndBytes)
medium prompt OR reasoning/creative keywords      →  balanced (8-bit BitsAndBytes)
forced quality flag OR long context               →  quality  (FP16 full precision)
```

All routing decisions are logged to `results/routing_log.jsonl` and surfaced in the dashboard's routing log panel. The `/router/explain` endpoint returns the routing decision with reasoning for any given prompt before generation.

---

## Repo Structure

```
llm-inference-lab/
│
├── api/
│   └── app.py                       FastAPI server — all endpoints, SSE streaming, routing
│
├── benchmarks/
│   ├── runner.py                    Orchestrates all Phase 5 experiments
│   ├── batching_comparison-vllm.py  vLLM vs manual batching with forced generation length
│   ├── locustfile.py                Locust user behaviour (POST /generate, GET /health)
│   ├── load_test_runner.py          Automated Locust sweeps → summary artifacts
│   ├── mlflow_integration.py        MLflow logging (Windows-safe file:// tracking URI)
│   └── merge_scored_results.py      Merges *_scored.json → canonical *_results.json
│
├── dashboard/
│   ├── src/App.jsx                  React dashboard — comparison table, charts, routing log
│   └── public/data/
│       ├── dashboard_bundle.json    Committed benchmark artifacts (loaded by Vercel)
│       └── load_test_summary.json   Committed load test results
│
├── inference/
│   ├── model_manager.py             Tiered lazy loader with VRAM tracking + per-tier locks
│   └── adaptive_router.py          Prompt-complexity heuristics → tier selection
│
├── optimization/
│   ├── kv_cache_experiment.py       KV cache scaling across context lengths
│   └── batching.py                 Manual/static batching baseline
│
├── results/                         All experiment artifacts (JSON, PNG, JSONL)
│   ├── kv_cache_experiment-results/
│   ├── batching-results/
│   ├── vllm/
│   ├── load-testing/run_*/
│   ├── router/
│   ├── routing_log.jsonl
│   ├── phase3_findings.md
│   └── phase4_findings.md
│
├── notebooks/                       Colab notebooks for GPU experiments
├── tests/                           API and unit tests
├── Dockerfile                       API image (uvicorn, healthcheck)
├── docker-compose.yml               api + mlflow + dashboard (one command)
└── requirements.txt
```

---

## Quick Start

### Option 1 — Docker (recommended, one command)

```bash
git clone https://github.com/Rana-Hassan7272/llm-inference-lab.git
cd llm-inference-lab
docker compose up -d --build
```

| Service | URL |
|---|---|
| API + Swagger docs | http://localhost:8000/docs |
| MLflow tracking UI | http://localhost:5000 |
| React dashboard | http://localhost:5173 |

### Option 2 — Local dev

```bash
# Backend
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload

# Dashboard (separate terminal)
cd dashboard
npm ci
npm run dev                     # http://localhost:5173
```

---

## API Reference

### POST `/generate` — Blocking generation

```bash
curl -s -X POST "https://llm-inference-lab.onrender.com/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Explain the attention mechanism",
    "max_tokens": 64,
    "temperature": 0.8,
    "top_p": 0.9,
    "top_k": 40
  }'
```

Response includes `total_time_ms`, `tokens_per_second`, `mem_gb`, `tier_used`, and the full generated text.

### POST `/generate/stream` — SSE token streaming

```bash
curl -N -X POST "https://llm-inference-lab.onrender.com/generate/stream" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain recursion briefly", "max_tokens": 64, "temperature": 0.8}'
```

Stream format:
```
data: {"routing": {"tier": "balanced", "reason": "reasoning keyword detected"}}
data: {"token": "Recursion"}
data: {"token": " is"}
...
data: {"done": true, "tok_per_sec": 15.48, "total_ms": 4134, "token_count": 64, "mem_gb": 1.94}
```

### GET `/benchmark/{tier}` — Quick benchmark

```bash
curl https://llm-inference-lab.onrender.com/benchmark/fast
curl https://llm-inference-lab.onrender.com/benchmark/balanced
curl https://llm-inference-lab.onrender.com/benchmark/quality
```

### GET `/router/explain` — Routing decision preview

```bash
curl "https://llm-inference-lab.onrender.com/router/explain?prompt=Write+a+poem"
# → {"tier": "balanced", "reason": "creative keyword detected", "confidence": 0.82}
```

### GET `/health`

```bash
curl https://llm-inference-lab.onrender.com/health
```

---

## Running Experiments

### KV Cache scaling

```bash
python optimization/kv_cache_experiment.py
# Outputs: results/kv_cache_experiment-results/kv_cache_results.json
#          results/kv_cache_experiment-results/kv_cache_speedup.png
```

### Static batching baseline

```bash
python optimization/batching.py
# Outputs: results/batching-results/batching_results.json
#          results/batching-results/batching_throughput.png
```

### vLLM comparison

```bash
python benchmarks/batching_comparison-vllm.py
# Outputs: results/vllm/vllm_results.json
#          results/vllm/vllm_comparison.png
```

### Load testing (requires running API)

```bash
# Start API first, then:
python benchmarks/load_test_runner.py
# Outputs: results/load-testing/run_{timestamp}/load_test_summary.json
#          results/load-testing/run_{timestamp}/throughput_trend.csv
```

### Full experiment suite

```bash
python benchmarks/runner.py
# Orchestrates all experiments, skips CUDA-only on CPU, UTF-8 safe on Windows
```

### MLflow tracking

```bash
python benchmarks/mlflow_integration.py
# Then open http://localhost:5000 to view all runs
```

---

## Cloud Deployment

### API → Render

1. New → Web Service → Connect this repo
2. Runtime: **Docker**
3. Environment variables:
   ```
   MODEL_ID=TinyLlama/TinyLlama-1.1B-Chat-v1.0
   HF_TOKEN=<your token if needed>
   ```
4. Health check path: `/health`
5. Render uses `Dockerfile` automatically (`EXPOSE 8000`, uvicorn CMD)

### Dashboard → Vercel

1. New Project → Import this repo
2. Root directory: `dashboard`
3. Build command: `npm ci && npm run build`
4. Output directory: `dist`
5. Commit benchmark artifacts to `dashboard/public/data/`:
   - `dashboard_bundle.json` (benchmark results)
   - `load_test_summary.json` (load test run)

The dashboard auto-detects and renders whichever artifacts are present. No backend connection required at read time — all metrics are served as static JSON.

---

## Notes on Precision Tiers and Quality

TinyLlama in 4-bit and 8-bit quantization produces noisier outputs on open-ended generation, especially with chat prompt formatting and greedy decoding. This is an expected and **documented** tradeoff — the project's value is in measuring and exposing it, not hiding it.

What the tiering demonstrates:
- 4-bit: lowest VRAM (~0.7 GB), highest TPS, acceptable quality for factual short-form
- 8-bit: balanced VRAM (~1.2 GB), moderate TPS, better coherence on reasoning tasks
- FP16: full model precision (~2.0 GB), lowest TPS, best quality for creative/long-form

For demo purposes, use `force_tier=quality` (FP16) or tune `temperature` and `top_p` for more coherent outputs at lower tiers.

Full quality/latency notes: [`results/phase4_findings.md`](results/phase4_findings.md)

---

### Ablation — Precision vs VRAM, TTFT, TPS, Quality

Aggregated over the 20-question set in `results/*_results.json` (TinyLlama-1.1B on Tesla T4). TTFT is mean time-to-first-token; TPS is mean tokens/sec; Quality is the mean of the 1–5 manual scores captured in the scored JSONs.

| Tier | VRAM (GB) | Avg TTFT (ms) | Avg TPS (tok/s) | Avg Quality (1–5) |
|---|---:|---:|---:|---:|
| 4-bit (fast) | 2.19 | 74 | 14.3 | 3.10 |
| 8-bit (balanced) | 2.16 | 150 | 7.1 | 1.55 |
| FP16 (quality) | 2.16 | 103 | 26.7 | 1.50 |

Notes:
- These are dataset averages; FP16 TTFT includes an initial warm-up outlier on the first prompt, which increases the mean.
- Quality scores come from manual 1–5 ratings merged into the canonical files (see `benchmarks/merge_scored_results.py`). They are indicative rather than a formal BLEU/perplexity evaluation.

If you want a stricter metric (BLEU/perplexity) on a fixed test set, we can add an automatic scorer and re-run to populate the "Quality" column with objective scores.

---

## Technology Stack

| Layer | Technology |
|---|---|
| Inference | Hugging Face Transformers + Accelerate |
| Quantization | BitsAndBytes (4-bit NF4, 8-bit LLM.int8) |
| Dynamic batching | vLLM (PagedAttention) |
| API server | FastAPI + Uvicorn |
| Streaming | SSE via `TextIteratorStreamer` thread |
| Load testing | Locust |
| Experiment tracking | MLflow (local file store) |
| Frontend | React + Vite + Recharts |
| Containerisation | Docker + Docker Compose |
| API hosting | Render (Docker) |
| Dashboard hosting | Vercel (static Vite build) |

---

## Project Phases

| Phase | Scope | Key Output |
|---|---|---|
| 1–2 | Environment, baselines | Stable generation pipeline, timing validated |
| 3 | Optimization experiments | KV cache 10.25×, static batching 8.66×, vLLM 3.5× benchmarks |
| 4 | Production API | Streaming server, adaptive router, tier benchmarks on T4 |
| 5 | Automation & observability | Load testing, MLflow tracking, React dashboard |
| 6 | Docker & deployment | Dockerised stack, Render API, Vercel dashboard — live |

---

<div align="center">

Built with PyTorch · FastAPI · vLLM · React · Docker

[Live Dashboard](https://llm-inference-lab.vercel.app/) · [API](https://llm-inference-lab.onrender.com/docs) · [Repo](https://github.com/Rana-Hassan7272/llm-inference-lab)

</div>

<!-- ABLATION_TABLE_START -->

### Ablation — Precision vs VRAM, TTFT, TPS, Quality (Auto + BLEU-1)

Computed from `results/*_results.json`. Quality is a heuristic factual-correctness score (1–5). BLEU-1 uses a small reference set for factual prompts only.

| Tier | VRAM (GB) | Avg TTFT (ms) | Avg TPS (tok/s) | Auto Quality (1–5) | BLEU-1 (0–100) |
|---|---:|---:|---:|---:|---:|
| 4-bit (fast) | 2.19 | 74 | 14.06 | 0.75 | 8.08 |
| 8-bit (balanced) | 2.16 | 152 | 7.16 | 0.25 | 1.90 |
| FP16 (quality) | 2.16 | 103 | 26.67 | 0.25 | 1.73 |

Notes: This is a lightweight proxy; BLEU computed only on factual prompts. BLEU coverage: ~35.0% of prompts have references.

<!-- ABLATION_TABLE_END -->
