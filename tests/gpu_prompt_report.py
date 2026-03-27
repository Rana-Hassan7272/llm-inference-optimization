"""
Run one prompt against the FastAPI endpoint and print a compact report.

Intended use: GPU runtime (Colab, Paperspace, local CUDA box) with API server running.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any, Dict
from urllib import error, request


DEFAULT_PROMPT = "To be, or not to be,"


def _post_json(url: str, payload: Dict[str, Any], timeout_s: int) -> Dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url=url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _get_json(url: str, timeout_s: int) -> Dict[str, Any]:
    req = request.Request(url=url, method="GET")
    with request.urlopen(req, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect one prompt inference report from API.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="API base URL")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Prompt text")
    parser.add_argument("--max-new-tokens", type=int, default=64, help="max_new_tokens to request")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p (sent for compatibility)")
    parser.add_argument("--top-k", type=int, default=40, help="Top-k (sent if endpoint accepts it)")
    parser.add_argument("--timeout", type=int, default=900, help="HTTP timeout in seconds")
    parser.add_argument("--force-tier", default=None, help="Optional: fast|balanced|quality")
    args = parser.parse_args()

    generate_url = f"{args.base_url.rstrip('/')}/generate"
    health_url = f"{args.base_url.rstrip('/')}/health"
    status_url = f"{args.base_url.rstrip('/')}/status"

    payload: Dict[str, Any] = {
        "prompt": args.prompt,
        "max_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
    }
    if args.force_tier:
        payload["force_tier"] = args.force_tier

    t0 = time.perf_counter()
    try:
        result = _post_json(generate_url, payload, args.timeout)
    except error.HTTPError as e:
        # If endpoint rejects extra sampling args, retry with minimal payload.
        if e.code in (400, 422):
            fallback = {
                "prompt": args.prompt,
                "max_tokens": args.max_new_tokens,
                "temperature": args.temperature,
            }
            if args.force_tier:
                fallback["force_tier"] = args.force_tier
            result = _post_json(generate_url, fallback, args.timeout)
        else:
            print(f"Request failed with HTTP {e.code}: {e.read().decode('utf-8', errors='ignore')}")
            return 1
    except Exception as e:
        print(f"Request failed: {e}")
        return 1
    elapsed_wall = time.perf_counter() - t0

    # Try to fetch device + memory context.
    health: Dict[str, Any] = {}
    status: Dict[str, Any] = {}
    try:
        health = _get_json(health_url, 30)
    except Exception:
        pass
    try:
        status = _get_json(status_url, 30)
    except Exception:
        pass

    text = (result.get("text") or "").strip()
    text_sample = text[:220].replace("\n", " ")
    if len(text) > 220:
        text_sample += " ..."

    total_ms = _safe_float(result.get("total_ms"))
    total_s = total_ms / 1000.0 if total_ms > 0 else elapsed_wall
    tok_per_sec = _safe_float(result.get("tok_per_sec"))
    tokens_generated = int(round(tok_per_sec * total_s)) if tok_per_sec > 0 and total_s > 0 else 0

    # Prefer endpoint metric for memory, then fallback to status info if present.
    mem_gb = result.get("mem_gb")
    if mem_gb is None:
        mem_gb = status.get("vram_used_gb")

    gpu_name = health.get("gpu_name")
    gpu_available = health.get("gpu_available")
    device_str = "GPU" if gpu_available else "CPU"
    if gpu_name:
        device_str = f"{device_str} ({gpu_name})"

    print("=== PROMPT INFERENCE REPORT ===")
    print(f"prompt: {args.prompt}")
    print(f"max_new_tokens: {args.max_new_tokens}")
    print(f"temperature: {args.temperature}")
    print(f"top_p: {args.top_p}")
    print(f"top_k: {args.top_k}")
    print("---")
    print(f"total_generation_time_s: {total_s:.3f}")
    print(f"tokens_generated: {tokens_generated}")
    print(f"tokens_per_second: {tok_per_sec:.2f}")
    print(f"output_sample: {text_sample or '[empty]'}")
    print(f"device: {device_str}")
    print(f"memory_usage_gb: {mem_gb if mem_gb is not None else 'n/a'}")
    print("source: API endpoint /generate")
    return 0


if __name__ == "__main__":
    sys.exit(main())
