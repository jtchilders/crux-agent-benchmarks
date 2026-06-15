#!/usr/bin/env python3
"""
ALCF Inference Endpoint Benchmark
==================================
Fires N parallel async chat-completion calls to the Sophia vLLM endpoint and
measures time-to-first-token (TTFT), total latency, and aggregate token throughput.

Usage:
    python benchmark.py [options]

    --model         Model ID (default: openai/gpt-oss-20b)
    --concurrency   Number of parallel calls per wave (default: 8)
    --waves         Number of waves to run (default: 3)
    --max-tokens    Max tokens per response (default: 256)
    --prompt        Prompt to use (default: a medium-complexity coding task)
    --prompt-file   Path to a file containing the prompt (overrides --prompt)
    --base-url      Inference API base URL (default: Sophia vLLM)
    --auth-script   Path to inference_auth_token.py (default: auto-detect)
    --output        Output JSON path (default: results/benchmark_<timestamp>.json)
    --sweep         Run a sweep of concurrency levels: 1,2,4,8,16,32
    --no-stream     Disable streaming (measures total latency only, no TTFT)
    --timeout       Per-call timeout in seconds (default: 120)
"""

import argparse
import asyncio
import json
import os
import sys
import time
import importlib.util
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median, quantiles

import httpx

try:
    from prompts import PROMPTS as PROMPT_LIBRARY
except ImportError:
    PROMPT_LIBRARY = {}

# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------

def load_auth_script(auth_script_path: str | None) -> object:
    """Import inference_auth_token.py from a known location."""
    candidates = []
    if auth_script_path:
        candidates.append(auth_script_path)
    # Auto-detect common locations
    candidates += [
        os.path.expanduser("~/crux/inference_auth_token.py"),
        os.path.expanduser("~/inference_auth_token.py"),
        str(Path(__file__).parent / "inference_auth_token.py"),
    ]
    for path in candidates:
        if os.path.exists(path):
            spec = importlib.util.spec_from_file_location("inference_auth_token", path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    raise FileNotFoundError(
        f"inference_auth_token.py not found. Searched: {candidates}\n"
        "Download from: https://raw.githubusercontent.com/argonne-lcf/inference-endpoints/refs/heads/main/inference_auth_token.py"
    )


def get_token(auth_mod) -> str:
    return auth_mod.get_access_token()


# ---------------------------------------------------------------------------
# Default prompt (medium-complexity coding task, ~80 tokens input)
# ---------------------------------------------------------------------------

DEFAULT_PROMPT = (
    "Write a Python function that reads a CSV file using the standard library "
    "(no pandas), computes the mean and standard deviation for each numeric column, "
    "and returns a dictionary mapping column names to (mean, stdev) tuples. "
    "Include error handling for missing files and non-numeric values. "
    "Add a brief docstring and type hints."
)

# ---------------------------------------------------------------------------
# Core async call
# ---------------------------------------------------------------------------

async def call_once(
    client: httpx.AsyncClient,
    base_url: str,
    token: str,
    model: str,
    prompt: str,
    max_tokens: int,
    stream: bool,
    timeout: float,
    call_id: int,
) -> dict:
    """Fire one streaming chat-completion call and record timing metrics."""
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": stream,
        "temperature": 0.2,
    }

    result = {
        "call_id": call_id,
        "start_time": time.monotonic(),
        "ttft_s": None,
        "total_latency_s": None,
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
        "error": None,
        "success": False,
        "reasoning_tokens": 0,
    }

    t0 = result["start_time"]

    try:
        if stream:
            first_token = False
            collected_text = []
            reasoning_tokens = 0
            payload["stream_options"] = {"include_usage": True}
            async with client.stream(
                "POST", url, headers=headers, json=payload, timeout=timeout
            ) as resp:
                resp.raise_for_status()
                async for raw_line in resp.aiter_lines():
                    if not raw_line.startswith("data: "):
                        continue
                    data_str = raw_line[6:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue
                    choices = chunk.get("choices", [])
                    delta = choices[0].get("delta", {}) if choices else {}
                    # Support both content and reasoning tokens (reasoning models)
                    content = delta.get("content") or ""
                    reasoning = delta.get("reasoning") or ""
                    any_token = content or reasoning
                    if any_token and not first_token:
                        result["ttft_s"] = time.monotonic() - t0
                        first_token = True
                    if content:
                        collected_text.append(content)
                    if reasoning:
                        reasoning_tokens += len(reasoning.split())
                    # usage arrives in the final chunk with stream_options
                    usage = chunk.get("usage")
                    if usage:
                        result["prompt_tokens"] = usage.get("prompt_tokens")
                        result["completion_tokens"] = usage.get("completion_tokens")
                        result["total_tokens"] = usage.get("total_tokens")

            # Fallback: count collected text words if usage not returned
            result["completion_tokens"] = result["completion_tokens"] or len(collected_text)
            result["reasoning_tokens"] = reasoning_tokens
        else:
            resp = await client.post(url, headers=headers, json=payload, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            usage = data.get("usage", {})
            result["prompt_tokens"] = usage.get("prompt_tokens")
            result["completion_tokens"] = usage.get("completion_tokens")
            result["total_tokens"] = usage.get("total_tokens")

        result["total_latency_s"] = time.monotonic() - t0
        result["success"] = True

    except Exception as e:
        result["total_latency_s"] = time.monotonic() - t0
        result["error"] = str(e)

    return result


# ---------------------------------------------------------------------------
# Wave runner
# ---------------------------------------------------------------------------

async def run_wave(
    base_url: str,
    token: str,
    model: str,
    prompt: str,
    max_tokens: int,
    concurrency: int,
    stream: bool,
    timeout: float,
    wave_id: int,
) -> list[dict]:
    """Launch `concurrency` calls simultaneously and wait for all to finish."""
    limits = httpx.Limits(max_connections=concurrency + 4, max_keepalive_connections=concurrency)
    async with httpx.AsyncClient(limits=limits) as client:
        tasks = [
            call_once(client, base_url, token, model, prompt, max_tokens, stream, timeout, i)
            for i in range(concurrency)
        ]
        results = await asyncio.gather(*tasks)
    for r in results:
        r["wave_id"] = wave_id
    return list(results)


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------

def percentile(data: list[float], p: float) -> float:
    if not data:
        return float("nan")
    data = sorted(data)
    if len(data) == 1:
        return data[0]
    qs = quantiles(data, n=100)
    idx = max(0, min(int(p) - 1, len(qs) - 1))
    return qs[idx]


def summarize(calls: list[dict], concurrency: int, waves: int) -> dict:
    successful = [c for c in calls if c["success"]]
    failed = [c for c in calls if not c["success"]]

    latencies = [c["total_latency_s"] for c in successful]
    ttfts = [c["ttft_s"] for c in successful if c["ttft_s"] is not None]
    comp_tokens = [c["completion_tokens"] for c in successful if c["completion_tokens"]]

    # Aggregate throughput: total completion tokens / total elapsed wall time
    # Wall time = max latency across a wave * n_waves (sequential waves)
    # We approximate as sum(completion_tokens) / sum(latencies) for concurrent calls
    total_tokens = sum(comp_tokens) if comp_tokens else 0

    # Per-wave wall time = max latency in that wave
    wave_ids = sorted(set(c.get("wave_id", 0) for c in successful))
    total_wall_time = 0.0
    for wid in wave_ids:
        wave_calls = [c for c in successful if c.get("wave_id") == wid]
        if wave_calls:
            total_wall_time += max(c["total_latency_s"] for c in wave_calls)

    agg_tps = total_tokens / total_wall_time if total_wall_time > 0 else 0.0

    return {
        "n_calls": len(calls),
        "n_success": len(successful),
        "n_failed": len(failed),
        "concurrency": concurrency,
        "waves": waves,
        "ttft_mean_s": round(mean(ttfts), 3) if ttfts else None,
        "ttft_p50_s": round(percentile(ttfts, 50), 3) if ttfts else None,
        "ttft_p95_s": round(percentile(ttfts, 95), 3) if ttfts else None,
        "ttft_p99_s": round(percentile(ttfts, 99), 3) if ttfts else None,
        "total_latency_mean_s": round(mean(latencies), 3) if latencies else None,
        "total_latency_p50_s": round(median(latencies), 3) if latencies else None,
        "total_latency_p95_s": round(percentile(latencies, 95), 3) if latencies else None,
        "total_latency_p99_s": round(percentile(latencies, 99), 3) if latencies else None,
        "total_completion_tokens": total_tokens,
        "total_wall_time_s": round(total_wall_time, 3),
        "aggregate_tokens_per_sec": round(agg_tps, 2),
        "errors": [c["error"] for c in failed] if failed else [],
    }


def print_summary(summary: dict, config: dict) -> None:
    print("\n" + "=" * 60)
    print(f"  BENCHMARK RESULTS")
    print(f"  Model:       {config['model']}")
    print(f"  Concurrency: {summary['concurrency']}  Waves: {summary['waves']}")
    print(f"  Prompt len:  ~{config['prompt_tokens_approx']} chars input")
    print(f"  Max tokens:  {config['max_tokens']}")
    print("=" * 60)
    print(f"  Calls:       {summary['n_success']}/{summary['n_calls']} succeeded")
    if summary["ttft_p50_s"] is not None:
        print(f"  TTFT:        p50={summary['ttft_p50_s']}s  p95={summary['ttft_p95_s']}s  p99={summary['ttft_p99_s']}s")
    print(f"  Latency:     p50={summary['total_latency_p50_s']}s  p95={summary['total_latency_p95_s']}s  p99={summary['total_latency_p99_s']}s")
    print(f"  Tokens out:  {summary['total_completion_tokens']} total")
    print(f"  Wall time:   {summary['total_wall_time_s']}s")
    print(f"  Throughput:  {summary['aggregate_tokens_per_sec']} tok/s aggregate")
    if summary["n_failed"]:
        print(f"  Errors ({summary['n_failed']}):")
        for e in summary["errors"][:5]:
            print(f"    - {e}")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="ALCF Inference Endpoint Benchmark")
    parser.add_argument("--model", default="openai/gpt-oss-20b",
                        help="Model ID on the inference endpoint")
    parser.add_argument("--concurrency", type=int, default=8,
                        help="Number of parallel calls per wave")
    parser.add_argument("--waves", type=int, default=3,
                        help="Number of sequential waves to run")
    parser.add_argument("--max-tokens", type=int, default=256,
                        help="Max completion tokens per call")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT,
                        help="Prompt text to send")
    parser.add_argument("--prompt-size", default=None,
                        choices=["short", "medium", "long"],
                        help="Use a named prompt from prompts.py (short/medium/long)")
    parser.add_argument("--prompt-file", default=None,
                        help="File containing the prompt (overrides --prompt)")
    parser.add_argument("--base-url",
                        default="https://inference-api.alcf.anl.gov/resource_server/sophia/vllm/v1",
                        help="Inference API base URL")
    parser.add_argument("--auth-script", default=None,
                        help="Path to inference_auth_token.py")
    parser.add_argument("--output", default=None,
                        help="Output JSON path (default: results/benchmark_<timestamp>.json)")
    parser.add_argument("--sweep", action="store_true",
                        help="Run concurrency sweep: 1,2,4,8,16,32")
    parser.add_argument("--sweep-levels", default="1,2,4,8,16,32",
                        help="Comma-separated concurrency levels for --sweep")
    parser.add_argument("--no-stream", action="store_true",
                        help="Disable streaming (no TTFT measurement)")
    parser.add_argument("--timeout", type=float, default=120.0,
                        help="Per-call timeout in seconds")
    return parser.parse_args()


async def main_async(args):
    # Load auth
    print("Loading auth token...")
    auth_mod = load_auth_script(args.auth_script)
    token = get_token(auth_mod)
    print("Token acquired.")

    # Prompt
    prompt = args.prompt
    if args.prompt_size and PROMPT_LIBRARY:
        prompt = PROMPT_LIBRARY[args.prompt_size]
    elif args.prompt_file:
        with open(args.prompt_file) as f:
            prompt = f.read().strip()
    prompt_label = args.prompt_size or ("file" if args.prompt_file else "custom")

    stream = not args.no_stream

    # Output dir
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    if args.sweep:
        levels = [int(x) for x in args.sweep_levels.split(",")]
    else:
        levels = [args.concurrency]

    all_results = []

    for concurrency in levels:
        if args.sweep:
            print(f"\n{'─'*60}")
            print(f"  Sweep: concurrency={concurrency}")
            print(f"{'─'*60}")

        all_calls = []
        for wave in range(args.waves):
            print(f"  Wave {wave+1}/{args.waves} | concurrency={concurrency} ...", end=" ", flush=True)
            wave_t0 = time.monotonic()

            # Refresh token each wave (they're cheap)
            token = get_token(auth_mod)

            calls = await run_wave(
                base_url=args.base_url,
                token=token,
                model=args.model,
                prompt=prompt,
                max_tokens=args.max_tokens,
                concurrency=concurrency,
                stream=stream,
                timeout=args.timeout,
                wave_id=wave,
            )
            wave_elapsed = time.monotonic() - wave_t0
            n_ok = sum(1 for c in calls if c["success"])
            print(f"done in {wave_elapsed:.1f}s ({n_ok}/{len(calls)} ok)")
            all_calls.extend(calls)

        config = {
            "model": args.model,
            "prompt_label": prompt_label,
            "concurrency": concurrency,
            "waves": args.waves,
            "max_tokens": args.max_tokens,
            "stream": stream,
            "base_url": args.base_url,
            "prompt_tokens_approx": len(prompt),
            "timestamp": timestamp,
        }

        summary = summarize(all_calls, concurrency, args.waves)
        print_summary(summary, config)

        run_result = {
            "config": config,
            "summary": summary,
            "calls": all_calls,
        }
        all_results.append(run_result)

    # Write output
    if args.output:
        out_path = Path(args.output)
    elif args.sweep:
        out_path = results_dir / f"sweep_{timestamp}.json"
    else:
        out_path = results_dir / f"benchmark_c{args.concurrency}_{timestamp}.json"

    with open(out_path, "w") as f:
        json.dump(all_results if args.sweep else all_results[0], f, indent=2)
    print(f"Results written to: {out_path}")


def main():
    args = parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
