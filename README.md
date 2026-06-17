# crux-agent-benchmarks

Throughput and latency benchmarks for ALCF inference services, designed to
characterize how many parallel LLM agent calls can be sustained from Crux compute
nodes. Supports two backends:

- **Sophia** — ALCF's on-premise A100 vLLM cluster (open-weight models)
- **Argo** — ALCF's hosted API proxy to commercial models (OpenAI, Anthropic, Google)

## Goals

- Measure time-to-first-token (TTFT) and total latency at varying concurrency levels
- Measure aggregate token throughput (tokens/sec) across N parallel calls
- Find the concurrency ceiling before rate limiting or significant latency degradation
- Inform agent framework design: how many parallel workers can we realistically run?

## Structure

```
benchmark.py          # Main async benchmark runner (--backend sophia|argo)
run_benchmark.pbs     # PBS job script — single model, Sophia
run_full_sweep.pbs    # PBS job script — all Sophia models × prompts × concurrencies
run_argo_sweep.pbs    # PBS job script — Argo models sweep
plot_results.py       # Generate throughput/latency/TTFT/error plots
prompts.py            # Short/medium/long prompt definitions
requirements.txt      # Python dependencies
results/              # Output JSON + summary files (gitignored)
```

## Setup

On Crux (uses the pre-existing venv in ~/crux/alcf-inference-venv):

```bash
cd ~/crux
git clone https://github.com/jtchilders/crux-agent-benchmarks.git
cd crux-agent-benchmarks
# venv already has openai + globus_sdk installed
source ~/crux/alcf-inference-venv/bin/activate
```

## Running

### Sophia — interactive smoke test

```bash
source ~/crux/alcf-inference-venv/bin/activate
python benchmark.py --concurrency 4 --model openai/gpt-oss-20b --max-tokens 256
```

### Sophia — batch full sweep

```bash
qsub run_full_sweep.pbs
```

### Argo — interactive smoke test

```bash
source ~/crux/alcf-inference-venv/bin/activate
python benchmark.py --backend argo --api-key jchilders --model "GPT-5" --concurrency 4
```

### Argo — batch full sweep

```bash
qsub run_argo_sweep.pbs
```

**Note:** Some Argo models (Claude/Anthropic) reject the `temperature` parameter.
The benchmark auto-detects these and omits it. Argo also returns bonus
`latency_checkpoint` data (engine TTFT, service TTFT, etc.) which is captured
in the per-call results under `argo_latency`.

## Authentication

### Sophia backend (default)

Uses `~/crux/inference_auth_token.py` (Globus auth, already authenticated).
Tokens are valid 48h and auto-refresh. No action needed unless you see auth errors,
in which case re-run `python ~/crux/inference_auth_token.py authenticate`.

### Argo backend

Argo uses a simple static API key — your ANL username. Pass it via `--api-key`
or set the `ARGO_API_KEY` environment variable:

```bash
export ARGO_API_KEY="jchilders"
python benchmark.py --backend argo --model "GPT-5" --sweep
```

No token minting or refresh needed.

## Output

Results are written to `results/benchmark_<timestamp>.json` with per-call stats
and a summary block:

```json
{
  "config": { "model": "...", "concurrency": 16, ... },
  "summary": {
    "n_calls": 16,
    "n_success": 16,
    "ttft_p50_s": 0.82,
    "ttft_p95_s": 1.43,
    "total_latency_p50_s": 3.21,
    "total_latency_p95_s": 5.87,
    "aggregate_tokens_per_sec": 412.3
  },
  "calls": [ ... ]
}
```
