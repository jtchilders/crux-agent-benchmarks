# crux-agent-benchmarks

Throughput and latency benchmarks for the ALCF Inference Endpoints, designed to
characterize how many parallel LLM agent calls can be sustained from Crux compute
nodes against the Sophia (A100) vLLM cluster.

## Goals

- Measure time-to-first-token (TTFT) and total latency at varying concurrency levels
- Measure aggregate token throughput (tokens/sec) across N parallel calls
- Find the concurrency ceiling before rate limiting or significant latency degradation
- Inform agent framework design: how many parallel workers can we realistically run?

## Structure

```
benchmark.py          # Main async benchmark runner
run_benchmark.pbs     # PBS job script (single Crux node)
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

### Interactive (from login node, quick smoke test)

```bash
source ~/crux/alcf-inference-venv/bin/activate
python benchmark.py --concurrency 4 --model openai/gpt-oss-20b --max-tokens 256
```

### Batch (PBS job, full sweep)

```bash
qsub run_benchmark.pbs
```

## Authentication

The benchmark uses `~/crux/inference_auth_token.py` (already authenticated).
Tokens are valid 48h and auto-refresh. No action needed unless you see auth errors,
in which case re-run `python ~/crux/inference_auth_token.py authenticate`.

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
