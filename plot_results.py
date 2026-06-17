#!/usr/bin/env python3
"""
Plot benchmark results: concurrency vs throughput, latency, TTFT, and errors.

Usage:
    python plot_results.py [--results-dir results/] [--output-dir plots/]

Generates per-prompt-size figures for:
  - Aggregate tokens/sec (mean ± std dev across waves)
  - Median total latency with p95 band and std dev error bars
  - Median TTFT with std dev error bars
  - Error rate (% 429s / failures)

Plus a 4-panel overview figure.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    from matplotlib.lines import Line2D
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("matplotlib not available — text output only")

# ─── display config ────────────────────────────────────────────────────────────

MODEL_LABELS = {
    # Sophia models
    "openai/gpt-oss-20b":                       "gpt-oss-20b",
    "openai/gpt-oss-120b":                      "gpt-oss-120b",
    "google/gemma-4-E4B-it":                    "gemma-4-E4B",
    "google/gemma-4-31B-it":                    "gemma-4-31B",
    "google/gemma-3-27b-it":                    "gemma-3-27B",
    "meta-llama/Meta-Llama-3.1-8B-Instruct":   "Llama-3.1-8B",
    "meta-llama/Meta-Llama-3.1-70B-Instruct":  "Llama-3.1-70B",
    "meta-llama/Llama-3.3-70B-Instruct":       "Llama-3.3-70B",
    "mistralai/Mixtral-8x22B-Instruct-v0.1":   "Mixtral-8x22B",
    # Argo models
    "Claude Sonnet 4.6":                        "Claude Sonnet 4.6",
    "Claude Opus 4.7":                          "Claude Opus 4.7",
    "GPT-5":                                    "GPT-5",
    "GPT-4.1":                                  "GPT-4.1",
    "GPT-4.1-nano":                             "GPT-4.1-nano",
    "Gemini 2.5 Flash":                         "Gemini 2.5 Flash",
    "Gemini 2.5 Pro":                           "Gemini 2.5 Pro",
}

COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
    "#c49c94",
]

PROMPT_ORDER  = ["short", "medium", "long"]
PROMPT_TITLES = {
    "short":  "Short prompt (~30 tok input)",
    "medium": "Medium prompt (~80 tok input)",
    "long":   "Long prompt (~400 tok input)",
}

CONC_LEVELS = [1, 2, 4, 8, 16, 32]

# ─── data loading ──────────────────────────────────────────────────────────────

def load_results(results_dir: Path) -> dict:
    """
    Returns nested dict:
      data[prompt_label][model_id][concurrency] = {
          'tps':      float  (aggregate tok/s for that run),
          'latencies': [float, ...]  (per-call total_latency_s, successful only),
          'ttfts':    [float, ...]   (per-call ttft_s, successful only),
          'n_calls':  int,
          'n_success': int,
          'n_failed': int,
          'errors':   [str, ...],
      }
    """
    data = defaultdict(lambda: defaultdict(dict))

    for f in sorted(results_dir.glob("sweep_*.json")):
        try:
            with open(f) as fh:
                runs = json.load(fh)
        except Exception as e:
            print(f"  Warning: could not load {f}: {e}")
            continue

        if isinstance(runs, dict):
            runs = [runs]

        for run in runs:
            cfg  = run.get("config", {})
            summ = run.get("summary", {})
            calls = run.get("calls", [])

            model        = cfg.get("model", "unknown")
            prompt_label = cfg.get("prompt_label", "custom")
            concurrency  = summ.get("concurrency")
            tps          = summ.get("aggregate_tokens_per_sec")

            if concurrency is None or tps is None:
                continue

            successful = [c for c in calls if c.get("success")]
            failed     = [c for c in calls if not c.get("success")]

            latencies = [c["total_latency_s"] for c in successful if c.get("total_latency_s") is not None]
            ttfts     = [c["ttft_s"]          for c in successful if c.get("ttft_s")          is not None]
            errors    = [c.get("error", "") or "" for c in failed]

            data[prompt_label][model][concurrency] = {
                "tps":       tps,
                "latencies": latencies,
                "ttfts":     ttfts,
                "n_calls":   summ.get("n_calls", len(calls)),
                "n_success": summ.get("n_success", len(successful)),
                "n_failed":  summ.get("n_failed",  len(failed)),
                "errors":    errors,
            }

    return data


def model_series(model_data: dict, metric_fn) -> tuple[list, list, list]:
    """
    Returns (xs, ys, yerrs) sorted by concurrency for plotting.
    metric_fn(entry) -> (y_value, y_err)  or None to skip.
    """
    xs, ys, yerrs = [], [], []
    for c in sorted(model_data.keys()):
        result = metric_fn(model_data[c])
        if result is None:
            continue
        y, ye = result
        xs.append(c)
        ys.append(y)
        yerrs.append(ye)
    return xs, ys, yerrs


# ─── metric extractors ─────────────────────────────────────────────────────────

def m_tps(entry):
    return (entry["tps"], 0)   # single value per concurrency level; no wave breakdown in JSON


def m_latency_median(entry):
    lats = entry["latencies"]
    if not lats:
        return None
    return (float(np.median(lats)), float(np.std(lats)))


def m_latency_p95(entry):
    lats = entry["latencies"]
    if not lats:
        return None
    return (float(np.percentile(lats, 95)), 0)


def m_ttft_median(entry):
    ttfts = entry["ttfts"]
    if not ttfts:
        return None
    return (float(np.median(ttfts)), float(np.std(ttfts)))


def m_error_rate(entry):
    n = entry["n_calls"]
    if not n:
        return None
    return (100.0 * entry["n_failed"] / n, 0)


# ─── per-prompt 4-panel figure ────────────────────────────────────────────────

def plot_prompt_figure(prompt_label: str, model_data: dict, output_path: Path):
    models = sorted(model_data.keys())
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"ALCF Sophia Inference — {PROMPT_TITLES.get(prompt_label, prompt_label)}",
        fontsize=14, fontweight="bold"
    )

    panels = [
        (axes[0, 0], "Aggregate Throughput",       "tok / sec",       m_tps,            False),
        (axes[0, 1], "Total Latency (median ± σ)", "seconds",         m_latency_median, True),
        (axes[1, 0], "TTFT (median ± σ)",          "seconds",         m_ttft_median,    True),
        (axes[1, 1], "Error Rate",                  "% of calls failed", m_error_rate,  False),
    ]

    legend_handles = []

    for i, (ax, title, ylabel, metric_fn, show_p95) in enumerate(panels):
        for j, model in enumerate(models):
            label  = MODEL_LABELS.get(model, model.split("/")[-1])
            color  = COLORS[j % len(COLORS)]
            entry  = model_data[model]

            xs, ys, yerrs = model_series(entry, metric_fn)
            if not xs:
                continue

            yerrs_arr = np.array(yerrs)

            if metric_fn == m_tps:
                ax.plot(xs, ys, "o-", color=color, label=label, linewidth=2, markersize=5)
            else:
                ax.errorbar(xs, ys, yerr=yerrs_arr,
                            fmt="o-", color=color, label=label,
                            linewidth=2, markersize=5,
                            capsize=4, capthick=1.5, elinewidth=1.2)

            # overlay p95 latency as a faint dashed line
            if show_p95 and metric_fn == m_latency_median:
                xs95, ys95, _ = model_series(entry, m_latency_p95)
                ax.plot(xs95, ys95, "--", color=color, linewidth=1, alpha=0.4)

            if i == 0:
                legend_handles.append(
                    Line2D([0], [0], color=color, linewidth=2, label=label)
                )

        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Parallel calls (concurrency)", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())
        ax.set_xticks(CONC_LEVELS)
        ax.grid(True, which="major", linestyle="--", alpha=0.5)
        ax.grid(True, which="minor", linestyle=":", alpha=0.25)

        if metric_fn == m_error_rate:
            ax.set_ylim(bottom=0)
            ax.axhline(y=25, color="red", linestyle=":", linewidth=1, alpha=0.6)
            ax.text(1.1, 26, "25% threshold", fontsize=7, color="red", alpha=0.7)

    # shared legend below panels
    fig.legend(handles=legend_handles, loc="lower center",
               ncol=5, fontsize=9, framealpha=0.9,
               bbox_to_anchor=(0.5, -0.04))

    note = ("Error bars = ±1σ across calls  |  Dashed lines on latency = p95  |  "
            "ALCF Crux → Sophia vLLM  |  max_tokens=256")
    fig.text(0.5, -0.01, note, ha="center", va="bottom", fontsize=7, color="gray")

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ─── overview 3×4 grid ────────────────────────────────────────────────────────

def plot_overview(data: dict, output_path: Path):
    prompts = [p for p in PROMPT_ORDER if p in data]
    if not prompts:
        return

    all_models = sorted(set(m for pl in data.values() for m in pl))
    metrics = [
        ("Throughput (tok/s)",        m_tps,           False),
        ("Latency median ± σ (s)",    m_latency_median, True),
        ("TTFT median ± σ (s)",       m_ttft_median,    True),
        ("Error rate (%)",            m_error_rate,     False),
    ]

    nrows = len(metrics)
    ncols = len(prompts)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), squeeze=False)
    fig.suptitle("ALCF Sophia Inference — All Models Overview", fontsize=14, fontweight="bold")

    for col, prompt_label in enumerate(prompts):
        model_data = data[prompt_label]
        axes[0, col].set_title(PROMPT_TITLES.get(prompt_label, prompt_label),
                               fontsize=10, fontweight="bold")

        for row, (ylabel, metric_fn, show_err) in enumerate(metrics):
            ax = axes[row, col]
            for j, model in enumerate(all_models):
                if model not in model_data:
                    continue
                label = MODEL_LABELS.get(model, model.split("/")[-1])
                color = COLORS[j % len(COLORS)]
                xs, ys, yerrs = model_series(model_data[model], metric_fn)
                if not xs:
                    continue
                if show_err:
                    ax.errorbar(xs, ys, yerr=np.array(yerrs),
                                fmt="o-", color=color, label=label,
                                linewidth=1.5, markersize=4,
                                capsize=3, capthick=1.2, elinewidth=1)
                else:
                    ax.plot(xs, ys, "o-", color=color, label=label,
                            linewidth=1.5, markersize=4)

            ax.set_xscale("log", base=2)
            ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
            ax.set_xticks(CONC_LEVELS)
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.tick_params(labelsize=8)
            if col == 0:
                ax.set_ylabel(ylabel, fontsize=9)
            if row == nrows - 1:
                ax.set_xlabel("Concurrency", fontsize=9)

    # shared legend
    handles = [Line2D([0], [0], color=COLORS[j % len(COLORS)], linewidth=2,
                      label=MODEL_LABELS.get(m, m.split("/")[-1]))
               for j, m in enumerate(all_models)]
    fig.legend(handles=handles, loc="lower center", ncol=5,
               fontsize=8, framealpha=0.9, bbox_to_anchor=(0.5, -0.03))

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ─── text summary ─────────────────────────────────────────────────────────────

def print_text_table(data: dict):
    for prompt_label in PROMPT_ORDER:
        if prompt_label not in data:
            continue
        print(f"\n{'='*85}")
        print(f"  Prompt: {PROMPT_TITLES.get(prompt_label, prompt_label)}")
        print(f"{'='*85}")
        print(f"  {'Model':<22} {'C':>4}  {'tok/s':>7}  {'lat_med':>7}  {'lat_σ':>6}  "
              f"{'lat_p95':>7}  {'ttft_med':>8}  {'ttft_σ':>6}  {'err%':>5}")
        print(f"  {'-'*22} {'----':>4}  {'-------':>7}  {'-------':>7}  {'------':>6}  "
              f"{'-------':>7}  {'--------':>8}  {'------':>6}  {'-----':>5}")

        for model in sorted(data[prompt_label].keys()):
            label = MODEL_LABELS.get(model, model.split("/")[-1])
            for c in sorted(data[prompt_label][model].keys()):
                e = data[prompt_label][model][c]
                lats  = e["latencies"]
                ttfts = e["ttfts"]
                lat_med = np.median(lats)  if lats  else float("nan")
                lat_std = np.std(lats)     if lats  else float("nan")
                lat_p95 = np.percentile(lats, 95) if lats else float("nan")
                ttft_med = np.median(ttfts) if ttfts else float("nan")
                ttft_std = np.std(ttfts)    if ttfts else float("nan")
                err_pct  = 100.0 * e["n_failed"] / e["n_calls"] if e["n_calls"] else 0
                flag = " ⚠" if err_pct > 0 else ""
                print(f"  {label:<22} {c:>4}  {e['tps']:>7.1f}  {lat_med:>7.3f}  "
                      f"{lat_std:>6.3f}  {lat_p95:>7.3f}  {ttft_med:>8.3f}  "
                      f"{ttft_std:>6.3f}  {err_pct:>4.0f}%{flag}")


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Plot inference benchmark results")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--output-dir",  default="plots")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir  = Path(args.output_dir)

    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        sys.exit(1)

    print(f"Loading results from {results_dir}/ ...")
    data = load_results(results_dir)

    if not data:
        print("No results found.")
        sys.exit(1)

    print_text_table(data)

    if not HAS_MPL:
        print("\nmatplotlib not available — text table only.")
        return

    output_dir.mkdir(exist_ok=True)
    print(f"\nGenerating plots → {output_dir}/")

    for prompt_label in PROMPT_ORDER:
        if prompt_label not in data:
            continue
        plot_prompt_figure(prompt_label, data[prompt_label],
                           output_dir / f"throughput_{prompt_label}.png")

    plot_overview(data, output_dir / "throughput_overview.png")
    print("\nDone.")


if __name__ == "__main__":
    main()
