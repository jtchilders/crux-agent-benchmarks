#!/usr/bin/env python3
"""
Plot benchmark results: parallel calls (x) vs aggregate tokens/sec (y).

Usage:
    python plot_results.py [--results-dir results/] [--output-dir plots/]

Generates one figure per prompt size (short/medium/long), with one line
per model. Also generates a combined overview figure.
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend for compute nodes
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    import numpy as np
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# Short display names for long model IDs
MODEL_LABELS = {
    "openai/gpt-oss-20b":                       "gpt-oss-20b",
    "openai/gpt-oss-120b":                      "gpt-oss-120b",
    "google/gemma-4-E4B-it":                    "gemma-4-E4B",
    "google/gemma-4-31B-it":                    "gemma-4-31B",
    "google/gemma-3-27b-it":                    "gemma-3-27B",
    "meta-llama/Meta-Llama-3.1-8B-Instruct":   "Llama-3.1-8B",
    "meta-llama/Meta-Llama-3.1-70B-Instruct":  "Llama-3.1-70B",
    "meta-llama/Llama-3.3-70B-Instruct":       "Llama-3.3-70B",
    "mistralai/Mixtral-8x22B-Instruct-v0.1":   "Mixtral-8x22B",
}

# Color cycle — distinct enough for 9 models
COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22",
]

PROMPT_ORDER = ["short", "medium", "long"]
PROMPT_TITLES = {
    "short":  "Short prompt (~30 tok input)",
    "medium": "Medium prompt (~80 tok input)",
    "long":   "Long prompt (~400 tok input)",
}


def load_results(results_dir: Path) -> dict:
    """
    Returns nested dict:
      data[prompt_label][model_id] = sorted list of (concurrency, agg_tps, success_rate)
    """
    data = defaultdict(lambda: defaultdict(list))

    for f in sorted(results_dir.glob("sweep_*.json")):
        try:
            with open(f) as fh:
                runs = json.load(fh)
        except Exception as e:
            print(f"  Warning: could not load {f}: {e}")
            continue

        # File may be a list (sweep) or a single dict
        if isinstance(runs, dict):
            runs = [runs]

        for run in runs:
            cfg = run.get("config", {})
            summ = run.get("summary", {})
            model = cfg.get("model", "unknown")
            prompt_label = cfg.get("prompt_label", "custom")
            concurrency = summ.get("concurrency")
            tps = summ.get("aggregate_tokens_per_sec")
            n_calls = summ.get("n_calls", 1)
            n_success = summ.get("n_success", 0)
            success_rate = n_success / n_calls if n_calls else 0

            if concurrency is not None and tps is not None:
                data[prompt_label][model].append((concurrency, tps, success_rate))

    # Sort each series by concurrency
    for pl in data:
        for model in data[pl]:
            data[pl][model].sort(key=lambda x: x[0])

    return data


def plot_prompt_figure(prompt_label: str, model_data: dict, output_path: Path):
    fig, ax = plt.subplots(figsize=(10, 6))

    models = sorted(model_data.keys())
    for i, model in enumerate(models):
        series = model_data[model]
        xs = [s[0] for s in series]
        ys = [s[1] for s in series]
        success = [s[2] for s in series]
        label = MODEL_LABELS.get(model, model.split("/")[-1])
        color = COLORS[i % len(COLORS)]

        ax.plot(xs, ys, "o-", color=color, label=label, linewidth=2, markersize=6)

        # Mark degraded points (success rate < 1.0) with open markers
        bad_xs = [x for x, s in zip(xs, success) if s < 1.0]
        bad_ys = [y for y, s in zip(ys, success) if s < 1.0]
        if bad_xs:
            ax.plot(bad_xs, bad_ys, "o", color=color,
                    markersize=10, markerfacecolor="none", markeredgewidth=2)

    ax.set_xlabel("Parallel calls (concurrency)", fontsize=12)
    ax.set_ylabel("Aggregate tokens / second", fontsize=12)
    ax.set_title(
        f"ALCF Sophia Inference Throughput\n{PROMPT_TITLES.get(prompt_label, prompt_label)}",
        fontsize=13, fontweight="bold"
    )
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax.set_xticks([1, 2, 4, 8, 16, 32])

    ax.legend(loc="upper left", fontsize=9, framealpha=0.85)
    ax.grid(True, which="major", linestyle="--", alpha=0.5)
    ax.grid(True, which="minor", linestyle=":", alpha=0.3)

    # Footnote
    fig.text(0.99, 0.01,
             "Open markers = <100% success rate  |  ALCF Crux → Sophia vLLM",
             ha="right", va="bottom", fontsize=7, color="gray")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_overview(data: dict, output_path: Path):
    """3-panel figure: one column per prompt size, best model highlighted."""
    prompts_present = [p for p in PROMPT_ORDER if p in data]
    if not prompts_present:
        return

    fig, axes = plt.subplots(1, len(prompts_present), figsize=(6 * len(prompts_present), 6),
                             sharey=False)
    if len(prompts_present) == 1:
        axes = [axes]

    all_models = sorted(set(m for pl in data.values() for m in pl))

    for ax, prompt_label in zip(axes, prompts_present):
        model_data = data[prompt_label]
        for i, model in enumerate(all_models):
            if model not in model_data:
                continue
            series = model_data[model]
            xs = [s[0] for s in series]
            ys = [s[1] for s in series]
            label = MODEL_LABELS.get(model, model.split("/")[-1])
            color = COLORS[i % len(COLORS)]
            ax.plot(xs, ys, "o-", color=color, label=label, linewidth=1.8, markersize=5)

        ax.set_xlabel("Concurrency", fontsize=11)
        ax.set_title(PROMPT_TITLES.get(prompt_label, prompt_label), fontsize=10)
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.set_xticks([1, 2, 4, 8, 16, 32])
        ax.grid(True, linestyle="--", alpha=0.4)
        if ax is axes[0]:
            ax.set_ylabel("Aggregate tokens / second", fontsize=11)

    # Shared legend under the figure
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5,
               fontsize=8, framealpha=0.85,
               bbox_to_anchor=(0.5, -0.08))

    fig.suptitle("ALCF Sophia Inference Throughput — All Models", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def print_text_table(data: dict):
    """ASCII summary table for environments without matplotlib."""
    for prompt_label in PROMPT_ORDER:
        if prompt_label not in data:
            continue
        print(f"\n{'='*70}")
        print(f"  Prompt: {PROMPT_TITLES.get(prompt_label, prompt_label)}")
        print(f"{'='*70}")
        print(f"  {'Model':<35} {'C':>4}  {'tok/s':>8}  {'ok%':>6}")
        print(f"  {'-'*35} {'----':>4}  {'------':>8}  {'----':>6}")
        for model, series in sorted(data[prompt_label].items()):
            label = MODEL_LABELS.get(model, model.split("/")[-1])
            for conc, tps, sr in series:
                flag = " ⚠" if sr < 1.0 else ""
                print(f"  {label:<35} {conc:>4}  {tps:>8.1f}  {sr*100:>5.0f}%{flag}")


def main():
    parser = argparse.ArgumentParser(description="Plot inference benchmark results")
    parser.add_argument("--results-dir", default="results", help="Directory with JSON results")
    parser.add_argument("--output-dir", default="plots", help="Directory for output figures")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)

    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        sys.exit(1)

    print(f"Loading results from {results_dir}...")
    data = load_results(results_dir)

    if not data:
        print("No results found.")
        sys.exit(1)

    print(f"Found data for prompts: {sorted(data.keys())}")
    for pl in data:
        print(f"  {pl}: {sorted(data[pl].keys())}")

    # Always print text table
    print_text_table(data)

    if not HAS_MPL:
        print("\nmatplotlib not available — text table only.")
        print("Install with: pip install matplotlib numpy")
        return

    output_dir.mkdir(exist_ok=True)
    print(f"\nGenerating plots in {output_dir}/...")

    # Per-prompt figures
    for prompt_label in PROMPT_ORDER:
        if prompt_label not in data:
            continue
        out = output_dir / f"throughput_{prompt_label}.png"
        plot_prompt_figure(prompt_label, data[prompt_label], out)

    # Overview figure
    plot_overview(data, output_dir / "throughput_overview.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
