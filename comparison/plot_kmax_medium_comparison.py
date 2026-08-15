#!/usr/bin/env python3
"""Plot four-method medium-graph comparison as P out-degree k_max varies."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parent.parent
COLORS = {
    "win": "#299d8f",
    "tie": "#e9c46a",
    "loss": "#d87659",
    "maxsat": "#299d8f",
    "ga": "#d87659",
    "khouzani": "#e9c46a",
    "zenitani": "#8a6bbf",
}


def _percentages(counts):
    total = counts["win"] + counts["tie"] + counts["loss"]
    if total == 0:
        return None
    return [100.0 * counts[key] / total for key in ("win", "tie", "loss")]


def plot(payload, output_prefix):
    cases = sorted(payload["summary"]["cases"], key=lambda item: item["k_max"])
    if not cases:
        raise RuntimeError("no cases found in result JSON")

    labels = [rf"$k_{{max}}={case['k_max']}$" for case in cases]
    positions = list(range(len(cases)))
    fig, (quality_ax, time_ax) = plt.subplots(1, 2, figsize=(12.8, 4.8))
    fig.patch.set_facecolor("white")

    baselines = [
        ("GA", "ga", -0.27),
        ("Kh", "khouzani", 0.0),
        ("Zen", "zenitani", 0.27),
    ]
    width = 0.25
    for label, method, offset in baselines:
        for index, case in enumerate(cases):
            values = _percentages(case["pairwise_vs_maxsat"][method])
            x = positions[index] + offset
            if values is not None:
                win, tie, loss = values
                quality_ax.bar(x, win, width, color=COLORS["win"], edgecolor="white")
                quality_ax.bar(
                    x, tie, width, bottom=win, color=COLORS["tie"], edgecolor="white"
                )
                quality_ax.bar(
                    x,
                    loss,
                    width,
                    bottom=win + tie,
                    color=COLORS["loss"],
                    edgecolor="white",
                )
            quality_ax.text(x, -4, label, ha="center", va="top", fontsize=8)

    quality_ax.set_ylim(0, 100)
    quality_ax.set_xticks(positions)
    quality_ax.set_xticklabels(labels)
    quality_ax.tick_params(axis="x", pad=22, length=0)
    quality_ax.set_ylabel("percentage (%)")
    quality_ax.set_title("Baselines vs. MaxSAT")
    quality_ax.legend(
        handles=[
            Patch(facecolor=COLORS["win"], label="baseline wins"),
            Patch(facecolor=COLORS["tie"], label="tie"),
            Patch(facecolor=COLORS["loss"], label="MaxSAT wins"),
        ],
        frameon=False,
        fontsize=8,
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
    )

    methods = [
        ("MaxSAT", "maxsat"),
        ("GA-BP median", "ga"),
        ("Khouzani", "khouzani"),
        ("Zenitani", "zenitani"),
    ]
    group_width = 0.19
    for method_index, (label, method) in enumerate(methods):
        xs = [
            position + (method_index - 1.5) * group_width
            for position in positions
        ]
        times = [case["mean_wall_time_s"].get(method) for case in cases]
        heights = [max(value, 1e-3) if value is not None else 1e-3 for value in times]
        bars = time_ax.bar(
            xs,
            heights,
            group_width,
            color=COLORS[method],
            edgecolor="white",
            label=label,
        )
        for bar, value in zip(bars, times):
            if value is None:
                bar.set_alpha(0.15)

    time_ax.set_yscale("log")
    time_ax.set_xticks(positions)
    time_ax.set_xticklabels(labels)
    time_ax.set_ylabel("mean complete-flow time (s, log)")
    time_ax.set_title("Runtime per method")
    time_ax.legend(frameon=False, fontsize=8)

    for axis in (quality_ax, time_ax):
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.grid(False)

    fig.tight_layout()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_prefix.with_suffix(".png")
    pdf_path = output_prefix.with_suffix(".pdf")
    fig.savefig(png_path, dpi=180, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png_path, pdf_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="comparison/kmax_medium_results/kmax_medium_comparison.json",
    )
    parser.add_argument(
        "--output-prefix",
        default="comparison/kmax_medium_results/kmax_medium_comparison",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_prefix = Path(args.output_prefix)
    if not input_path.is_absolute():
        input_path = ROOT / input_path
    if not output_prefix.is_absolute():
        output_prefix = ROOT / output_prefix
    with open(input_path, encoding="utf-8") as handle:
        payload = json.load(handle)
    paths = plot(payload, output_prefix)
    print(f"saved: {paths[0]}")
    print(f"saved: {paths[1]}")


if __name__ == "__main__":
    main()
