#!/usr/bin/env python3
"""Plot the four-method structured-medium k_max experiment."""

import argparse
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parent.parent
METHODS = ("maxsat", "ga", "khouzani", "zenitani")
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


def _protocol_time(method_name, method, expected_ga_runs):
    """Return the recorded complete protocol time for one graph.

    GA is one method invocation consisting of all independent repetitions, so
    its time is their sum rather than the selected/median repetition's time.
    """
    if method.get("status") != "complete":
        return None
    if method.get("complete_protocol_time_s") is not None:
        return float(method["complete_protocol_time_s"])
    if method.get("protocol_wall_time_s") is not None:
        return float(method["protocol_wall_time_s"])
    if method_name == "ga":
        runs = method.get("runs", [])
        if len(runs) != expected_ga_runs:
            return None
        return sum(
            float(run["wall_time_s"])
            + float(
                run.get(
                    "final_evaluation_time_s",
                    run.get("evaluation_time_s", 0.0),
                )
            )
            for run in runs
        )
    return float(method["wall_time_s"]) + float(
        method.get(
            "final_evaluation_time_s",
            method.get("evaluation_time_s", 0.0),
        )
    )


def _mean_protocol_times(payload):
    expected_ga_runs = int(payload.get("config", {}).get("ga_runs", 5))
    result = {}
    for case in payload.get("cases", []):
        by_method = {}
        for method_name in METHODS:
            values = []
            for row in case.get("graphs", []):
                method = row.get("methods", {}).get(method_name)
                if not method:
                    continue
                value = _protocol_time(method_name, method, expected_ga_runs)
                if value is not None:
                    values.append(value)
            by_method[method_name] = statistics.fmean(values) if values else None
        result[case["k_max"]] = by_method
    return result


def _validate_structured_payload(payload):
    experiment = str(payload.get("config", {}).get("experiment", "")).lower()
    if "structured" not in experiment:
        raise RuntimeError(
            "input is not a structured k_max result (config.experiment mismatch)"
        )


def plot(payload, output_prefix):
    _validate_structured_payload(payload)
    cases = sorted(payload["summary"]["cases"], key=lambda item: item["k_max"])
    if not cases:
        raise RuntimeError("no cases found in result JSON")

    protocol_times = _mean_protocol_times(payload)
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
    quality_ax.set_title("Structured medium: baselines vs. MaxSAT")
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
        ("GA-BP (5 runs)", "ga"),
        ("Khouzani", "khouzani"),
        ("Zenitani", "zenitani"),
    ]
    group_width = 0.19
    for method_index, (label, method) in enumerate(methods):
        xs = [
            position + (method_index - 1.5) * group_width
            for position in positions
        ]
        times = [protocol_times[case["k_max"]][method] for case in cases]
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
    time_ax.set_ylabel("mean full-protocol time (s, log)")
    time_ax.set_title("Structured medium: runtime per method")
    time_ax.legend(frameon=False, fontsize=8)
    time_ax.text(
        0.5,
        -0.16,
        "GA-BP time sums all five independent runs per graph.",
        transform=time_ax.transAxes,
        ha="center",
        va="top",
        fontsize=8,
    )

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
        default=(
            "comparison/kmax_structured_medium_results/"
            "kmax_structured_medium_comparison.json"
        ),
    )
    parser.add_argument(
        "--output-prefix",
        default=(
            "comparison/kmax_structured_medium_results/"
            "kmax_structured_medium_comparison"
        ),
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
