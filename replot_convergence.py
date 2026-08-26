#!/usr/bin/env python3
"""Replot convergence curves with 4 subplots: robust step-fill convergence curves."""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


WCNF_INFO = {
    0.1: {"vars": 790, "clauses": 2126},
    0.2: {"vars": 1580, "clauses": 4246},
    0.5: {"vars": 3950, "clauses": 10594},
    0.8: {"vars": 6320, "clauses": 17656},
    1.0: {"vars": 7900, "clauses": 22712},
    1.2: {"vars": 9480, "clauses": 27692},
    1.5: {"vars": 11850, "clauses": 34796},
    2.0: {"vars": 15800, "clauses": 46478},
}

# Scales to plot (4 subplots)
SELECTED_SCALES = [1.0, 1.2, 1.5, 2.0]

# Set to 300.0 to force all subplots to have x-axis up to 300s
# When None, uses the maximum time from the data
FORCE_X_MAX_SECONDS = None
# FORCE_X_MAX_SECONDS = 300.0


def _ensure_matplotlib_font():
    matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False


def _clean_time_objective(time_hist, obj_hist):
    """
    Sort by time and merge duplicate timestamps.

    If multiple objective values have the same timestamp, keep the last one.
    """
    time_hist = np.asarray(time_hist, dtype=float)
    obj_hist = np.asarray(obj_hist, dtype=float)

    if time_hist.size != obj_hist.size:
        raise ValueError(
            f"time and objective length mismatch: {time_hist.size} vs {obj_hist.size}"
        )

    if time_hist.size == 0:
        return time_hist, obj_hist

    order = np.argsort(time_hist, kind="stable")
    time_hist = time_hist[order]
    obj_hist = obj_hist[order]

    clean_t = []
    clean_y = []

    for t, y in zip(time_hist, obj_hist):
        if clean_t and np.isclose(t, clean_t[-1]):
            clean_y[-1] = y
        else:
            clean_t.append(t)
            clean_y.append(y)

    return np.asarray(clean_t, dtype=float), np.asarray(clean_y, dtype=float)


def _make_post_step_xy(time_hist, obj_hist, x_max=None):
    """
    Convert point history to explicit post-step coordinates.

    For points:
        t0, t1, t2
        y0, y1, y2

    post-step means:
        y0 is held on [t0, t1)
        y1 is held on [t1, t2)
        y2 is held after t2
    """
    time_hist, obj_hist = _clean_time_objective(time_hist, obj_hist)

    if time_hist.size == 0:
        return time_hist, obj_hist

    if x_max is not None and x_max > time_hist[-1]:
        time_hist = np.append(time_hist, float(x_max))
        obj_hist = np.append(obj_hist, obj_hist[-1])

    # Explicit stair coordinates
    step_x = np.repeat(time_hist, 2)[1:]
    step_y = np.repeat(obj_hist, 2)[:-1]

    return step_x, step_y


def _get_data_max_time(results, selected_scales):
    max_time = 0.0

    for scale in selected_scales:
        data = results.get(scale)
        if not data:
            continue

        time_hist = data.get("time", [])
        if time_hist:
            max_time = max(max_time, float(np.max(np.asarray(time_hist, dtype=float))))

    return max_time


def plot_convergence(results, out_dir):
    _ensure_matplotlib_font()

    line_colors = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0"]

    if FORCE_X_MAX_SECONDS is not None:
        x_max = float(FORCE_X_MAX_SECONDS)
    else:
        x_max = _get_data_max_time(results, SELECTED_SCALES)

    if x_max <= 0:
        x_max = 1.0

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()

    for idx, scale in enumerate(SELECTED_SCALES):
        ax = axes[idx]
        color = line_colors[idx]

        data = results.get(scale)
        info = WCNF_INFO.get(scale, {})

        title = f"Variable={info.get('vars', '?')}, clauses={info.get('clauses', '?')}"

        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Time (s)", fontsize=9)
        ax.set_ylabel(r"$\mathcal{F}(\mathbf{s})$", fontsize=9)
        ax.axhline(0, linewidth=0.8, color="black", alpha=0.55)
        ax.grid(False)
        ax.set_xlim(0, x_max)

        if data is None:
            ax.text(
                0.5,
                0.5,
                "No data",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=10,
            )
            continue

        time_hist = data.get("time", [])
        obj_hist = data.get("objective", [])

        if not time_hist or not obj_hist:
            ax.set_title(f"{title}\n(no convergence data)", fontsize=10)
            continue

        step_x, step_y = _make_post_step_xy(
            time_hist,
            obj_hist,
            x_max=x_max,
        )

        # Positive region: fill only where y > 0
        y_pos = np.where(step_y > 0, step_y, 0.0)

        # Negative region: fill only where y < 0
        y_neg = np.where(step_y < 0, step_y, 0.0)

        ax.fill_between(
            step_x,
            0,
            y_pos,
            alpha=0.2,
            color=color,
            linewidth=0,
        )

        ax.fill_between(
            step_x,
            0,
            y_neg,
            alpha=0.2,
            color=color,
            linewidth=0,
        )

        # Staircase curve
        ax.plot(
            step_x,
            step_y,
            linewidth=1.8,
            color=color,
        )

        # Original sample points
        clean_t, clean_y = _clean_time_objective(time_hist, obj_hist)
        ax.plot(
            clean_t,
            clean_y,
            marker="o",
            linestyle="None",
            markersize=4,
            color=color,
        )

        y_min = min(float(np.min(step_y)), 0.0)
        y_max = max(float(np.max(step_y)), 0.0)

        y_range = y_max - y_min
        margin = 0.05 * y_range if y_range > 0 else 1.0

        ax.set_ylim(y_min - margin, y_max + margin)

    plt.tight_layout()

    out_paths = [
        out_dir / "convergence_objective_v2.png",
        out_dir / "convergence_objective_v2.pdf",
    ]
    for out_path in out_paths:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    for out_path in out_paths:
        print(f"Convergence curve saved: {out_path}")


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="scaling_results/scaling_summary.json",
        help="scaling summary JSON generated by batch_run_and_analyze.py",
    )
    parser.add_argument(
        "--output-dir",
        default="scaling_results",
        help="directory for convergence_objective_v2.png and .pdf",
    )
    args = parser.parse_args()

    json_path = Path(args.input)
    out_dir = Path(args.output_dir)
    if not json_path.is_absolute():
        json_path = root / json_path
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(json_path, "r", encoding="utf-8") as f:
        all_results = json.load(f)

    all_results = {float(k): v for k, v in all_results.items()}

    plot_convergence(all_results, out_dir)


if __name__ == "__main__":
    main()
