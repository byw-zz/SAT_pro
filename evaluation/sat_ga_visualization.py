#!/usr/bin/env python3
"""
SAT vs GA Performance Visualization

Generates:
1. scale_comparison.png / scale_comparison.pdf
   - Single horizontal stacked bar chart
   - GA wins are split into SAT placed 2nd / 3rd / 4th+
   - Rank segments are shown as percentages of the total number of graphs

2. detailed_comparison.png / detailed_comparison.pdf
   - Per-graph objective curves
   - Per-graph objective difference bars

3. analysis_report.txt
   - Text summary

Main improvements:
- scale_comparison uses a single merged horizontal stacked bar figure.
- Tighter layout with compact figure size and adjusted spacing.
- Legends are placed above each subplot to avoid covering bars.
- Cleaner layout with constrained_layout.
- Softer, report-friendly color palette.
- More robust handling of missing categories and zero totals.
- Optional CLI arguments for base/output directories.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


# =========================
# Global plotting style
# =========================

plt.rcParams.update({
    "font.size": 10,
    "figure.dpi": 120,
    "savefig.dpi": 180,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.unicode_minus": False,
})

# Softer report-friendly palette
COLORS = {
    "sat": "#4E79A7",       # muted blue
    "ga": "#59A14F",        # muted green
    "tie": "#C9CED6",       # soft gray-blue
    "rank2": "#E15759",     # soft red
    "rank3": "#F28E2B",     # orange
    "rank_ge4": "#B07AA1",  # muted purple
    "bg": "#FFFFFF",
    "grid": "#D9DEE7",
    "text": "#25364D",
    "axis": "#6B7280",
    "zero": "#374151",
}


Category = Tuple[str, str, str]


CATEGORIES: List[Category] = [
    ("small", "random", "Small-Scale\nRandom"),
    ("small", "structured", "Small-Scale\nStructured"),
    ("medium", "random", "Medium-Scale\nRandom"),
    ("medium", "structured", "Medium-Scale\nStructured"),
]


DATA_FILES = {
    ("small", "random"): "random_10_100graphs.json",
    ("small", "structured"): "structured_12_17_100graphs.json",
    ("medium", "random"): "random_100_10graphs.json",
    ("medium", "structured"): "structured_151_199_10graphs.json",
}


# =========================
# Data loading / extraction
# =========================

def load_json(filepath: str) -> Dict[str, Any]:
    """Load a JSON data file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_mean(values: List[float]) -> float:
    """Return mean for non-empty list; otherwise 0."""
    return float(np.mean(values)) if values else 0.0


def extract_metrics(data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract summary metrics required by the plots and report."""
    args = data.get("args", {})
    summary = data.get("summary", {})
    results = data.get("results", [])

    graph_type = args.get("graph_type", "unknown")

    if results:
        graph_stats = results[0].get("graph_stats", {})
        total_nodes = graph_stats.get("total", 0)
        nP = graph_stats.get("nP", 0)
        nE = graph_stats.get("nE", 0)
        nC = graph_stats.get("nC", 0)
        nD = graph_stats.get("nD", 0)
        edges = graph_stats.get("edges", 0)
    else:
        total_nodes = nP = nE = nC = nD = edges = 0

    rank2 = rank3 = rank_ge4 = 0
    for r in results:
        if r.get("winner") == "GA":
            rank = r.get("maxsat_rank_in_ga_all")
            if rank == 2:
                rank2 += 1
            elif rank == 3:
                rank3 += 1
            else:
                rank_ge4 += 1

    ga_objectives: List[float] = []
    sat_objectives: List[float] = []
    objective_diffs: List[float] = []

    for r in results:
        ga_obj = r.get("bp_objective", 0)
        sat_obj = r.get("maxsat_objective", 0)
        ga_objectives.append(ga_obj)
        sat_objectives.append(sat_obj)
        objective_diffs.append(ga_obj - sat_obj)

    metrics = {
        "graph_type": graph_type,
        "total_nodes": total_nodes,
        "nP": nP,
        "nE": nE,
        "nC": nC,
        "nD": nD,
        "edges": edges,
        "num_graphs": len(results),
        "num_sat_better": summary.get("num_sat_better", 0),
        "num_ga_better": summary.get("num_ga_better", 0),
        "num_tie": summary.get("num_tie", 0),
        "avg_bp_time": summary.get("avg_bp_time_s", 0),
        "avg_maxsat_time": summary.get("avg_maxsat_time_s", 0),
        "rank2": rank2,
        "rank3": rank3,
        "rank_ge4": rank_ge4,
        "ga_objectives": ga_objectives,
        "sat_objectives": sat_objectives,
        "objective_diffs": objective_diffs,
        "avg_ga_objective": safe_mean(ga_objectives),
        "avg_sat_objective": safe_mean(sat_objectives),
        "avg_objective_diff": safe_mean(objective_diffs),
        "results": results,
    }

    return metrics


# =========================
# Plot helper functions
# =========================

def style_axis(ax: plt.Axes, grid_axis: str = "y") -> None:
    """Apply consistent styling to an axis."""
    ax.set_facecolor(COLORS["bg"])
    ax.spines["top"].set_color(COLORS["axis"])
    ax.spines["right"].set_color(COLORS["axis"])
    ax.spines["left"].set_color(COLORS["axis"])
    ax.spines["bottom"].set_color(COLORS["axis"])
    ax.tick_params(colors=COLORS["text"])
    ax.yaxis.label.set_color(COLORS["text"])
    ax.xaxis.label.set_color(COLORS["text"])
    ax.title.set_color(COLORS["text"])

    if grid_axis:
        ax.grid(
            axis=grid_axis,
            linestyle="-",
            linewidth=0.7,
            alpha=0.70,
            color=COLORS["grid"],
        )
        ax.set_axisbelow(True)


def add_bar_labels(
    ax: plt.Axes,
    bars,
    offset: float = 0.25,
    fontsize: int = 8,
) -> None:
    """Add integer value labels above bars."""
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + offset,
                f"{int(height)}",
                ha="center",
                va="bottom",
                fontsize=fontsize,
                color=COLORS["text"],
            )


def collect_category_data(
    all_metrics: Dict[Tuple[str, str], Dict[str, Any]]
) -> Tuple[List[str], Dict[str, Dict[str, Any]]]:
    """Collect plotting data for categories that exist in all_metrics."""
    labels: List[str] = []
    data: Dict[str, Dict[str, Any]] = {}

    for scale, gtype, label in CATEGORIES:
        key = (scale, gtype)
        if key not in all_metrics:
            continue

        m = all_metrics[key]
        total = m.get("num_graphs", 0)
        if total <= 0:
            continue

        labels.append(label)
        data[label] = {
            "sat_better": m.get("num_sat_better", 0),
            "ga_better": m.get("num_ga_better", 0),
            "tie": m.get("num_tie", 0),
            "total": total,
            "rank2": m.get("rank2", 0),
            "rank3": m.get("rank3", 0),
            "rank_ge4": m.get("rank_ge4", 0),
        }

    return labels, data


# =========================
# Main comparison figure
# =========================

def plot_scale_comparison(
    all_metrics: Dict[Tuple[str, str], Dict[str, Any]],
    output_dir: str
) -> None:
    """Generate a clustered grouped-bar layout with the legend placed above the axes.

    Layout features:
    - Vertical grouped bars
    - Two higher-level groups: Small / Medium
    - Each higher-level group contains Random / Structured
    - 5 methods per subgroup:
        SAT wins, Tie, SAT placed 2nd, SAT placed 3rd, SAT placed 4th+
    - Legend is moved above the plotting area to avoid covering bars
    - No title, no extra x-axis label
    """

    labels, data = collect_category_data(all_metrics)
    if not labels:
        print("  Skipped scale_comparison.png: no valid category data.")
        return

    ordered_keys = [
        "Small-Scale\nRandom",
        "Small-Scale\nStructured",
        "Medium-Scale\nRandom",
        "Medium-Scale\nStructured",
    ]
    ordered_labels = [label for label in ordered_keys if label in data]
    if not ordered_labels:
        print("  Skipped scale_comparison.png: no valid ordered category data.")
        return

    sat_wins = np.array([
        data[label]["sat_better"] / data[label]["total"] * 100
        for label in ordered_labels
    ])
    ties = np.array([
        data[label]["tie"] / data[label]["total"] * 100
        for label in ordered_labels
    ])
    sat_rank2 = np.array([
        data[label]["rank2"] / data[label]["total"] * 100
        for label in ordered_labels
    ])
    sat_rank3 = np.array([
        data[label]["rank3"] / data[label]["total"] * 100
        for label in ordered_labels
    ])
    sat_rank_ge4 = np.array([
        data[label]["rank_ge4"] / data[label]["total"] * 100
        for label in ordered_labels
    ])

    plot_colors = {
        "sat_win": "#F4C04B",     # SAT wins
        "tie": "#D2DCBB",         # Tie
        "rank2": "#B0CED8",       # SAT placed 2nd
        "rank3": "#64BBE0",       # SAT placed 3rd
        "rank_ge4": "#559FCF",    # SAT placed 4th+
    }

    fig, ax = plt.subplots(1, 1, figsize=(9.2, 4.8), constrained_layout=False)
    fig.patch.set_facecolor(COLORS["bg"])
    style_axis(ax, grid_axis="y")
    ax.set_axisbelow(True)

    # Leave extra top margin for the legend placed above the axes
    fig.subplots_adjust(
        left=0.10,
        right=0.98,
        bottom=0.22,
        top=0.78,
    )

    # Two major groups: Small / Medium
    # Each group contains Random / Structured
    centers = np.array([0.0, 0.9, 2.5, 3.4])
    subgroup_labels = ["Random", "Structured", "Random", "Structured"]

    bar_width = 0.12
    offsets = np.array([-2, -1, 0, 1, 2]) * bar_width

    ax.bar(
        centers + offsets[0],
        sat_wins,
        width=bar_width,
        label="MaxSAT wins",
        color=plot_colors["sat_win"],
        edgecolor="white",
        linewidth=0.8,
    )
    ax.bar(
        centers + offsets[1],
        ties,
        width=bar_width,
        label="Tie",
        color=plot_colors["tie"],
        edgecolor="white",
        linewidth=0.8,
    )
    ax.bar(
        centers + offsets[2],
        sat_rank2,
        width=bar_width,
        label="Rank 2",
        color=plot_colors["rank2"],
        edgecolor="white",
        linewidth=0.8,
    )
    ax.bar(
        centers + offsets[3],
        sat_rank3,
        width=bar_width,
        label="Rank 3",
        color=plot_colors["rank3"],
        edgecolor="white",
        linewidth=0.8,
    )
    ax.bar(
        centers + offsets[4],
        sat_rank_ge4,
        width=bar_width,
        label="Rank≥4",
        color=plot_colors["rank_ge4"],
        edgecolor="white",
        linewidth=0.8,
    )

    ax.set_ylabel("Percentage (%)")
    ax.set_xlabel("")
    ax.set_xticks(centers)
    ax.set_xticklabels(subgroup_labels)
    ax.set_ylim(0, 100)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.set_xlim(-0.45, 4.0)

    # Higher-level group labels
    trans = ax.get_xaxis_transform()
    ax.text((centers[0] + centers[1]) / 2, -0.16, "Small", ha="center", va="top", transform=trans, fontsize=12, fontweight="bold")
    ax.text((centers[2] + centers[3]) / 2, -0.16, "Medium", ha="center", va="top", transform=trans, fontsize=12, fontweight="bold")

    # Put legend above the axes so it does not cover any bars
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.2, 0.9),
        ncol=3,
        frameon=True,
        facecolor="white",
        edgecolor=COLORS["grid"],
        handlelength=1.8,
        columnspacing=0.9,
        borderpad=0.4,
    )

    output_path_png = os.path.join(output_dir, "scale_comparison.png")
    output_path_pdf = os.path.join(output_dir, "scale_comparison.pdf")

    plt.savefig(
        output_path_png,
        facecolor=COLORS["bg"],
        bbox_inches="tight",
        pad_inches=0.08,
    )
    plt.savefig(
        output_path_pdf,
        facecolor=COLORS["bg"],
        bbox_inches="tight",
        pad_inches=0.08,
    )
    plt.close(fig)

    print("  Saved: scale_comparison.png")
    print("  Saved: scale_comparison.pdf")

# =========================
# Detailed figure
# =========================

def plot_detailed_comparison(
    all_metrics: Dict[Tuple[str, str], Dict[str, Any]],
    output_dir: str
) -> None:
    """Generate detailed per-graph comparison for each category."""

    detailed_categories = [
        ("small", "random", "Small-Scale Random"),
        ("small", "structured", "Small-Scale Structured"),
        ("medium", "random", "Medium-Scale Random"),
        ("medium", "structured", "Medium-Scale Structured"),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(18, 8.2), constrained_layout=True)
    fig.patch.set_facecolor(COLORS["bg"])

    for idx, (scale, gtype, label) in enumerate(detailed_categories):
        key = (scale, gtype)

        ax_obj = axes[0, idx]
        style_axis(ax_obj, grid_axis="both")

        ax_diff = axes[1, idx]
        style_axis(ax_diff, grid_axis="y")

        if key not in all_metrics:
            for ax in (ax_obj, ax_diff):
                ax.text(
                    0.5,
                    0.5,
                    "No Data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=12,
                    color=COLORS["text"],
                )
            ax_obj.set_title(label, fontweight="bold")
            ax_diff.set_title("No Data")
            continue

        m = all_metrics[key]
        n = m.get("num_graphs", 0)
        x = np.arange(n)

        ga_objectives = m.get("ga_objectives", [])
        sat_objectives = m.get("sat_objectives", [])
        diffs = m.get("objective_diffs", [])

        ax_obj.fill_between(x, ga_objectives, alpha=0.16, color=COLORS["ga"])
        ax_obj.fill_between(x, sat_objectives, alpha=0.16, color=COLORS["sat"])
        ax_obj.plot(
            x,
            ga_objectives,
            "o-",
            color=COLORS["ga"],
            label="GA",
            linewidth=1.6,
            markersize=4,
        )
        ax_obj.plot(
            x,
            sat_objectives,
            "s-",
            color=COLORS["sat"],
            label="SAT",
            linewidth=1.6,
            markersize=4,
        )

        ax_obj.set_xlabel("Graph ID")
        ax_obj.set_ylabel("Objective")
        ax_obj.set_title(f"{label}\nObjective per Graph", fontweight="bold")
        ax_obj.legend(
            loc="upper left",
            bbox_to_anchor=(0.2, 1.02),
            ncol=2,
            frameon=False,
            borderaxespad=0.0,
        )

        diff_colors = [
            COLORS["ga"] if d > 0 else COLORS["sat"] if d < 0 else COLORS["tie"]
            for d in diffs
        ]

        ax_diff.bar(
            x,
            diffs,
            width=0.68,
            color=diff_colors,
            alpha=0.92,
            edgecolor="white",
            linewidth=0.6,
        )
        ax_diff.axhline(
            y=0,
            color=COLORS["zero"],
            linestyle="-",
            linewidth=1.1,
            alpha=0.85,
        )

        ax_diff.set_xlabel("Graph ID")
        ax_diff.set_ylabel("GA - SAT")
        ax_diff.set_title(f"Objective Difference (n={n})", fontweight="bold")

        sat_w = m.get("num_sat_better", 0)
        ga_w = m.get("num_ga_better", 0)
        tie = m.get("num_tie", 0)

        ax_diff.text(
            0.98,
            0.03,
            f"SAT: {sat_w}   GA: {ga_w}   Tie: {tie}",
            transform=ax_diff.transAxes,
            fontsize=8.5,
            ha="right",
            va="bottom",
            color=COLORS["text"],
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                edgecolor=COLORS["grid"],
                alpha=0.95,
            ),
        )

    fig.suptitle(
        "Per-Graph Performance Breakdown",
        fontsize=16,
        fontweight="bold",
        color=COLORS["text"],
    )

    output_path_png = os.path.join(output_dir, "detailed_comparison.png")
    output_path_pdf = os.path.join(output_dir, "detailed_comparison.pdf")
    plt.savefig(output_path_png, facecolor=COLORS["bg"], bbox_inches="tight", pad_inches=0.12)
    plt.savefig(output_path_pdf, facecolor=COLORS["bg"], bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print("  Saved: detailed_comparison.png")
    print("  Saved: detailed_comparison.pdf")


# =========================
# Report
# =========================

def generate_report(
    all_metrics: Dict[Tuple[str, str], Dict[str, Any]],
    output_dir: str
) -> str:
    """Generate text report."""
    report: List[str] = []
    report.append("=" * 80)
    report.append("       SAT vs GA Performance Analysis Report")
    report.append("=" * 80)

    categories = [
        ("small", "random", "Small-Scale Random"),
        ("small", "structured", "Small-Scale Structured"),
        ("medium", "random", "Medium-Scale Random"),
        ("medium", "structured", "Medium-Scale Structured"),
    ]

    for scale, gtype, label in categories:
        key = (scale, gtype)
        if key not in all_metrics:
            continue

        m = all_metrics[key]

        sat_wins = m.get("num_sat_better", 0)
        ga_wins = m.get("num_ga_better", 0)
        ties = m.get("num_tie", 0)

        if sat_wins > ga_wins:
            winner = "SAT"
        elif ga_wins > sat_wins:
            winner = "GA"
        else:
            winner = "Tie"

        report.append("")
        report.append("=" * 40)
        report.append(f"  {label.upper()}")
        report.append("=" * 40)
        report.append(f"  Total Nodes: {m.get('total_nodes', 0)}")
        report.append(f"  Edges: {m.get('edges', 0)}")
        report.append(f"  Graphs: {m.get('num_graphs', 0)}")
        report.append("")
        report.append("  Performance:")
        report.append(f"    SAT Wins: {sat_wins} | GA Wins: {ga_wins} | Ties: {ties}")
        report.append(f"    Winner: {winner}")
        report.append("")
        report.append("  Runtime:")
        report.append(f"    GA: {m.get('avg_bp_time', 0):.4f}s")
        report.append(f"    SAT: {m.get('avg_maxsat_time', 0) * 1000:.4f}ms")
        report.append("")
        report.append("  Objective:")
        report.append(f"    GA: {m.get('avg_ga_objective', 0):.4f}")
        report.append(f"    SAT: {m.get('avg_sat_objective', 0):.4f}")
        report.append(
            f"    Diff (SAT - GA): "
            f"{m.get('avg_sat_objective', 0) - m.get('avg_ga_objective', 0):.4f}"
        )

    report.append("")
    report.append("=" * 80)

    report_text = "\n".join(report)

    report_path = os.path.join(output_dir, "analysis_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    print("  Saved: analysis_report.txt")
    return report_text


# =========================
# CLI / main
# =========================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate SAT vs GA performance visualizations."
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default=str(Path(__file__).parent.parent),
        help="Directory containing input JSON files. Default: project root",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save outputs. Default: <base-dir>/evaluation",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    base_dir = Path(args.base_dir)
    output_dir = Path(args.output_dir) if args.output_dir else base_dir / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("SAT vs GA Visualization Analysis")
    print("=" * 50)
    print(f"Base directory:   {base_dir}")
    print(f"Output directory: {output_dir}")
    print("=" * 50)

    all_metrics: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for key, filename in DATA_FILES.items():
        filepath = base_dir / filename
        scale, gtype = key

        if not filepath.exists():
            print(f"Missing: {filename}")
            continue

        print(f"Loading: {filename}")
        data = load_json(str(filepath))
        metrics = extract_metrics(data)
        all_metrics[key] = metrics

        print(
            f"  -> {scale}/{gtype}: "
            f"{metrics['total_nodes']} nodes, "
            f"SAT: {metrics['num_sat_better']} "
            f"GA: {metrics['num_ga_better']} "
            f"Tie: {metrics['num_tie']}"
        )

    if not all_metrics:
        print("Error: no data files found.")
        return

    print("\nGenerating visualizations...")
    print("=" * 50)

    plot_scale_comparison(all_metrics, str(output_dir))
    plot_detailed_comparison(all_metrics, str(output_dir))
    generate_report(all_metrics, str(output_dir))

    print("\nDone.")
    print(f"All outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()