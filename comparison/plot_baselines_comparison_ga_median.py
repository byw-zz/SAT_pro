#!/usr/bin/env python3
"""Four-way baseline comparison using the median of five GA runs per graph.

The expensive GA runs are not repeated.  This analysis reads the raw ``runs``
stored by ``rerun_ga_singlepoint_5runs.py`` and uses, for each graph:

* quality: the median of the five independently seeded GA objectives;
* time: the median of the five individual GA running times.

The other methods and evaluators are identical to ``plot_baselines_comparison``.
Small graphs use VE and medium graphs use BP in the stored comparison datasets.
Besides the figure, a JSON file records the per-dataset values and all pairwise
win/tie/loss counts used by the analysis.
"""

import json
import os
from pathlib import Path
from statistics import median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parent.parent
EXPECTED_GA_RUNS = 5
TIE_ABS_TOL = 1e-9

# label, Khouzani results, Zenitani results, stored MaxSAT dataset, five-run GA data
DSETS = [
    (
        "structured\nsmall",
        "comparison/khouzani_out_s_small.json",
        "comparison/structured_12_17_100graphs_zenitani.json",
        "structured_12_17_100graphs.json",
        "rerun_ga_singlepoint_5runs/structured_12_17_100graphs_ga_singlepoint_5runs.json",
    ),
    (
        "random\nsmall",
        "comparison/khouzani_out_r_small.json",
        "comparison/random_10_100graphs_zenitani.json",
        "random_10_100graphs.json",
        "rerun_ga_singlepoint_5runs/random_10_100graphs_ga_singlepoint_5runs.json",
    ),
    (
        "structured\nmedium",
        "comparison/khouzani_out_s_large.json",
        "comparison/structured_151_199_10graphs_zenitani.json",
        "structured_151_199_10graphs.json",
        "rerun_ga_singlepoint_5runs/structured_151_199_10graphs_ga_singlepoint_5runs.json",
    ),
    (
        "random\nmedium",
        "comparison/khouzani_out_r_large.json",
        "comparison/random_100_10graphs_zenitani.json",
        "random_100_10graphs.json",
        "rerun_ga_singlepoint_5runs/random_100_10graphs_ga_singlepoint_5runs.json",
    ),
]

INK = "#0b0b0b"
MUTED = "#898781"
SURF = "#fcfcfb"
TEAL = "#299d8f"
YELLOW = "#e9c46a"
CORAL = "#d87659"
VIOLET = "#8a6bbf"
WIN = TEAL
TIE = YELLOW
LOSS = CORAL
C_SAT = TEAL
C_GA = CORAL
C_KH = YELLOW
C_ZEN = VIOLET


def _read_json(relative_path):
    with open(ROOT / relative_path, encoding="utf-8") as handle:
        return json.load(handle)


def _wtl(rows, akey, bkey, tol_abs=TIE_ABS_TOL):
    """Return method A win/tie/loss counts; larger objective is better."""
    wins = ties = losses = 0
    for row in rows:
        a = row.get(akey)
        b = row.get(bkey)
        if a is None or b is None:
            continue
        if a > b + tol_abs:
            wins += 1
        elif a < b - tol_abs:
            losses += 1
        else:
            ties += 1
    return [wins, ties, losses]


def _median_ga_row(row):
    runs = row.get("runs", [])
    if len(runs) != EXPECTED_GA_RUNS:
        raise ValueError(
            f"graph {row.get('graph_id')} has {len(runs)} GA runs; "
            f"expected {EXPECTED_GA_RUNS}"
        )
    objectives = [run["objective"] for run in runs]
    times = [run["ga_time_s"] for run in runs]
    return {
        "graph_id": row["graph_id"],
        "ga_obj": median(objectives),
        "ga_time_s": median(times),
        "ga_objectives": objectives,
        "ga_times_s": times,
    }


def _mean(values):
    return sum(values) / len(values) if values else None


def load():
    datasets = []
    for label, kh_path, zen_path, stored_path, ga_path in DSETS:
        stored = _read_json(stored_path)
        kh_rows = _read_json(kh_path)["rows"]
        zen_rows = _read_json(zen_path)["rows"]
        ga_data = _read_json(ga_path)
        ga_rows = [_median_ga_row(row) for row in ga_data["results"]]

        stored_by_id = {row["graph_id"]: row for row in stored["results"]}
        kh_by_id = {row["graph_id"]: row for row in kh_rows}
        zen_by_id = {row["graph_id"]: row for row in zen_rows}
        ga_by_id = {row["graph_id"]: row for row in ga_rows}
        graph_ids = set(stored_by_id)
        for method, rows_by_id in (
            ("Khouzani", kh_by_id),
            ("Zenitani", zen_by_id),
            ("GA", ga_by_id),
        ):
            if set(rows_by_id) != graph_ids:
                missing = sorted(graph_ids - set(rows_by_id))
                extra = sorted(set(rows_by_id) - graph_ids)
                raise ValueError(
                    f"{label.replace(os.linesep, ' ')} {method} graph IDs mismatch: "
                    f"missing={missing}, extra={extra}"
                )

        rows = []
        for graph_id in sorted(graph_ids):
            stored_row = stored_by_id[graph_id]
            ga_row = ga_by_id[graph_id]
            kh_row = kh_by_id[graph_id]
            zen_row = zen_by_id[graph_id]
            rows.append(
                {
                    "graph_id": graph_id,
                    "maxsat_obj": stored_row.get("maxsat_objective"),
                    "ga_obj": ga_row["ga_obj"],
                    "khouzani_obj": kh_row.get("khouzani_obj"),
                    "zenitani_obj": zen_row.get("zenitani_obj"),
                    "ga_time_s": ga_row["ga_time_s"],
                    "ga_objectives": ga_row["ga_objectives"],
                    "ga_times_s": ga_row["ga_times_s"],
                }
            )

        summary = stored["summary"]
        pairwise = {
            "ga_vs_maxsat": _wtl(rows, "ga_obj", "maxsat_obj"),
            "khouzani_vs_maxsat": _wtl(rows, "khouzani_obj", "maxsat_obj"),
            "zenitani_vs_maxsat": _wtl(rows, "zenitani_obj", "maxsat_obj"),
            "ga_vs_khouzani": _wtl(rows, "ga_obj", "khouzani_obj"),
            "ga_vs_zenitani": _wtl(rows, "ga_obj", "zenitani_obj"),
            "khouzani_vs_zenitani": _wtl(
                rows, "khouzani_obj", "zenitani_obj"
            ),
        }
        datasets.append(
            {
                "label": label,
                "n": len(rows),
                "pairwise_wtl": pairwise,
                "mean_time_s": {
                    "maxsat": summary.get("avg_maxsat_time_s"),
                    "ga_median_run": _mean([row["ga_time_s"] for row in rows]),
                    "khouzani": _mean(
                        [row["khouzani_milp_ms"] / 1000.0 for row in kh_rows]
                    ),
                    "zenitani": _mean(
                        [row["zenitani_wall_ms"] / 1000.0 for row in zen_rows]
                    ),
                },
                "rows": rows,
            }
        )
    return datasets


def _percentages(counts):
    total = sum(counts) or 1
    return [100.0 * value / total for value in counts]


def plot(data):
    labels = [dataset["label"] for dataset in data]
    x_positions = list(range(len(data)))
    fig, (quality_ax, time_ax) = plt.subplots(1, 2, figsize=(13.2, 4.8))
    fig.patch.set_facecolor(SURF)

    quality_ax.set_facecolor(SURF)
    bar_width = 0.26
    columns = [
        (-bar_width - 0.02, "GA", "ga_vs_maxsat"),
        (0.0, "Kh", "khouzani_vs_maxsat"),
        (bar_width + 0.02, "Zen", "zenitani_vs_maxsat"),
    ]
    for offset, method, key in columns:
        for index, dataset in enumerate(data):
            win, tie, loss = _percentages(dataset["pairwise_wtl"][key])
            x_value = x_positions[index] + offset
            quality_ax.bar(
                x_value, win, bar_width, color=WIN, edgecolor=SURF, linewidth=1.1
            )
            quality_ax.bar(
                x_value,
                tie,
                bar_width,
                bottom=win,
                color=TIE,
                edgecolor=SURF,
                linewidth=1.1,
            )
            quality_ax.bar(
                x_value,
                loss,
                bar_width,
                bottom=win + tie,
                color=LOSS,
                edgecolor=SURF,
                linewidth=1.1,
            )
            quality_ax.text(
                x_value,
                -4,
                method,
                ha="center",
                va="top",
                fontsize=7.5,
                color=MUTED,
            )
    quality_ax.set_ylim(0, 100)
    quality_ax.set_xticks(x_positions)
    quality_ax.set_xticklabels(labels, fontsize=9, color=INK)
    quality_ax.tick_params(axis="x", length=0, pad=22)
    quality_ax.set_ylabel("percentage (%)", fontsize=10, color=INK)
    quality_ax.set_title(
        "Baseline vs. exact MaxSAT (GA: median of five runs)",
        fontsize=11,
        color=INK,
        pad=8,
    )
    for spine in ("top", "right", "left"):
        quality_ax.spines[spine].set_visible(False)
    quality_ax.spines["bottom"].set_color(MUTED)
    quality_ax.tick_params(colors=MUTED)
    quality_ax.legend(
        handles=[
            Patch(facecolor=WIN, label="baseline wins"),
            Patch(facecolor=TIE, label="objective tie"),
            Patch(facecolor=LOSS, label="MaxSAT wins"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.06),
        ncol=3,
        frameon=False,
        fontsize=8.5,
        handlelength=1.1,
    )

    time_ax.set_facecolor(SURF)
    group_width = 0.2
    series = [
        ("MaxSAT", C_SAT, "maxsat"),
        ("GA-BP median", C_GA, "ga_median_run"),
        ("Khouzani", C_KH, "khouzani"),
        ("Zenitani", C_ZEN, "zenitani"),
    ]
    for series_index, (name, color, key) in enumerate(series):
        xs = [
            position + (series_index - 1.5) * group_width
            for position in x_positions
        ]
        ys = [max(dataset["mean_time_s"][key], 1e-3) for dataset in data]
        time_ax.bar(
            xs,
            ys,
            group_width,
            color=color,
            edgecolor=SURF,
            linewidth=1.0,
            label=name,
        )
    time_ax.set_yscale("log")
    time_ax.set_ylim(1e-3, 3e4)
    time_ax.set_xticks(x_positions)
    time_ax.set_xticklabels(labels, fontsize=9, color=INK)
    time_ax.set_ylabel("mean solve time (s, log)", fontsize=10, color=INK)
    time_ax.set_title("Solve time per method", fontsize=11, color=INK, pad=8)
    for spine in ("top", "right", "left"):
        time_ax.spines[spine].set_visible(False)
    time_ax.spines["bottom"].set_color(MUTED)
    time_ax.tick_params(colors=MUTED, length=0)
    time_ax.legend(
        loc="upper left", frameon=False, fontsize=8.5, handlelength=1.1
    )

    fig.tight_layout()
    png_path = ROOT / "comparison" / "baselines_comparison_ga_median.png"
    pdf_path = ROOT / "comparison" / "baselines_comparison_ga_median.pdf"
    fig.savefig(png_path, dpi=170, bbox_inches="tight", facecolor=SURF)
    fig.savefig(pdf_path, bbox_inches="tight", facecolor=SURF)
    plt.close(fig)
    return png_path, pdf_path


def save_analysis(data):
    output_path = ROOT / "comparison" / "baselines_comparison_ga_median.json"
    payload = {
        "ga_aggregation": {
            "runs_per_graph": EXPECTED_GA_RUNS,
            "quality": "median objective across five independent runs per graph",
            "time": "median single-run time per graph, then mean across graphs",
        },
        "comparison": {
            "tie_rule": "absolute final-objective difference <= 1e-9",
            "tie_abs": TIE_ABS_TOL,
            "tie_rel": 0.0,
        },
        "wtl_order": ["first_method_win", "tie", "first_method_loss"],
        "datasets": data,
    }
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return output_path


def main():
    data = load()
    analysis_path = save_analysis(data)
    png_path, pdf_path = plot(data)
    print(f"saved: {analysis_path}")
    print(f"saved: {png_path}")
    print(f"saved: {pdf_path}")
    for dataset in data:
        label = dataset["label"].replace("\n", " ")
        print(
            f"{label:<20} n={dataset['n']:>3} "
            f"GA-vs-MaxSAT={dataset['pairwise_wtl']['ga_vs_maxsat']} "
            f"GA-time={dataset['mean_time_s']['ga_median_run']:.3f}s"
        )


if __name__ == "__main__":
    main()
