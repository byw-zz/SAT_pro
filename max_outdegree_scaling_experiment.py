#!/usr/bin/env python3
"""Compare MaxHS convergence under several graph out-degree limits."""

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from batch_run_and_analyze import (
    extract_stats_from_bn,
    generate_bn_and_values,
    generate_wcnf_and_save,
    parse_ub_from_log,
    run_maxhs,
    scale_params,
    ub_to_objective,
)
from replot_convergence import _clean_time_objective, _make_post_step_xy


DEFAULT_SCALES = [1.0, 1.2, 1.5, 2.0]
DEFAULT_MAX_CHILDREN = [5, 7, 10]
DEFAULT_TARGET_P_MEANS = [4.0, 5.6, 8.0]
COLORS = {5: "#4E79A7", 7: "#F28E2B", 10: "#59A14F"}


def p_outdegree_stats(bn):
    """Return descriptive statistics and histogram for P-node out-degrees."""
    out_degree = bn.get("out_degree", {})
    degrees = np.asarray([out_degree.get(p, 0) for p in bn.get("P", [])], dtype=int)
    counts = Counter(int(x) for x in degrees)
    if degrees.size == 0:
        return {
            "count": 0,
            "min": 0,
            "max": 0,
            "mean": 0.0,
            "std": 0.0,
            "median": 0.0,
            "p90": 0.0,
            "at_limit_count": 0,
            "at_limit_ratio": 0.0,
            "histogram": {},
        }

    limit = int(bn.get("meta", {}).get("max_children", max(degrees)))
    at_limit = int(np.count_nonzero(degrees == limit))
    return {
        "count": int(degrees.size),
        "min": int(np.min(degrees)),
        "max": int(np.max(degrees)),
        "mean": float(np.mean(degrees)),
        "std": float(np.std(degrees)),
        "median": float(np.median(degrees)),
        "p90": float(np.percentile(degrees, 90)),
        "at_limit_count": at_limit,
        "at_limit_ratio": float(at_limit / degrees.size),
        "histogram": {str(k): counts[k] for k in sorted(counts)},
    }


def _attach_experiment_meta(bn, max_children, target_p_mean):
    bn.setdefault("meta", {})["max_children"] = int(max_children)
    bn["meta"]["target_p_mean_outdegree"] = float(target_p_mean)


def run_case(scale, max_children, target_p_mean, seed, timeout, out_dir, reuse):
    params = scale_params(scale)
    params["max_children"] = max_children
    params["nE"] = int(round(params["nP"] * target_p_mean))
    mean_tag = str(target_p_mean).replace(".", "p")
    stem = f"graph_s{scale:.1f}_mc{max_children}_pm{mean_tag}"
    wcnf_path = out_dir / f"{stem}.wcnf"
    meta_path = out_dir / f"{stem}.meta.json"
    log_path = out_dir / f"{stem}.log"

    if reuse and wcnf_path.exists() and meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        bn = meta["bn"]
        values_table = meta["values_table"]
        print(f"  Reusing {wcnf_path}")
    else:
        bn, values_table = generate_bn_and_values(params, seed)
        _attach_experiment_meta(bn, max_children, target_p_mean)
        generate_wcnf_and_save(params, seed, wcnf_path, bn, values_table)

    stats = extract_stats_from_bn(bn, values_table)
    degree_stats = p_outdegree_stats(bn)

    log_text = run_maxhs(wcnf_path, timeout)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(log_text)

    ub_data = parse_ub_from_log(log_text)
    if ub_data["initial_ub"] is None:
        print("  WARNING: no convergence points parsed")
        objective = []
    else:
        objective = ub_to_objective(ub_data["ub"], stats["c_benefit_sum"])

    return {
        "scale": scale,
        "max_children": max_children,
        "target_p_mean_outdegree": target_p_mean,
        "params": params,
        "n_vars": sum(len(bn.get(t, [])) for t in ("P", "E", "C", "D")),
        "n_edges": len(bn.get("edges", [])),
        **stats,
        "time": ub_data["time"],
        "ub": ub_data["ub"],
        "objective": objective,
        "initial_ub": ub_data["initial_ub"],
        "final_ub": ub_data["final_ub"],
        "last_improvement_time": ub_data["total_time"],
        "n_points": ub_data["n_points"],
        "p_outdegree": degree_stats,
        "wcnf": str(wcnf_path),
        "log": str(log_path),
    }


def plot_convergence(results, scales, limits, timeout, out_dir):
    """Plot four scale panels with one staircase curve per out-degree limit."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True)
    axes = axes.flatten()
    x_max = float(timeout)

    for ax, scale in zip(axes, scales):
        scale_results = [
            r for r in results
            if np.isclose(r["scale"], scale) and r["max_children"] in limits
        ]
        for result in sorted(scale_results, key=lambda x: x["max_children"]):
            times = result["time"]
            objectives = result["objective"]
            if not times or not objectives:
                continue
            limit = result["max_children"]
            step_x, step_y = _make_post_step_xy(times, objectives, x_max=x_max)
            clean_t, clean_y = _clean_time_objective(times, objectives)
            color = COLORS.get(limit)
            ax.plot(
                step_x,
                step_y,
                linewidth=1.8,
                color=color,
                label=(rf"$d_{{max}}={limit}$, "
                       rf"$\bar{{d}}_P={result['target_p_mean_outdegree']:g}$"),
            )
            ax.plot(clean_t, clean_y, marker="o", linestyle="None", markersize=3, color=color)

        ax.set_title(f"scale={scale:.1f}", fontsize=11)
        ax.set_xlabel("Time (s)", fontsize=9)
        ax.set_ylabel(r"$\mathcal{F}(\mathbf{s})$", fontsize=9)
        ax.axhline(0, linewidth=0.8, color="black", alpha=0.55)
        ax.set_xlim(0, x_max)
        ax.grid(False)
        ax.legend(fontsize=9)

    fig.tight_layout()
    png_path = out_dir / "convergence_objective_p_outdegree.png"
    pdf_path = out_dir / "convergence_objective_p_outdegree.pdf"
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def plot_p_outdegree_distribution(results, scales, limits, out_dir):
    """Plot P-node out-degree proportions for every scale and limit."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()

    for ax, scale in zip(axes, scales):
        for limit in limits:
            result = next(
                (r for r in results if np.isclose(r["scale"], scale)
                 and r["max_children"] == limit),
                None,
            )
            if result is None:
                continue
            hist = result["p_outdegree"]["histogram"]
            x = np.asarray([int(k) for k in hist], dtype=int)
            y = np.asarray([hist[str(k)] for k in x], dtype=float)
            y /= y.sum()
            ax.plot(
                x,
                y,
                marker="o",
                linewidth=1.7,
                color=COLORS.get(limit),
                label=(rf"$d_{{max}}={limit}$, "
                       rf"$\bar{{d}}_P={result['target_p_mean_outdegree']:g}$"),
            )

        ax.set_title(f"scale={scale:.1f}", fontsize=11)
        ax.set_xlabel("P-node out-degree", fontsize=9)
        ax.set_ylabel("Proportion", fontsize=9)
        ax.set_xticks(range(1, max(limits) + 1))
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=9)

    fig.tight_layout()
    png_path = out_dir / "p_outdegree_distribution_by_target.png"
    pdf_path = out_dir / "p_outdegree_distribution_by_target.pdf"
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def main():
    parser = argparse.ArgumentParser(
        description="Run MaxHS scale experiments for several maximum out-degree limits."
    )
    parser.add_argument("--scales", type=float, nargs="+", default=DEFAULT_SCALES)
    parser.add_argument(
        "--max-children-values", type=int, nargs="+", default=DEFAULT_MAX_CHILDREN
    )
    parser.add_argument(
        "--target-p-mean-values", type=float, nargs="+", default=DEFAULT_TARGET_P_MEANS
    )
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="p_outdegree_scaling_results")
    parser.add_argument("--reuse", action="store_true")
    parser.add_argument(
        "--resume", action="store_true",
        help="Reuse completed cases from max_outdegree_summary.json",
    )
    args = parser.parse_args()

    if len(args.max_children_values) != len(args.target_p_mean_values):
        parser.error("--max-children-values and --target-p-mean-values must have equal lengths")
    cases = list(zip(args.max_children_values, args.target_p_mean_values))
    for max_children, target_p_mean in cases:
        if not 1.0 <= target_p_mean <= max_children:
            parser.error(
                f"target P mean {target_p_mean} must be in [1, {max_children}]"
            )

    out_dir = Path(__file__).parent / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = out_dir / "max_outdegree_summary.json"
    results = []
    if args.resume and summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as f:
            results = json.load(f).get("results", [])

    completed = {
        (
            round(float(r["scale"]), 10),
            int(r["max_children"]),
            round(float(r["target_p_mean_outdegree"]), 10),
        )
        for r in results
    }
    total = len(args.scales) * len(cases)
    case = 0
    for scale in args.scales:
        for max_children, target_p_mean in cases:
            case += 1
            key = (
                round(float(scale), 10),
                int(max_children),
                round(float(target_p_mean), 10),
            )
            if key in completed:
                print(
                    f"[{case}/{total}] scale={scale:.1f}, max_children={max_children}, "
                    f"target_P_mean={target_p_mean:g}: completed, skipping"
                )
                continue
            print(
                f"[{case}/{total}] scale={scale:.1f}, "
                f"max_children={max_children}, target_P_mean={target_p_mean:g}, "
                f"timeout={args.timeout}s"
            )
            result = run_case(
                scale,
                max_children,
                target_p_mean,
                args.seed,
                args.timeout,
                out_dir,
                args.reuse,
            )
            results.append(result)
            degree = result["p_outdegree"]
            print(
                f"  P out-degree: mean={degree['mean']:.3f}, std={degree['std']:.3f}, "
                f"max={degree['max']}, at-limit={degree['at_limit_ratio']:.2%}; "
                f"points={result['n_points']}"
            )

            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump({"results": results}, f, ensure_ascii=False, indent=2)

    convergence_paths = plot_convergence(
        results, args.scales, args.max_children_values, args.timeout, out_dir
    )
    distribution_paths = plot_p_outdegree_distribution(
        results, args.scales, args.max_children_values, out_dir
    )
    print(f"Convergence plots: {convergence_paths[0]}, {convergence_paths[1]}")
    print(f"P out-degree plots: {distribution_paths[0]}, {distribution_paths[1]}")


if __name__ == "__main__":
    main()
