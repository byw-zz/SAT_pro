#!/usr/bin/env python3
"""Run the isolated d_max=5 normal-P-out-degree convergence experiment."""

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
    generate_wcnf_and_save,
    parse_ub_from_log,
    run_maxhs,
    ub_to_objective,
)
from generate_graph.config import FIXED_E_PROBS
from generate_graph.normal_outdegree_random_graph import generate_bn_normal_p_outdegree
from generate_graph.numerical_generation import generate_node_values
from replot_convergence import _clean_time_objective, _make_post_step_xy


DEFAULT_SCALES = [1.0, 1.2, 1.5, 2.0]


def scale_params(scale):
    return {
        "nP": int(round(1000 * scale)),
        "nC": int(round(3000 * scale)),
        "nD": int(round(1500 * scale)),
        "max_children": 5,
        "p_mean": 3.0,
        "p_std": 0.8,
    }


def degree_stats(bn):
    degrees = np.asarray(
        [bn.get("out_degree", {}).get(p, 0) for p in bn.get("P", [])],
        dtype=float,
    )
    counts = Counter(int(d) for d in degrees)
    std = float(np.std(degrees))
    centered = degrees - float(np.mean(degrees))
    skewness = float(np.mean(centered ** 3) / std ** 3) if std > 0 else 0.0
    return {
        "count": int(degrees.size),
        "mean": float(np.mean(degrees)),
        "std": std,
        "median": float(np.median(degrees)),
        "min": int(np.min(degrees)),
        "max": int(np.max(degrees)),
        "skewness": skewness,
        "histogram": {str(k): counts[k] for k in sorted(counts)},
    }


def run_case(scale, seed, timeout, out_dir, reuse):
    params = scale_params(scale)
    stem = f"graph_s{scale:.1f}_dmax5_normal"
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
        bn = generate_bn_normal_p_outdegree(
            nP=params["nP"],
            nC=params["nC"],
            nD=params["nD"],
            max_children=params["max_children"],
            p_mean=params["p_mean"],
            p_std=params["p_std"],
            p_EP=0.25,
            seed=seed,
        )
        values_table = generate_node_values(
            bn,
            seed=seed,
            fixed_e_probs=FIXED_E_PROBS,
            p_loss_range=(50, 500),
            c_benefit_range=(10, 50),
            d_cost_range=(50, 100),
        )
        generate_wcnf_and_save(params, seed, wcnf_path, bn, values_table)

    utility_stats = extract_stats_from_bn(bn, values_table)
    p_stats = degree_stats(bn)
    log_text = run_maxhs(wcnf_path, timeout)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(log_text)

    ub_data = parse_ub_from_log(log_text)
    objective = (
        ub_to_objective(ub_data["ub"], utility_stats["c_benefit_sum"])
        if ub_data["initial_ub"] is not None
        else []
    )
    return {
        "scale": scale,
        "params": params,
        "nP": len(bn["P"]),
        "nE": len(bn["E"]),
        "nC": len(bn["C"]),
        "nD": len(bn["D"]),
        "n_vars": sum(len(bn[t]) for t in ("P", "E", "C", "D")),
        "n_edges": len(bn["edges"]),
        "p_outdegree": p_stats,
        "target_distribution": bn["meta"]["p_outdegree"],
        **utility_stats,
        "time": ub_data["time"],
        "ub": ub_data["ub"],
        "objective": objective,
        "initial_ub": ub_data["initial_ub"],
        "final_ub": ub_data["final_ub"],
        "last_improvement_time": ub_data["total_time"],
        "n_points": ub_data["n_points"],
        "wcnf": str(wcnf_path),
        "log": str(log_path),
    }


def plot_convergence(results, timeout, out_dir):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True)
    for ax, result in zip(axes.flatten(), sorted(results, key=lambda r: r["scale"])):
        if result["time"] and result["objective"]:
            x, y = _make_post_step_xy(
                result["time"], result["objective"], x_max=float(timeout)
            )
            clean_x, clean_y = _clean_time_objective(result["time"], result["objective"])
            ax.plot(x, y, color="#4E79A7", linewidth=1.8)
            ax.plot(clean_x, clean_y, "o", color="#4E79A7", markersize=3)
        ax.axhline(0, color="black", linewidth=0.8, alpha=0.55)
        ax.set_xlim(0, timeout)
        ax.set_title(
            f"scale={result['scale']:.1f}, variables={result['n_vars']}",
            fontsize=11,
        )
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(r"$\mathcal{F}(\mathbf{s})$")
        ax.grid(False)
    fig.tight_layout()
    paths = (
        out_dir / "convergence_objective_dmax5_normal.png",
        out_dir / "convergence_objective_dmax5_normal.pdf",
    )
    fig.savefig(paths[0], dpi=180, bbox_inches="tight")
    fig.savefig(paths[1], bbox_inches="tight")
    plt.close(fig)
    return paths


def plot_degree_distribution(results, out_dir):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True, sharey=True)
    for ax, result in zip(axes.flatten(), sorted(results, key=lambda r: r["scale"])):
        hist = result["p_outdegree"]["histogram"]
        n = result["p_outdegree"]["count"]
        x = np.arange(1, 6, dtype=int)
        observed = np.asarray([hist.get(str(k), 0) / n for k in x])
        target = np.asarray(result["target_distribution"]["probabilities"])
        ax.bar(x, observed, width=0.65, color="#A0CBE8", label="Observed")
        ax.plot(x, target, "o-", color="#E15759", linewidth=1.6, label="Target normal")
        stats = result["p_outdegree"]
        ax.set_title(
            f"scale={result['scale']:.1f}, mean={stats['mean']:.2f}, "
            f"std={stats['std']:.2f}, skew={stats['skewness']:.3f}",
            fontsize=10,
        )
        ax.set_xlabel("P-node out-degree")
        ax.set_ylabel("Proportion")
        ax.set_xticks(range(1, 6))
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=8)
    fig.tight_layout()
    paths = (
        out_dir / "p_outdegree_distribution_dmax5_normal.png",
        out_dir / "p_outdegree_distribution_dmax5_normal.pdf",
    )
    fig.savefig(paths[0], dpi=180, bbox_inches="tight")
    fig.savefig(paths[1], bbox_inches="tight")
    plt.close(fig)
    return paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scales", type=float, nargs="+", default=DEFAULT_SCALES)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--output-dir", default="normal_outdegree_dmax5_results")
    parser.add_argument("--reuse", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    out_dir = Path(__file__).parent / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "normal_outdegree_summary.json"
    results = []
    if args.resume and summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as f:
            results = json.load(f).get("results", [])
    completed = {round(float(r["scale"]), 10) for r in results}

    for index, scale in enumerate(args.scales, start=1):
        if round(float(scale), 10) in completed:
            print(f"[{index}/{len(args.scales)}] scale={scale:.1f}: completed, skipping")
            continue
        print(f"[{index}/{len(args.scales)}] scale={scale:.1f}, d_max=5, mean=3, std=0.8")
        result = run_case(scale, args.seed, args.timeout, out_dir, args.reuse)
        results.append(result)
        stats = result["p_outdegree"]
        print(
            f"  P distribution: mean={stats['mean']:.3f}, std={stats['std']:.3f}, "
            f"skew={stats['skewness']:.4f}; convergence points={result['n_points']}"
        )
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump({"results": results}, f, ensure_ascii=False, indent=2)

    convergence = plot_convergence(results, args.timeout, out_dir)
    distribution = plot_degree_distribution(results, out_dir)
    print(f"Convergence plots: {convergence[0]}, {convergence[1]}")
    print(f"Distribution plots: {distribution[0]}, {distribution[1]}")


if __name__ == "__main__":
    main()
