#!/usr/bin/env python3
"""Run normal-P-out-degree experiments for d_max=7 and d_max=10."""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
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
from normal_outdegree_experiment import degree_stats
from replot_convergence import _clean_time_objective, _make_post_step_xy


DEFAULT_SCALES = [1.0, 1.2, 1.5, 2.0]
CASES = {
    7: {"mean": 4.0, "std": 1.2, "color": "#F28E2B"},
    10: {"mean": 5.5, "std": 1.8, "color": "#59A14F"},
}


def graph_params(scale, max_children):
    case = CASES[max_children]
    return {
        "nP": int(round(1000 * scale)),
        "nC": int(round(3000 * scale)),
        "nD": int(round(1500 * scale)),
        "max_children": max_children,
        "p_mean": case["mean"],
        "p_std": case["std"],
    }


def run_case(scale, max_children, seed, timeout, out_dir, reuse):
    params = graph_params(scale, max_children)
    stem = f"graph_s{scale:.1f}_dmax{max_children}_normal"
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
            max_children=max_children,
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

    utility = extract_stats_from_bn(bn, values_table)
    p_stats = degree_stats(bn)
    log_text = run_maxhs(wcnf_path, timeout)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(log_text)
    ub_data = parse_ub_from_log(log_text)
    objective = (
        ub_to_objective(ub_data["ub"], utility["c_benefit_sum"])
        if ub_data["initial_ub"] is not None
        else []
    )
    return {
        "scale": scale,
        "max_children": max_children,
        "params": params,
        "nP": len(bn["P"]),
        "nE": len(bn["E"]),
        "nC": len(bn["C"]),
        "nD": len(bn["D"]),
        "n_vars": sum(len(bn[t]) for t in ("P", "E", "C", "D")),
        "n_edges": len(bn["edges"]),
        "p_outdegree": p_stats,
        "target_distribution": bn["meta"]["p_outdegree"],
        **utility,
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


def plot_convergence(results, scales, timeout, out_dir):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True)
    for ax, scale in zip(axes.flatten(), scales):
        for result in sorted(
            (r for r in results if np.isclose(r["scale"], scale)),
            key=lambda r: r["max_children"],
        ):
            if not result["time"] or not result["objective"]:
                continue
            dmax = result["max_children"]
            x, y = _make_post_step_xy(
                result["time"], result["objective"], x_max=float(timeout)
            )
            clean_x, clean_y = _clean_time_objective(result["time"], result["objective"])
            label = (
                rf"$d_{{max}}={dmax}$, $\mu={result['params']['p_mean']:g}$, "
                rf"$\sigma={result['params']['p_std']:g}$"
            )
            color = CASES[dmax]["color"]
            ax.plot(x, y, color=color, linewidth=1.8, label=label)
            ax.plot(clean_x, clean_y, "o", color=color, markersize=3)
        ax.axhline(0, color="black", linewidth=0.8, alpha=0.55)
        ax.set_xlim(0, timeout)
        ax.set_title(f"scale={scale:.1f}", fontsize=11)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(r"$\mathcal{F}(\mathbf{s})$")
        ax.legend(fontsize=8)
        ax.grid(False)
    fig.tight_layout()
    paths = (
        out_dir / "convergence_objective_dmax7_10_normal.png",
        out_dir / "convergence_objective_dmax7_10_normal.pdf",
    )
    fig.savefig(paths[0], dpi=180, bbox_inches="tight")
    fig.savefig(paths[1], bbox_inches="tight")
    plt.close(fig)
    return paths


def plot_distributions(results, scales, out_dir):
    fig, axes = plt.subplots(2, len(scales), figsize=(18, 8), sharey="row")
    for row, dmax in enumerate(sorted(CASES)):
        for col, scale in enumerate(scales):
            ax = axes[row, col]
            result = next(
                r for r in results
                if r["max_children"] == dmax and np.isclose(r["scale"], scale)
            )
            hist = result["p_outdegree"]["histogram"]
            n = result["p_outdegree"]["count"]
            x = np.arange(1, dmax + 1, dtype=int)
            observed = np.asarray([hist.get(str(k), 0) / n for k in x])
            target = np.asarray(result["target_distribution"]["probabilities"])
            ax.bar(x, observed, width=0.68, color="#A0CBE8", label="Observed")
            ax.plot(x, target, "o-", color="#E15759", linewidth=1.5, label="Target normal")
            stats = result["p_outdegree"]
            ax.set_title(
                f"dmax={dmax}, scale={scale:.1f}\n"
                f"mean={stats['mean']:.2f}, std={stats['std']:.2f}, "
                f"skew={stats['skewness']:.3f}",
                fontsize=9,
            )
            ax.set_xlabel("P-node out-degree")
            if col == 0:
                ax.set_ylabel("Proportion")
            ax.set_xticks(x)
            ax.grid(axis="y", alpha=0.25)
            ax.legend(fontsize=7)
    fig.tight_layout()
    paths = (
        out_dir / "p_outdegree_distribution_dmax7_10_normal.png",
        out_dir / "p_outdegree_distribution_dmax7_10_normal.pdf",
    )
    fig.savefig(paths[0], dpi=180, bbox_inches="tight")
    fig.savefig(paths[1], bbox_inches="tight")
    plt.close(fig)
    return paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scales", type=float, nargs="+", default=DEFAULT_SCALES)
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="normal_outdegree_dmax7_10_results")
    parser.add_argument("--jobs", type=int, default=1)
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
    completed = {
        (round(float(r["scale"]), 10), int(r["max_children"])) for r in results
    }

    pending = []
    total = len(args.scales) * len(CASES)
    index = 0
    for scale in args.scales:
        for dmax in sorted(CASES):
            index += 1
            key = (round(float(scale), 10), dmax)
            if key in completed:
                print(f"[{index}/{total}] scale={scale:.1f}, dmax={dmax}: completed, skipping")
                continue
            case = CASES[dmax]
            print(
                f"[{index}/{total}] queued scale={scale:.1f}, dmax={dmax}, "
                f"mean={case['mean']}, std={case['std']}, timeout={args.timeout}s"
            )
            pending.append((index, scale, dmax))

    def save_result(index, result):
        nonlocal results
        results.append(result)
        results.sort(key=lambda r: (float(r["scale"]), int(r["max_children"])))
        stats = result["p_outdegree"]
        print(
            f"[{index}/{total}] completed scale={result['scale']:.1f}, "
            f"dmax={result['max_children']}: mean={stats['mean']:.3f}, "
            f"std={stats['std']:.3f}, skew={stats['skewness']:.4f}; "
            f"points={result['n_points']}",
            flush=True,
        )
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump({"results": results}, f, ensure_ascii=False, indent=2)

    if args.jobs <= 1:
        for index, scale, dmax in pending:
            result = run_case(scale, dmax, args.seed, args.timeout, out_dir, args.reuse)
            save_result(index, result)
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as executor:
            futures = {
                executor.submit(
                    run_case,
                    scale,
                    dmax,
                    args.seed,
                    args.timeout,
                    out_dir,
                    args.reuse,
                ): index
                for index, scale, dmax in pending
            }
            for future in as_completed(futures):
                index = futures[future]
                result = future.result()
                save_result(index, result)

    convergence = plot_convergence(results, args.scales, args.timeout, out_dir)
    distributions = plot_distributions(results, args.scales, out_dir)
    print(f"Convergence plots: {convergence[0]}, {convergence[1]}")
    print(f"Distribution plots: {distributions[0]}, {distributions[1]}")


if __name__ == "__main__":
    main()
