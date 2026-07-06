#!/usr/bin/env python3
"""Batch generate WCNF files at different scales and run MaxHS to collect convergence data (objective version)"""

import argparse
import re
import subprocess
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

MAXHS_BIN = "MaxHS/build/release/bin/maxhs"
from generate_graph.config import FIXED_E_PROBS

SCALES = [0.1, 0.2, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0]


def scale_params(scale):
    """Scale parameters proportionally, maintaining constraint: nC <= 3*nD, nE >= nP"""
    nP = max(5, int(1000 * scale))
    nE = max(nP, int(2400 * scale))
    nD = max(0, int(1500 * scale))
    nC = min(max(5, int(3000 * scale)), 3 * nD)
    mc = max(3, int(5 * scale))
    return {"nP": nP, "nE": nE, "nC": nC, "nD": nD, "max_children": mc}


def generate_bn_and_values(params, seed):
    """Generate BN graph and node values table"""
    from generate_graph.random_graph import generate_bn_dag_multi_pe
    from generate_graph.numerical_generation import generate_node_values

    bn = generate_bn_dag_multi_pe(
        nP=params["nP"],
        nE=params["nE"],
        nC=params["nC"],
        nD=params["nD"],
        max_children=params["max_children"],
        p_EP=0.25,
        seed=seed,
    )

    values_table = generate_node_values(
        bn,
        seed=seed,
        fixed_e_probs=FIXED_E_PROBS,
        p_loss_range=(50, 500),
        p_benefit_range=(5, 80),
        c_benefit_range=(10, 50),
        d_cost_range=(50, 100),
    )
    return bn, values_table


def extract_stats_from_bn(bn, values_table):
    """Extract C_benefit_sum, P_loss_sum, baseline_objective from bn and values_table.

    P_loss_sum = sum_p ( P_comp_max(p) * P_loss(p) )
    where P_comp_max = 1 - prod(1 - E_prob_i) (noisy-OR, all child E active)
    baseline_objective = C_benefit_sum - P_loss_sum
    """
    from collections import defaultdict

    vindex = {row["node"]: row for row in values_table}
    node_type = bn["node_type"]

    parents = defaultdict(list)
    children = defaultdict(list)
    for u, v in bn["edges"]:
        parents[v].append(u)
        children[u].append(v)

    c_benefit_sum = sum(
        vindex.get(c, {}).get("C_benefit", 0.0)
        for c in bn.get("C", [])
    )

    p_loss_sum = 0.0
    for p in bn.get("P", []):
        p_loss = vindex.get(p, {}).get("P_loss", 0.0)
        e_children = [e for e in children.get(p, []) if node_type.get(e) == "E"]
        if not e_children:
            p_comp_max = 0.0
        else:
            prob_not_comp = 1.0
            for e in e_children:
                prob_e = vindex.get(e, {}).get("E_prob", 0.0)
                prob_not_comp *= (1.0 - prob_e)
            p_comp_max = 1.0 - prob_not_comp
        p_loss_sum += p_comp_max * p_loss

    baseline_objective = c_benefit_sum - p_loss_sum

    return {
        "c_benefit_sum": c_benefit_sum,
        "p_loss_sum": p_loss_sum,
        "baseline_objective": baseline_objective,
    }


def generate_wcnf_and_save(params, seed, wcnf_path, bn, values_table):
    """Convert bn to CNF and export WCNF, also save bn + values_table to JSON"""
    from graph2sat.graph2sat import bn_to_maxsat_cnf, export_to_wcnf

    cnf_data = bn_to_maxsat_cnf(bn, values_table, initial_true_nodes=None, initial_false_nodes=None)
    export_to_wcnf(cnf_data, str(wcnf_path))

    meta_path = wcnf_path.with_suffix(".meta.json")
    with open(meta_path, "w") as f:
        json.dump({
            "bn": bn,
            "values_table": values_table,
        }, f, ensure_ascii=False, indent=2)


def run_maxhs(wcnf_path, timeout):
    """Run MaxHS and return log"""
    cmd = [MAXHS_BIN, "-cpu-lim=" + str(timeout), "-verb=2",
           "-printSoln", "-printBstSoln", str(wcnf_path)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout + 10)
        return result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return ""


def parse_ub_from_log(log_text):
    """Extract UB convergence data from MaxHS log"""
    ub_history = []
    time_history = []
    initial_ub = None
    current_time = 0.0

    for line in log_text.splitlines():
        line = line.strip()
        if not line.startswith("c "):
            continue
        content = line[2:]

        tm = re.search(r"Elapsed time ([\d.]+)", content)
        if tm:
            current_time = float(tm.group(1))

        init = re.search(r"Init Bnds: LB =([\d.]+) UB = ([\d.]+)", content)
        if init:
            initial_ub = float(init.group(2))
            current_time = 0.0
            time_history.append(current_time)
            ub_history.append(initial_ub)
            continue

        cplex = re.search(r"after CPLEX bnds: LB =([\d.]+) UB = ([\d.]+) GAP", content)
        if cplex:
            ub = float(cplex.group(2))
            if initial_ub and ub < ub_history[-1]:
                time_history.append(current_time)
                ub_history.append(ub)
            continue

        sat = re.search(r"(?:New SAT solution found \([^)]+\), cost|New UB found) = ([\d.]+)", content)
        if sat:
            cost = float(sat.group(1))
            if initial_ub and cost < initial_ub and cost < ub_history[-1]:
                time_history.append(current_time)
                ub_history.append(cost)

    return {
        "time": time_history,
        "ub": ub_history,
        "initial_ub": initial_ub,
        "final_ub": ub_history[-1] if ub_history else None,
        "total_time": time_history[-1] if time_history else 0.0,
        "n_points": len(time_history),
    }


def load_stats_from_meta(wcnf_path):
    """Load C_benefit_sum, P_loss_sum, baseline_objective from .meta.json"""
    meta_path = wcnf_path.with_suffix(".meta.json")
    with open(meta_path, "r") as f:
        meta = json.load(f)
    return extract_stats_from_bn(meta["bn"], meta["values_table"])


def ub_to_objective(ub_history, c_benefit_sum):
    """Convert UB list to objective value list: objective = C_benefit_sum - UB / 100"""
    return [c_benefit_sum - ub / 100.0 for ub in ub_history]


def _ensure_matplotlib_font():
    """Try to set matplotlib font to avoid Chinese character issues"""
    try:
        font_list = matplotlib.font_manager.findSystemFonts()
        if font_list:
            matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans"]
            matplotlib.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass


def plot_convergence(results, out_dir):
    """Plot convergence curves for all scales (objective values)"""
    _ensure_matplotlib_font()

    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    axes = axes.flatten()

    for idx, scale in enumerate(SCALES):
        ax = axes[idx]
        data = results.get(scale)
        if data is None:
            ax.set_title(f"scale={scale}\n(no data)", fontsize=10)
            ax.set_xlabel("Time (s)", fontsize=8)
            ax.set_ylabel(r"$\mathcal{F}(s)$", fontsize=8)
            ax.grid(True, alpha=0.3)
            continue

        time_hist = data.get("time", [])
        obj_hist = data.get("objective", [])

        if not time_hist or not obj_hist:
            ax.set_title(f"scale={scale}\n(no convergence data)", fontsize=10)
            ax.grid(True, alpha=0.3)
            continue

        baseline = data.get("baseline_objective")
        c_benefit_sum = data.get("c_benefit_sum", 0)
        p_loss_sum = data.get("p_loss_sum", 0)

        ax.plot(time_hist, obj_hist, marker="o", markersize=3,
                linewidth=1.5, color="#2196F3", label=r"$\mathcal{F}(s)$")

        if baseline is not None:
            ax.axhline(y=baseline, color="#F44336", linestyle="--",
                       linewidth=1.2, label=f"Baseline ({baseline:.1f})")

        ax.axhline(y=c_benefit_sum, color="#4CAF50", linestyle=":",
                   linewidth=1.0, label=f"C_benefit_sum ({c_benefit_sum:.1f})")

        ax.set_title(f"scale={scale}  (P_loss_sum={p_loss_sum:.1f})", fontsize=9)
        ax.set_xlabel("Time (s)", fontsize=7)
        ax.set_ylabel(r"$\mathcal{F}(s)$", fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)
        if idx == 0:
            ax.legend(fontsize=6, loc="best")

    fig.suptitle(r"MaxHS Convergence: $\mathcal{F}(s)$ vs Time  "
                 r"($ \mathcal{F}(s)$ = C_benefit_sum - UB/100, Baseline = C_benefit_sum - P_loss_sum)",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    out_path = out_dir / "convergence_objective.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Convergence curve saved: {out_path}")


def plot_comparison_bar(results, out_dir):
    """Plot comparison bar chart for C_benefit_sum, P_loss_sum, baseline_objective across all scales"""
    _ensure_matplotlib_font()

    c_vals = []
    p_vals = []
    bl_vals = []
    labels = []

    for scale in SCALES:
        data = results.get(scale)
        if data is None:
            c_vals.append(0)
            p_vals.append(0)
            bl_vals.append(0)
        else:
            c_vals.append(data.get("c_benefit_sum", 0))
            p_vals.append(data.get("p_loss_sum", 0))
            bl_vals.append(data.get("baseline_objective", 0))
        labels.append(f"s={scale}")

    x = range(len(SCALES))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 6))
    bars1 = ax.bar([i - width for i in x], c_vals, width, label="C_benefit_sum", color="#2196F3", alpha=0.85)
    bars2 = ax.bar(x, p_vals, width, label="P_loss_sum", color="#FF5722", alpha=0.85)
    bars3 = ax.bar([i + width for i in x], bl_vals, width, label="baseline_objective", color="#4CAF50", alpha=0.85)

    for bar in bars1 + bars2 + bars3:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                    f"{h:.0f}", ha="center", va="bottom", fontsize=6)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_xlabel("Scale", fontsize=10)
    ax.set_ylabel("Value", fontsize=10)
    ax.set_title("C_benefit_sum vs P_loss_sum vs baseline_objective", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    out_path = out_dir / "comparison_bar.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Comparison bar chart saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Batch generate WCNF and run MaxHS (objective convergence version)")
    parser.add_argument("--timeout", type=int, default=60, help="Per-run timeout in seconds")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", default="scaling_results", help="Results directory")
    args = parser.parse_args()

    base_dir = Path(__file__).parent
    out_dir = base_dir / args.output_dir
    out_dir.mkdir(exist_ok=True)

    all_results = {}

    header = (f"{'Scale':>8} | {'nP':>6} | {'nE':>6} | {'nC':>6} | {'nD':>6} | "
              f"{'C_ben_sum':>12} | {'P_loss_sum':>12} | {'BaselineObj':>12} | "
              f"{'Time(s)':>8} | {'Points':>6}")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for scale in SCALES:
        params = scale_params(scale)
        wcnf_file = out_dir / f"graph_s{scale}.wcnf"
        log_file = out_dir / f"graph_s{scale}.log"

        print(f"\n>>> Scale scale={scale}  nP={params['nP']} nE={params['nE']} nC={params['nC']} nD={params['nD']}")

        if not wcnf_file.exists() or not wcnf_file.with_suffix(".meta.json").exists():
            print(f"  Generating BN graph and node values...")
            bn, values_table = generate_bn_and_values(params, args.seed)
            nP = len(bn.get("P", []))
            nC = len(bn.get("C", []))
            nD = len(bn.get("D", []))
            nE = len(bn.get("E", []))
            print(f"  BN: nP={nP} nE={nE} nC={nC} nD={nD}")

            print(f"  Exporting WCNF ...")
            generate_wcnf_and_save(params, args.seed, wcnf_file, bn, values_table)
            print(f"  WCNF generated: {wcnf_file.stat().st_size / 1024:.1f} KB")
        else:
            print(f"  Using existing WCNF file: {wcnf_file}")

        print(f"  Extracting C_benefit_sum and P_loss_sum ...")
        stats = load_stats_from_meta(wcnf_file)
        c_benefit_sum = stats["c_benefit_sum"]
        p_loss_sum = stats["p_loss_sum"]
        baseline_objective = stats["baseline_objective"]
        print(f"  C_benefit_sum = {c_benefit_sum:.2f}  P_loss_sum = {p_loss_sum:.2f}  "
              f"baseline_objective = {baseline_objective:.2f}")

        print(f"  Running MaxHS (timeout={args.timeout}s) ...")
        log_text = run_maxhs(wcnf_file, args.timeout)

        with open(log_file, "w") as f:
            f.write(log_text)
        print(f"  Log saved: {log_file}")

        ub_data = parse_ub_from_log(log_text)

        if ub_data["initial_ub"] is None:
            print(f"  [WARNING] Cannot parse log")
            continue

        time_hist = ub_data["time"]
        ub_hist = ub_data["ub"]
        obj_hist = ub_to_objective(ub_hist, c_benefit_sum)
        initial_obj = c_benefit_sum - ub_data["initial_ub"] / 100.0
        final_obj = obj_hist[-1] if obj_hist else initial_obj

        row = (f"  {scale:>7.1f} | {params['nP']:>6} | {params['nE']:>6} | "
               f"{params['nC']:>6} | {params['nD']:>6} | "
               f"{c_benefit_sum:>12,.0f} | {p_loss_sum:>12,.0f} | "
               f"{baseline_objective:>12,.0f} | {ub_data['total_time']:>8.2f} | "
               f"{ub_data['n_points']:>6}")
        print(row)

        all_results[scale] = {
            "time": time_hist,
            "ub": ub_hist,
            "objective": obj_hist,
            "c_benefit_sum": c_benefit_sum,
            "p_loss_sum": p_loss_sum,
            "baseline_objective": baseline_objective,
            "initial_objective": initial_obj,
            "final_objective": final_obj,
            "initial_ub": ub_data["initial_ub"],
            "final_ub": ub_data["final_ub"],
            "total_time": ub_data["total_time"],
            "n_points": ub_data["n_points"],
        }

    if not all_results:
        print("No successful runs, exiting.")
        return all_results, out_dir

    plot_convergence(all_results, out_dir)
    plot_comparison_bar(all_results, out_dir)

    serializable = {}
    for k, v in all_results.items():
        serializable[str(k)] = {}
        for kk, vv in v.items():
            if isinstance(vv, list):
                serializable[str(k)][kk] = [float(x) if isinstance(x, (int, float)) else x for x in vv]
            elif isinstance(vv, (int, float)) and not isinstance(vv, bool):
                serializable[str(k)][kk] = float(vv)
            else:
                serializable[str(k)][kk] = vv

    json_path = out_dir / "scaling_summary.json"
    with open(json_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nSummary saved: {json_path}")

    return all_results, out_dir


if __name__ == "__main__":
    results, out_dir = main()
