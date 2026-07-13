#!/usr/bin/env python3
"""OAT sensitivity analysis of the utility model for MaxSAT defense optimization.

Reviewer point addressed
------------------------
The utility model relies on manually assigned benefit / cost / risk values, and
those values directly influence the selected defense strategy. This experiment
tests how ROBUST the optimization outcome is to perturbations of those values.

Design (as chosen)
------------------
* One-At-a-Time (OAT) *local* sensitivity.
* Multiplicative scaling +/- x% applied to ONE parameter class at a time
  (C_benefit, P_loss, D_cost, E_prob); the other classes are held fixed.
* Solver: MaxSAT (MaxHS) re-solves the optimal defense set under each perturbation.

Only parameters that enter the objective are perturbed
(objective = sum C_benefit - sum P(compromise)*P_loss - sum D_cost).

Metrics per (graph, parameter, scale)
--------------------------------------
* strategy robustness : Jaccard(D*_perturbed, D*_baseline) and #flipped defenses.
* optimality regret   : how sub-optimal the BASELINE strategy is once the
                        parameters change, evaluated with the SAME estimator
                        (exact VE for small graphs, BP otherwise):
                            regret = (obj_s(D*_s) - obj_s(D*_base)) / |obj_s(D*_s)|
                        regret ~ 0  => the baseline strategy is still (near) optimal
                                       under the perturbed values => robust.

Requires the `sat_pro` env (pgmpy, pymoo, networkx) and a MaxHS binary.

Example
-------
  conda run -n sat_pro python sensitivity_analysis.py \
      --graph-type random --graphs 20 \
      --scales 0.7 0.8 0.9 1.1 1.2 1.3 \
      --out sensitivity_out/
"""

import argparse
import contextlib
import csv
import io
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from generate_graph.config import FIXED_E_PROBS
from generate_graph.random_graph import generate_bn_dag_multi_pe
from generate_graph.structured_graph import generate_bn_from_root_and_reverse
from generate_graph.numerical_generation import generate_node_values as gen_random_values
from generate_graph.number_generation import generate_node_values as gen_structured_values
from comparison.bp_core import (
    find_best_defense_maxsat,
    extract_D_state_from_solution,
    compute_objective_bp_style,
)
from result_analysis.exact_analysis import run_exact_analysis

VAL_RANGES = dict(
    p_loss_range=(50, 500),
    c_benefit_range=(10, 50), d_cost_range=(50, 100),
)

# parameter class -> (node type it lives on, field name in the value row)
PARAM_FIELD = {
    "C_benefit": ("C", "C_benefit"),
    "P_loss":    ("P", "P_loss"),
    "D_cost":    ("D", "D_cost"),
    "E_prob":    ("E", "E_prob"),
}

EPS = 1e-9


# --------------------------------------------------------------------------- #
def make_graph(gtype, gid, args, rng):
    """Generate one graph + value table using the existing generators/params."""
    if gtype == "random":
        bn = generate_bn_dag_multi_pe(
            nP=args.nP, nE=args.nE, nC=args.nC, nD=args.nD,
            max_children=5, p_EP=0.25, seed=args.seed + gid,
        )
        vt = gen_random_values(bn, seed=args.seed + gid, fixed_e_probs=FIXED_E_PROBS, **VAL_RANGES)
    else:
        target_nP = rng.randint(args.nP_min, args.nP_max)
        bn = generate_bn_from_root_and_reverse(
            nP=target_nP, seed=args.seed + gid, extra_p_edge_prob=0.25,
            max_extra_p_out_per_node=2, c_parents_per_e_range=(1, 3),
            max_e_children_per_c=5, max_c_children_per_d=3,
        )
        vt = gen_structured_values(
            bn, seed=args.seed + gid, fixed_e_probs=FIXED_E_PROBS, **VAL_RANGES,
            use_level_scaling=True, p_alpha_max=0.8, cd_beta_max=0.5, cd_floor=0.2,
        )
    return bn, vt


def scale_values(values_table, param, s):
    """Return a copy of values_table with one parameter class multiplicatively scaled."""
    ntype, field = PARAM_FIELD[param]
    out = []
    for row in values_table:
        r = dict(row)
        if r.get("type") == ntype and r.get(field) is not None:
            v = r[field] * s
            if field == "E_prob":                 # keep probabilities in (0, 1)
                v = min(max(v, 1e-6), 0.999)
            r[field] = v
        out.append(r)
    return out


def defended_set(d_state):
    return frozenset(d for d, v in (d_state or {}).items() if v)


def jaccard(a, b):
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union) if union else 1.0


def solve_maxsat_strategy(bn, values, wcnf_path, maxhs_bin, timeout):
    # silence graph2sat's "WCNF file exported" chatter on every solve
    with contextlib.redirect_stdout(io.StringIO()):
        node_state, nbt, _cnf, _t = find_best_defense_maxsat(
            bn, values, wcnf_path, maxhs_bin, timeout=timeout
        )
    if node_state is None:
        return None
    return extract_D_state_from_solution(bn, nbt)


def eval_objective(bn, values, d_state, use_ve, bp_kwargs):
    if use_ve:
        return run_exact_analysis(bn, values, D_state=d_state)["objective"]
    return compute_objective_bp_style(bn, values, d_state, **bp_kwargs)["objective"]


# --------------------------------------------------------------------------- #
def run(args):
    import random
    rng = random.Random(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    wcnf_path = out_dir / "_tmp.wcnf"
    bp_kwargs = dict(bp_max_iters=args.bp_max_iters, bp_damping=args.bp_damping, bp_tol=args.bp_tol)

    rows = []
    t0 = time.time()
    for gid in range(1, args.graphs + 1):
        bn, vt = make_graph(args.graph_type, gid, args, rng)
        nP, nC, nE = len(bn["P"]), len(bn["C"]), len(bn["E"])
        if args.eval == "ve":
            use_ve = True
        elif args.eval == "bp":
            use_ve = False
        else:  # auto: exact VE for small graphs, BP otherwise (matches batch_comparison)
            use_ve = (nP + nC + nE) < 100

        d_base = solve_maxsat_strategy(bn, vt, wcnf_path, args.maxhs_bin, args.timeout)
        if d_base is None:
            print(f"[graph {gid}] baseline MaxHS returned no solution (timeout?), skipping")
            continue
        set_base = defended_set(d_base)

        for param in args.params:
            for s in args.scales:
                vt_s = scale_values(vt, param, s)
                d_s = solve_maxsat_strategy(bn, vt_s, wcnf_path, args.maxhs_bin, args.timeout)
                if d_s is None:
                    continue
                set_s = defended_set(d_s)
                jac = jaccard(set_s, set_base)
                flipped = len(set_s ^ set_base)

                # optimality regret: baseline strategy vs re-optimized, under perturbed params
                obj_s_opt = eval_objective(bn, vt_s, d_s, use_ve, bp_kwargs)
                obj_s_base = eval_objective(bn, vt_s, d_base, use_ve, bp_kwargs)
                regret = (obj_s_opt - obj_s_base) / max(abs(obj_s_opt), EPS)

                rows.append({
                    "graph_id": gid, "nP": nP, "nE": nE, "nC": nC, "nD": len(bn["D"]),
                    "use_ve": use_ve, "param": param, "scale": s,
                    "jaccard": round(jac, 4), "flipped": flipped,
                    "n_def_base": len(set_base), "n_def_scaled": len(set_s),
                    "obj_scaled_opt": round(obj_s_opt, 4),
                    "obj_scaled_base_strategy": round(obj_s_base, 4),
                    "optimality_regret": round(regret, 6),
                })
        print(f"[graph {gid}/{args.graphs}] done  (P={nP} E={nE} C={nC} D={len(bn['D'])}, use_ve={use_ve})")

    # ---- write per-row CSV ----
    csv_path = out_dir / "sensitivity_results.csv"
    if rows:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    summary = summarize(rows, args.params, args.scales)
    print_summary(summary, args.scales)
    _plot(summary, args.params, args.scales, out_dir, args.no_plot)

    import json
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump({"args": {k: v for k, v in vars(args).items()}, "summary": summary}, f,
                  ensure_ascii=False, indent=2, default=str)
    print(f"\nRows: {len(rows)}  CSV: {csv_path}  ({time.time()-t0:.1f}s)")


def summarize(rows, params, scales):
    """Aggregate per parameter and per (parameter, scale)."""
    def agg(subset):
        n = len(subset)
        if n == 0:
            return None
        return {
            "n": n,
            "mean_jaccard": round(sum(r["jaccard"] for r in subset) / n, 4),
            "strategy_unchanged_frac": round(sum(1 for r in subset if r["jaccard"] == 1.0) / n, 4),
            "mean_flipped": round(sum(r["flipped"] for r in subset) / n, 3),
            "mean_abs_regret": round(sum(abs(r["optimality_regret"]) for r in subset) / n, 6),
            "max_abs_regret": round(max(abs(r["optimality_regret"]) for r in subset), 6),
        }

    out = {"by_param": {}, "by_param_scale": {}}
    for p in params:
        out["by_param"][p] = agg([r for r in rows if r["param"] == p])
        for s in scales:
            out["by_param_scale"][f"{p}@{s}"] = agg([r for r in rows if r["param"] == p and r["scale"] == s])
    return out


def print_summary(summary, scales):
    print("\n" + "=" * 84)
    print("OAT SENSITIVITY SUMMARY  (per parameter, aggregated over graphs & scales)")
    print("=" * 84)
    print(f"{'param':<12} {'n':>5} {'mean_jaccard':>13} {'unchanged%':>11} "
          f"{'mean_flip':>10} {'mean|regret|':>13} {'max|regret|':>12}")
    print("-" * 84)
    for p, a in summary["by_param"].items():
        if not a:
            continue
        print(f"{p:<12} {a['n']:>5} {a['mean_jaccard']:>13} "
              f"{a['strategy_unchanged_frac']*100:>10.1f}% {a['mean_flipped']:>10} "
              f"{a['mean_abs_regret']:>13} {a['max_abs_regret']:>12}")
    print("=" * 84)
    print("Reading: mean_jaccard->1 and mean|regret|->0 means the selected defense")
    print("strategy is robust to that parameter; lower jaccard / higher regret = sensitive.")


def _plot(summary, params, scales, out_dir, no_plot):
    if no_plot:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[plot skipped] {e}")
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    xs = sorted(scales)
    for p in params:
        jac = [summary["by_param_scale"].get(f"{p}@{s}", {}) for s in xs]
        ax.plot(xs, [(d or {}).get("mean_jaccard") for d in jac], marker="o", label=p)
    ax.set_xlabel("multiplicative scale of the parameter")
    ax.set_ylabel("mean Jaccard(D*_perturbed, D*_baseline)")
    ax.set_title("Defense-strategy stability under parameter perturbation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "sensitivity.png", dpi=150)
    print(f"Plot: {out_dir / 'sensitivity.png'}")


def main():
    p = argparse.ArgumentParser(description="OAT sensitivity analysis of the utility model.",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--graph-type", choices=["random", "structured"], default="random")
    p.add_argument("--graphs", type=int, default=20, help="number of graphs to average over")
    p.add_argument("--scales", type=float, nargs="+", default=[0.7, 0.8, 0.9, 1.1, 1.2, 1.3])
    p.add_argument("--params", nargs="+", default=["C_benefit", "P_loss", "D_cost", "E_prob"],
                   choices=list(PARAM_FIELD.keys()))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default="sensitivity_out")
    p.add_argument("--maxhs-bin", type=str,
                   default=os.path.expanduser("~/MaxHS/build/release/bin/maxhs"))
    p.add_argument("--timeout", type=int, default=120)
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--eval", choices=["auto", "ve", "bp"], default="auto",
                   help="objective estimator for regret: auto=exact VE if small else BP; "
                        "ve=force exact VE; bp=force BP (needed for deep structured graphs where VE is intractable)")
    # random-family sizes (match the existing experiments)
    p.add_argument("--nP", type=int, default=10)
    p.add_argument("--nE", type=int, default=24)
    p.add_argument("--nC", type=int, default=30)
    p.add_argument("--nD", type=int, default=15)
    # structured-family size range
    p.add_argument("--nP-min", type=int, default=12)
    p.add_argument("--nP-max", type=int, default=17)
    # BP evaluation params (used only when a graph is too big for exact VE)
    p.add_argument("--bp-max-iters", type=int, default=100)
    p.add_argument("--bp-damping", type=float, default=0.2)
    p.add_argument("--bp-tol", type=float, default=1e-6)
    args = p.parse_args()

    if not os.path.exists(args.maxhs_bin):
        alt = "MaxHS/build/release/bin/maxhs"
        if os.path.exists(alt):
            args.maxhs_bin = alt
        else:
            p.error(f"MaxHS binary not found at {args.maxhs_bin} (nor {alt}); pass --maxhs-bin")
    run(args)


if __name__ == "__main__":
    main()
