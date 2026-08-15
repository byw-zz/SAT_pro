"""Add a Khouzani-MILP column to an existing batch_comparison dataset.

The four existing datasets (structured_12_17 / structured_151_199 / random_10 /
random_100) already hold MaxSAT and GA-BP results per graph, and GA-BP is far too
slow to re-run on the large ones (thousands of seconds per graph). Since every
graph is regenerated deterministically from (seed + graph_id), we replay the exact
same graphs, run ONLY the Khouzani baseline on each, and compare against the stored
maxsat_objective / bp_objective(=GA) under the SAME evaluator the dataset used
(VE for small graphs where nP+nC+nE < 100, else BP).
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent.parent))

from comparison.batch_comparison import (
    generate_structured_graph_with_np_range,
    generate_random_graph,
)
from comparison.khouzani_baseline import find_best_defense_khouzani


def _defended(D_state):
    return sorted(d for d, v in D_state.items() if v)


def run_dataset(dataset_json, n_budget=15, limit=None, verify=True, milp_time_limit=10,
                method="rowgen", threads=1):
    d = json.load(open(dataset_json))
    args = SimpleNamespace(**d["args"])
    # Historical comparison datasets consumed one now-removed P_benefit draw
    # per P node. Preserve that RNG stream when replaying their numeric values.
    args.legacy_rng_compat = True
    stored = d["results"]

    # replay the exact generation loop from batch_comparison.main
    rng = random.Random(args.seed)
    rows = []
    for res in stored:
        gid = res["graph_id"]
        if args.graph_type == "structured":
            bn, vt = generate_structured_graph_with_np_range(args, rng, gid)
        else:
            bn, vt = generate_random_graph(args, gid)

        nP, nC, nE, nD = len(bn["P"]), len(bn["C"]), len(bn["E"]), len(bn["D"])
        gs = res.get("graph_stats", {})
        if verify and gs:
            ok = (nP == gs.get("nP") and nE == gs.get("nE")
                  and nC == gs.get("nC") and nD == gs.get("nD"))
            if not ok:
                raise RuntimeError(
                    f"graph {gid} regeneration MISMATCH: "
                    f"got (P{nP},E{nE},C{nC},D{nD}) vs stored {gs}")

        use_ve = (nP + nC + nE) < 100
        mode = "exact" if use_ve else "bp"

        kh = find_best_defense_khouzani(bn, vt, n_budget=n_budget, eval_mode=mode,
                                        milp_time_limit=milp_time_limit,
                                        method=method, threads=threads)

        rows.append({
            "graph_id": gid,
            "nP": nP, "nE": nE, "nC": nC, "nD": nD,
            "eval": mode,
            "khouzani_obj": kh["best_objective"],
            "khouzani_defended": _defended(kh["best_D_state"]),
            "khouzani_n_def": sum(1 for v in kh["best_D_state"].values() if v),
            "khouzani_milp_ms": kh["milp_time_s"] * 1000.0,
            "khouzani_wall_ms": kh["runtime_s"] * 1000.0,
            "khouzani_milp_optimal": f"{kh['n_milp_optimal']}/{kh['n_milp_solves']}",
            "maxsat_obj": res.get("maxsat_objective"),
            "maxsat_defended": res.get("maxsat_defended"),
            "maxsat_timeout": res.get("maxsat_timeout"),
            "ga_obj": res.get("bp_objective"),
            "ga_defended": res.get("bp_defended"),
        }, )
        print(f"  [{gid:>3}] nP={nP:>3} eval={mode:<5} "
              f"kh={kh['best_objective']:>10.2f} "
              f"sat={res.get('maxsat_objective')} "
              f"ga={res.get('bp_objective')} "
              f"milp={kh['milp_time_s']*1000:>6.0f}ms", flush=True)
        if limit and len(rows) >= limit:
            break

    return args, rows


def summarize(rows, tol=0.5):
    """Pairwise win/tie/loss of Khouzani vs MaxSAT and vs GA (higher obj = better)."""
    def cmp(a, b):
        if a is None or b is None:
            return "na"
        if a > b + tol:
            return "win"
        if a < b - tol:
            return "loss"
        return "tie"

    out = {"n": len(rows)}
    for other in ("maxsat", "ga"):
        w = t = l = na = 0
        for r in rows:
            c = cmp(r["khouzani_obj"], r[f"{other}_obj"])
            w += c == "win"; t += c == "tie"; l += c == "loss"; na += c == "na"
        out[f"vs_{other}"] = {"kh_win": w, "tie": t, "kh_loss": l, "na": na}
    milp = [r["khouzani_milp_ms"] for r in rows]
    out["milp_ms"] = {"min": min(milp), "max": max(milp),
                      "mean": sum(milp) / len(milp)} if milp else {}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset_json")
    ap.add_argument("--n-budget", type=int, default=10)
    ap.add_argument("--milp-time-limit", type=int, default=60, help="CBC seconds per solve")
    ap.add_argument("--method", choices=["rowgen", "bigm"], default="bigm")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--no-verify", action="store_true")
    a = ap.parse_args()

    print(f"=== {a.dataset_json} ===", flush=True)
    t0 = time.time()
    args, rows = run_dataset(a.dataset_json, n_budget=a.n_budget,
                             limit=a.limit, verify=not a.no_verify,
                             milp_time_limit=a.milp_time_limit,
                             method=a.method, threads=a.threads)
    summ = summarize(rows)
    summ["elapsed_s"] = time.time() - t0
    print("\n--- summary ---")
    print(json.dumps(summ, ensure_ascii=False, indent=2))

    out = a.out or (Path(a.dataset_json).stem + "_khouzani.json")
    json.dump({"dataset": a.dataset_json, "args": vars(args),
               "khouzani_config": {
                   "method": a.method,
                   "n_budget": a.n_budget,
                   "milp_time_limit": a.milp_time_limit,
                   "threads": a.threads,
                   "legacy_rng_compat": args.legacy_rng_compat,
               },
               "rows": rows, "summary": summ},
              open(out, "w"), ensure_ascii=False, indent=2)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
