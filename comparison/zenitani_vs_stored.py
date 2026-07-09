"""Add a Zenitani gradient-descent column to an existing batch_comparison dataset.

Mirrors `khouzani_vs_stored.py`: the four datasets (structured_12_17 /
structured_151_199 / random_10 / random_100) already store MaxSAT and GA-BP
results per graph. Each graph is regenerated deterministically from
(seed + graph_id), so we replay the exact same graphs, run ONLY the Zenitani
baseline (Algorithm 1 one-pass + Algorithm 3 iterative refinement), and compare
against the stored maxsat_objective / bp_objective(=GA) under the SAME evaluator
the dataset used (VE for small graphs where nP+nC+nE < 100, else BP).

Large (BP) graphs cap the Algorithm-3 neighbour random-sample at s=20 to bound
the refinement cost at O(r*nD*s); small (VE) graphs keep the whole front (exact).
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
from comparison.zenitani_baseline import find_best_defense_zenitani


# Large (BP) graphs get a lighter Algorithm 3: BP costs ~1.5 s/eval on ~1000-node
# graphs, so sample/n_iter are trimmed to keep the neighbour-evaluation count (and
# thus wall time) tractable. max_k already caps Algorithm 1 at O(max_k*nD).
LARGE_SAMPLE = 5    # s: neighbour random-sample per defence.
LARGE_N_ITER = 5    # r: Algorithm-3 rounds; a few growth/refine rounds past max_k.
# Algorithm-1 descent cap (max #controls) PER DATASET SCALE.
#   structured_151_199: quality was max_k-INSENSITIVE (10 vs 12 -> same n_def=13,
#     same obj~2750, both far below GA/MaxSAT), so keep it low (10) to save Alg1 cost.
#   random_100: optima deploy ~34-40 (max 52); 40 covers the mean, hit_max_k flags
#     the few graphs whose optimum exceeds ~45 (max_k + n_iter growth).
LARGE_MAX_K = {"structured": 10, "random": 40}


def _defended(D_state):
    return sorted(d for d, v in D_state.items() if v)


def run_dataset(dataset_json, n_iter=20, limit=None, verify=True):
    d = json.load(open(dataset_json))
    args = SimpleNamespace(**d["args"])
    stored = d["results"]

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
        # small (VE) graphs: full, exact settings; large (BP) graphs: trimmed for cost
        sample = None if use_ve else LARGE_SAMPLE
        max_k = None if use_ve else LARGE_MAX_K.get(args.graph_type, 15)
        g_n_iter = n_iter if use_ve else LARGE_N_ITER
        bp_fast = not use_ve                        # closed-form BP on big graphs (identical, faster)

        t = time.time()
        zen = find_best_defense_zenitani(bn, vt, eval_mode=mode, n_iter=g_n_iter,
                                         sample=sample, max_k=max_k, seed=args.seed,
                                         bp_fast=bp_fast)
        wall_ms = (time.time() - t) * 1000.0

        rows.append({
            "graph_id": gid,
            "nP": nP, "nE": nE, "nC": nC, "nD": nD,
            "eval": mode,
            "zen_sample": sample,
            "zenitani_obj": zen["best_objective"],
            "zenitani_defended": _defended(zen["best_D_state"]),
            "zenitani_n_def": sum(1 for v in zen["best_D_state"].values() if v),
            "zenitani_n_evals": zen["n_evals"],
            "zenitani_n_iter": zen["n_iter"],
            "zenitani_max_k": zen["max_k"],
            "zenitani_hit_max_k": zen["hit_max_k"],
            "zenitani_wall_ms": wall_ms,
            "maxsat_obj": res.get("maxsat_objective"),
            "maxsat_defended": res.get("maxsat_defended"),
            "maxsat_timeout": res.get("maxsat_timeout"),
            "ga_obj": res.get("bp_objective"),
            "ga_defended": res.get("bp_defended"),
        })
        print(f"  [{gid:>3}] nP={nP:>3} nD={nD:>3} eval={mode:<5} "
              f"zen={zen['best_objective']:>10.2f} "
              f"sat={res.get('maxsat_objective')} "
              f"ga={res.get('bp_objective')} "
              f"ev={zen['n_evals']:>5} {wall_ms:>7.0f}ms", flush=True)
        if limit and len(rows) >= limit:
            break

    return args, rows


def summarize(rows, tol=0.5):
    """Pairwise win/tie/loss of Zenitani vs MaxSAT and vs GA (higher obj = better)."""
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
            c = cmp(r["zenitani_obj"], r[f"{other}_obj"])
            w += c == "win"; t += c == "tie"; l += c == "loss"; na += c == "na"
        out[f"vs_{other}"] = {"zen_win": w, "tie": t, "zen_loss": l, "na": na}
    ev = [r["zenitani_n_evals"] for r in rows]
    ms = [r["zenitani_wall_ms"] for r in rows]
    out["n_evals"] = {"min": min(ev), "max": max(ev), "mean": sum(ev) / len(ev)} if ev else {}
    out["wall_ms"] = {"min": min(ms), "max": max(ms), "mean": sum(ms) / len(ms)} if ms else {}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset_json")
    ap.add_argument("--n-iter", type=int, default=20, help="Algorithm-3 rounds (r)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--no-verify", action="store_true")
    a = ap.parse_args()

    print(f"=== {a.dataset_json} ===", flush=True)
    t0 = time.time()
    args, rows = run_dataset(a.dataset_json, n_iter=a.n_iter,
                             limit=a.limit, verify=not a.no_verify)
    summ = summarize(rows)
    summ["elapsed_s"] = time.time() - t0
    print("\n--- summary ---")
    print(json.dumps(summ, ensure_ascii=False, indent=2))

    out = a.out or (Path(a.dataset_json).stem + "_zenitani.json")
    json.dump({"dataset": a.dataset_json, "args": vars(args),
               "rows": rows, "summary": summ},
              open(out, "w"), ensure_ascii=False, indent=2)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
