"""Batch compare BP+GA and MaxSAT on multiple graphs."""

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from generate_graph.random_graph import generate_bn_dag_multi_pe
from generate_graph.structured_graph import generate_bn_from_root_and_reverse
from generate_graph.numerical_generation import generate_node_values as gen_random_values
from generate_graph.number_generation import generate_node_values as gen_structured_values

from result_analysis.exact_analysis import run_exact_analysis

from comparison.bp_core import (
    MAXHS_BIN,
    FIXED_E_PROBS,
    find_best_defense_bp,
    find_best_defense_maxsat,
    extract_D_state_from_solution,
    compute_objective_bp_style,
)
from comparison.khouzani_baseline import find_best_defense_khouzani


def _winner(obj_a, name_a, obj_b, name_b, eps=1e-9):
    """Higher objective wins; TIE within eps. Returns None if either is missing."""
    if obj_a is None or obj_b is None:
        return None
    if abs(obj_a - obj_b) <= eps:
        return "TIE"
    return name_a if obj_a > obj_b else name_b


def generate_structured_graph_with_np_range(args, rng, graph_id):
    """Generate structured graph with nP in [nP_min, nP_max]"""
    target_nP = rng.randint(args.nP_min, args.nP_max)

    bn = generate_bn_from_root_and_reverse(
        nP=target_nP,
        seed=args.seed + graph_id,
        extra_p_edge_prob=0.25,
        max_extra_p_out_per_node=args.max_extra_p_out_per_node,
        c_parents_per_e_range=(1, 3),
        max_e_children_per_c=5,
        max_c_children_per_d=3,
    )

    values_table = gen_structured_values(
        bn, seed=args.seed + graph_id, fixed_e_probs=FIXED_E_PROBS,
        p_loss_range=(50, 500),
        c_benefit_range=(10, 50), d_cost_range=(50, 100),
        use_level_scaling=not args.no_level_scaling,
        p_alpha_max=args.p_alpha_max,
        cd_beta_max=args.cd_beta_max,
        cd_floor=args.cd_floor,
    )
    return bn, values_table


def generate_random_graph(args, graph_id):
    """Generate random graph"""
    bn = generate_bn_dag_multi_pe(
        nP=args.nP, nE=args.nE, nC=args.nC, nD=args.nD,
        max_children=args.max_children, p_EP=args.p_ep,
        seed=args.seed + graph_id,
    )
    values_table = gen_random_values(
        bn, seed=args.seed + graph_id, fixed_e_probs=FIXED_E_PROBS,
        p_loss_range=(50, 500),
        c_benefit_range=(10, 50), d_cost_range=(50, 100),
    )
    return bn, values_table


def defended_tuple(D_state):
    return tuple(sorted([d for d, v in D_state.items() if v]))


def find_rank_in_ga_results(target_D_state, ga_results):
    """Find rank of target_D_state in all evaluated GA solutions"""
    target = defended_tuple(target_D_state)
    for rank, (obj, D_state, detail) in enumerate(ga_results, start=1):
        if defended_tuple(D_state) == target:
            return rank
    return None


def run_single_comparison(bn, values_table, args, graph_id, top_n=10):
    """Compare both methods on a single graph"""
    nP = len(bn.get("P", []))
    nC = len(bn.get("C", []))
    nE = len(bn.get("E", []))
    use_ve = (nP + nC + nE) < 100

    if use_ve:
        bp_result = find_best_defense_bp(
            bn, values_table,
            population_size=args.population_size,
            genmax=args.genmax,
            seed=args.seed + graph_id,
            bp_max_iters=args.bp_max_iters,
            bp_damping=args.bp_damping,
            bp_tol=args.bp_tol,
            top_n=top_n,
            return_all=True,
        )
        bp_D_state = bp_result["best_D_state"]
        bp_eval = run_exact_analysis(bn, values_table, D_state=bp_D_state)
    else:
        bp_result = find_best_defense_bp(
            bn, values_table,
            population_size=args.population_size,
            genmax=args.genmax,
            seed=args.seed + graph_id,
            bp_max_iters=args.bp_max_iters,
            bp_damping=args.bp_damping,
            bp_tol=args.bp_tol,
            top_n=top_n,
            return_all=True,
        )
        bp_D_state = bp_result["best_D_state"]
        bp_eval = compute_objective_bp_style(
            bn, values_table, bp_D_state,
            bp_max_iters=args.bp_max_iters,
            bp_damping=args.bp_damping,
            bp_tol=args.bp_tol,
        )

    wcnf_path = Path(args.output_dir) / f"graph_{graph_id:03d}.wcnf"
    t0 = time.time()
    node_state, node_state_by_type, cnf_data, maxhs_out = find_best_defense_maxsat(
        bn, values_table, wcnf_path, MAXHS_BIN, timeout=args.maxhs_timeout
    )
    maxsat_solve_time = time.time() - t0

    ga_top_n = bp_result.get("top_n_results", [])
    ga_all_results = bp_result.get("all_results", [])

    if node_state is None:
        maxsat_D_state = None
        maxsat_bp_eval = None
        maxsat_timeout = True
        maxsat_rank_in_ga_top_n = None
        maxsat_rank_in_ga_all = None
    else:
        maxsat_D_state = extract_D_state_from_solution(bn, node_state_by_type)
        if use_ve:
            maxsat_bp_eval = run_exact_analysis(bn, values_table, D_state=maxsat_D_state)
        else:
            maxsat_bp_eval = compute_objective_bp_style(
                bn, values_table, maxsat_D_state,
                bp_max_iters=args.bp_max_iters,
                bp_damping=args.bp_damping,
                bp_tol=args.bp_tol,
            )
        maxsat_timeout = False
        maxsat_rank_in_ga_all = find_rank_in_ga_results(maxsat_D_state, ga_all_results)
        maxsat_rank_in_ga_top_n = None
        if maxsat_rank_in_ga_all is not None and maxsat_rank_in_ga_all <= top_n:
            maxsat_rank_in_ga_top_n = maxsat_rank_in_ga_all

    # --- Khouzani MILP interdiction baseline (3rd method) ---
    # Scored with the SAME per-graph evaluator (VE if use_ve else BP) as GA/MaxSAT.
    # big-M dual, single-thread (deterministic), 60s/solve cap; n_budget=10.
    t_kh = time.time()
    kh_result = find_best_defense_khouzani(
        bn, values_table, n_budget=10,
        eval_mode=("exact" if use_ve else "bp"),
        method="bigm", threads=1, milp_time_limit=60,
    )
    khouzani_solve_time = time.time() - t_kh
    khouzani_D_state = kh_result["best_D_state"]
    khouzani_defended = sorted([d for d, v in khouzani_D_state.items() if v])
    khouzani_objective = kh_result["best_objective"]

    khouzani_vs_ga = _winner(khouzani_objective, "Khouzani", bp_eval["objective"], "GA")
    khouzani_vs_maxsat = (None if maxsat_timeout
                          else _winner(khouzani_objective, "Khouzani",
                                       maxsat_bp_eval["objective"], "MaxSAT"))

    bp_defended = sorted([d for d, v in bp_D_state.items() if v])
    maxsat_defended = sorted([d for d, v in maxsat_D_state.items() if v]) if maxsat_D_state else []

    strategies_same = (bp_defended == maxsat_defended)
    strategies_different = not strategies_same

    if maxsat_timeout:
        winner = "UNKNOWN_TIMEOUT"
        objective_diff = None
    else:
        bp_obj = bp_eval["objective"]
        sat_obj = maxsat_bp_eval["objective"]
        objective_diff = bp_obj - sat_obj
        eps = 1e-9

        if abs(objective_diff) <= eps:
            winner = "TIE"
        elif objective_diff > 0:
            winner = "GA"
        else:
            winner = "MaxSAT"

    result = {
        "graph_id": graph_id,
        "graph_stats": {
            "nP": len(bn.get("P", [])),
            "nE": len(bn.get("E", [])),
            "nC": len(bn.get("C", [])),
            "nD": len(bn.get("D", [])),
            "total": len(bn.get("P", [])) + len(bn.get("E", [])) + len(bn.get("C", [])) + len(bn.get("D", [])),
            "edges": len(bn.get("edges", [])),
        },
        "use_ve": use_ve,
        "bp_time_s": bp_result["total_time_s"],
        "bp_evaluated": bp_result["unique_evaluated"],
        "bp_objective": bp_eval["objective"],
        "bp_defended": bp_defended,
        "bp_defended_count": len(bp_defended),

        "maxsat_time_s": None if maxsat_timeout else maxsat_solve_time,
        "maxsat_timeout": maxsat_timeout,
        "maxsat_objective": maxsat_bp_eval["objective"] if maxsat_bp_eval else None,
        "maxsat_defended": maxsat_defended,
        "maxsat_defended_count": len(maxsat_defended),

        "khouzani_time_s": khouzani_solve_time,
        "khouzani_milp_time_s": kh_result["milp_time_s"],
        "khouzani_objective": khouzani_objective,
        "khouzani_defended": khouzani_defended,
        "khouzani_defended_count": len(khouzani_defended),
        "khouzani_milp_optimal": f"{kh_result['n_milp_optimal']}/{kh_result['n_milp_solves']}",
        "khouzani_vs_ga": khouzani_vs_ga,
        "khouzani_vs_maxsat": khouzani_vs_maxsat,

        "strategies_same": strategies_same,
        "strategies_different": strategies_different,
        "winner": winner,
        "objective_diff_ga_minus_sat": objective_diff,

        "ga_top_n": [
            {
                "rank": rank,
                "objective": obj,
                "defended": sorted([d for d, v in D_state.items() if v]),
                "C_benefit": detail.get("C_benefit", 0.0),
                "P_expected_loss": detail.get("P_expected_loss", 0.0),
                "D_cost": detail.get("D_cost", 0.0),
            }
            for rank, (obj, D_state, detail) in enumerate(ga_top_n, start=1)
        ],

        "maxsat_rank_in_ga_top_n": maxsat_rank_in_ga_top_n,
        "maxsat_rank_in_ga_all": maxsat_rank_in_ga_all,
    }

    return result


def summarize_results(results):
    """Summarize comparison results across all graphs"""
    summary = {
        "num_graphs": len(results),
        "num_same_strategy": 0,
        "num_different_strategy": 0,
        "num_ga_better": 0,
        "num_sat_better": 0,
        "num_tie": 0,
        "num_timeout": 0,
        "avg_bp_time_s": None,
        "avg_maxsat_time_s": None,
        "avg_khouzani_time_s": None,
        # Khouzani win/tie/loss (higher objective = better)
        "khouzani_vs_ga": {"kh_win": 0, "tie": 0, "kh_loss": 0},
        "khouzani_vs_maxsat": {"kh_win": 0, "tie": 0, "kh_loss": 0},
        "ga_better_cases": [],
    }

    bp_times = []
    sat_times = []
    kh_times = []

    for r in results:
        bp_times.append(r["bp_time_s"])
        if r.get("khouzani_time_s") is not None:
            kh_times.append(r["khouzani_time_s"])
        for key, other in (("khouzani_vs_ga", "GA"), ("khouzani_vs_maxsat", "MaxSAT")):
            w = r.get(key)
            if w == "Khouzani":
                summary[key]["kh_win"] += 1
            elif w == "TIE":
                summary[key]["tie"] += 1
            elif w == other:
                summary[key]["kh_loss"] += 1
        if not r["maxsat_timeout"] and r["maxsat_time_s"] is not None:
            sat_times.append(r["maxsat_time_s"])
        else:
            summary["num_timeout"] += 1

        if r["strategies_same"]:
            summary["num_same_strategy"] += 1
        else:
            summary["num_different_strategy"] += 1

        if r["winner"] == "GA":
            summary["num_ga_better"] += 1
            summary["ga_better_cases"].append({
                "graph_id": r["graph_id"],
                "ga_objective": r["bp_objective"],
                "sat_objective": r["maxsat_objective"],
                "sat_rank_in_ga_all": r["maxsat_rank_in_ga_all"],
                "sat_rank_in_ga_top_n": r["maxsat_rank_in_ga_top_n"],
                "ga_defended": r["bp_defended"],
                "sat_defended": r["maxsat_defended"],
            })
        elif r["winner"] == "MaxSAT":
            summary["num_sat_better"] += 1
        elif r["winner"] == "TIE":
            summary["num_tie"] += 1

    if bp_times:
        summary["avg_bp_time_s"] = sum(bp_times) / len(bp_times)
    if sat_times:
        summary["avg_maxsat_time_s"] = sum(sat_times) / len(sat_times)
    if kh_times:
        summary["avg_khouzani_time_s"] = sum(kh_times) / len(kh_times)

    return summary


def print_single_result(r):
    """Print single graph comparison result"""
    method = "VE" if r.get("use_ve") else "GA"
    print("=" * 100)
    print(f"[Graph {r['graph_id']:02d}] ({method}) "
          f"P={r['graph_stats']['nP']} E={r['graph_stats']['nE']} "
          f"C={r['graph_stats']['nC']} D={r['graph_stats']['nD']} "
          f"total={r['graph_stats']['total']} edges={r['graph_stats']['edges']}")
    print(f"GA     : time={r['bp_time_s']:.2f}s  evaluated={r['bp_evaluated']}  "
          f"obj={r['bp_objective']:.6f}  defended={r['bp_defended']}")
    if r["maxsat_timeout"]:
        print("MaxSAT  : TIMEOUT")
    else:
        print(f"MaxSAT  : time={r['maxsat_time_s']:.2f}s  "
              f"obj={r['maxsat_objective']:.6f}  defended={r['maxsat_defended']}")

    print(f"Strategies same: {'Yes' if r['strategies_same'] else 'No'}")
    winner_label = "VE" if r.get("use_ve") else "GA"
    print(f"Winner           : {r['winner']} ({winner_label})")
    if r["objective_diff_ga_minus_sat"] is not None:
        method_label = "VE" if r.get("use_ve") else "GA"
        print(f"{method_label} - SAT objective = {r['objective_diff_ga_minus_sat']:.6f}")

    if r["winner"] == "GA":
        rank_all = r["maxsat_rank_in_ga_all"]
        rank_top_n = r["maxsat_rank_in_ga_top_n"]
        if rank_all is None:
            print("When GA is better, SAT solution not in GA evaluated set")
        else:
            print(f"When GA is better, SAT rank in all GA evaluations: {rank_all}")
            if rank_top_n is not None:
                print(f"Also in GA Top-{len(r['ga_top_n'])}, rank {rank_top_n}")
            else:
                print(f"Not in GA Top-{len(r['ga_top_n'])}")

    print(f"GA Top-{len(r['ga_top_n'])}:")
    for item in r["ga_top_n"]:
        print(f"  rank={item['rank']:>2d}  "
              f"obj={item['objective']:>12.6f}  "
              f"defended={item['defended']}")


def print_summary(summary):
    """Print summary results"""
    print("\n" + "#" * 100)
    print("Overall Summary")
    print("#" * 100)
    print(f"Graph count              : {summary['num_graphs']}")
    print(f"Same strategy count       : {summary['num_same_strategy']}")
    print(f"Different strategy count : {summary['num_different_strategy']}")
    print(f"GA better count          : {summary['num_ga_better']}")
    print(f"MaxSAT better count      : {summary['num_sat_better']}")
    print(f"Tie count                : {summary['num_tie']}")
    print(f"MaxSAT timeout count     : {summary['num_timeout']}")
    print(f"Average BP+GA time       : {summary['avg_bp_time_s']}")
    print(f"Average MaxSAT time      : {summary['avg_maxsat_time_s']}")

    if summary["ga_better_cases"]:
        print("\nGA better cases:")
        for case in summary["ga_better_cases"]:
            print(f"  Graph {case['graph_id']:02d}: "
                  f"SAT rank in GA all = {case['sat_rank_in_ga_all']}, "
                  f"SAT rank in GA top_n = {case['sat_rank_in_ga_top_n']}")


def main():
    parser = argparse.ArgumentParser(description="Batch compare BP+GA vs MaxSAT")

    parser.add_argument("graph_type", choices=["structured", "random"])
    parser.add_argument("--num-graphs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--output-dir", type=str, default="comparison/batch_outputs")
    parser.add_argument("--save-result", action="store_true")
    parser.add_argument("--result-json", type=str, default="comparison/batch_compare_results.json")

    parser.add_argument("--maxhs-timeout", type=int, default=120)
    parser.add_argument("--population-size", type=int, default=100)
    parser.add_argument("--genmax", type=int, default=50)
    parser.add_argument("--bp-max-iters", type=int, default=100)
    parser.add_argument("--bp-damping", type=float, default=0.2)
    parser.add_argument("--bp-tol", type=float, default=1e-6)
    parser.add_argument("--top-n", type=int, default=10)

    parser.add_argument("--nP-min", type=int, default=None, help="structured: nP lower bound")
    parser.add_argument("--nP-max", type=int, default=None, help="structured: nP upper bound")
    parser.add_argument("--no-level-scaling", action="store_true")
    parser.add_argument("--p-alpha-max", type=float, default=0.8)
    parser.add_argument("--cd-beta-max", type=float, default=0.5)
    parser.add_argument("--cd-floor", type=float, default=0.2)
    parser.add_argument("--max-extra-p-out-per-node", type=int, default=2)

    parser.add_argument("--nP", type=int, default=None)
    parser.add_argument("--nE", type=int, default=None)
    parser.add_argument("--nC", type=int, default=None)
    parser.add_argument("--nD", type=int, default=None)
    parser.add_argument("--max-children", type=int, default=5)
    parser.add_argument("--p-ep", type=float, default=0.25)

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.graph_type == "structured":
        if args.nP_min is None or args.nP_max is None:
            parser.error("structured mode requires --nP-min and --nP-max")
        if args.nP_min >= args.nP_max:
            parser.error("--nP-min must be less than --nP-max")
    else:
        if None in (args.nP, args.nE, args.nC, args.nD):
            parser.error("random mode requires --nP --nE --nC --nD")

    rng = random.Random(args.seed)
    results = []

    for graph_id in range(1, args.num_graphs + 1):
        try:
            if args.graph_type == "structured":
                bn, values_table = generate_structured_graph_with_np_range(args, rng, graph_id)
            else:
                bn, values_table = generate_random_graph(args, graph_id)

            r = run_single_comparison(
                bn, values_table, args,
                graph_id=graph_id,
                top_n=args.top_n
            )
            results.append(r)
            print_single_result(r)

        except Exception as e:
            print("=" * 100)
            print(f"[Graph {graph_id:02d}] Failed: {e}")
            results.append({
                "graph_id": graph_id,
                "error": str(e)
            })

    valid_results = [r for r in results if "error" not in r]
    summary = summarize_results(valid_results)
    print_summary(summary)

    if args.save_result:
        save_data = {
            "args": vars(args),
            "summary": summary,
            "results": results,
        }
        with open(args.result_json, "w", encoding="utf-8") as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
        print(f"\nResult saved to: {args.result_json}")


if __name__ == "__main__":
    main()
