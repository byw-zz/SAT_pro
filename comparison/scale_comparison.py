"""Compare BP+GA and MaxSAT across different graph scales."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from generate_graph.random_graph import generate_bn_dag_multi_pe
from generate_graph.structured_graph import generate_bn_from_root_and_reverse
from generate_graph.numerical_generation import generate_node_values as gen_random_values
from generate_graph.number_generation import generate_node_values as gen_structured_values

from comparison.bp_core import (
    MAXHS_BIN,
    FIXED_E_PROBS,
    find_best_defense_bp,
    find_best_defense_maxsat,
    extract_D_state_from_solution,
    compute_objective_bp_style,
)


def run_single_comparison(bn, values_table, args, top_n=10):
    """Compare both methods for a single configuration"""

    bp_result = find_best_defense_bp(
        bn, values_table,
        population_size=args.population_size,
        genmax=args.genmax,
        seed=args.seed,
        bp_max_iters=args.bp_max_iters,
        bp_damping=args.bp_damping,
        bp_tol=args.bp_tol,
        top_n=top_n,
    )

    bp_D_state = bp_result['best_D_state']
    bp_eval = compute_objective_bp_style(
        bn, values_table, bp_D_state,
        bp_max_iters=args.bp_max_iters,
        bp_damping=args.bp_damping,
        bp_tol=args.bp_tol,
    )

    wcnf_path = Path(args.output).resolve()
    node_state, node_state_by_type, cnf_data, maxsat_parse_time = find_best_defense_maxsat(
        bn, values_table, wcnf_path, MAXHS_BIN, timeout=args.maxhs_timeout
    )

    ga_top_n = bp_result.get("top_n_results", [])

    if node_state is None:
        maxsat_D_state = None
        maxsat_bp_eval = None
        maxsat_timeout = True
        maxsat_in_top_n_rank = None
        maxsat_in_top_n = False
    else:
        maxsat_D_state = extract_D_state_from_solution(bn, node_state_by_type)
        maxsat_bp_eval = compute_objective_bp_style(
            bn, values_table, maxsat_D_state,
            bp_max_iters=args.bp_max_iters,
            bp_damping=args.bp_damping,
            bp_tol=args.bp_tol,
        )
        maxsat_timeout = False

        maxsat_defended_tuple = tuple(sorted([d for d, v in maxsat_D_state.items() if v]))
        maxsat_in_top_n = False
        maxsat_in_top_n_rank = None
        for rank, (obj, D_state, _) in enumerate(ga_top_n, start=1):
            ga_defended_tuple = tuple(sorted([d for d, v in D_state.items() if v]))
            if ga_defended_tuple == maxsat_defended_tuple:
                maxsat_in_top_n = True
                maxsat_in_top_n_rank = rank
                break

    bp_defended = sorted([d for d, v in bp_D_state.items() if v])
    maxsat_defended = sorted([d for d, v in maxsat_D_state.items() if v]) if maxsat_D_state else []

    strategies_different = (bp_defended != maxsat_defended)

    return {
        "bp_time_s": bp_result['total_time_s'],
        "bp_evaluated": bp_result['unique_evaluated'],
        "bp_objective": bp_eval['objective'],
        "bp_defended": bp_defended,
        "bp_defended_count": len(bp_defended),
        "ga_top_n": ga_top_n,
        "maxsat_time_s": None if maxsat_timeout else maxsat_parse_time,
        "maxsat_timeout": maxsat_timeout,
        "maxsat_objective": maxsat_bp_eval['objective'] if maxsat_bp_eval else None,
        "maxsat_defended": maxsat_defended,
        "maxsat_defended_count": len(maxsat_defended),
        "maxsat_in_top_n": maxsat_in_top_n,
        "maxsat_in_top_n_rank": maxsat_in_top_n_rank,
        "strategies_different": strategies_different,
        "objective_diff": (bp_eval['objective'] - maxsat_bp_eval['objective']) if maxsat_bp_eval else None,
    }


def defended_tuple(D_state):
    return tuple(sorted([d for d, v in D_state.items() if v]))


def print_single_result(result, actual_counts, graph_type, cfg):
    """Print single scale comparison result"""
    bp_time = result['bp_time_s']
    maxsat_time = result['maxsat_time_s']
    maxsat_status = f"{maxsat_time:.2f}s" if not result['maxsat_timeout'] else "TIMEOUT"

    print(f"  BP+GA:    {bp_time:>7.2f}s | evaluated {result['bp_evaluated']:>4d} | "
          f"obj={result['bp_objective']:>10.4f} | defended {result['bp_defended_count']:>2d} nodes")
    print(f"     BP strategy:      {result['bp_defended']}")
    print(f"  MaxSAT:   {maxsat_status:>7} | "
          f"obj={str(result['maxsat_objective'] or 'N/A'):>10} | "
          f"defended {result['maxsat_defended_count']:>2d} nodes")
    print(f"     MaxSAT strategy:  {result['maxsat_defended']}")

    ga_top_n = result.get("ga_top_n", [])
    if ga_top_n:
        print(f"\n  === GA Top-{len(ga_top_n)} strategies (by objective) ===")
        print(f"  {'Rank':>4} | {'Defended nodes':<30} | {'C_benefit':>10} | {'P_loss':>10} | {'D_cost':>8} | {'Objective':>12}")
        print(f"  {'-'*4}-+-{'-'*30}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}-+-{'-'*12}")
        for rank, (obj, D_state, r) in enumerate(ga_top_n, start=1):
            defended = sorted([d for d, v in D_state.items() if v])
            defended_str = str(defended)
            defended_str = defended_str[:28] + ".." if len(defended_str) > 30 else defended_str
            marker = " *" if rank == 1 else "  "
            print(f"  {marker}{rank:>2} | {defended_str:<30} | "
                  f"{r.get('C_benefit', 0):>10.2f} | {r.get('P_expected_loss', 0):>10.2f} | "
                  f"{r.get('D_cost', 0):>8.2f} | {obj:>12.4f}")

    if not result['maxsat_timeout']:
        if result['maxsat_in_top_n']:
            print(f"\n  *** MaxSAT result in GA Top-{len(ga_top_n)} at rank {result['maxsat_in_top_n_rank']} ***")
        else:
            print(f"\n  *** MaxSAT result NOT in GA Top-{len(ga_top_n)} ***")
    else:
        print(f"\n  (MaxSAT timeout, cannot determine rank)")

    if result['strategies_different']:
        print(f"  *** Strategies different! ***")
    else:
        obj_diff = result['objective_diff']
        if obj_diff is not None:
            print(f"  Same strategy, objective diff: {obj_diff:.6f}")


def print_summary_table(results, graph_type):
    """Print summary table"""
    print("\n" + "=" * 80)
    print("Summary Results")
    print("=" * 80)

    print(f"\n{'P':>4} {'E':>5} {'C':>5} {'D':>4} {'Total':>6} {'BPTime':>8} {'MaxSAT':>8} {'Diff':>6} {'DiffStrat':>8}")
    print("-" * 55)
    for r in results:
        cfg = r["config"]
        if graph_type == "structured":
            actual = r.get("actual_counts", {})
            nP_s, nE_s, nC_s, nD_s = actual.get("nP", "?"), actual.get("nE", "?"), \
                actual.get("nC", "?"), actual.get("nD", "?")
            total = actual.get("total", "?")
        else:
            nP_s, nE_s, nC_s, nD_s = cfg["nP"], cfg["nE"], cfg["nC"], cfg["nD"]
            total = cfg["total"]
        bp_t = r["bp_time_s"]
        ms_t = f"{r['maxsat_time_s']:.2f}s" if not r["maxsat_timeout"] else "TIMEOUT"
        diff = f"{r['objective_diff']:.4f}" if r["objective_diff"] is not None else "N/A"
        different = "YES" if r["strategies_different"] else "no"
        scale_s = f"{cfg['scale']:.1f}" if cfg.get("scale") else "-"
        print(f"{nP_s:>4} {nE_s:>5} {nC_s:>5} {nD_s:>4} {str(total):>6} "
              f"{bp_t:>7.2f}s {ms_t:>8} {diff:>6} {different:>8}")


def main():
    parser = argparse.ArgumentParser(
        description="Scale comparison experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage:
  1. Specify graph type with --graph-type
  2. Pass parameters based on type

structured mode: --graph-type structured --nP <count> [--scale <float>] [--seed N]
  - nP: root node count, nE/nC/nD auto-calculated
  - scale: optional, float like 3.0, actual nP = nP * scale

random mode:   --graph-type random --nP <count> --nE <count> --nC <count> --nD <count>
  - all node counts must be specified manually
        """,
    )
    parser.add_argument("graph_type", choices=["random", "structured"],
                        help="Graph generation type")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", "-o", default="comparison/scale_test.wcnf")
    parser.add_argument("--maxhs-timeout", type=int, default=120)
    parser.add_argument("--population-size", type=int, default=100)
    parser.add_argument("--genmax", type=int, default=50)
    parser.add_argument("--bp-max-iters", type=int, default=50)
    parser.add_argument("--bp-damping", type=float, default=0.5)
    parser.add_argument("--bp-tol", type=float, default=1e-6)
    parser.add_argument("--save-result", action="store_true")

    parser.add_argument("--nP", type=int, default=None,
                        help="P node count (if --scale also specified, actual nP = nP * scale)")
    parser.add_argument("--scale", type=float, default=None,
                        help="Scale factor for all node counts")

    parser.add_argument("--no-level-scaling", action="store_true")
    parser.add_argument("--p-alpha-max", type=float, default=0.8)
    parser.add_argument("--cd-beta-max", type=float, default=0.5)
    parser.add_argument("--cd-floor", type=float, default=0.2)
    parser.add_argument("--max-extra-p-out-per-node", type=int, default=2,
                        help="Max P extra outgoing edges")

    parser.add_argument("--nE", type=int, default=None)
    parser.add_argument("--nC", type=int, default=None)
    parser.add_argument("--nD", type=int, default=None)
    parser.add_argument("--max-children", type=int, default=5,
                        help="Max children per node")
    parser.add_argument("--p-ep", type=float, default=0.25,
                        help="P->E edge probability")

    args = parser.parse_args()

    graph_type = args.graph_type

    if graph_type == "structured":
        if args.nP is None:
            parser.error("--nP is required in structured mode")
        nP = args.nP
        if args.scale is not None:
            nP = int(round(nP * args.scale))
        cfg = {
            "nP": nP,
            "scale": args.scale,
            "total": None,
        }
        configs = [cfg]
        scale_note = f" (from {args.nP} * {args.scale})" if args.scale else ""
        print(f"[structured] nP={nP}{scale_note}")
    else:
        base_nP = args.nP
        base_nE = args.nE
        base_nC = args.nC
        base_nD = args.nD

        if None in (base_nP, base_nE, base_nC, base_nD):
            parser.error("random mode requires --nP --nE --nC --nD")

        if args.scale is not None:
            nP = int(round(base_nP * args.scale))
            nE = int(round(base_nE * args.scale))
            nC = int(round(base_nC * args.scale))
            nD = int(round(base_nD * args.scale))
            scale_note = f" (base {base_nP}/{base_nE}/{base_nC}/{base_nD} * {args.scale})"
        else:
            nP, nE, nC, nD = base_nP, base_nE, base_nC, base_nD
            scale_note = ""

        cfg = {
            "nP": nP, "nE": nE, "nC": nC, "nD": nD,
            "total": nP + nE + nC + nD,
            "scale": args.scale,
        }
        configs = [cfg]
        print(f"[random]{scale_note} -> nP={nP} nE={nE} nC={nC} nD={nD} total={cfg['total']}")

    print("=" * 80)
    print("Scale Comparison Experiment")
    print(f"Graph generator: {'layered-structured' if graph_type == 'structured' else 'random-multi-parent-E-DAG'}")
    print(f"MaxSAT timeout: {args.maxhs_timeout}s")
    print(f"Total {len(configs)} scale levels")
    print("=" * 80)

    results = []
    found_difference = False

    for i, cfg in enumerate(configs):
        try:
            if graph_type == "structured":
                bn = generate_bn_from_root_and_reverse(
                    nP=cfg["nP"],
                    seed=args.seed,
                    extra_p_edge_prob=0.25,
                    max_extra_p_out_per_node=args.max_extra_p_out_per_node,
                    c_parents_per_e_range=(1, 3),
                    max_e_children_per_c=5,
                    max_c_children_per_d=3,
                )
                actual_nP = len(bn["P"])
                actual_nE = len(bn["E"])
                actual_nC = len(bn["C"])
                actual_nD = len(bn["D"])
                total_nodes = actual_nP + actual_nE + actual_nC + actual_nD
                scale_info = f"scale={cfg['scale']:.1f}" if cfg["scale"] else f"nP={actual_nP}"
                print(f"\n[{i+1}/{len(configs)}] Scale: {scale_info} | "
                      f"P={actual_nP} E={actual_nE} C={actual_nC} D={actual_nD} | "
                      f"total={total_nodes}")
                values_table = gen_structured_values(
                    bn, seed=args.seed, fixed_e_probs=FIXED_E_PROBS,
                    p_loss_range=(50, 500), p_benefit_range=(5, 80),
                    c_benefit_range=(10, 50), d_cost_range=(50, 100),
                    use_level_scaling=not args.no_level_scaling,
                    p_alpha_max=args.p_alpha_max,
                    cd_beta_max=args.cd_beta_max,
                    cd_floor=args.cd_floor,
                )
            else:
                bn = generate_bn_dag_multi_pe(
                    nP=cfg["nP"], nE=cfg["nE"],
                    nC=cfg["nC"], nD=cfg["nD"],
                    max_children=args.max_children, p_EP=args.p_ep, seed=args.seed,
                )
                total_nodes = cfg["total"]
                print(f"\n[{i+1}/{len(configs)}] Scale: "
                      f"P={cfg['nP']} E={cfg['nE']} C={cfg['nC']} D={cfg['nD']} | "
                      f"total={total_nodes}")
                values_table = gen_random_values(
                    bn, seed=args.seed, fixed_e_probs=FIXED_E_PROBS,
                    p_loss_range=(50, 500), p_benefit_range=(5, 80),
                    c_benefit_range=(10, 50), d_cost_range=(50, 100),
                )
        except Exception as e:
            print(f"  [SKIP] Graph generation failed: {e}")
            continue

        try:
            result = run_single_comparison(bn, values_table, args)
        except Exception as e:
            print(f"  [SKIP] Comparison failed: {e}")
            continue

        print_single_result(result, None, graph_type, cfg)

        if result['strategies_different']:
            found_difference = True

        result["config"] = cfg
        if graph_type == "structured":
            result["actual_counts"] = {
                "nP": actual_nP, "nE": actual_nE,
                "nC": actual_nC, "nD": actual_nD,
                "total": total_nodes,
            }
        results.append(result)

    print_summary_table(results, graph_type)

    if found_difference:
        print("\nConclusion: In tested scale range, strategies differed between two methods")
    else:
        print("\nConclusion: In tested scale range, strategies were always the same")

    if args.save_result:
        result_path = "comparison/scale_comparison_results.json"
        save_data = {
            "graph_type": graph_type,
            "seed": args.seed,
            "maxhs_timeout": args.maxhs_timeout,
            "population_size": args.population_size,
            "genmax": args.genmax,
            "bp_max_iters": args.bp_max_iters,
            "bp_damping": args.bp_damping,
            "bp_tol": args.bp_tol,
            "configs": configs,
            "results": results,
        }
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
        print(f"\nResult saved to: {result_path}")

    return results


if __name__ == "__main__":
    main()
