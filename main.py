"""Main entry: generate attack graph -> export WCNF -> MaxHS solve -> objective analysis"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from graph2sat.graph2sat import export_to_wcnf

MAXHS_BIN = "MaxHS/build/release/bin/maxhs"

FIXED_E_PROBS = [
    0.02, 0.05, 0.10, 0.12, 0.15, 0.18,
    0.20, 0.25, 0.30, 0.32, 0.35, 0.38,
    0.40, 0.45, 0.50, 0.55, 0.60, 0.65,
    0.70, 0.75, 0.80, 0.85, 0.90, 0.95,
]


def _get_cnf_converter(sat_type):
    if sat_type == "pro":
        from graph2sat.graph2sat import bn_to_maxsat_cnf
    else:
        from graph2sat.garaph2sat_without_pro import bn_to_maxsat_cnf
    return bn_to_maxsat_cnf


def _solve_maxsat(wcnf_path, maxhs_bin, timeout=None, extra_args=None):
    """Call MaxHS to solve WCNF file, return solver output. Returns empty on timeout."""
    cmd = [maxhs_bin]
    if extra_args:
        cmd.extend(extra_args)
    cmd.append(str(wcnf_path))

    kwargs = {"capture_output": True, "text": True}
    if timeout:
        kwargs["timeout"] = timeout

    try:
        result = subprocess.run(cmd, **kwargs)
        return result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        print(f"[WARNING] MaxHS timeout ({timeout}s), returning empty solution")
        return ""


def _run_pipeline(args, bn, values_table, return_stats=False):
    """Pipeline: CNF conversion -> export WCNF -> MaxHS solve -> parse -> objective calculation."""
    maxhs_bin = args.maxhs_bin
    bn_to_maxsat_cnf = _get_cnf_converter(args.sat)
    cnf_data = bn_to_maxsat_cnf(bn, values_table, initial_true_nodes=None, initial_false_nodes=None)

    wcnf_path = Path(args.output).resolve()
    export_to_wcnf(cnf_data, str(wcnf_path))
    print(f"[1/4] WCNF generated: {wcnf_path}")

    from collections import defaultdict
    from rusult_analysis.one_analysisi import build_value_index
    vindex = build_value_index(values_table)
    node_type = bn["node_type"]
    children = defaultdict(list)
    for u, v in bn["edges"]:
        children[u].append(v)

    c_benefit_sum = sum(
        (vindex.get(c, {}).get("C_benefit") or 0.0)
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

    if return_stats:
        return {
            "c_benefit_sum": c_benefit_sum,
            "p_loss_sum": p_loss_sum,
            "baseline_objective": baseline_objective,
            "wcnf_path": str(wcnf_path),
            "n_vars": len(cnf_data["var_map"]),
        }

    if args.no_solve:
        print(f"[2/4] Skip solving (--no-solve)")
        print(f"[3/4] Skip analysis (--no-solve)")
        return

    print(f"[2/4] Running MaxHS solver ...")
    extra = [
        "-printSoln",
        "-printBstSoln",
        "-verb=0",
        f"-cpu-lim={args.maxhs_timeout}",
    ]
    maxhs_out = _solve_maxsat(wcnf_path, maxhs_bin, timeout=args.maxhs_timeout + 5, extra_args=extra)
    print(f"[2/4] MaxHS solving complete")

    from rusult_analysis.one_analysisi import (
        parse_maxhs_solution_str,
        interpret_solution,
        compute_objective_from_solution,
    )

    n_vars = len(cnf_data["var_map"])
    assignment = parse_maxhs_solution_str(maxhs_out, n_vars)

    if not assignment:
        print("[3/4] No valid solution found (assignment empty)", file=sys.stderr)
        return

    node_state, node_state_by_type = interpret_solution(bn, cnf_data, assignment)
    print(f"[3/4] Solution parsed, {len(assignment)} variable assignments")

    if args.show_p_state:
        print("\n=== P Node States ===")
        from rusult_analysis.one_analysisi import build_value_index, build_adjacency
        vindex = build_value_index(values_table)
        parents, children = build_adjacency(bn)
        node_type = bn["node_type"]
        p_states_printed = []
        for p, active_p in node_state_by_type.get("P", {}).items():
            p_info = vindex.get(p, {})
            p_loss = p_info.get("P_loss", 0.0)
            p_benefit = p_info.get("P_benefit", 0.0)

            if active_p:
                e_children = [
                    e for e in children.get(p, [])
                    if node_type.get(e) == "E" and node_state.get(e, False)
                ]
                prob_not_comp = 1.0
                for e in e_children:
                    row_e = vindex.get(e, {})
                    prob_e = row_e.get("E_prob") or 0.0
                    prob_not_comp *= (1.0 - prob_e)
                p_comp = 1.0 - prob_not_comp if e_children else 0.0
                expected_loss = p_comp * p_loss
                expected_gain = (1 - p_comp) * p_benefit
            else:
                p_comp = 0.0
                expected_loss = 0.0
                expected_gain = 0.0

            p_states_printed.append({
                "name": p,
                "active": active_p,
                "P_loss": p_loss,
                "P_benefit": p_benefit,
                "P_comp": p_comp,
                "expected_loss": expected_loss,
                "expected_gain": expected_gain,
            })
        p_states_printed.sort(key=lambda x: x["expected_loss"], reverse=True)
        print(f"{'Node':<8} {'Active':<6} {'P_loss':>10} {'P_benefit':>10} {'P_comp':>8} {'ExpLoss':>12} {'ExpGain':>12}")
        print("-" * 70)
        for item in p_states_printed:
            print(f"{item['name']:<8} {'T' if item['active'] else 'F':<6} "
                  f"{item['P_loss']:>10.2f} {item['P_benefit']:>10.2f} "
                  f"{item['P_comp']:>8.4f} {item['expected_loss']:>12.2f} {item['expected_gain']:>12.2f}")
        total_loss = sum(item["expected_loss"] for item in p_states_printed)
        total_gain = sum(item["expected_gain"] for item in p_states_printed)
        print("-" * 70)
        print(f"{'Total':<8} {'':6} {'':>10} {'':>10} {'':>8} {total_loss:>12.2f} {total_gain:>12.2f}")
        print()

    obj = compute_objective_from_solution(bn, values_table, node_state, node_state_by_type)
    print(f"[4/4] Objective calculation complete:")
    print(f"       C_benefit       = {obj['C_benefit']:.4f}")
    print(f"       P_expected_loss = {obj['P_expected_loss']:.4f}")
    print(f"       D_cost          = {obj['D_cost']:.4f}")
    print(f"       ─────────────────────")
    print(f"       objective       = {obj['objective']:.4f}")

    if args.save_result:
        result_path = wcnf_path.with_suffix(".result.json")
        result_data = {
            "objective": obj["objective"],
            "C_benefit": obj["C_benefit"],
            "P_expected_loss": obj["P_expected_loss"],
            "D_cost": obj["D_cost"],
            "assignment": {str(k): v for k, v in assignment.items()},
            "node_state": node_state,
        }
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)
        print(f"       Result saved to: {result_path}")


def parse_maxhs_solution_str(maxhs_output, n_vars):
    """Parse solution vector from MaxHS output (stdout+stderr).
    Supports two formats:
      1) DIMACS:   v 1 -2 3 0
      2) bit string: v 11100101...
    """
    assignment = {}
    for line in maxhs_output.splitlines():
        line = line.strip()
        if not line.startswith("v "):
            continue

        data = line[2:].strip()

        if set(data) <= {"0", "1"} and len(data) > 0:
            for i, ch in enumerate(data, start=1):
                if i > n_vars:
                    break
                assignment[i] = (ch == "1")
        else:
            for tok in data.split():
                if tok == "0":
                    continue
                lit = int(tok)
                v = abs(lit)
                if v > n_vars:
                    continue
                assignment[v] = (lit > 0)

    return assignment


def run_random(args):
    from generate_graph.random_graph import generate_bn_dag_multi_pe
    from generate_graph.numerical_generation import generate_node_values

    bn = generate_bn_dag_multi_pe(
        nP=args.nP,
        nE=args.nE,
        nC=args.nC,
        nD=args.nD,
        max_children=args.max_children,
        p_EP=args.p_EP,
        seed=args.seed,
    )

    values_table = generate_node_values(
        bn,
        seed=args.seed,
        fixed_e_probs=FIXED_E_PROBS,
        p_loss_range=(args.p_loss_lo, args.p_loss_hi),
        p_benefit_range=(args.p_benefit_lo, args.p_benefit_hi),
        c_benefit_range=(args.c_benefit_lo, args.c_benefit_hi),
        d_cost_range=(args.d_cost_lo, args.d_cost_hi),
    )

    return _run_pipeline(args, bn, values_table)


def run_structured(args):
    from generate_graph.structured_graph import generate_bn_from_root_and_reverse
    from generate_graph.number_generation import generate_node_values

    bn = generate_bn_from_root_and_reverse(
        nP=args.nP,
        seed=args.seed,
        extra_p_edge_prob=args.extra_p_edge_prob,
        max_extra_p_out_per_node=args.max_extra_p_out_per_node,
        c_parents_per_e_range=(args.c_parents_min, args.c_parents_max),
        max_e_children_per_c=args.max_e_children_per_c,
        max_c_children_per_d=args.max_c_children_per_d,
        nC=args.nC,
        nD=args.nD,
    )

    values_table = generate_node_values(
        bn,
        seed=args.seed,
        fixed_e_probs=FIXED_E_PROBS,
        p_loss_range=(args.p_loss_lo, args.p_loss_hi),
        p_benefit_range=(args.p_benefit_lo, args.p_benefit_hi),
        c_benefit_range=(args.c_benefit_lo, args.c_benefit_hi),
        d_cost_range=(args.d_cost_lo, args.d_cost_hi),
        use_level_scaling=args.use_level_scaling,
        p_alpha_max=args.p_alpha_max,
        cd_beta_max=args.cd_beta_max,
        cd_floor=args.cd_floor,
    )

    return _run_pipeline(args, bn, values_table)


def main():
    parser = argparse.ArgumentParser(
        description="Generate attack graph -> export WCNF -> MaxHS solve -> objective analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--type", "-t", choices=["random", "structured"], required=True,
        help="Graph generation type: random=random graph, structured=structured graph"
    )
    parser.add_argument(
        "--sat", choices=["pro", "without_pro"], default="pro",
        help="SAT conversion type: pro=graph2sat.graph2sat (with P_benefit/noisy-OR), "
             "without_pro=graph2sat.garaph2sat_without_pro (without)"
    )
    parser.add_argument(
        "--output", "-o", default="attack_graph.wcnf",
        help="WCNF output filename"
    )
    parser.add_argument(
        "--seed", "-s", type=int, default=42,
        help="Random seed"
    )

    parser.add_argument(
        "--no-solve", action="store_true",
        help="Only generate WCNF file, skip MaxHS solving"
    )
    parser.add_argument(
        "--maxhs-timeout", type=int, default=300,
        help="MaxHS solver timeout in seconds"
    )
    parser.add_argument(
        "--maxhs-verbose", action="store_true",
        help="Show MaxHS solving progress"
    )
    parser.add_argument(
        "--save-result", action="store_true",
        help="Save solving result (variable assignments & objective) as JSON"
    )
    parser.add_argument(
        "--show-p-state", action="store_true",
        help="Print activation status and calculation details for each P node"
    )
    parser.add_argument(
        "--maxhs-bin", default=MAXHS_BIN,
        help="MaxHS executable path"
    )

    parser.add_argument("--nP", type=int, default=100, help="Number of P nodes")
    parser.add_argument("--nC", type=int, default=None, help="Number of C nodes (optional for structured)")
    parser.add_argument("--nD", type=int, default=None, help="Number of D nodes (optional for structured)")

    parser.add_argument("--p-loss-lo", type=float, default=50, help="P_loss lower bound")
    parser.add_argument("--p-loss-hi", type=float, default=500, help="P_loss upper bound")
    parser.add_argument("--p-benefit-lo", type=float, default=5, help="P_benefit lower bound")
    parser.add_argument("--p-benefit-hi", type=float, default=80, help="P_benefit upper bound")
    parser.add_argument("--c-benefit-lo", type=float, default=10, help="C_benefit lower bound")
    parser.add_argument("--c-benefit-hi", type=float, default=50, help="C_benefit upper bound")
    parser.add_argument("--d-cost-lo", type=float, default=50, help="D_cost lower bound")
    parser.add_argument("--d-cost-hi", type=float, default=100, help="D_cost upper bound")

    parser.add_argument("--nE", type=int, default=None, help="Number of E nodes (required for random)")
    parser.add_argument("--max-children", type=int, default=5, help="Maximum out-degree")
    parser.add_argument("--p-EP", type=float, default=0.25, help="E->P edge probability")

    parser.add_argument("--extra-p-edge-prob", type=float, default=0.25, help="Extra P->P edge probability")
    parser.add_argument("--max-extra-p-out-per-node", type=int, default=2, help="Maximum extra P out-degree per node")
    parser.add_argument("--c-parents-min", type=int, default=1, help="Minimum C parents per E")
    parser.add_argument("--c-parents-max", type=int, default=3, help="Maximum C parents per E")
    parser.add_argument("--max-e-children-per-c", type=int, default=5, help="Maximum E children per C")
    parser.add_argument("--max-c-children-per-d", type=int, default=3, help="Maximum C children per D")
    parser.add_argument("--no-level-scaling", action="store_true", help="Disable level scaling")
    parser.add_argument("--p-alpha-max", type=float, default=0.8, help="P_loss positive correlation max scale")
    parser.add_argument("--cd-beta-max", type=float, default=0.5, help="C/D negative correlation max reduction")
    parser.add_argument("--cd-floor", type=float, default=0.2, help="C/D negative correlation floor")

    args = parser.parse_args()

    if args.type == "random":
        if args.nE is None:
            parser.error("--nE is required in random mode")
        if args.nP < 2:
            parser.error("--nP must be at least 2")
        if args.nE < args.nP:
            parser.error(f"--nE({args.nE}) must be >= --nP({args.nP})")

    if args.type == "structured" and args.c_parents_max < args.c_parents_min:
        parser.error(f"--c-parents-max({args.c_parents_max}) must be >= --c-parents-min({args.c_parents_min})")

    args.use_level_scaling = not args.no_level_scaling

    if args.type == "random":
        run_random(args)
    else:
        run_structured(args)


if __name__ == "__main__":
    main()
