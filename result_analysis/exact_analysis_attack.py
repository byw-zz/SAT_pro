"""MulVAL attack graph defense strategy exact analysis using pgmpy Variable Elimination."""

import argparse
import os
import sys
import time
from collections import defaultdict
from itertools import product

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

from mulval_graph import (
    csv_to_bn_dict,
    DEFENSE_PLAN,
    add_defenses_to_bn,
    specified_p_loss,
    specified_c_benefit,
    specified_d_cost,
    cve_to_exploitability,
    generate_node_values_attackgraph_with_specified_values_per_node,
)


def build_c_cpd(node, forced_zero_C):
    """C node CPD: root node with prior"""
    values = [[1.0], [0.0]] if node in forced_zero_C else [[0.0], [1.0]]
    return TabularCPD(
        variable=node, variable_card=2, values=values, state_names={node: [0, 1]},
    )


def build_e_cpd(node, parents):
    """E node CPD: strict AND"""
    if not parents:
        return TabularCPD(
            variable=node, variable_card=2, values=[[1.0], [0.0]], state_names={node: [0, 1]},
        )
    parent_assignments = list(product([0, 1], repeat=len(parents)))
    row0, row1 = [], []
    for assign in parent_assignments:
        p1 = 1.0 if all(x == 1 for x in assign) else 0.0
        row0.append(1.0 - p1)
        row1.append(p1)
    state_names = {node: [0, 1]}
    for p in parents:
        state_names[p] = [0, 1]
    return TabularCPD(
        variable=node, variable_card=2, values=[row0, row1],
        evidence=parents, evidence_card=[2] * len(parents), state_names=state_names,
    )


def build_p_cpd(node, parents, vindex):
    """P node CPD: noisy-OR"""
    if not parents:
        return TabularCPD(
            variable=node, variable_card=2, values=[[0.0], [1.0]], state_names={node: [0, 1]},
        )
    parent_assignments = list(product([0, 1], repeat=len(parents)))
    row0, row1 = [], []
    for assign in parent_assignments:
        prob_not_active = 1.0
        for e_node, e_val in zip(parents, assign):
            e_prob = (vindex.get(e_node, {}) or {}).get("E_prob", 0.0) or 0.0
            prob_not_active *= (1.0 - e_prob) ** e_val
        p1 = max(0.0, min(1.0, 1.0 - prob_not_active))
        row0.append(1.0 - p1)
        row1.append(p1)
    state_names = {node: [0, 1]}
    for p in parents:
        state_names[p] = [0, 1]
    return TabularCPD(
        variable=node, variable_card=2, values=[row0, row1],
        evidence=parents, evidence_card=[2] * len(parents), state_names=state_names,
    )


def build_pgmpy_model(node_type, parents_f, vindex, forced_zero_C, all_remaining_nodes):
    """Build pgmpy DiscreteBayesianNetwork"""
    node_set = set(all_remaining_nodes)
    edges, cpds = [], []

    model = DiscreteBayesianNetwork()
    model.add_nodes_from(all_remaining_nodes)

    for child in all_remaining_nodes:
        for parent in parents_f.get(child, []):
            if parent in node_set:
                edges.append((parent, child))
    if edges:
        model.add_edges_from(edges)

    for node in all_remaining_nodes:
        ntype = node_type.get(node, "")
        parents = [p for p in parents_f.get(node, []) if p in node_set]
        if ntype == "C":
            cpd = build_c_cpd(node, forced_zero_C)
        elif ntype == "E":
            cpd = build_e_cpd(node, parents)
        elif ntype == "P":
            cpd = build_p_cpd(node, parents, vindex)
        else:
            continue
        cpds.append(cpd)

    model.add_cpds(*cpds)
    model.check_model()
    return model


def ve_exact_inference(node_type, parents_f, vindex, query_nodes,
                       forced_zero_C, all_remaining_nodes):
    """Run pgmpy VariableElimination"""
    model = build_pgmpy_model(node_type, parents_f, vindex, forced_zero_C, all_remaining_nodes)
    infer = VariableElimination(model)
    marginals = {}
    for q in query_nodes:
        result = infer.query(variables=[q], show_progress=False)
        marginals[q] = float(result.values[1])
    return marginals


def run_exact_analysis(bn, values_table, D_state=None):
    """Execute exact probability analysis."""
    node_type = bn["node_type"]

    reversed_edges = [(v, u) for u, v in bn["edges"]]
    parents = defaultdict(list)
    children = defaultdict(list)
    for u, v in reversed_edges:
        parents[v].append(u)
        children[u].append(v)

    if D_state is None:
        D_state = {d: False for d in bn.get("D", [])}

    forced_zero_C = set()
    for d, defended in D_state.items():
        if defended:
            for c in children.get(d, []):
                forced_zero_C.add(c)

    removed = set(bn.get("D", []))
    remaining_set = set(bn["P"] + bn["E"] + bn["C"] + bn["D"]) - removed

    rem_P = [p for p in bn["P"] if p in remaining_set]
    rem_E = [e for e in bn["E"] if e in remaining_set]
    rem_C = [c for c in bn["C"] if c in remaining_set]

    filtered_edges = [(u, v) for u, v in reversed_edges if u in remaining_set and v in remaining_set]

    parents_f = defaultdict(list)
    for u, v in filtered_edges:
        parents_f[v].append(u)

    vindex = {row["node"]: row for row in values_table}

    t0 = time.time()
    marginals = ve_exact_inference(
        node_type=node_type, parents_f=parents_f, vindex=vindex, query_nodes=rem_P,
        forced_zero_C=forced_zero_C, all_remaining_nodes=list(remaining_set),
    )
    ve_time = time.time() - t0

    total_C_benefit = 0.0
    total_P_expected_loss = 0.0
    total_P_expected_gain = 0.0
    total_D_cost = 0.0

    for c in rem_C:
        if c in forced_zero_C:
            continue
        row = vindex.get(c, {})
        total_C_benefit += row.get("C_benefit", 0.0) or 0.0

    for p in rem_P:
        row_p = vindex.get(p, {})
        p_loss = row_p.get("P_loss", 0.0) or 0.0
        p_benefit = row_p.get("P_benefit", 0.0) or 0.0
        p_marg = marginals.get(p, 0.0)
        total_P_expected_loss += p_marg * p_loss
        total_P_expected_gain += (1.0 - p_marg) * p_benefit

    for d, defended in D_state.items():
        if defended:
            row = vindex.get(d, {})
            total_D_cost += row.get("D_cost", 0.0) or 0.0

    objective = total_C_benefit - total_P_expected_loss - total_D_cost

    return {
        "objective": objective,
        "C_benefit": total_C_benefit,
        "P_expected_loss": total_P_expected_loss,
        "P_expected_gain": total_P_expected_gain,
        "D_cost": total_D_cost,
        "P_marginals": dict(marginals),
        "remaining_nodes": {"P": rem_P, "E": rem_E, "C": rem_C, "D": []},
        "D_state": dict(D_state),
        "forced_zero_C": list(forced_zero_C),
        "marginals": dict(marginals),
        "_ve_time_ms": ve_time * 1000,
    }


def find_best_defense_strategy(bn, values_table, max_enum=1 << 15):
    """Enumerate all defense combinations to find best strategy"""
    D_nodes = bn.get("D", [])
    nD = len(D_nodes)
    n_strategies = 1 << nD

    if n_strategies > max_enum:
        print(f"[WARNING] nD={nD}, strategies={n_strategies} > {max_enum},"
              f" enumeration may be slow", flush=True)

    best_obj = float("-inf")
    best_D_state = None
    best_result = None
    all_results = []

    t_start = time.time()
    for bits in range(n_strategies):
        D_state = {d: bool((bits >> i) & 1) for i, d in enumerate(D_nodes)}
        result = run_exact_analysis(bn, values_table, D_state=D_state)
        all_results.append(result)

        if result["objective"] > best_obj:
            best_obj = result["objective"]
            best_D_state = D_state
            best_result = result

        if (bits + 1) % 100 == 0:
            elapsed = time.time() - t_start
            rate = (bits + 1) / elapsed if elapsed > 0 else 0
            eta = (n_strategies - bits - 1) / rate if rate > 0 else 0
            print(f"  Progress {bits+1}/{n_strategies} "
                  f"({elapsed:.1f}s elapsed, ~{eta:.1f}s remaining)", flush=True)

    total_time = time.time() - t_start
    return {
        "best_objective": best_obj,
        "best_D_state": best_D_state,
        "best_result": best_result,
        "all_results": all_results,
        "total_strategies": n_strategies,
        "total_time_s": total_time,
    }


def print_comparison_table(all_results):
    print("\n" + "=" * 90)
    print(f"{'#':>3} | {'D_state':<30} | {'C_benefit':>10} | {'P_loss':>10} | {'D_cost':>8} | {'Objective':>12}")
    print("-" * 90)
    sorted_results = sorted(enumerate(all_results), key=lambda x: x[1]["objective"], reverse=True)
    for rank, (idx, r) in enumerate(sorted_results):
        d_str = str({d: int(v) for d, v in r["D_state"].items()})
        d_str = d_str[:28] + ".." if len(d_str) > 30 else d_str
        marker = " *" if rank == 0 else "  "
        print(f"{marker}{idx:>2} | {d_str:<30} | "
              f"{r['C_benefit']:>10.2f} | {r['P_expected_loss']:>10.2f} | "
              f"{r['D_cost']:>8.2f} | {r['objective']:>12.4f}")
    print("=" * 90)


def main():
    parser = argparse.ArgumentParser(
        description="MulVAL attack graph defense strategy exact analysis (pgmpy Variable Elimination)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--arcs", required=True, help="ARCS.CSV file path")
    parser.add_argument("--vertices", required=True, help="VERTICES.CSV file path")
    parser.add_argument(
        "--arcs-order",
        choices=["dest_src", "src_dest"],
        default="src_dest",
        help="ARCS CSV column order"
    )
    parser.add_argument(
        "--initial-true",
        default="C4,P2",
        help="Nodes initially true (comma-separated)"
    )
    parser.add_argument(
        "--enumerate", "-e",
        action="store_true",
        help="Enumerate all D strategies (2^nD, suggest nD <= 15)"
    )
    parser.add_argument(
        "--max-enum",
        type=int, default=1 << 15,
        help="Enumeration limit"
    )

    args = parser.parse_args()

    print(f"[1/3] Reading ARCS: {args.arcs}")
    print(f"[1/3] Reading VERTICES: {args.vertices}")
    bn = csv_to_bn_dict(args.vertices, args.arcs, arcs_order=args.arcs_order)
    print(f"[1/3] Initial nodes: P={len(bn['P'])}, E={len(bn['E'])}, C={len(bn['C'])}, D={len(bn['D'])}, edges={len(bn['edges'])}")

    print(f"[1/3] Adding defense nodes (DEFENSE_PLAN)...")
    try:
        add_defenses_to_bn(bn, DEFENSE_PLAN)
    except Exception as e:
        print(f"[ERROR] Failed to add defense nodes: {e}", file=sys.stderr)
        sys.exit(1)
    print(f"[1/3] After adding: P={len(bn['P'])}, E={len(bn['E'])}, C={len(bn['C'])}, D={len(bn['D'])}")

    print(f"[2/3] Generating node values...")
    values_table = generate_node_values_attackgraph_with_specified_values_per_node(
        bn,
        specified_p_loss=specified_p_loss,
        specified_c_benefit=specified_c_benefit,
        specified_d_cost=specified_d_cost,
        cve_to_exploitability=cve_to_exploitability,
        e_default_no_cve=1.0,
    )

    print("=" * 65)
    print(f"Initial true nodes: {args.initial_true}")
    print(f"P={len(bn['P'])}  E={len(bn['E'])}  C={len(bn['C'])}  D={len(bn['D'])}  edges={len(bn['edges'])}")
    print("=" * 65)

    print(f"[3/3] Starting analysis...")
    if args.enumerate:
        nD = len(bn.get("D", []))
        n_strategies = 1 << nD
        print(f"\n[ENUMERATION MODE] nD={nD}, strategies=2^{nD}={n_strategies}", flush=True)

        enum_result = find_best_defense_strategy(bn, values_table, max_enum=args.max_enum)

        print(f"\nSearch complete! Total time: {enum_result['total_time_s']:.2f}s")
        print_comparison_table(enum_result["all_results"])

        br = enum_result["best_result"]
        bd = enum_result["best_D_state"]
        defended_nodes = [d for d, v in bd.items() if v]

        print(f"\n* Best defense combination ({len(defended_nodes)} nodes):")
        print(f"  Defended nodes: {sorted(defended_nodes)}")
        print(f"  C_benefit        = {br['C_benefit']:.4f}")
        print(f"  P_expected_loss  = {br['P_expected_loss']:.4f}")
        print(f"  P_expected_gain  = {br['P_expected_gain']:.4f}")
        print(f"  D_cost           = {br['D_cost']:.4f}")
        print(f"  objective        = {br['objective']:.4f}")

        if br.get("_ve_time_ms"):
            print(f"\nVE single run: {br['_ve_time_ms']:.2f}ms")
        print(f"Forced-zero C nodes: {sorted(br['forced_zero_C'])}")

        print(f"\nP node marginal probabilities P(P=1):")
        for p, prob in sorted(br["P_marginals"].items()):
            print(f"  {p}: {prob:.6f}")
    else:
        result = run_exact_analysis(bn, values_table, D_state=None)

        defended = [d for d, v in result["D_state"].items() if v]
        print(f"\nD state (default all undefended: D=0):")
        print(f"  Defended: {defended if defended else 'none'}")

        print(f"\nObjective:")
        print(f"  C_benefit        = {result['C_benefit']:.4f}")
        print(f"  P_expected_loss  = {result['P_expected_loss']:.4f}")
        print(f"  P_expected_gain  = {result['P_expected_gain']:.4f}")
        print(f"  D_cost           = {result['D_cost']:.4f}")
        print(f"  objective        = {result['objective']:.4f}")

        if result.get("_ve_time_ms"):
            print(f"\nVE inference time: {result['_ve_time_ms']:.2f}ms")

        if result["remaining_nodes"]["P"]:
            print(f"\nP node marginal probabilities P(P=1):")
            for p, prob in sorted(result["P_marginals"].items()):
                print(f"  {p}: {prob:.6f}")


if __name__ == "__main__":
    main()
