"""Genetic algorithm for defense strategy optimization."""

import argparse
import os
import random
import sys
import time
from collections import defaultdict
from itertools import product

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generate_graph.random_graph import generate_bn_dag_multi_pe
from generate_graph.structured_graph import generate_bn_from_root_and_reverse
from generate_graph.numerical_generation import generate_node_values as gen_random_values
from generate_graph.number_generation import generate_node_values as gen_structured_values

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination


from generate_graph.config import FIXED_E_PROBS


def build_c_cpd(node, forced_zero_C):
    """C node CPD"""
    values = [[1.0], [0.0]] if node in forced_zero_C else [[0.0], [1.0]]
    return TabularCPD(variable=node, variable_card=2, values=values, state_names={node: [0, 1]})


def build_e_cpd(node, parents):
    """E node CPD: strict AND"""
    if not parents:
        return TabularCPD(variable=node, variable_card=2, values=[[1.0], [0.0]], state_names={node: [0, 1]})
    parent_assignments = list(product([0, 1], repeat=len(parents)))
    row0, row1 = [], []
    for assign in parent_assignments:
        p1 = 1.0 if all(x == 1 for x in assign) else 0.0
        row0.append(1.0 - p1)
        row1.append(p1)
    state_names = {node: [0, 1]}
    for p in parents:
        state_names[p] = [0, 1]
    return TabularCPD(variable=node, variable_card=2, values=[row0, row1],
                     evidence=parents, evidence_card=[2] * len(parents), state_names=state_names)


def build_p_cpd(node, parents, vindex):
    """P node CPD: noisy-OR"""
    if not parents:
        return TabularCPD(variable=node, variable_card=2, values=[[0.0], [1.0]], state_names={node: [0, 1]})
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
    return TabularCPD(variable=node, variable_card=2, values=[row0, row1],
                     evidence=parents, evidence_card=[2] * len(parents), state_names=state_names)


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


def ve_exact_inference(node_type, parents_f, vindex, query_nodes, forced_zero_C, all_remaining_nodes):
    """Run pgmpy VariableElimination"""
    model = build_pgmpy_model(node_type, parents_f, vindex, forced_zero_C, all_remaining_nodes)
    infer = VariableElimination(model)
    marginals = {}
    for q in query_nodes:
        result = infer.query(variables=[q], show_progress=False)
        marginals[q] = float(result.values[1])
    return marginals


def run_exact_analysis(bn, values_table, D_state=None):
    """Execute exact probability analysis"""
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
        node_type, parents_f, vindex, rem_P, forced_zero_C, list(remaining_set))
    ve_time = time.time() - t0
    total_C_benefit = total_P_expected_loss = total_D_cost = 0.0
    for c in rem_C:
        if c in forced_zero_C:
            continue
        row = vindex.get(c, {})
        total_C_benefit += row.get("C_benefit", 0.0) or 0.0
    for p in rem_P:
        row_p = vindex.get(p, {})
        p_loss = row_p.get("P_loss", 0.0) or 0.0
        p_marg = marginals.get(p, 0.0)
        total_P_expected_loss += p_marg * p_loss
    for d, defended in D_state.items():
        if defended:
            row = vindex.get(d, {})
            total_D_cost += row.get("D_cost", 0.0) or 0.0
    objective = total_C_benefit - total_P_expected_loss - total_D_cost
    return {
        "objective": objective,
        "C_benefit": total_C_benefit,
        "P_expected_loss": total_P_expected_loss,
        "D_cost": total_D_cost,
        "P_marginals": dict(marginals),
        "remaining_nodes": {"P": rem_P, "E": rem_E, "C": rem_C, "D": []},
        "D_state": dict(D_state),
        "forced_zero_C": list(forced_zero_C),
        "marginals": dict(marginals),
        "_ve_time_ms": ve_time * 1000,
    }


def binary_tournament_select(evaluated_population, rng):
    """Binary tournament selection"""
    if not evaluated_population:
        raise ValueError("Empty population")
    a = rng.choice(evaluated_population)
    b = rng.choice(evaluated_population)
    return a if a["result"]["objective"] >= b["result"]["objective"] else b


def single_point_crossover(parent1, parent2, crossover_prob, rng):
    if len(parent1) != len(parent2):
        raise ValueError("Parent length mismatch")
    n = len(parent1)
    if n <= 1 or rng.random() >= crossover_prob:
        return tuple(parent1), tuple(parent2)
    point = rng.randint(1, n - 1)
    return (tuple(list(parent1[:point]) + list(parent2[point:])),
           tuple(list(parent2[:point]) + list(parent1[point:])))

def mutate_bits(bits, mutation_prob, rng):
    return tuple(1 - b if rng.random() < mutation_prob else b for b in bits)


def find_best_defense_strategy_ga(
    bn, values_table,
    population_size=100, genmax=50,
    crossover_prob=0.8, mutation_prob=0.01, seed=42,
):
    """Search best defense strategy using GA"""
    rng = random.Random(seed)
    D_nodes = bn.get("D", [])
    nD = len(D_nodes)
    if nD == 0:
        result = run_exact_analysis(bn, values_table, D_state={})
        return {"best_objective": result["objective"], "best_D_state": {},
                "best_result": result, "final_population": [], "history": [result],
                "unique_evaluated": 1, "total_time_s": 0.0}
    cache = {}

    def evaluate(bits):
        key = tuple(int(x) for x in bits)
        if key not in cache:
            D_state = {d: bool(key[i]) for i, d in enumerate(D_nodes)}
            cache[key] = run_exact_analysis(bn, values_table, D_state=D_state)
        return {"genes": key,
                "D_state": {d: bool(key[i]) for i, d in enumerate(D_nodes)},
                "result": cache[key]}

    population = [tuple(rng.randint(0, 1) for _ in range(nD)) for _ in range(population_size)]
    history = []
    best = None
    t_start = time.time()
    for gen in range(genmax + 1):
        evaluated = [evaluate(bits) for bits in population]
        history.extend(item["result"] for item in evaluated)
        gen_best = max(evaluated, key=lambda x: x["result"]["objective"])
        if best is None or gen_best["result"]["objective"] > best["result"]["objective"]:
            best = gen_best
        mating_pool_size = max(1, population_size // 2)
        mating_pool = [binary_tournament_select(evaluated, rng) for _ in range(mating_pool_size)]
        offspring = []
        while len(offspring) < mating_pool_size:
            p1 = rng.choice(mating_pool)["genes"]
            p2 = rng.choice(mating_pool)["genes"]
            c1, c2 = single_point_crossover(p1, p2, crossover_prob, rng)
            offspring.append(mutate_bits(c1, mutation_prob, rng))
            if len(offspring) >= mating_pool_size:
                break
            offspring.append(mutate_bits(c2, mutation_prob, rng))
        population = [ind["genes"] for ind in mating_pool] + offspring[:mating_pool_size]
        if (gen + 1) % 10 == 0 or gen == genmax:
            print(f"  [GA] gen {gen:>3}/{genmax} | best={best['result']['objective']:.4f} | evaluated={len(cache)}", flush=True)
    total_time = time.time() - t_start
    return {
        "best_objective": best["result"]["objective"],
        "best_D_state": best["D_state"],
        "best_result": best["result"],
        "unique_evaluated": len(cache),
        "total_time_s": total_time,
    }


def main():
    parser = argparse.ArgumentParser(description="GA defense strategy optimization")
    parser.add_argument("--type", "-t", choices=["random", "structured"], required=True)
    parser.add_argument("--seed", "-s", type=int, default=42)
    parser.add_argument("--nP", type=int, default=5)
    parser.add_argument("--nC", type=int, default=None)
    parser.add_argument("--nD", type=int, default=None)
    parser.add_argument("--nE", type=int, default=None)
    parser.add_argument("--max-children", type=int, default=5)
    parser.add_argument("--p-EP", type=float, default=0.25)
    parser.add_argument("--extra-p-edge-prob", type=float, default=0.25)
    parser.add_argument("--max-extra-p-out-per-node", type=int, default=2)
    parser.add_argument("--c-parents-min", type=int, default=1)
    parser.add_argument("--c-parents-max", type=int, default=3)
    parser.add_argument("--max-e-children-per-c", type=int, default=5)
    parser.add_argument("--max-c-children-per-d", type=int, default=3)
    parser.add_argument("--p-loss-lo", type=float, default=50)
    parser.add_argument("--p-loss-hi", type=float, default=500)
    parser.add_argument("--c-benefit-lo", type=float, default=10)
    parser.add_argument("--c-benefit-hi", type=float, default=50)
    parser.add_argument("--d-cost-lo", type=float, default=50)
    parser.add_argument("--d-cost-hi", type=float, default=100)
    parser.add_argument("--population-size", type=int, default=100)
    parser.add_argument("--genmax", type=int, default=50)
    parser.add_argument("--crossover-prob", type=float, default=0.8)
    parser.add_argument("--mutation-prob", type=float, default=0.01)
    parser.add_argument("--no-level-scaling", action="store_true")
    parser.add_argument("--p-alpha-max", type=float, default=0.8)
    parser.add_argument("--cd-beta-max", type=float, default=0.5)
    parser.add_argument("--cd-floor", type=float, default=0.2)
    args = parser.parse_args()

    if args.type == "random":
        if args.nE is None:
            parser.error("--nE required in random mode")
        bn = generate_bn_dag_multi_pe(
            nP=args.nP, nE=args.nE,
            nC=args.nC or max(2, args.nP), nD=args.nD or 0,
            max_children=args.max_children, p_EP=args.p_EP, seed=args.seed)
        values_table = gen_random_values(
            bn, seed=args.seed, fixed_e_probs=FIXED_E_PROBS,
            p_loss_range=(args.p_loss_lo, args.p_loss_hi),
            c_benefit_range=(args.c_benefit_lo, args.c_benefit_hi),
            d_cost_range=(args.d_cost_lo, args.d_cost_hi))
    else:
        bn = generate_bn_from_root_and_reverse(
            nP=args.nP, seed=args.seed,
            extra_p_edge_prob=args.extra_p_edge_prob,
            max_extra_p_out_per_node=args.max_extra_p_out_per_node,
            c_parents_per_e_range=(args.c_parents_min, args.c_parents_max),
            max_e_children_per_c=args.max_e_children_per_c,
            max_c_children_per_d=args.max_c_children_per_d, nC=args.nC, nD=args.nD)
        values_table = gen_structured_values(
            bn, seed=args.seed, fixed_e_probs=FIXED_E_PROBS,
            p_loss_range=(args.p_loss_lo, args.p_loss_hi),
            c_benefit_range=(args.c_benefit_lo, args.c_benefit_hi),
            d_cost_range=(args.d_cost_lo, args.d_cost_hi),
            use_level_scaling=not args.no_level_scaling,
            p_alpha_max=args.p_alpha_max,
            cd_beta_max=args.cd_beta_max, cd_floor=args.cd_floor)
    print("=" * 60)
    print(f"Graph: {args.type} | Seed: {args.seed}")
    print(f"P={len(bn['P'])} E={len(bn['E'])} C={len(bn['C'])} D={len(bn['D'])}")
    print("=" * 60)
    result = find_best_defense_strategy_ga(
        bn, values_table,
        population_size=args.population_size, genmax=args.genmax,
        crossover_prob=args.crossover_prob, mutation_prob=args.mutation_prob, seed=args.seed)
    bd = result["best_D_state"]
    br = result["best_result"]
    defended = [d for d, v in bd.items() if v]
    print(f"\n* Best: {defended}")
    print(f"  C_benefit={br['C_benefit']:.2f} P_loss={br['P_expected_loss']:.2f} D_cost={br['D_cost']:.2f}")
    print(f"  objective={br['objective']:.4f}")
    print(f"  Time: {result['total_time_s']:.2f}s | Evaluated: {result['unique_evaluated']}")


if __name__ == "__main__":
    main()
