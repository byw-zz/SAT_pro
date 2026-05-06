"""BP approximate inference for defense strategy optimization using NSGA-II."""

import argparse
import random
import sys
import time
from collections import defaultdict
from itertools import product

import numpy as np

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.optimize import minimize

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generate_graph.random_graph import generate_bn_dag_multi_pe
from generate_graph.structured_graph import generate_bn_from_root_and_reverse
from generate_graph.numerical_generation import generate_node_values as gen_random_values
from generate_graph.number_generation import generate_node_values as gen_structured_values


FIXED_E_PROBS = [
    0.02, 0.05, 0.10, 0.12, 0.15, 0.18,
    0.20, 0.25, 0.30, 0.32, 0.35, 0.38,
    0.40, 0.45, 0.50, 0.55, 0.60, 0.65,
    0.70, 0.75, 0.80, 0.85, 0.90, 0.95,
]


def _normalize2(msg):
    s = msg[0] + msg[1]
    if s <= 0:
        return [0.5, 0.5]
    return [msg[0] / s, msg[1] / s]


class Factor:
    def __init__(self, name, vars_, table):
        self.name = name
        self.vars = tuple(vars_)
        self.table = dict(table)

    def __repr__(self):
        return f"Factor(name={self.name}, vars={self.vars}, size={len(self.table)})"


def build_c_factor(node, forced_zero_C):
    """C node factor: root node with prior"""
    if node in forced_zero_C:
        table = {(0,): 1.0, (1,): 0.0}
    else:
        table = {(0,): 0.0, (1,): 1.0}
    return Factor(name=f"phi_{node}", vars_=(node,), table=table)


def build_e_factor(node, parents):
    """E node factor: strict AND"""
    if not parents:
        return Factor(name=f"phi_{node}", vars_=(node,), table={(0,): 1.0, (1,): 0.0})

    vars_ = tuple(parents) + (node,)
    table = {}
    for parent_assign in product([0, 1], repeat=len(parents)):
        p_e1 = 1.0 if all(x == 1 for x in parent_assign) else 0.0
        p_e0 = 1.0 - p_e1
        table[parent_assign + (0,)] = p_e0
        table[parent_assign + (1,)] = p_e1
    return Factor(name=f"phi_{node}", vars_=vars_, table=table)


def build_p_factor(node, parents, vindex):
    """P node factor: noisy-OR"""
    if not parents:
        return Factor(name=f"phi_{node}", vars_=(node,), table={(0,): 0.0, (1,): 1.0})

    vars_ = tuple(parents) + (node,)
    table = {}
    for parent_assign in product([0, 1], repeat=len(parents)):
        prob_not_active = 1.0
        for e_node, e_val in zip(parents, parent_assign):
            e_prob = (vindex.get(e_node, {}) or {}).get("E_prob", 0.0) or 0.0
            prob_not_active *= (1.0 - e_prob) ** e_val
        p1 = 1.0 - prob_not_active
        p1 = max(0.0, min(1.0, p1))
        p0 = 1.0 - p1
        table[parent_assign + (0,)] = p0
        table[parent_assign + (1,)] = p1
    return Factor(name=f"phi_{node}", vars_=vars_, table=table)


def build_all_factors(node_type, parents_f, vindex, forced_zero_C, all_remaining_nodes):
    """Build all local factors"""
    node_set = set(all_remaining_nodes)
    factors = []
    for node in all_remaining_nodes:
        ntype = node_type.get(node, "")
        parents = [p for p in parents_f.get(node, []) if p in node_set]
        if ntype == "C":
            factors.append(build_c_factor(node, forced_zero_C))
        elif ntype == "E":
            factors.append(build_e_factor(node, parents))
        elif ntype == "P":
            factors.append(build_p_factor(node, parents, vindex))
    return factors


def loopy_bp_marginals(factors, query_nodes, max_iters=50, damping=0.5, tol=1e-6):
    """Run loopy BP on factor graph"""
    var_to_factors = defaultdict(list)
    factor_map = {}
    for f in factors:
        factor_map[f.name] = f
        for v in f.vars:
            var_to_factors[v].append(f.name)

    m_vf = {}
    m_fv = {}
    for v, fnames in var_to_factors.items():
        for fname in fnames:
            m_vf[(v, fname)] = [0.5, 0.5]
            m_fv[(fname, v)] = [0.5, 0.5]

    converged = False
    last_delta = float("inf")

    for it in range(1, max_iters + 1):
        max_delta = 0.0

        new_m_vf = {}
        for v, fnames in var_to_factors.items():
            for fname in fnames:
                msg = [1.0, 1.0]
                for other_fname in fnames:
                    if other_fname == fname:
                        continue
                    incoming = m_fv[(other_fname, v)]
                    msg[0] *= incoming[0]
                    msg[1] *= incoming[1]
                msg = _normalize2(msg)
                old = m_vf[(v, fname)]
                if damping > 0:
                    msg = [damping * old[0] + (1.0 - damping) * msg[0],
                          damping * old[1] + (1.0 - damping) * msg[1]]
                    msg = _normalize2(msg)
                delta = max(abs(msg[0] - old[0]), abs(msg[1] - old[1]))
                max_delta = max(max_delta, delta)
                new_m_vf[(v, fname)] = msg
        m_vf = new_m_vf

        new_m_fv = {}
        for fname, f in factor_map.items():
            vars_ = f.vars
            for target_v in vars_:
                tidx = vars_.index(target_v)
                out = [0.0, 0.0]
                for full_assign in product([0, 1], repeat=len(vars_)):
                    phi = f.table.get(tuple(full_assign), 0.0)
                    if phi == 0.0:
                        continue
                    weight = phi
                    for idx, var in enumerate(vars_):
                        if var == target_v:
                            continue
                        incoming = m_vf[(var, fname)]
                        weight *= incoming[full_assign[idx]]
                    out[full_assign[tidx]] += weight
                out = _normalize2(out)
                old = m_fv[(fname, target_v)]
                if damping > 0:
                    out = [damping * old[0] + (1.0 - damping) * out[0],
                          damping * old[1] + (1.0 - damping) * out[1]]
                    out = _normalize2(out)
                delta = max(abs(out[0] - old[0]), abs(out[1] - old[1]))
                max_delta = max(max_delta, delta)
                new_m_fv[(fname, target_v)] = out
        m_fv = new_m_fv
        last_delta = max_delta

        if max_delta < tol:
            converged = True
            break

    marginals = {}
    for v in query_nodes:
        belief = [1.0, 1.0]
        for fname in var_to_factors.get(v, []):
            incoming = m_fv[(fname, v)]
            belief[0] *= incoming[0]
            belief[1] *= incoming[1]
        belief = _normalize2(belief)
        marginals[v] = belief[1]

    return marginals, {"iters": it, "max_delta": last_delta, "converged": converged}


def bp_approx_inference(node_type, parents_f, vindex, query_nodes,
                        forced_zero_C, all_remaining_nodes,
                        max_iters=50, damping=0.5, tol=1e-6):
    """BP approximate inference entry point"""
    factors = build_all_factors(
        node_type=node_type, parents_f=parents_f, vindex=vindex,
        forced_zero_C=forced_zero_C, all_remaining_nodes=all_remaining_nodes,
    )
    return loopy_bp_marginals(
        factors=factors, query_nodes=query_nodes,
        max_iters=max_iters, damping=damping, tol=tol,
    )


def run_bp_analysis(bn, values_table, D_state=None, bp_max_iters=10, bp_damping=0, bp_tol=1e-6):
    """Execute approximate probability analysis based on loopy BP"""
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
    marginals, bp_info = bp_approx_inference(
        node_type=node_type, parents_f=parents_f, vindex=vindex, query_nodes=rem_P,
        forced_zero_C=forced_zero_C, all_remaining_nodes=list(remaining_set),
        max_iters=bp_max_iters, damping=bp_damping, tol=bp_tol,
    )
    infer_time = time.time() - t0

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
        "_bp_time_ms": infer_time * 1000,
        "_bp_iters": bp_info["iters"],
        "_bp_converged": bp_info["converged"],
        "_bp_max_delta": bp_info["max_delta"],
    }


class DefenseStrategyProblem(ElementwiseProblem):
    """Defense strategy optimization problem (pymoo interface)"""

    def __init__(self, bn, values_table, D_nodes, cache,
                 bp_max_iters=50, bp_damping=0.5, bp_tol=1e-6):
        self.bn = bn
        self.values_table = values_table
        self.D_nodes = list(D_nodes)
        self.cache = cache
        self.bp_max_iters = bp_max_iters
        self.bp_damping = bp_damping
        self.bp_tol = bp_tol
        n_var = len(self.D_nodes)
        super().__init__(
            n_var=n_var, n_obj=1, n_ieq_constr=0,
            xl=np.zeros(n_var, dtype=int), xu=np.ones(n_var, dtype=int), vtype=int,
        )

    def _evaluate(self, x, out, *args, **kwargs):
        bits = tuple(int(v) for v in np.asarray(x).tolist())
        if bits not in self.cache:
            D_state = {d: bool(bits[i]) for i, d in enumerate(self.D_nodes)}
            self.cache[bits] = run_bp_analysis(
                self.bn, self.values_table, D_state=D_state,
                bp_max_iters=self.bp_max_iters,
                bp_damping=self.bp_damping, bp_tol=self.bp_tol,
            )
        result = self.cache[bits]
        out["F"] = [-result["objective"]]


def find_best_defense_strategy_nsga2(
    bn, values_table,
    population_size=100, genmax=50,
    crossover_prob=0.8, mutation_prob=0.01,
    seed=42, bp_max_iters=50, bp_damping=0.5, bp_tol=1e-6,
):
    """Search defense strategy using NSGA-II"""
    D_nodes = bn.get("D", [])
    nD = len(D_nodes)

    if nD == 0:
        result = run_bp_analysis(
            bn, values_table, D_state={},
            bp_max_iters=bp_max_iters, bp_damping=bp_damping, bp_tol=bp_tol,
        )
        return {
            "best_objective": result["objective"],
            "best_D_state": {},
            "best_result": result,
            "final_population": [],
            "history": [result],
            "unique_evaluated": 1,
            "total_time_s": 0.0,
        }

    cache = {}
    problem = DefenseStrategyProblem(
        bn=bn, values_table=values_table, D_nodes=D_nodes, cache=cache,
        bp_max_iters=bp_max_iters, bp_damping=bp_damping, bp_tol=bp_tol,
    )
    algorithm = NSGA2(
        pop_size=population_size,
        sampling=BinaryRandomSampling(),
        crossover=TwoPointCrossover(prob=crossover_prob),
        mutation=BitflipMutation(prob=mutation_prob),
        eliminate_duplicates=True,
    )

    t_start = time.time()
    res = minimize(problem, algorithm, ("n_gen", genmax), seed=seed, verbose=False)
    total_time = time.time() - t_start

    best_bits = None
    best_result = None
    best_obj = -float("inf")

    for bits, result in cache.items():
        obj = result["objective"]
        if obj > best_obj:
            best_obj = obj
            best_bits = bits
            best_result = result

    best_D_state = {d: bool(best_bits[i]) for i, d in enumerate(D_nodes)} if best_bits else {}

    final_population = []
    if hasattr(res, "pop") and res.pop is not None:
        X = res.pop.get("X")
        if X is not None:
            for row in X:
                bits = tuple(int(v) for v in np.asarray(row).tolist())
                result = cache[bits]
                final_population.append({
                    "genes": bits,
                    "D_state": {d: bool(bits[i]) for i, d in enumerate(D_nodes)},
                    "result": result,
                })

    history = list(cache.values())

    return {
        "best_objective": best_obj,
        "best_D_state": best_D_state,
        "best_result": best_result,
        "final_population": final_population,
        "history": history,
        "unique_evaluated": len(cache),
        "total_time_s": total_time,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Bayesian network defense strategy approximate analysis (Loopy BP + NSGA-II)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--type", "-t", choices=["random", "structured"],
                        required=True, help="Graph generation type")
    parser.add_argument("--seed", "-s", type=int, default=42)
    parser.add_argument("--nP", type=int, default=5, help="P node count")
    parser.add_argument("--nC", type=int, default=None)
    parser.add_argument("--nD", type=int, default=None)

    parser.add_argument("--nE", type=int, default=None)
    parser.add_argument("--max-children", type=int, default=5)
    parser.add_argument("--p-EP", type=float, default=0.25)

    parser.add_argument("--extra-p-edge-prob", type=float, default=0.25)
    parser.add_argument("--max-extra-p-out-per-node", type=int, default=4)
    parser.add_argument("--c-parents-min", type=int, default=1)
    parser.add_argument("--c-parents-max", type=int, default=3)
    parser.add_argument("--max-e-children-per-c", type=int, default=5)
    parser.add_argument("--max-c-children-per-d", type=int, default=3)

    parser.add_argument("--p-loss-lo", type=float, default=50)
    parser.add_argument("--p-loss-hi", type=float, default=500)
    parser.add_argument("--p-benefit-lo", type=float, default=5)
    parser.add_argument("--p-benefit-hi", type=float, default=80)
    parser.add_argument("--c-benefit-lo", type=float, default=10)
    parser.add_argument("--c-benefit-hi", type=float, default=50)
    parser.add_argument("--d-cost-lo", type=float, default=50)
    parser.add_argument("--d-cost-hi", type=float, default=100)

    parser.add_argument("--population-size", type=int, default=100, help="Population size N")
    parser.add_argument("--genmax", type=int, default=50, help="Max generations")
    parser.add_argument("--crossover-prob", type=float, default=0.8, help="Crossover probability")
    parser.add_argument("--mutation-prob", type=float, default=0.01, help="Bit mutation probability")
    parser.add_argument("--no-level-scaling", action="store_true")
    parser.add_argument("--p-alpha-max", type=float, default=0.8)
    parser.add_argument("--cd-beta-max", type=float, default=0.5)
    parser.add_argument("--cd-floor", type=float, default=0.2)

    parser.add_argument("--bp-max-iters", type=int, default=10, help="BP max iterations")
    parser.add_argument("--bp-damping", type=float, default=0, help="BP damping coefficient [0,1)")
    parser.add_argument("--bp-tol", type=float, default=1e-6, help="BP convergence threshold")

    args = parser.parse_args()

    if args.type == "random":
        if args.nE is None:
            parser.error("--nE required in random mode")

        bn = generate_bn_dag_multi_pe(
            nP=args.nP, nE=args.nE,
            nC=args.nC or max(2, args.nP),
            nD=args.nD or 0,
            max_children=args.max_children, p_EP=args.p_EP, seed=args.seed,
        )
        values_table = gen_random_values(
            bn, seed=args.seed, fixed_e_probs=FIXED_E_PROBS,
            p_loss_range=(args.p_loss_lo, args.p_loss_hi),
            p_benefit_range=(args.p_benefit_lo, args.p_benefit_hi),
            c_benefit_range=(args.c_benefit_lo, args.c_benefit_hi),
            d_cost_range=(args.d_cost_lo, args.d_cost_hi),
        )
    else:
        bn = generate_bn_from_root_and_reverse(
            nP=args.nP, seed=args.seed,
            extra_p_edge_prob=args.extra_p_edge_prob,
            max_extra_p_out_per_node=args.max_extra_p_out_per_node,
            c_parents_per_e_range=(args.c_parents_min, args.c_parents_max),
            max_e_children_per_c=args.max_e_children_per_c,
            max_c_children_per_d=args.max_c_children_per_d,
            nC=args.nC, nD=args.nD,
        )
        values_table = gen_structured_values(
            bn, seed=args.seed, fixed_e_probs=FIXED_E_PROBS,
            p_loss_range=(args.p_loss_lo, args.p_loss_hi),
            p_benefit_range=(args.p_benefit_lo, args.p_benefit_hi),
            c_benefit_range=(args.c_benefit_lo, args.c_benefit_hi),
            d_cost_range=(args.d_cost_lo, args.d_cost_hi),
            use_level_scaling=not args.no_level_scaling,
            p_alpha_max=args.p_alpha_max,
            cd_beta_max=args.cd_beta_max,
            cd_floor=args.cd_floor,
        )

    print("=" * 70)
    print(f"Graph type: {args.type} | Seed: {args.seed}")
    print(f"P={len(bn['P'])} E={len(bn['E'])} C={len(bn['C'])} D={len(bn['D'])} edges={len(bn['edges'])}")
    print("=" * 70)

    print(
        f"\n[NSGA-II MODE] nD={len(bn.get('D', []))}, "
        f"population={args.population_size}, generations={args.genmax}, crossover={args.crossover_prob}",
        flush=True
    )

    ga_result = find_best_defense_strategy_nsga2(
        bn, values_table,
        population_size=args.population_size,
        genmax=args.genmax,
        crossover_prob=args.crossover_prob,
        mutation_prob=args.mutation_prob,
        seed=args.seed,
        bp_max_iters=args.bp_max_iters,
        bp_damping=args.bp_damping,
        bp_tol=args.bp_tol,
    )

    print(f"\nSearch complete! Total time: {ga_result['total_time_s']:.2f}s | Unique evaluations: {ga_result['unique_evaluated']}", flush=True)

    br = ga_result["best_result"]
    bd = ga_result["best_D_state"]
    defended_nodes = [d for d, v in bd.items() if v]

    print(f"\n* Best defense combination ({len(defended_nodes)} nodes):")
    print(f"  Defended nodes: {defended_nodes}")
    print(f"  C_benefit       = {br['C_benefit']:.4f}")
    print(f"  P_expected_loss = {br['P_expected_loss']:.4f}")
    print(f"  P_expected_gain = {br['P_expected_gain']:.4f}")
    print(f"  D_cost          = {br['D_cost']:.4f}")
    print(f"  objective       = {br['objective']:.4f}")
    print(f"  BP time         = {br['_bp_time_ms']:.2f}ms")
    print(f"  BP iterations   = {br['_bp_iters']}")
    print(f"  BP converged    = {br['_bp_converged']}")
    print(f"  BP max delta   = {br['_bp_max_delta']:.3e}")
    print(f"  Forced-zero C   = {br['forced_zero_C']}")

    print(f"\nP node approximate marginal probabilities P(P=1):")
    for p, prob in sorted(br["P_marginals"].items()):
        print(f"  {p}: {prob:.6f}")


if __name__ == "__main__":
    main()
