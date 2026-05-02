"""BP+GA core shared code for comparison."""

import subprocess
import time
from collections import defaultdict
from itertools import product
from pathlib import Path

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.optimize import minimize

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from graph2sat.graph2sat import export_to_wcnf


MAXHS_BIN = "/home/wzz/MaxHS/build/release/bin/maxhs"

FIXED_E_PROBS = [
    0.02, 0.05, 0.10, 0.12, 0.15, 0.18,
    0.20, 0.25, 0.30, 0.32, 0.35, 0.38,
    0.40, 0.45, 0.50, 0.55, 0.60, 0.65,
    0.70, 0.75, 0.80, 0.85, 0.90, 0.95,
]


def _normalize2(msg):
    """Normalize message"""
    s = msg[0] + msg[1]
    if s <= 0:
        return [0.5, 0.5]
    return [msg[0] / s, msg[1] / s]


class Factor:
    """Discrete binary factor"""
    def __init__(self, name, vars_, table):
        self.name = name
        self.vars = tuple(vars_)
        self.table = dict(table)


def build_c_factor(node, forced_zero_C):
    """C node factor: root node with prior distribution"""
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
    """P node factor: Noisy-OR"""
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
    """Build all factors"""
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
    """Loopy Belief Propagation message passing"""
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
                    msg = [
                        damping * old[0] + (1.0 - damping) * msg[0],
                        damping * old[1] + (1.0 - damping) * msg[1],
                    ]
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
                    out = [
                        damping * old[0] + (1.0 - damping) * out[0],
                        damping * old[1] + (1.0 - damping) * out[1],
                    ]
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


def run_bp_analysis(bn, values_table, D_state=None, bp_max_iters=50, bp_damping=0.5, bp_tol=1e-6):
    """
    Execute approximate probability analysis based on Loopy BP.
    Returns:
        objective = C_benefit - P_expected_loss - D_cost
    """
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
        "D_state": dict(D_state),
        "forced_zero_C": list(forced_zero_C),
        "_bp_time_ms": infer_time * 1000,
        "_bp_converged": bp_info["converged"],
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
                bp_max_iters=self.bp_max_iters, bp_damping=self.bp_damping, bp_tol=self.bp_tol,
            )
        result = self.cache[bits]
        out["F"] = [-result["objective"]]


def find_best_defense_bp(bn, values_table, population_size=100, genmax=50,
                         seed=42, bp_max_iters=50, bp_damping=0.5, bp_tol=1e-6,
                         top_n=10, return_all=False):
    """
    Search for optimal defense strategy using BP + GA.

    Returns:
        {
            "best_objective": ...,
            "best_D_state": {...},
            "best_result": {...},
            "unique_evaluated": ...,
            "total_time_s": ...,
            "top_n_results": [...],
            "all_results": [...]  # if return_all=True
        }
    """
    D_nodes = bn.get("D", [])
    nD = len(D_nodes)

    if nD == 0:
        result = run_bp_analysis(bn, values_table, D_state={},
                                 bp_max_iters=bp_max_iters, bp_damping=bp_damping, bp_tol=bp_tol)
        return {
            "best_objective": result["objective"],
            "best_D_state": {},
            "best_result": result,
            "unique_evaluated": 1,
            "total_time_s": 0.0,
            "top_n_results": [(result["objective"], {}, result)],
            "all_results": [(result["objective"], {}, result)],
        }

    cache = {}
    problem = DefenseStrategyProblem(
        bn=bn, values_table=values_table, D_nodes=D_nodes,
        cache=cache, bp_max_iters=bp_max_iters,
        bp_damping=bp_damping, bp_tol=bp_tol
    )
    algorithm = NSGA2(
        pop_size=population_size,
        sampling=BinaryRandomSampling(),
        crossover=TwoPointCrossover(prob=0.8),
        mutation=BitflipMutation(prob=0.01),
        eliminate_duplicates=True
    )

    t_start = time.time()
    minimize(problem, algorithm, ("n_gen", genmax), seed=seed, verbose=False)
    total_time = time.time() - t_start

    sorted_cache = sorted(
        [(bits, r) for bits, r in cache.items()],
        key=lambda x: x[1]["objective"],
        reverse=True
    )

    best_obj = sorted_cache[0][1]["objective"] if sorted_cache else -float("inf")
    best_bits = sorted_cache[0][0] if sorted_cache else None
    best_result = sorted_cache[0][1] if sorted_cache else None
    best_D_state = {d: bool(best_bits[i]) for i, d in enumerate(D_nodes)} if best_bits else {}

    top_n_results = []
    all_results = []
    for bits, r in sorted_cache:
        D_state = {d: bool(bits[i]) for i, d in enumerate(D_nodes)}
        item = (r["objective"], D_state, r)
        all_results.append(item)
        if len(top_n_results) < top_n:
            top_n_results.append(item)

    out = {
        "best_objective": best_obj,
        "best_D_state": best_D_state,
        "best_result": best_result,
        "unique_evaluated": len(cache),
        "total_time_s": total_time,
        "top_n_results": top_n_results,
    }
    if return_all:
        out["all_results"] = all_results
    return out


def get_cnf_converter(sat_type):
    """Get CNF converter"""
    if sat_type == "pro":
        from graph2sat.graph2sat import bn_to_maxsat_cnf
    else:
        from graph2sat.garaph2sat_without_pro import bn_to_maxsat_cnf
    return bn_to_maxsat_cnf


def solve_maxsat(wcnf_path, maxhs_bin, timeout=None, extra_args=None):
    """Call MaxHS solver"""
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
        return ""


def parse_maxhs_solution_str(maxhs_output, n_vars):
    """Parse MaxHS output"""
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


def interpret_solution(bn, cnf_data, assignment):
    """Interpret solution"""
    var_map = cnf_data["var_map"]
    rev_var_map = {vid: node for node, vid in var_map.items()}
    node_state = {}
    for vid, node in rev_var_map.items():
        val = assignment.get(vid, False)
        node_state[node] = val
    node_type = bn["node_type"]
    node_state_by_type = {"P": {}, "E": {}, "C": {}, "D": {}}
    for node, val in node_state.items():
        t = node_type[node]
        node_state_by_type[t][node] = val
    return node_state, node_state_by_type


def find_best_defense_maxsat(bn, values_table, wcnf_path, maxhs_bin, timeout=120):
    """
    Solve optimal defense strategy using MaxSAT (MaxHS).
    Returns:
        (node_state, node_state_by_type, cnf_data, solve_time)
    """
    bn_to_maxsat_cnf = get_cnf_converter("pro")
    cnf_data = bn_to_maxsat_cnf(bn, values_table, initial_true_nodes=None)
    export_to_wcnf(cnf_data, str(wcnf_path))

    extra = ["-printSoln", "-printBstSoln", "-verb=0", f"-cpu-lim={timeout}"]
    maxhs_out = solve_maxsat(wcnf_path, maxhs_bin, timeout=timeout + 5, extra_args=extra)

    n_vars = len(cnf_data["var_map"])
    assignment = parse_maxhs_solution_str(maxhs_out, n_vars)

    if not assignment:
        return None, None, None, 0.0

    t0 = time.time()
    node_state, node_state_by_type = interpret_solution(bn, cnf_data, assignment)
    solve_time = time.time() - t0

    return node_state, node_state_by_type, cnf_data, solve_time


def extract_D_state_from_solution(bn, node_state_by_type):
    """Extract D state from solution"""
    D_state = {}
    for d in bn.get("D", []):
        D_state[d] = node_state_by_type["D"].get(d, False)
    return D_state


def compute_objective_bp_style(bn, values_table, D_state, bp_max_iters=50, bp_damping=0.5, bp_tol=1e-6):
    """Calculate objective for given D_state in BP style"""
    return run_bp_analysis(
        bn, values_table, D_state=D_state,
        bp_max_iters=bp_max_iters, bp_damping=bp_damping, bp_tol=bp_tol
    )
