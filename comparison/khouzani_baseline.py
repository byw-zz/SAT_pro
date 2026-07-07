"""Khouzani et al. (EJOR 2019) MILP baseline, adapted to the P/E/C/D model.

Faithful reproduction of the min-max shortest-path-interdiction MILP
(Khouzani, Liu, Malacaria 2019, Prop. 2 / Prop. 3), with the defence
"weakening rate" p_ecl fixed to 0: a selected control fully blocks (interdicts)
every attack edge it covers.

Graph reduction (P->E->C->D  ==>  P-only attack graph):
  - C nodes are dropped (they only wire D to E);
  - E nodes are contracted into direct  P_i -> P_j  attack edges carrying
    pi_e = E_prob (the baseline exploitation probability);
  - each attack edge is annotated with the set of D controls that can interdict
    it  ->  D controls edge_E  iff  D -> C -> E  in the original graph;
  - edges are oriented in attack-flow direction (source P0 -> targets).
This is exactly: reverse the complete graph  P->E->C->D  to  D->C->E->P,
then contract out C and E, leaving D as an interdiction control on P->P edges.

The MILP is solved for a sweep of direct-cost budgets (epsilon-constraint, as in
the paper). Each budget yields a defence = a set of selected D nodes. Every such
defence is then scored with the SAME BP/VE objective used for MaxSAT and GA
(`run_exact_analysis` / `compute_objective_bp_style`), so all methods are compared
on one ruler. The optimisation objective of Khouzani (worst-path risk) differs
from the evaluation objective (sum of expected losses); this mismatch is
intentional -- see memory: objective-mismatch-intentional.
"""

import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import pulp

sys.path.insert(0, str(Path(__file__).parent.parent))

from comparison.bp_core import compute_objective_bp_style


def _get_evaluator(eval_mode):
    """Return the shared objective evaluator; import pgmpy-backed exact lazily."""
    if eval_mode == "exact":
        from result_analysis.exact_analysis import run_exact_analysis
        return run_exact_analysis
    return compute_objective_bp_style


_EPS_PROB = 1e-12       # floor for log() of a probability
_TIE_BREAK = 1e-6       # tiny nudge: prefer fewer controls among equal-risk optima


SUPER_SOURCE = "__SRC__"


def build_khouzani_attack_graph(bn, values_table):
    """Reduce the P/E/C/D graph to a Khouzani P-only attack graph.

    Works for BOTH generators. The shared evaluator (`run_exact_analysis` /
    `run_bp_analysis`) reverses the stored edges uniformly, i.e. a node's
    probabilistic parents are its stored children. Compromise therefore flows
    OPPOSITE to the stored edges, so for every exploit E the attack edge is

        (P among stored-children of E)  ->  (P among stored-parents of E)

    - structured (edges pre-reversed): stored-child = P_in, stored-parent = P_out
      => attack P_in -> P_out;
    - random (edges natural): stored-child = P_out, stored-parent = P_in
      => attack P_out -> P_in  (verified: defending D lowers exactly the P that
      is the stored-parent of the exploits it controls).

    Two virtual-source cases feed a single SUPER_SOURCE `s`:
    - a "dead-end" exploit with no P among its stored-children (random only) fires
      from its controls alone and compromises its stored-parent P: edge s -> Phead;
    - an always-active privilege (no stored edge P->E, e.g. structured P0) is
      compromised unconditionally: edge s -> P with prob 1, no controls.

    Returns:
        {
          "V":       [P nodes] + [SUPER_SOURCE],
          "source":  SUPER_SOURCE,
          "edges":   [ {"e", "tail", "head", "pi", "ctrl": [D,...]}, ... ],
          "targets": {P: impact(=P_loss)},   # every real P node
          "d_cost":  {D: cost},
          "D":       [D nodes],
        }
    """
    node_type = bn["node_type"]
    vindex = {row["node"]: row for row in values_table}

    children = defaultdict(list)
    parents = defaultdict(list)
    for u, v in bn["edges"]:
        children[u].append(v)
        parents[v].append(u)

    edges = []
    for e in bn["E"]:
        p_children = [p for p in children[e] if node_type.get(p) == "P"]
        p_parents = [p for p in parents[e] if node_type.get(p) == "P"]
        if not p_parents:
            # No privilege is compromised by this exploit -> irrelevant to risk.
            continue
        head = p_parents[0]                       # P compromised by E
        tail = p_children[0] if p_children else SUPER_SOURCE  # dead-end -> source

        pi = vindex.get(e, {}).get("E_prob", None)
        if pi is None:
            raise RuntimeError(f"Exploit {e} has no E_prob in values_table.")

        # D controls edge_E  iff  E -> C -> D  in the stored graph
        ctrl = set()
        for c in children[e]:
            if node_type.get(c) != "C":
                continue
            for d in children[c]:
                if node_type.get(d) == "D":
                    ctrl.add(d)

        edges.append({"e": e, "tail": tail, "head": head,
                      "pi": float(pi), "ctrl": sorted(ctrl)})

    # Always-active privileges (no exploit launched from them in the factor graph,
    # i.e. no stored P->E edge) are compromised unconditionally from the source.
    for p in bn["P"]:
        if not any(node_type.get(c) == "E" for c in children[p]):
            edges.append({"e": None, "tail": SUPER_SOURCE, "head": p,
                          "pi": 1.0, "ctrl": []})

    targets = {p: float(vindex.get(p, {}).get("P_loss", 0.0) or 0.0)
               for p in bn["P"]}
    d_cost = {d: float(vindex.get(d, {}).get("D_cost", 0.0) or 0.0) for d in bn["D"]}

    return {
        "V": list(bn["P"]) + [SUPER_SOURCE],
        "source": SUPER_SOURCE,
        "edges": edges,
        "targets": targets,
        "d_cost": d_cost,
        "D": list(bn["D"]),
    }


def _big_M(agraph):
    """A finite big-M large enough that interdicting one edge makes its
    constraint non-binding. A simple s->t path uses at most |V| edges, so the
    longest achievable path length is bounded by |V|*max|log pi| + max|log imp|;
    this is much tighter than summing over all edges, which tightens the LP
    relaxation and speeds up branch-and-bound."""
    max_lp = max((abs(math.log(min(max(ed["pi"], _EPS_PROB), 1.0)))
                  for ed in agraph["edges"]), default=0.0)
    max_imp = max((abs(math.log(max(imp, _EPS_PROB)))
                   for imp in agraph["targets"].values()), default=0.0)
    return len(agraph["V"]) * max_lp + max_imp + 1.0


def solve_khouzani_milp(agraph, budget, M=None, solver=None, msg=False, time_limit=None):
    """Solve the dualised shortest-path-interdiction MILP for one cost budget.

    min_{lambda,x}  lambda_source - lambda_sink        (sink potential fixed to 0)
    s.t.  lambda_tail(e) - lambda_head(e) >= log(pi_e) - M * sum_{D in ctrl(e)} x_D
          lambda_t       - 0             >= log(impact_t)        for each target t
          sum_D cost_D * x_D <= budget
          x_D in {0,1}
    Risk R = exp(objective). Returns {status, D_selected, risk, milp_obj}.
    """
    edges = agraph["edges"]
    source = agraph["source"]
    if M is None:
        M = _big_M(agraph)

    prob = pulp.LpProblem("khouzani_interdiction", pulp.LpMinimize)

    lam = {p: pulp.LpVariable(f"lam_{p}", lowBound=None) for p in agraph["V"]}
    x = {d: pulp.LpVariable(f"x_{d}", cat="Binary") for d in agraph["D"]}

    # objective: worst-path risk in log space, with a tiny preference for fewer
    # controls to break ties among equal-risk optima (fairer to the baseline).
    prob += lam[source] + _TIE_BREAK * pulp.lpSum(x[d] for d in agraph["D"])

    for ed in edges:
        w = math.log(min(max(ed["pi"], _EPS_PROB), 1.0))
        ctrl_sum = pulp.lpSum(x[d] for d in ed["ctrl"]) if ed["ctrl"] else 0
        prob += lam[ed["tail"]] - lam[ed["head"]] >= w - M * ctrl_sum

    for t, imp in agraph["targets"].items():
        prob += lam[t] >= math.log(max(imp, _EPS_PROB))

    prob += pulp.lpSum(agraph["d_cost"][d] * x[d] for d in agraph["D"]) <= budget

    if solver is None:
        solver = pulp.PULP_CBC_CMD(msg=msg, timeLimit=time_limit)
    prob.solve(solver)

    status = pulp.LpStatus[prob.status]
    # read the incumbent even if CBC stopped at the time limit (not "Optimal")
    D_selected = {d for d in agraph["D"]
                  if x[d].value() is not None and x[d].value() > 0.5}
    milp_obj = pulp.value(prob.objective)
    risk = None
    if lam[source].value() is not None:
        risk = math.exp(lam[source].value())
    return {"status": status, "D_selected": D_selected, "risk": risk, "milp_obj": milp_obj}


SINK = "__SINK__"


def _attack_arcs(agraph):
    """All arcs of the attack graph incl. target->sink, as
    (tail, head, logw, frozenset(controlling D)). Target arcs carry log(impact)
    and have no controls."""
    arcs = []
    for ed in agraph["edges"]:
        w = math.log(min(max(ed["pi"], _EPS_PROB), 1.0))
        arcs.append((ed["tail"], ed["head"], w, frozenset(ed["ctrl"])))
    for t, imp in agraph["targets"].items():
        arcs.append((t, SINK, math.log(max(imp, _EPS_PROB)), frozenset()))
    return arcs


def _longest_path(arcs, nodes, source, blocked_D):
    """Longest-log-weight path source->SINK on the DAG of arcs whose controls are
    all un-selected. Returns (best_logw, [arc,...]) or (None, None) if unreachable.
    Both attack graphs are DAGs (structured: P-DAG; random: arcs strictly decrease
    P-rank), so a topological longest path is exact and O(V+E)."""
    adj = defaultdict(list)
    indeg = defaultdict(int)
    allnodes = set(nodes) | {SINK}
    live = []
    for (u, v, w, ctrl) in arcs:
        if ctrl & blocked_D:          # arc interdicted -> removed
            continue
        adj[u].append((v, w, (u, v, w, ctrl)))
        indeg[v] += 1
        indeg.setdefault(u, indeg.get(u, 0))
        allnodes.add(u); allnodes.add(v)
        live.append((u, v))
    # Kahn topological order
    from collections import deque
    q = deque([n for n in allnodes if indeg.get(n, 0) == 0])
    topo = []
    seen_indeg = dict(indeg)
    while q:
        n = q.popleft(); topo.append(n)
        for (v, w, arc) in adj.get(n, []):
            seen_indeg[v] -= 1
            if seen_indeg[v] == 0:
                q.append(v)
    NEG = float("-inf")
    dist = {n: NEG for n in allnodes}
    prev = {n: None for n in allnodes}
    dist[source] = 0.0
    for n in topo:
        if dist[n] == NEG:
            continue
        for (v, w, arc) in adj.get(n, []):
            if dist[n] + w > dist[v]:
                dist[v] = dist[n] + w
                prev[v] = arc
    if dist.get(SINK, NEG) == NEG:
        return None, None
    path = []
    cur = SINK
    while prev.get(cur) is not None:
        arc = prev[cur]; path.append(arc); cur = arc[0]
    path.reverse()
    return dist[SINK], path


def solve_khouzani_rowgen(agraph, budget, threads=1, max_iters=2000, msg=False):
    """Exact Israeli-Wood row generation for the worst-path-interdiction problem
    (the SPNI method the paper cites). Avoids the weak global big-M: each generated
    path P adds  t >= logw(P) - |logw(P)| * sum_{D in controls(P)} x_D  (tight per-path
    big-M), and the separation oracle is a DAG longest path. Deterministic and, on
    the dense random nP=100 instances, far faster than the big-M dual.

    Returns {status, D_selected, risk, milp_obj, iters}.
    """
    arcs = _attack_arcs(agraph)
    nodes = list(agraph["V"])
    source = agraph["source"]
    d_cost = agraph["d_cost"]

    prob = pulp.LpProblem("khouzani_rowgen", pulp.LpMinimize)
    x = {d: pulp.LpVariable(f"x_{d}", cat="Binary") for d in agraph["D"]}
    t = pulp.LpVariable("t", lowBound=None)
    prob += t + _TIE_BREAK * pulp.lpSum(x[d] for d in agraph["D"])
    prob += pulp.lpSum(d_cost[d] * x[d] for d in agraph["D"]) <= budget

    def add_path_cut(logw, path):
        controls = set()
        for (_u, _v, _w, ctrl) in path:
            controls |= ctrl
        bigM = abs(logw) + 1.0
        rhs = logw - bigM * (pulp.lpSum(x[d] for d in controls) if controls else 0)
        prob.addConstraint(t >= rhs)

    # seed with the un-interdicted worst path so t is bounded below
    seed_w, seed_p = _longest_path(arcs, nodes, source, frozenset())
    if seed_w is None:
        return {"status": "NoPath", "D_selected": set(), "risk": 0.0,
                "milp_obj": None, "iters": 0}
    add_path_cut(seed_w, seed_p)

    solver = pulp.PULP_CBC_CMD(msg=msg, threads=threads)
    status = "Optimal"
    it = 0
    for it in range(1, max_iters + 1):
        prob.solve(solver)
        status = pulp.LpStatus[prob.status]
        sel = frozenset(d for d in agraph["D"]
                        if x[d].value() is not None and x[d].value() > 0.5)
        worst_w, worst_p = _longest_path(arcs, nodes, source, sel)
        t_val = t.value() if t.value() is not None else float("-inf")
        if worst_w is None:                     # all target paths interdicted
            break
        if worst_w <= t_val + 1e-6:             # master already covers true worst path
            break
        add_path_cut(worst_w, worst_p)          # separate and repeat

    sel = {d for d in agraph["D"]
           if x[d].value() is not None and x[d].value() > 0.5}
    final_w, _ = _longest_path(arcs, nodes, source, frozenset(sel))
    risk = math.exp(final_w) if final_w is not None else 0.0
    return {"status": status, "D_selected": sel, "risk": risk,
            "milp_obj": pulp.value(prob.objective), "iters": it}


def find_best_defense_khouzani(bn, values_table, n_budget=20, eval_mode="exact",
                               solver=None, msg=False, return_all=False,
                               milp_time_limit=None, mip_gap=None, threads=1,
                               method="rowgen"):
    """Run the Khouzani MILP over a budget sweep and pick the defence that is
    best under the shared BP/VE objective.

    For reproducibility, solving is DETERMINISTIC: single-threaded CBC stopped by
    a fixed relative MIP gap (`mip_gap`, CBC gapRel) rather than a wall-clock time
    limit. On the dense big-M interdiction instances (random nP=100) proving true
    optimality can take >200s per budget, but a gap-bounded incumbent is reached
    fast, is identical across runs, and -- being scored by BP/VE like every method
    -- yields a faithful "within-gap-of-optimal" Khouzani. `milp_time_limit` is
    only a non-deterministic safety cap; leave it None (or well above the gap-stop
    time) so it never binds. `n_milp_optimal` reports how many solves CBC proved
    optimal at the chosen gap.

    Returns:
        {
          "best_D_state": {D: bool},
          "best_objective": float,
          "best_eval": <evaluator dict>,
          "best_budget": float,
          "pareto": [ {budget, D_selected, risk, objective, ...}, ... ],
          "runtime_s", "milp_time_s", "n_milp_solves",
          "attack_graph": agraph,
        }
    """
    t0 = time.time()
    agraph = build_khouzani_attack_graph(bn, values_table)
    M = _big_M(agraph)

    total_cost = sum(agraph["d_cost"].values())
    if not agraph["D"] or total_cost <= 0:
        budgets = [0.0]
    elif n_budget <= 1:
        budgets = [total_cost]
    else:
        budgets = [total_cost * k / (n_budget - 1) for k in range(n_budget)]

    evaluator = _get_evaluator(eval_mode)

    # big-M dual solver (only built/used when method="bigm")
    bigm_solver = None
    if method == "bigm" and solver is None:
        kw = {"msg": msg, "threads": threads}
        if mip_gap is not None:
            kw["gapRel"] = mip_gap
        if milp_time_limit is not None:
            kw["timeLimit"] = milp_time_limit
        bigm_solver = pulp.PULP_CBC_CMD(**kw)
    elif method == "bigm":
        bigm_solver = solver

    seen = {}          # frozenset(D_selected) -> (D_state, eval_dict, milp_sol)
    pareto = []
    milp_time = 0.0
    n_solves = 0
    n_optimal = 0      # solves that finished exactly (rowgen: always; bigm: proved optimal)
    best = None        # (D_state, eval_dict, milp_sol, budget)

    for b in budgets:
        ts = time.time()
        if method == "rowgen":
            sol = solve_khouzani_rowgen(agraph, b, threads=threads, msg=msg)
        else:
            sol = solve_khouzani_milp(agraph, b, M=M, solver=bigm_solver, msg=msg)
        milp_time += time.time() - ts
        n_solves += 1
        n_optimal += (sol["status"] == "Optimal")

        key = frozenset(sol["D_selected"])
        if key not in seen:
            D_state = {d: (d in sol["D_selected"]) for d in bn["D"]}
            ev = evaluator(bn, values_table, D_state=D_state)
            seen[key] = (D_state, ev, sol)
            pareto.append({
                "budget": b,
                "D_selected": sorted(sol["D_selected"]),
                "n_defended": len(sol["D_selected"]),
                "risk": sol["risk"],
                "objective": ev["objective"],
                "C_benefit": ev["C_benefit"],
                "P_expected_loss": ev["P_expected_loss"],
                "D_cost": ev["D_cost"],
            })

        D_state, ev, sol = seen[key]
        if best is None or ev["objective"] > best[1]["objective"]:
            best = (D_state, ev, sol, b)

    runtime = time.time() - t0
    best_D_state, best_eval, best_sol, best_budget = best

    pareto.sort(key=lambda r: r["budget"])
    result = {
        "best_D_state": best_D_state,
        "best_objective": best_eval["objective"],
        "best_eval": best_eval,
        "best_budget": best_budget,
        "pareto": pareto,
        "runtime_s": runtime,
        "milp_time_s": milp_time,
        "n_milp_solves": n_solves,
        "n_milp_optimal": n_optimal,
    }
    if return_all:
        result["attack_graph"] = agraph
    return result


# --------------------------------------------------------------------------- #
# Self-check / verification on a tiny structured graph.
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    from generate_graph.structured_graph import generate_bn_from_root_and_reverse
    from generate_graph.number_generation import generate_node_values
    from generate_graph.config import FIXED_E_PROBS

    nP, seed = 4, 7
    bn = generate_bn_from_root_and_reverse(nP=nP, seed=seed)
    values_table = generate_node_values(bn, seed=seed, fixed_e_probs=FIXED_E_PROBS)
    vindex = {r["node"]: r for r in values_table}

    agraph = build_khouzani_attack_graph(bn, values_table)

    print(f"=== Reduced Khouzani attack graph (nP={nP}, seed={seed}) ===")
    print(f"source = {agraph['source']}")
    print(f"{'attack edge':<16}{'exploit':<8}{'pi_e':>8}   controls (D)")
    print("-" * 55)
    for ed in agraph["edges"]:
        print(f"{ed['tail']+'->'+ed['head']:<16}{str(ed['e']):<8}{ed['pi']:>8.3f}   {ed['ctrl']}")
    print(f"\ntargets (impact=P_loss): "
          + ", ".join(f"{t}={imp:.0f}" for t, imp in agraph["targets"].items()))
    print("D_cost: " + ", ".join(f"{d}={c:.0f}" for d, c in agraph["d_cost"].items()))

    EVAL_MODE = "exact"
    print(f"\n=== MILP budget sweep (evaluator={EVAL_MODE}) ===")
    res = find_best_defense_khouzani(bn, values_table, n_budget=12, eval_mode=EVAL_MODE)
    print(f"{'budget':>8}{'risk':>10}{'objective':>12}{'D_cost':>8}  defended")
    print("-" * 60)
    for r in res["pareto"]:
        risk = f"{r['risk']:.4f}" if r["risk"] is not None else "n/a"
        print(f"{r['budget']:>8.1f}{risk:>10}{r['objective']:>12.2f}"
              f"{r['D_cost']:>8.0f}  {r['D_selected']}")

    print(f"\nBEST Khouzani defence: {sorted(d for d,v in res['best_D_state'].items() if v)}")
    print(f"  objective = {res['best_objective']:.4f}  (budget={res['best_budget']:.1f})")
    print(f"  runtime = {res['runtime_s']*1000:.1f} ms, "
          f"MILP solves = {res['n_milp_solves']}")

    # sanity baselines under the same evaluator
    _eval = _get_evaluator(EVAL_MODE)
    no_def = _eval(bn, values_table, D_state={d: False for d in bn["D"]})
    all_def = _eval(bn, values_table, D_state={d: True for d in bn["D"]})
    print(f"\n  [ref] no-defence  objective = {no_def['objective']:.4f}")
    print(f"  [ref] all-defence objective = {all_def['objective']:.4f}")
