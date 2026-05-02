import itertools
from collections import defaultdict


def build_adjacency(bn):
    """Build parents/children mappings from bn['edges']."""
    parents = defaultdict(list)
    children = defaultdict(list)
    for u, v in bn["edges"]:
        parents[v].append(u)
        children[u].append(v)
    return parents, children


def build_var_map(bn):
    """Assign a CNF variable number (1..N) to each node."""
    var_map = {}
    idx = 1
    for t in ["P", "E", "C", "D"]:
        for n in bn.get(t, []):
            var_map[n] = idx
            idx += 1
    return var_map


def bn_to_maxsat_cnf(bn, values_table):
    """
    Convert Bayesian graph + node value table to Max-SAT CNF representation.

    Args:
        bn: Return value from generate_bn_dag_multi_pe(...)
        values_table: Output from generate_node_values(bn), or any list of dicts
                    like [{'node': name, 'type': 'P', 'P_loss': ... , ...}, ...]

    Returns:
        result = {
            'var_map': {node_name: var_id},
            'hard_clauses': [ [lit, ...], ... ],
            'soft_clauses': [ (weight, [lit, ...]), ... ]
        }
    """
    node_type = bn["node_type"]
    parents, children = build_adjacency(bn)
    var_map = build_var_map(bn)

    value_dict = {row["node"]: row for row in values_table}

    hard_clauses = []
    soft_clauses = []

    if "P0" in var_map:
        p0_children = children.get("P0", [])
        if len(p0_children) == 0:
            hard_clauses.append([var_map["P0"]])

    # Rule 1 & 3: Two hard clauses for each exploit node e
    for e in bn["E"]:
        v_e = var_map[e]

        p_parents = [u for u in parents[e] if node_type[u] == "P"]
        if len(p_parents) != 1:
            raise RuntimeError(f"Exploit node {e} does not have exactly 1 parent P (current: {len(p_parents)})")
        p_in = p_parents[0]
        v_p_in = var_map[p_in]

        c_children = [v for v in children[e] if node_type[v] == "C"]
        p_children = [v for v in children[e] if node_type[v] == "P"]

        # Rule 1: p ∪ ¬c ∪ ¬p'
        clause1 = [v_p_in]
        clause1 += [-var_map[c] for c in c_children]
        clause1 += [-var_map[p] for p in p_children]
        hard_clauses.append(clause1)

        # Rule 3: e ∪ ¬c ∪ ¬p'
        clause3 = [v_e]
        clause3 += [-var_map[c] for c in c_children]
        clause3 += [-var_map[p] for p in p_children]
        hard_clauses.append(clause3)

    # Rule 2: Condition node c combined with its child D nodes form hard clauses
    for c in bn["C"]:
        d_children = [v for v in children[c] if node_type[v] == "D"]
        if not d_children:
            continue
        v_c = var_map[c]
        clause_c = [v_c] + [var_map[d] for d in d_children]
        hard_clauses.append(clause_c)

    for c in bn["C"]:
        d_children = [v for v in children[c] if node_type[v] == "D"]
        if not d_children:
            continue
        v_c = var_map[c]
        for d in d_children:
            clause_c = [-v_c] + [-var_map[d]]
            hard_clauses.append(clause_c)

    # Rule 6: For each defense node d, if d is F then cost w_d
    for d in bn["D"]:
        row = value_dict.get(d, {})
        w_d = row.get("D_cost", None)
        if w_d is None:
            continue
        v_d = var_map[d]
        soft_clauses.append((float(w_d), [-v_d]))

    # Rule 7: For each condition node c, if c is T then benefit w_c
    for c in bn["C"]:
        row = value_dict.get(c, {})
        w_c = row.get("C_benefit", None)
        if w_c is None:
            continue
        v_c = var_map[c]
        soft_clauses.append((float(w_c), [v_c]))

    for p in bn["P"]:
        row_p = value_dict.get(p, {})
        w_p = row_p.get("P_loss", None)
        if w_p is None:
            continue
        v_p = var_map[p]
        soft_clauses.append((float(w_p), [-v_p]))

    return {
        "var_map": var_map,
        "hard_clauses": hard_clauses,
        "soft_clauses": soft_clauses,
    }
