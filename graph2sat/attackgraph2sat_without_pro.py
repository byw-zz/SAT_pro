import itertools
import random
from collections import defaultdict


# ---------------------------------------------------------------------
# 1) Graph helpers
# ---------------------------------------------------------------------

def build_adjacency(bn):
    parents = defaultdict(list)
    children = defaultdict(list)
    for u, v in bn["edges"]:
        parents[v].append(u)
        children[u].append(v)
    return parents, children


def build_var_map_from_raw_id_map(bn):
    """
    Construct CNF variable number mapping using bn["raw_id_map"].

    bn["raw_id_map"] structure is:
        { raw_id(int) : node_name(str) }

    CNF requires:
        var_map = { node_name : raw_id }
    """
    raw_id_map = bn.get("raw_id_map", None)
    if not isinstance(raw_id_map, dict) or not raw_id_map:
        raise ValueError("bn missing raw_id_map or raw_id_map is empty.")

    var_map = {}
    for rid, name in raw_id_map.items():
        rid_i = int(rid)
        if rid_i <= 0:
            raise ValueError(f"raw_id_map contains invalid raw_id={rid} (must be positive integer).")
        if not isinstance(name, str) or not name:
            raise ValueError(f"Node name in raw_id_map[{rid}] is invalid: {name}")
        if name in var_map and var_map[name] != rid_i:
            raise ValueError(f"Duplicate mapping for node name: {name} -> {var_map[name]} and {rid_i}")
        var_map[name] = rid_i

    return var_map


def sanity_check_bn_for_cnf(bn):
    """
    Check against current conventions:
    - Each E has exactly 1 parent P
    - noisy-OR uses children[p]∩E, so at least one P->E edge exists
    """
    node_type = bn["node_type"]
    parents, children = build_adjacency(bn)

    for e in bn.get("E", []):
        p_parents = [u for u in parents[e] if node_type.get(u) == "P"]
        if len(p_parents) != 1:
            raise RuntimeError(f"[Structure check failed] E={e} has {len(p_parents)} parent P nodes, expected 1.")

    any_pe = False
    for p in bn.get("P", []):
        e_children = [v for v in children[p] if node_type.get(v) == "E"]
        if e_children:
            any_pe = True
            break
    if not any_pe and bn.get("P"):
        raise RuntimeError("[Structure check failed] No P->E edges found, but you use children[p] for noisy-OR.")


# ---------------------------------------------------------------------
# 2) BN -> MaxSAT CNF
# ---------------------------------------------------------------------

def bn_to_maxsat_cnf_attackgraph(bn, values_table, initial_true_nodes=None):
    """
    Convert attack graph bn + values_table to Max-SAT CNF (hard clauses + soft clauses)
    - var_map uses bn["raw_id_map"] (raw_id -> node_name) inversion
    - noisy-OR uses children[p] of E (P -> E)
    - Does not use root_p; use initial_true_nodes for starting point constraints
    """
    sanity_check_bn_for_cnf(bn)

    node_type = bn["node_type"]
    parents, children = build_adjacency(bn)
    var_map = build_var_map_from_raw_id_map(bn)

    value_dict = {row["node"]: row for row in values_table}

    hard_clauses = []
    soft_clauses = []

    # Rule 1 & 3: Each E (assuming exactly 1 parent P)
    for e in bn.get("E", []):
        if e not in var_map:
            raise KeyError(f"var_map missing node {e} (raw_id_map not covering?)")
        v_e = var_map[e]

        p_parents = [u for u in parents[e] if node_type.get(u) == "P"]
        if len(p_parents) != 1:
            raise RuntimeError(f"Exploit node {e} does not have exactly 1 parent P (current: {len(p_parents)})")
        p_in = p_parents[0]
        if p_in not in var_map:
            raise KeyError(f"var_map missing node {p_in} (raw_id_map not covering?)")
        v_p_in = var_map[p_in]

        c_children = [v for v in children[e] if node_type.get(v) == "C"]
        p_children = [v for v in children[e] if node_type.get(v) == "P"]

        # Rule 1: p_in ∪ ¬c ∪ ¬p'
        clause1 = [v_p_in]
        clause1 += [-var_map[c] for c in c_children]
        clause1 += [-var_map[p] for p in p_children]
        hard_clauses.append(clause1)

        # Additional: (-p_in ∨ child)
        for c in c_children:
            hard_clauses.append([-v_p_in, var_map[c]])
        for p in p_children:
            hard_clauses.append([-v_p_in, var_map[p]])

        # Rule 3: e ∪ ¬c ∪ ¬p'
        clause3 = [v_e]
        clause3 += [-var_map[c] for c in c_children]
        clause3 += [-var_map[p] for p in p_children]
        hard_clauses.append(clause3)

        # Additional: (-e ∨ child)
        for c in c_children:
            hard_clauses.append([-v_e, var_map[c]])
        for p in p_children:
            hard_clauses.append([-v_e, var_map[p]])

    # Optional: starting point/known true nodes (unit hard clauses)
    if initial_true_nodes:
        for n in initial_true_nodes:
            if n not in var_map:
                raise KeyError(f"initial_true_nodes contains unknown node: {n}")
            hard_clauses.append([var_map[n]])

    # Rule 2: C and its child D
    # (c ∨ d1 ∨ d2 ...) and (¬c ∨ ¬d)
    for c in bn.get("C", []):
        d_children = [v for v in children[c] if node_type.get(v) == "D"]
        if not d_children:
            continue
        hard_clauses.append([var_map[c]] + [var_map[d] for d in d_children])

    for c in bn.get("C", []):
        d_children = [v for v in children[c] if node_type.get(v) == "D"]
        if c == "C6":
            hard_clauses.append([-var_map[c]])
        if not d_children:
            continue
        for d in d_children:
            hard_clauses.append([-var_map[c], -var_map[d]])

    # Rule 6: D cost ((-d))
    for d in bn.get("D", []):
        w_d = value_dict.get(d, {}).get("D_cost", None)
        if w_d is None:
            continue
        soft_clauses.append((float(w_d), [-var_map[d]]))

    # Rule 7: C benefit ((c))
    for c in bn.get("C", []):
        w_c = value_dict.get(c, {}).get("C_benefit", None)
        if w_c is None:
            continue
        soft_clauses.append((float(w_c), [var_map[c]]))

    # Rule 8: noisy-OR (using children[p] of E)
    for p in bn.get("P", []):
        w_p = value_dict.get(p, {}).get("P_loss", None)
        if w_p is None:
            continue
        soft_clauses.append((float(w_p), [-var_map[p]]))

    return {
        "var_map": var_map,
        "hard_clauses": hard_clauses,
        "soft_clauses": soft_clauses,
    }


# ---------------------------------------------------------------------
# 3) Export WCNF
# ---------------------------------------------------------------------

def export_to_wcnf(cnf_data, filename, hard_weight=1000000, soft_scale=1):
    """
    Export WCNF. Since var_id may not be continuous, n_vars takes max(var_id).
    """
    var_map = cnf_data["var_map"]
    hard_clauses = cnf_data["hard_clauses"]
    soft_clauses = cnf_data["soft_clauses"]

    if not var_map:
        raise ValueError("var_map is empty.")

    n_vars = max(var_map.values())
    n_clauses = len(hard_clauses) + len(soft_clauses)
    top = int(hard_weight)

    with open(filename, "w") as f:
        f.write(f"p wcnf {n_vars} {n_clauses} {top}\n")

        for clause in hard_clauses:
            f.write(str(top) + " " + " ".join(str(l) for l in clause) + " 0\n")

        for weight, clause in soft_clauses:
            w = float(weight)
            if w < 0:
                w = 0.0
            w_int = int(round(w * soft_scale))
            if w_int <= 0:
                continue
            f.write(str(w_int) + " " + " ".join(str(l) for l in clause) + " 0\n")

    print(f"WCNF file exported: {filename}")


# ---------------------------------------------------------------------
# 4) Optional: add random P true hard clauses
# ---------------------------------------------------------------------

def add_random_p_true_hard_clauses(cnf_data, bn, num_p_true, seed=None):
    if seed is not None:
        random.seed(seed)
    var_map = cnf_data["var_map"]
    P_nodes = list(bn.get("P", []))
    if not P_nodes:
        return cnf_data
    num = min(num_p_true, len(P_nodes))
    chosen = random.sample(P_nodes, num)
    for p in chosen:
        if p not in var_map:
            raise KeyError(f"P node {p} not in var_map.")
        cnf_data["hard_clauses"].append([var_map[p]])
    return cnf_data
