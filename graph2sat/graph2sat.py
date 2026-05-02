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


def bn_to_maxsat_cnf(bn, values_table, initial_true_nodes=None, initial_false_nodes=None):
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
        clause1 += [-v_e]
        hard_clauses.append(clause1)

        # Rule 3: e ∪ ¬c ∪ ¬p'
        clause3 = [v_e]
        clause3 += [-var_map[c] for c in c_children]
        clause3 += [-var_map[p] for p in p_children]
        hard_clauses.append(clause3)

    for p in bn["P"]:
        v_p = var_map[p]
        e_children = [u for u in children[p] if node_type[u] == "E"]
        if len(e_children) == 0:
            hard_clauses.append([v_p])
        else:
            clause = [-v_p]
            for e in e_children:
                v_e = var_map[e]
                clause += [v_e]
            hard_clauses.append(clause)

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

    if initial_true_nodes:
        for n in initial_true_nodes:
            if n not in var_map:
                raise KeyError(f"initial_true_nodes contains unknown node: {n}")
            hard_clauses.append([var_map[n]])
    if initial_false_nodes:
        for n in initial_false_nodes:
            if n not in var_map:
                raise KeyError(f"initial_false_nodes contains unknown node: {n}")
            hard_clauses.append([-var_map[n]])

    # Rule 6: For each defense node d, if d is F then cost w_d
    # Soft clause (d) with weight w_d
    for d in bn["D"]:
        row = value_dict.get(d, {})
        w_d = row.get("D_cost", None)
        if w_d is None:
            continue
        v_d = var_map[d]
        soft_clauses.append((float(w_d), [-v_d]))

    # Rule 7: For each condition node c, if c is T then benefit w_c
    # Soft clause (c) with weight w_c
    for c in bn["C"]:
        row = value_dict.get(c, {})
        w_c = row.get("C_benefit", None)
        if w_c is None:
            continue
        v_c = var_map[c]
        soft_clauses.append((float(w_c), [v_c]))

    # Rule 8: 2^k patterns for privilege node p in noisy-OR form
    #
    # Let p's parent exploit nodes be {e1, ..., ek}, exploit success prob is prob_i, p loss is w_p.
    # For each pattern α = (a1, ..., ak) ∈ {0,1}^k:
    #   - Active set S = { i | a_i = 1 }
    #   - If S is empty, attack success probability P_comp = 0
    #   - Otherwise P_comp = 1 - Π_{i∈S} (1 - prob_i)
    #   - Loss = P_comp * w_p
    #   - Gain = (1 - P_comp) * w_p
    #
    # In Max-SAT, we use a CNF clause that is uniquely "violated" by pattern α.
    # For pattern α, construct clause:
    #   C_α = (l1 ∨ ... ∨ lk)
    #   where: if a_i = 1, l_i = ¬e_i; if a_i = 0, l_i = e_i
    # This way, only when e_i == a_i does C_α become false, binding gain/loss to the pattern.
    for p in bn["P"]:
        row_p = value_dict.get(p, {})
        w_p = row_p.get("P_loss", None)
        w_p2 = row_p.get("P_benefit", None)
        if w_p is None:
            continue

        e_children = [u for u in children[p] if node_type[u] == "E"]
        k = len(e_children)
        if k == 0:
            continue

        probs = []
        for e in e_children:
            row_e = value_dict.get(e, {})
            prob_e = row_e.get("E_prob", None)
            if prob_e is None:
                raise RuntimeError(f"Exploit node {e} has no E_prob value, cannot construct noisy-OR soft clause.")
            probs.append(float(prob_e))

        for bits in itertools.product([0, 1], repeat=k):
            S = [i for i, b in enumerate(bits) if b == 1]

            if not S:
                P_comp = 0.0
            else:
                prod = 1.0
                for i in S:
                    prod *= (1.0 - probs[i])
                P_comp = 1.0 - prod

            reward = round((P_comp) * float(w_p), 2)
            reward2 = round(P_comp * float(w_p2), 2)

            clause = []
            for b, e in zip(bits, e_children):
                v_e = var_map[e]
                if b == 1:
                    clause.append(-v_e)
                else:
                    clause.append(v_e)

            soft_clauses.append((reward, clause))

    return {
        "var_map": var_map,
        "hard_clauses": hard_clauses,
        "soft_clauses": soft_clauses,
    }


def export_to_wcnf(cnf_data, filename, hard_weight=1000000):
    """
    Export Max-SAT CNF data to WCNF file (DIMACS Weighted CNF format).

    Args:
        cnf_data: { 'var_map': {...}, 'hard_clauses': [...], 'soft_clauses': [...] }
        filename: Output filename, e.g. "attack_graph.wcnf"
        hard_weight: Hard clause weight (must be integer, larger than all soft clause weights)
    """
    var_map = cnf_data["var_map"]
    hard_clauses = cnf_data["hard_clauses"]
    soft_clauses = cnf_data["soft_clauses"]

    n_vars = len(var_map)
    n_hard = len(hard_clauses)
    n_soft = len(soft_clauses)
    n_clauses = n_hard + n_soft

    hard_weight_int = int(hard_weight)

    with open(filename, "w") as f:
        f.write(f"p wcnf {n_vars} {n_clauses} {hard_weight_int}\n")

        for clause in hard_clauses:
            line = str(hard_weight_int) + " "
            line += " ".join(str(l) for l in clause)
            line += " 0\n"
            f.write(line)

        for weight, clause in soft_clauses:
            w = max(weight, 0.0)
            w_int = int(round(w * 100))
            if w_int <= 0:
                continue
            line = str(w_int) + " "
            line += " ".join(str(l) for l in clause)
            line += " 0\n"
            f.write(line)

    print(f"WCNF file successfully exported to: {filename}")


def add_hard_clauses_by_node_names(cnf_data, name_clause_list):
    """
    Add additional hard clauses by node names on top of existing cnf_data.

    Args:
        cnf_data: Dict returned by bn_to_maxsat_cnf
                {
                    'var_map': {node_name: var_id},
                    'hard_clauses': [...],
                    'soft_clauses': [...]
                }
        name_clause_list: e.g.
                [
                    [("p_in", False), ("c_x", True), ("p_out", True)],   # ¬p_in ∨ c_x ∨ p_out
                    [("p_7", True)],                                    # unit clause p_7
                    ...
                ]
    """
    var_map = cnf_data["var_map"]
    hard_clauses = cnf_data["hard_clauses"]

    for clause_spec in name_clause_list:
        clause = []
        for name, polarity in clause_spec:
            if name not in var_map:
                raise KeyError(f"Node {name} not in var_map, cannot construct hard clause.")
            vid = var_map[name]
            lit = vid if polarity else -vid
            clause.append(lit)
        hard_clauses.append(clause)

    return cnf_data


import random
from collections import defaultdict


def add_random_p_true_hard_clauses(cnf_data, bn, num_p_true, seed=None):
    """
    Randomly select P nodes, add unit hard clause (p) forcing it to True,
    and remove these P nodes' original Noisy-OR soft clauses.

    Args:
        cnf_data: Dict returned by bn_to_maxsat_cnf
        bn: Return value from generate_bn_dag_multi_pe(...)
        num_p_true: Number of P nodes to force as True
        seed: Random seed
    """
    if seed is not None:
        random.seed(seed)

    var_map = cnf_data["var_map"]
    hard_clauses = cnf_data["hard_clauses"]
    soft_clauses = cnf_data["soft_clauses"]
    node_type = bn["node_type"]

    all_p_nodes = list(bn.get("P", []))
    if not all_p_nodes:
        return cnf_data

    num = min(num_p_true, len(all_p_nodes))
    chosen_p_nodes = random.sample(all_p_nodes, num)

    children_map = defaultdict(list)
    if "edges" in bn:
        for u, v in bn["edges"]:
            children_map[u].append(v)

    signatures_to_remove = set()

    for p in chosen_p_nodes:
        if p not in var_map:
            continue

        v_p = var_map[p]
        hard_clauses.append([v_p])

        p_children = children_map.get(p, [])
        e_nodes = [u for u in p_children if node_type[u] == "E"]

        if e_nodes:
            e_var_ids = set(var_map[e] for e in e_nodes)
            signatures_to_remove.add(frozenset(e_var_ids))

    new_soft_clauses = []

    for weight, clause in soft_clauses:
        current_clause_vars = frozenset(abs(lit) for lit in clause)
        if current_clause_vars in signatures_to_remove:
            continue
        new_soft_clauses.append((weight, clause))

    cnf_data["soft_clauses"] = new_soft_clauses

    return cnf_data
