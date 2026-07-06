"""Parse MaxHS solution and compute objective."""

from graph2sat.graph2sat import build_adjacency
from graph2sat.graph2sat import build_var_map


def parse_maxhs_solution(sol_file, n_vars):
    """Parse MaxHS output file to extract solution vector."""
    assignment = {}

    with open(sol_file, "r") as f:
        for line in f:
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


from common import parse_maxhs_solution_str


def interpret_solution(bn, cnf_data, assignment):
    """Interpret each node's True/False based on MaxHS assignment and var_map."""
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

def build_value_index(values_table):
    """Convert values_table list to dict indexed by node name."""
    idx = {}
    for row in values_table:
        node = row["node"]
        idx[node] = row
    return idx

def compute_objective_from_solution(bn, values_table, node_state, node_state_by_type):
    """Calculate objective value based on MaxHS solution and original semantics."""
    parents, children = build_adjacency(bn)
    node_type = bn["node_type"]

    vindex = build_value_index(values_table)

    total_C_benefit = 0.0
    total_P_expected_loss = 0.0
    total_D_cost = 0.0

    for c, active in node_state_by_type["C"].items():
        if not active:
            continue
        row = vindex.get(c, {})
        benefit = row.get("C_benefit") or 0.0
        total_C_benefit += benefit

    for p, active_p in node_state_by_type["P"].items():
        if not active_p:
            continue
        row_p = vindex.get(p, {})
        p_loss = row_p.get("P_loss") or 0.0

        e_children = [
            e for e in children[p]
            if node_type.get(e) == "E" and node_state.get(e, False)
        ]

        prob_not_comp = 1.0
        for e in e_children:
            row_e = vindex.get(e, {})
            prob_e = row_e.get("E_prob") or 0.0
            prob_not_comp *= (1.0 - prob_e)

        if e_children:
            p_comp = 1.0 - prob_not_comp
        else:
            p_comp = 0.0

        total_P_expected_loss += p_comp * p_loss

    for d, active in node_state_by_type["D"].items():
        if not active:
            continue
        row = vindex.get(d, {})
        d_cost = row.get("D_cost") or 0.0
        total_D_cost += d_cost

    objective = total_C_benefit - total_P_expected_loss - total_D_cost

    return {
        "C_benefit": total_C_benefit,
        "P_expected_loss": total_P_expected_loss,
        "D_cost": total_D_cost,
        "objective": objective,
    }
