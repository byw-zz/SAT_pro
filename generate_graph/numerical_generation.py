import random
from collections import defaultdict


def _make_fixed_int_choices(value_range, k=5):
    """
    Generate k distinct fixed integer candidate values in closed interval [lo, hi].
    """
    lo, hi = value_range
    lo_i, hi_i = int(lo), int(hi)
    if lo_i > hi_i:
        lo_i, hi_i = hi_i, lo_i

    domain_size = hi_i - lo_i + 1
    if domain_size < k:
        raise ValueError(f"Interval {value_range} does not have enough integers to sample {k} distinct values.")

    return random.sample(range(lo_i, hi_i + 1), k)


def generate_node_values(
    bn,
    seed=None,
    p_loss_range=(50, 500),
    p_benefit_range=(5, 80),
    c_benefit_range=(10, 50),
    d_cost_range=(50, 100),
    fixed_e_probs=None,
    fixed_int_k=5,
    return_fixed_choices=False,
):
    """
    Generate value table for nodes in the Bayesian graph.
    P_loss / C_benefit / D_cost use fixed integer candidates (k=5) from the interval.
    """
    if seed is not None:
        random.seed(seed)

    if fixed_e_probs is None or len(fixed_e_probs) == 0:
        raise ValueError("fixed_e_probs must be provided (e.g., containing 24 fixed values between 0 and 1)")

    p_loss_choices = _make_fixed_int_choices(p_loss_range, k=fixed_int_k)
    c_benefit_choices = _make_fixed_int_choices(c_benefit_range, k=fixed_int_k)
    d_cost_choices = _make_fixed_int_choices(d_cost_range, k=fixed_int_k)

    values_table = []

    node_types_order = ["P", "E", "C", "D"]
    all_nodes = []
    for t in node_types_order:
        all_nodes.extend(bn.get(t, []))

    for node in all_nodes:
        t = bn["node_type"][node]
        row = {
            "node": node,
            "type": t,
            "P_loss": None,
            "P_benefit": None,
            "C_benefit": None,
            "D_cost": None,
            "E_prob": None
        }

        if t == "P":
            row["P_loss"] = random.choice(p_loss_choices)
            row["P_benefit"] = round(random.uniform(*p_benefit_range), 2)

        elif t == "C":
            row["C_benefit"] = random.choice(c_benefit_choices)

        elif t == "D":
            row["D_cost"] = random.choice(d_cost_choices)

        elif t == "E":
            row["E_prob"] = random.choice(fixed_e_probs)

        values_table.append(row)

    if return_fixed_choices:
        fixed_choices = {
            "P_loss_choices": p_loss_choices,
            "C_benefit_choices": c_benefit_choices,
            "D_cost_choices": d_cost_choices,
        }
        return values_table, fixed_choices

    return values_table


def print_values_table(values_table):
    headers = ["node", "type", "P_loss", "C_benefit", "D_cost", "E_prob"]
    col_widths = {h: len(h) for h in headers}

    for row in values_table:
        for h in headers:
            val = row[h]
            s = "" if val is None else str(val)
            col_widths[h] = max(col_widths[h], len(s))

    def fmt_row(row_dict_or_header):
        parts = []
        for h in headers:
            if isinstance(row_dict_or_header, dict):
                val = row_dict_or_header[h]
                s = "" if val is None else str(val)
            else:
                s = row_dict_or_header[h]
            parts.append(s.ljust(col_widths[h]))
        return " | ".join(parts)

    print(fmt_row({h: h for h in headers}))
    print("-" * (sum(col_widths.values()) + 3 * (len(headers) - 1)))

    for row in values_table:
        print(fmt_row(row))
