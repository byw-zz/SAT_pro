import random
import math
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


def _level_coef_log_bounded_inc(level: int, max_level: int, alpha_max: float = 0.8) -> float:
    """
    Normalized logarithmic positive correlation coefficient (with upper bound):
        coef_inc = 1 + alpha_max * log(1+level) / log(1+max_level)
    coef ∈ [1, 1+alpha_max]
    """
    level = 0 if level is None else int(level)
    max_level = max(0, int(max_level))
    if max_level <= 0:
        return 1.0
    return 1.0 + float(alpha_max) * (math.log1p(level) / math.log1p(max_level))


def _level_coef_log_bounded_dec(level: int, max_level: int, beta_max: float = 0.5, floor: float = 0.2) -> float:
    """
    Normalized logarithmic negative correlation coefficient (with lower bound):
        coef_dec_raw = 1 - beta_max * log(1+level) / log(1+max_level)
    coef ∈ [1-beta_max, 1]
    Apply floor as hard lower bound:
        coef_dec = max(coef_dec_raw, floor)

    Args:
      beta_max: Maximum reduction (e.g., 0.5 => lowest 0.5x, before floor)
      floor: Hard lower bound multiplier (e.g., 0.2)
    """
    level = 0 if level is None else int(level)
    max_level = max(0, int(max_level))
    if max_level <= 0:
        return 1.0

    raw = 1.0 - float(beta_max) * (math.log1p(level) / math.log1p(max_level))
    return max(raw, float(floor))


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

    # Level coefficient parameters
    use_level_scaling=True,

    # P: positive correlation, max scale up to (1 + p_alpha_max)
    p_alpha_max=0.8,

    # C/D: negative correlation, max scale down to (1 - cd_beta_max), protected by floor
    cd_beta_max=0.5,
    cd_floor=0.2,

    round_scaled_int=True,
):
    """
    Generate node values with level-based scaling:
      - P_loss: positive correlation with level (normalized log amplification, upper bound 1+p_alpha_max)
      - C_benefit: negative correlation with level (normalized log reduction, lower bound from 1-cd_beta_max and floor)
      - D_cost: negative correlation with level (same as C)
      - E_prob: unchanged
      - P_benefit: unchanged (not level-dependent)
    """
    if seed is not None:
        random.seed(seed)

    if fixed_e_probs is None or len(fixed_e_probs) == 0:
        raise ValueError("fixed_e_probs must be provided (e.g., containing 24 fixed values between 0 and 1)")

    p_loss_choices = _make_fixed_int_choices(p_loss_range, k=fixed_int_k)
    c_benefit_choices = _make_fixed_int_choices(c_benefit_range, k=fixed_int_k)
    d_cost_choices = _make_fixed_int_choices(d_cost_range, k=fixed_int_k)

    level_map = bn.get("level", {})
    max_level = max(level_map.values()) if level_map else 0

    def scale_for_PD(x_int: int, node: str):
        if not use_level_scaling:
            return int(x_int)
        lv = level_map.get(node, 0)
        coef = _level_coef_log_bounded_inc(lv, max_level, alpha_max=p_alpha_max)
        val = x_int * coef
        return int(round(val)) if round_scaled_int else val

    def scale_for_C(x_int: int, node: str):
        if not use_level_scaling:
            return int(x_int)
        lv = level_map.get(node, 0)
        coef = _level_coef_log_bounded_dec(lv, max_level, beta_max=cd_beta_max, floor=cd_floor)
        val = x_int * coef
        return int(round(val)) if round_scaled_int else val

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
            if node == "P0":
                row["P_loss"] = 0
            else:
                base = random.choice(p_loss_choices)
                row["P_loss"] = scale_for_PD(base, node)
            row["P_benefit"] = round(random.uniform(*p_benefit_range), 2)

        elif t == "C":
            base = random.choice(c_benefit_choices)
            row["C_benefit"] = scale_for_C(base, node)

        elif t == "D":
            base = random.choice(d_cost_choices)
            row["D_cost"] = scale_for_PD(base, node)

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
    headers = ["node", "type", "P_loss", "P_benefit", "C_benefit", "D_cost", "E_prob"]
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
