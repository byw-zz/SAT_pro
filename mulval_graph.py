"""Convert MulVAL attack graph to Max-SAT CNF, call MaxHS solver, parse results."""

import argparse
import sys
import subprocess
import json
from pathlib import Path

import pandas as pd
import networkx as nx

from collections import defaultdict

from graph2sat.attackgraph2sat import (
    bn_to_maxsat_cnf_attackgraph,
    export_to_wcnf as _export_wcnf_v1,
    build_adjacency,
)
from graph2sat.attackgraph2sat_without_pro import (
    bn_to_maxsat_cnf_attackgraph as bn_to_maxsat_cnf_v2,
    export_to_wcnf as _export_wcnf_v2,
)
import io, sys

def export_wcnf_v1(cnf_data, filename, hard_weight=1000000, soft_scale=1):
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    _export_wcnf_v1(cnf_data, filename, hard_weight, soft_scale)
    sys.stdout = old_stdout

def export_wcnf_v2(cnf_data, filename, hard_weight=1000000, soft_scale=1):
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    _export_wcnf_v2(cnf_data, filename, hard_weight, soft_scale)
    sys.stdout = old_stdout

MAXHS_BIN = "MaxHS/build/release/bin/maxhs"

def read_csv_maybe_noheader(path, min_cols):
    df = pd.read_csv(path, header=None)
    if df.shape[1] < min_cols:
        df2 = pd.read_csv(path, header=0)
        if df2.shape[1] >= min_cols:
            return df2
        raise ValueError(f"{path} has too few columns (need >= {min_cols}).")
    return df

def build_mapped_labels_and_types(verts_df):
    """
    Map labels to node types:
    - label contains 'rule'/'RULE' => E_i
    - label contains 'netAccess' OR 'execCode' => P_i
    - otherwise => C_i
    """
    e_i = p_i = c_i = 0
    id_to_mapped = {}
    id_to_type = {}

    ids = verts_df["id"].astype(int).tolist()
    labels = verts_df["label"].astype(str).tolist()

    for vid, lbl in zip(ids, labels):
        if ("rule" in lbl) or ("RULE" in lbl):
            e_i += 1
            id_to_mapped[vid] = f"E{e_i}"
            id_to_type[vid] = "E"
        elif ("netAccess" in lbl) or ("execCode" in lbl):
            p_i += 1
            id_to_mapped[vid] = f"P{p_i}"
            id_to_type[vid] = "P"
        else:
            c_i += 1
            id_to_mapped[vid] = f"C{c_i}"
            id_to_type[vid] = "C"

    return id_to_mapped, id_to_type

def csv_to_bn_dict(vertices_csv, arcs_csv, arcs_order="dest_src"):
    """Convert MulVAL VERTICES/ARCS to bn dict with same interface as generate_bn_dag_multi_pe."""
    arcs = read_csv_maybe_noheader(arcs_csv, min_cols=2).copy()
    verts = read_csv_maybe_noheader(vertices_csv, min_cols=2).copy()

    rename_map = {}
    if verts.shape[1] >= 1: rename_map[verts.columns[0]] = "id"
    if verts.shape[1] >= 2: rename_map[verts.columns[1]] = "label"
    if verts.shape[1] >= 3: rename_map[verts.columns[2]] = "kind"
    if verts.shape[1] >= 4: rename_map[verts.columns[3]] = "leaf_flag"
    verts.rename(columns=rename_map, inplace=True)

    verts["id"] = pd.to_numeric(verts["id"], errors="coerce").astype("Int64")
    verts = verts.dropna(subset=["id"]).copy()
    verts["id"] = verts["id"].astype(int)
    verts["label"] = verts["label"].astype(str)

    arcs = arcs.iloc[:, :2].copy()
    if arcs_order == "dest_src":
        arcs.columns = ["dest", "src"]
    elif arcs_order == "src_dest":
        arcs.columns = ["src", "dest"]
    else:
        raise ValueError("arcs_order must be 'dest_src' or 'src_dest'.")

    arcs["src"] = pd.to_numeric(arcs["src"], errors="coerce").astype("Int64")
    arcs["dest"] = pd.to_numeric(arcs["dest"], errors="coerce").astype("Int64")
    arcs = arcs.dropna(subset=["src", "dest"]).copy()
    arcs["src"] = arcs["src"].astype(int)
    arcs["dest"] = arcs["dest"].astype(int)

    id_to_mapped, id_to_type = build_mapped_labels_and_types(verts)
    id_to_raw = dict(zip(verts["id"].tolist(), verts["label"].tolist()))

    edges = []
    for s, d in arcs[["src", "dest"]].itertuples(index=False):
        if s not in id_to_mapped or d not in id_to_mapped:
            continue
        u = id_to_mapped[s]
        v = id_to_mapped[d]
        if u != v:
            edges.append((u, v))

    P_nodes = []
    E_nodes = []
    C_nodes = []
    D_nodes = []

    node_type = {}
    node_meta = {}

    for vid in verts["id"].tolist():
        n = id_to_mapped[vid]
        t = id_to_type[vid]
        node_type[n] = t
        node_meta[n] = {"vertex_id": vid, "raw_label": id_to_raw.get(vid, "")}

        if t == "P":
            P_nodes.append(n)
        elif t == "E":
            E_nodes.append(n)
        else:
            C_nodes.append(n)

    out_degree = defaultdict(int)
    for u, v in edges:
        out_degree[u] += 1

    bn = {
        "P": P_nodes,
        "E": E_nodes,
        "C": C_nodes,
        "D": D_nodes,
        "edges": edges,
        "node_type": node_type,
        "out_degree": dict(out_degree),
        "node_meta": node_meta,
        "raw_id_map": {vid: id_to_mapped[vid] for vid in verts["id"].tolist()},
    }
    return bn

def layered_layout_from_bn(bn):
    G = nx.DiGraph()
    for n in bn["P"] + bn["E"] + bn["C"] + bn["D"]:
        G.add_node(n, kind=bn["node_type"][n])
    G.add_edges_from(bn["edges"])

    types_order = ["P", "E", "C", "D"]
    nodes_by_type = {t: [] for t in types_order}
    for n in G.nodes():
        nodes_by_type[G.nodes[n]["kind"]].append(n)

    pos = {}
    y_gap = 1.5
    x_gap = 2.0
    for layer_idx, t in enumerate(types_order):
        layer_nodes = nodes_by_type[t]
        if not layer_nodes:
            continue
        N = len(layer_nodes)
        start_x = -(N - 1) * x_gap / 2.0
        y = -layer_idx * y_gap
        for i, n in enumerate(layer_nodes):
            pos[n] = (start_x + i * x_gap, y)
    return G, pos

def visualize_bn_like_generator(bn, layout="layered", show_raw=False):
    if layout == "layered":
        G, pos = layered_layout_from_bn(bn)
    else:
        G = nx.DiGraph()
        for n in bn["P"] + bn["E"] + bn["C"] + bn["D"]:
            G.add_node(n, kind=bn["node_type"][n])
        G.add_edges_from(bn["edges"])
        pos = nx.spring_layout(G, seed=0)

    color_map = {"P": "#1f77b4", "E": "#ff7f0e", "C": "#2ca02c", "D": "#d62728"}
    node_colors = [color_map.get(G.nodes[n]["kind"], "#999999") for n in G.nodes()]

    if show_raw and "node_meta" in bn:
        labels = {}
        for n in G.nodes():
            raw = bn["node_meta"].get(n, {}).get("raw_label", "")
            labels[n] = f"{n}\n{raw}"
    else:
        labels = {n: n for n in G.nodes()}

    plt.figure(figsize=(12, 7))
    nx.draw(
        G, pos,
        with_labels=True,
        labels=labels,
        node_color=node_colors,
        node_size=700,
        arrows=True,
        arrowstyle="->",
        arrowsize=14,
        font_size=8,
    )
    plt.title("BN from CSV (same bn-dict interface as generate_bn_dag_multi_pe)")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

DEFENSE_PLAN = {
    "D1_A": ["C19"],
    "D1_B": ["C30"],
    "D2": ["C18", "C29"],
    "D3_Ext": ["C17"],
    "D3_Int": ["C13", "C21"],
    "D4":     ["C1", "C20", "C22"],
    "D5_Vuln": ["C16"],
    "D5_Cfg":  ["C15"],
    "D5_ACL":  ["C14"],
    "D8_Vuln": ["C12"],
    "D8_Cfg":  ["C11"],
    "D9":      ["C2", "C7"],
    "D6": ["C6", "C5", "C3"],
    "D7": ["C10", "C9", "C8"],
    "D10": ["C28", "C27", "C23"],
    "D11": ["C26", "C25", "C24"],
    "D12": ["C4"],
}

import copy
import re
import networkx as nx
import matplotlib.pyplot as plt

def normalize_bn_for_visual(bn):
    bn2 = copy.deepcopy(bn)
    rename, drop = {}, set()

    for n in bn2.get("C", []):
        m = re.fullmatch(r"C(\d+)", n)
        if m:
            cid = int(m.group(1))
            if cid == 17:
                drop.add(n)
            elif cid > 17:
                rename[n] = f"C{cid-1}"

    for n in bn2.get("D", []):
        if n in {"D12", "D3_Ext"}:
            drop.add(n); continue
        if n == "D3_Int":
            rename[n] = "D3"; continue
        m = re.fullmatch(r"D(\d+)_([A-Za-z]).*", n)
        if m:
            rename[n] = f"D{m.group(1)}_{m.group(2)}"

    def map_node(x):
        if x in drop:
            return None
        return rename.get(x, x)

    new_edges = []
    for u, v in bn2.get("edges", []):
        nu, nv = map_node(u), map_node(v)
        if nu is None or nv is None:
            continue
        new_edges.append((nu, nv))
    seen = set()
    bn2["edges"] = [e for e in new_edges if not (e in seen or seen.add(e))]

    for k in ["P", "E", "C", "D"]:
        out = []
        for x in bn2.get(k, []):
            nx_ = map_node(x)
            if nx_ is None:
                continue
            out.append(nx_)
        seen = set()
        bn2[k] = [x for x in out if not (x in seen or seen.add(x))]

    if "node_type" in bn2:
        nt2 = {}
        for old, kind in bn2["node_type"].items():
            new = map_node(old)
            if new is None:
                continue
            nt2.setdefault(new, kind)
        bn2["node_type"] = nt2

    if "node_meta" in bn2:
        nm2 = {}
        for old, meta in bn2["node_meta"].items():
            new = map_node(old)
            if new is None:
                continue
            nm2.setdefault(new, meta)
        bn2["node_meta"] = nm2

    return bn2


def pretty_node_label(n: str) -> str:
    m = re.fullmatch(r"([PCEF D])(\d+)", n.replace(" ", ""))
    if m:
        head, num = m.group(1), m.group(2)
        return rf"$\mathrm{{{head}}}_{{{num}}}$"

    m = re.fullmatch(r"([PCEF D])(\d+)_([A-Za-z]).*", n.replace(" ", ""))
    if m:
        head, num, suf = m.group(1), m.group(2), m.group(3)
        return rf"$\mathrm{{{head}}}_{{{num}{suf}}}$"

    return n



def visualize_bn_like_generator(
    bn,
    layout="layered",
    show_raw=False,
    save_path=None,
    dpi=300
):
    bn = normalize_bn_for_visual(bn)

    if layout == "layered":
        G, pos = layered_layout_from_bn(bn)
    else:
        G = nx.DiGraph()
        for n in bn["P"] + bn["E"] + bn["C"] + bn["D"]:
            G.add_node(n, kind=bn["node_type"].get(n, "UNK"))
        G.add_edges_from(bn["edges"])
        pos = nx.spring_layout(G, seed=0, k=0.25)

    color_map = {"P": "#d62728", "E": "#ff7f0e", "C": "#1f77b4", "D":  "#2ca02c", "UNK": "#999999"}
    node_colors = [color_map.get(G.nodes[n].get("kind", "UNK"), "#999999") for n in G.nodes()]

    plt.rcParams["mathtext.default"] = "regular"

    if show_raw and "node_meta" in bn:
        labels = {}
        for n in G.nodes():
            raw = bn["node_meta"].get(n, {}).get("raw_label", "")
            base = pretty_node_label(n)
            labels[n] = f"{base}\n{raw}" if raw else base
    else:
        labels = {n: pretty_node_label(n) for n in G.nodes()}

    fig = plt.figure(figsize=(10, 5.5))
    nx.draw(
        G, pos,
        with_labels=True,
        labels=labels,
        node_color=node_colors,
        node_size=450,
        arrows=True,
        arrowstyle="->",
        arrowsize=10,
        font_size=7,
    )
    plt.axis("off")
    plt.margins(0)
    plt.tight_layout(pad=0.2)

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=dpi)
        plt.close(fig)
        return save_path
    else:
        plt.show()
        return None

def add_defenses_to_bn(
    bn,
    defense_plan,
    max_c_per_d=3,
    assign_raw_ids=True,
    d_meta_prefix="defense:",
):
    """Add defense nodes to existing bn."""
    if "node_type" not in bn or "edges" not in bn:
        raise ValueError("bn missing node_type or edges.")
    if "C" not in bn:
        raise ValueError("bn missing C list.")

    all_C = set(bn["C"])
    used_C = set()

    for d, c_list in defense_plan.items():
        if len(c_list) > max_c_per_d:
            raise ValueError(f"{d} covers {len(c_list)} C nodes, exceeds limit {max_c_per_d}.")
        for c in c_list:
            if c not in all_C:
                raise KeyError(f"{d} references non-existent condition node: {c}")
            if c in used_C:
                raise ValueError(f"Condition node {c} assigned to multiple defenses.")
            used_C.add(c)

    missing = [c for c in bn["C"] if c not in used_C]
    if missing:
        raise ValueError(f"Unassigned condition nodes: {missing[:20]}... (total {len(missing)})")

    bn.setdefault("D", [])
    bn.setdefault("node_meta", {})
    bn.setdefault("raw_id_map", {})

    max_raw = 0
    if assign_raw_ids and bn["raw_id_map"]:
        max_raw = max(int(x) for x in bn["raw_id_map"].keys())

    edges_set = set(tuple(e) for e in bn["edges"])

    for d, c_list in defense_plan.items():
        if d not in bn["node_type"]:
            bn["node_type"][d] = "D"
        if d not in bn["D"]:
            bn["D"].append(d)

        if d not in bn["node_meta"]:
            bn["node_meta"][d] = {
                "vertex_id": None,
                "raw_label": f"{d_meta_prefix}{d}"
            }

        if assign_raw_ids:
            max_raw += 1
            bn["raw_id_map"][max_raw] = d
            bn["node_meta"][d]["vertex_id"] = max_raw

        for c in c_list:
            edges_set.add((c, d))

    bn["edges"] = list(edges_set)

    out_degree = defaultdict(int)
    for u, v in bn["edges"]:
        out_degree[u] += 1
    bn["out_degree"] = dict(out_degree)

    return bn

import random
import re

def _make_fixed_int_choices(value_range, k=5):
    lo, hi = value_range
    lo_i, hi_i = int(lo), int(hi)
    if lo_i > hi_i:
        lo_i, hi_i = hi_i, lo_i

    domain_size = hi_i - lo_i + 1
    if domain_size < k:
        raise ValueError(f"Range {value_range} has insufficient integers for {k} unique samples.")

    return random.sample(range(lo_i, hi_i + 1), k)

_CVE_RE = re.compile(r"\bCVE-\d{4}-\d{4,7}\b", re.IGNORECASE)

def _extract_one_cve(text: str):
    if not text:
        return None
    cves = [m.group(0).upper() for m in _CVE_RE.finditer(text)]
    if not cves:
        return None
    seen = set()
    uniq = []
    for c in cves:
        if c not in seen:
            seen.add(c)
            uniq.append(c)
    if len(uniq) > 1:
        raise ValueError(f"Multiple CVEs found: {uniq}")
    return uniq[0]

def _build_children_index(edges):
    children = {}
    for u, v in edges:
        children.setdefault(u, []).append(v)
    return children

def _e_prob_from_children_cve(bn, e_node, children_index, cve_to_exploitability, default_no_cve=1.0, round_prob=4):
    """If child raw_label contains CVE, E_prob = exploitability/10, else 1.0"""
    node_meta = bn.get("node_meta", {})

    found_cve = None
    for child in children_index.get(e_node, []):
        raw = ""
        if isinstance(node_meta, dict):
            raw = (node_meta.get(child, {}) or {}).get("raw_label", "") or ""
        if not raw:
            raw = str(child)

        cve = _extract_one_cve(raw)
        if cve:
            if found_cve and cve != found_cve:
                raise ValueError(f"{e_node} children have multiple different CVEs: {found_cve}, {cve}")
            found_cve = cve

    if not found_cve:
        return float(default_no_cve)

    if found_cve not in cve_to_exploitability:
        raise ValueError(f"Missing exploitability score for {found_cve} (referenced by {e_node}).")

    p = float(cve_to_exploitability[found_cve]) / 10.0
    if p < 0.0: p = 0.0
    if p > 1.0: p = 1.0
    return round(p, round_prob)

def generate_node_values_attackgraph_with_specified_values_per_node(
    bn,
    specified_p_loss=None,
    specified_c_benefit=None,
    specified_d_cost=None,
    seed=None,
    cve_to_exploitability=None,
    e_default_no_cve=1.0,
    round_e_prob=4,
):
    """Generate node values for attack graph, allowing per-node specified values."""
    if seed is not None:
        random.seed(seed)

    if not isinstance(cve_to_exploitability, dict):
        raise ValueError("cve_to_exploitability (dict: CVE -> exploitability) must be provided.")

    children_index = _build_children_index(bn.get("edges", []))

    values_table = []
    all_nodes = []
    for t in ["P", "E", "C", "D"]:
        all_nodes.extend(bn.get(t, []))

    for node in all_nodes:
        t = bn["node_type"][node]
        row = {
            "node": node,
            "type": t,
            "P_loss": None,
            "C_benefit": None,
            "D_cost": None,
            "E_prob": None
        }

        if t == "P":
            row["P_loss"] = specified_p_loss.get(node, random.choice(_make_fixed_int_choices((200, 500), k=5)))

        elif t == "C":
            row["C_benefit"] = specified_c_benefit.get(node, random.choice(_make_fixed_int_choices((100, 500), k=5)))

        elif t == "D":
            row["D_cost"] = specified_d_cost.get(node, random.choice(_make_fixed_int_choices((100, 500), k=5)))

        elif t == "E":
            row["E_prob"] = _e_prob_from_children_cve(
                bn=bn,
                e_node=node,
                children_index=children_index,
                cve_to_exploitability=cve_to_exploitability,
                default_no_cve=e_default_no_cve,
                round_prob=round_e_prob,
            )

        values_table.append(row)

    return values_table

cve_to_exploitability = {
    "CVE-2021-41773": 3.9,
    "CVE-2018-15473": 3.9,
    "CVE-2021-44228": 3.9,
    "CVE-2019-9193": 1.2,
    "CVE-2019-0708": 3.9,
    "CVE-2020-8617": 2.2,
    "CVE-2017-7494": 3.9,
    "CVE-2020-0796": 3.9
}

specified_p_loss = {
    "P1": 15000,
    "P3": 8000,
    "P9": 6000,
    "P14": 4000,
    "P5": 3000, "P7": 3000,
    "P2": 1000, "P4": 1000, "P6": 1000, "P8": 1000,
    "P10": 1000, "P11": 1000, "P12": 1000, "P13": 1000, "P15": 1000
}

specified_c_benefit = {
    "C11": 10000, "C18": 10000, "C29": 10000,
    "C5": 6000, "C9": 6000, "C25": 6000,
    "C15": 4000, "C27": 4000,
    "C1": 200, "C2": 200, "C3": 200, "C7": 200, "C8": 200,
    "C13": 200, "C14": 200, "C17": 200, "C20": 200, "C21": 200,
    "C22": 200, "C23": 200, "C24": 200,
    "C4": 0, "C6": 0, "C10": 0, "C12": 0,
    "C16": 0, "C19": 0, "C26": 0, "C28": 0, "C30": 0
}

specified_d_cost = {
    "D1_A": 2500, "D1_B": 2500,
    "D8_Vuln": 2000,
    "D5_Vuln": 1500, "D11": 1200, "D6": 1200, "D7": 1200, "D10": 1200,
    "D2": 1000, "D5_Cfg": 800, "D8_Cfg": 800,
    "D3_Ext": 300, "D3_Int": 400, "D4": 400, "D5_ACL": 300, "D9": 400,
    "D12": 1500,
}

from common import parse_maxhs_solution_str


def interpret_solution(bn, cnf_data, assignment):
    """Interpret solution based on assignment and var_map."""
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
    """Convert values_table to dict indexed by node name."""
    return {row["node"]: row for row in values_table}


def compute_objective_from_solution(bn, values_table, node_state, node_state_by_type):
    """Calculate objective value from solution."""
    parents, children = build_adjacency(bn)
    node_type = bn["node_type"]
    vindex = build_value_index(values_table)

    total_C_benefit = 0.0
    total_P_expected_loss = 0.0
    total_D_cost = 0.0

    for c, active in node_state_by_type["C"].items():
        if not active:
            continue
        benefit = vindex.get(c, {}).get("C_benefit", 0.0)
        total_C_benefit += benefit

    for p, active_p in node_state_by_type["P"].items():
        if not active_p:
            continue
        p_loss = vindex.get(p, {}).get("P_loss", 0.0)
        e_children = [
            e for e in children[p]
            if node_type.get(e) == "E" and node_state.get(e, False)
        ]
        prob_not_comp = 1.0
        for e in e_children:
            prob_e = vindex.get(e, {}).get("E_prob", 0.0)
            prob_not_comp *= (1.0 - prob_e)
        p_comp = 1.0 - prob_not_comp if e_children else 0.0
        total_P_expected_loss += p_comp * p_loss

    for d, active in node_state_by_type["D"].items():
        if not active:
            continue
        d_cost = vindex.get(d, {}).get("D_cost", 0.0)
        total_D_cost += d_cost

    objective = total_C_benefit - total_P_expected_loss - total_D_cost

    return {
        "C_benefit": total_C_benefit,
        "P_expected_loss": total_P_expected_loss,
        "D_cost": total_D_cost,
        "objective": objective,
    }


def print_solution_summary(node_state_by_type, objective_info):
    """Print solution summary."""
    print("\n========== Solution Summary ==========")
    print(f"Objective value: {objective_info['objective']:.4f}")
    print(f"  C_benefit:       {objective_info['C_benefit']:.4f}")
    print(f"  P_expected_loss: {objective_info['P_expected_loss']:.4f}")
    print(f"  D_cost:          {objective_info['D_cost']:.4f}")

    for node_type_name in ["P", "E", "C", "D"]:
        nodes = node_state_by_type.get(node_type_name, {})
        if not nodes:
            continue
        true_nodes = [n for n, v in nodes.items() if v]
        false_nodes = [n for n, v in nodes.items() if not v]
        print(f"\nNode type [{node_type_name}]:")
        print(f"  True:  {len(true_nodes)} nodes: {sorted(true_nodes)}")
        print(f"  False: {len(false_nodes)} nodes: {sorted(false_nodes)}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert MulVAL CSV to Max-SAT CNF -> MaxHS solve -> output node states",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--arcs", required=True, help="ARCS.CSV file path")
    parser.add_argument("--vertices", required=True, help="VERTICES.CSV file path")
    parser.add_argument(
        "--arcs-order",
        choices=["dest_src", "src_dest"],
        default="src_dest",
        help="ARCS CSV column order"
    )

    parser.add_argument(
        "--sat",
        choices=["pro", "without_pro"],
        default="pro",
        help="SAT conversion function"
    )

    parser.add_argument(
        "--initial-true",
        default="C4,P2",
        help="Nodes initially true, comma-separated"
    )

    parser.add_argument(
        "--maxhs-bin",
        default=MAXHS_BIN,
        help="MaxHS executable path"
    )
    parser.add_argument(
        "--maxhs-timeout",
        type=int,
        default=300,
        help="MaxHS timeout in seconds"
    )
    parser.add_argument(
        "--no-solve",
        action="store_true",
        help="Only generate WCNF, skip MaxHS solving"
    )
    parser.add_argument(
        "--output",
        default="attack_graph.wcnf",
        help="WCNF output filename"
    )
    parser.add_argument(
        "--save-result",
        action="store_true",
        help="Save solving result as JSON"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Visualize attack graph before solving"
    )

    args = parser.parse_args()

    print(f"[1/5] Reading ARCS: {args.arcs}")
    print(f"[1/5] Reading VERTICES: {args.vertices}")
    bn = csv_to_bn_dict(args.vertices, args.arcs, arcs_order=args.arcs_order)

    print(f"[1/5] Node stats: P={len(bn['P'])}, E={len(bn['E'])}, C={len(bn['C'])}, D={len(bn['D'])}")
    print(f"[1/5] Edge count: {len(bn['edges'])}")

    print(f"[1/5] Adding defense nodes (DEFENSE_PLAN)...")
    try:
        add_defenses_to_bn(bn, DEFENSE_PLAN)
        print(f"[1/5] Node stats: P={len(bn['P'])}, E={len(bn['E'])}, C={len(bn['C'])}, D={len(bn['D'])}")
    except Exception as e:
        print(f"[ERROR] Failed to add defense nodes: {e}", file=sys.stderr)
        return

    if args.visualize:
        print("[2/5] Visualizing attack graph...")
        visualize_bn_like_generator(bn, layout="layered", show_raw=False)

    print("[2/5] Generating node values...")
    values_table = generate_node_values_attackgraph_with_specified_values_per_node(
        bn,
        specified_p_loss=specified_p_loss,
        specified_c_benefit=specified_c_benefit,
        specified_d_cost=specified_d_cost,
        cve_to_exploitability=cve_to_exploitability,
        e_default_no_cve=1.0,
    )

    print(f"[3/5] Using SAT conversion: {args.sat}")
    if args.sat == "pro":
        bn_to_maxsat_cnf = bn_to_maxsat_cnf_attackgraph
        export_wcnf = export_wcnf_v1
    else:
        bn_to_maxsat_cnf = bn_to_maxsat_cnf_v2
        export_wcnf = export_wcnf_v2

    initial_true_nodes = [n.strip() for n in args.initial_true.split(",") if n.strip()]

    cnf_data = bn_to_maxsat_cnf(bn, values_table, initial_true_nodes=initial_true_nodes)
    wcnf_path = Path(args.output).resolve()
    export_wcnf(cnf_data, str(wcnf_path))
    print(f"[3/5] WCNF exported: {wcnf_path}")

    if args.no_solve:
        print("[4/5] Skip solving (--no-solve)")
        print("[5/5] Skip analysis (--no-solve)")
        return

    print(f"[4/5] Calling MaxHS (timeout {args.maxhs_timeout}s)...")
    extra = [
        "-printSoln",
        "-printBstSoln",
        "-verb=0",
        f"-cpu-lim={args.maxhs_timeout}",
    ]
    cmd = [args.maxhs_bin] + extra + [str(wcnf_path)]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=args.maxhs_timeout + 5)
        maxhs_out = result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        print(f"[WARNING] MaxHS timeout ({args.maxhs_timeout}s)")
        maxhs_out = ""

    print("[5/5] Parsing solution...")
    n_vars = len(cnf_data["var_map"])
    assignment = parse_maxhs_solution_str(maxhs_out, n_vars)

    if not assignment:
        print("[ERROR] No valid solution found", file=sys.stderr)
        return

    node_state, node_state_by_type = interpret_solution(bn, cnf_data, assignment)
    objective_info = compute_objective_from_solution(bn, values_table, node_state, node_state_by_type)

    print_solution_summary(node_state_by_type, objective_info)

    if args.save_result:
        result_path = wcnf_path.with_suffix(".result.json")
        result_data = {
            "objective": objective_info["objective"],
            "C_benefit": objective_info["C_benefit"],
            "P_expected_loss": objective_info["P_expected_loss"],
            "D_cost": objective_info["D_cost"],
            "assignment": {str(k): v for k, v in assignment.items()},
            "node_state": node_state,
        }
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)
        print(f"\nResult saved to: {result_path}")


if __name__ == "__main__":
    main()
