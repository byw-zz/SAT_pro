# -*- coding: utf-8 -*-
import math
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# -----------------------------
# 1) Input file paths
# -----------------------------
ARCS_CSV = "ARCS.CSV"
VERTICES_CSV = "VERTICES.CSV"
VERTICES_MAPPED_CSV = "VERTICES_MAPPED.CSV"

# -----------------------------
# 2) DEFENSE_PLAN (Cxx → Dyy)
# -----------------------------
DEFENSE_PLAN = {
    # Core asset Admin Workstation (P1)
    "D1_A": ["C19"],  # Only fix BlueKeep (RDP)
    "D1_B": ["C30"],  # Only fix SMBGhost (SMB)

    "D2": ["C18", "C29"],  # Harden RDP/SMB service configuration

    "D3_Ext": ["C17"],          # Block external attack source direct to Admin
    "D3_Int": ["C13", "C21"],   # Block internal Gateway to Admin path (merge as D3)
    "D4": ["C1", "C20", "C22"], # Block DB/FileSrv lateral movement to Admin

    # Boundary gateway Edge Gateway (P9)
    "D5_Vuln": ["C16"],
    "D5_Cfg": ["C15"],
    "D5_ACL": ["C14"],

    # Core database DB Core (P3)
    "D8_Vuln": ["C12"],
    "D8_Cfg": ["C11"],
    "D9": ["C2", "C7"],

    # Other DMZ services
    "D6": ["C6", "C5", "C3"],
    "D7": ["C10", "C9", "C8"],
    "D10": ["C28", "C27", "C23"],
    "D11": ["C26", "C25", "C24"],

    "D12": ["C4"],  # Threat intelligence block (no merge)
}

# D3_Int unified as D3; D3_Ext and D12 not merged
DEF_NODE_ALIAS = {"D3_Int": "D3"}


def load_attack_graph(arcs_csv: str, vertices_csv: str, vertices_mapped_csv: str) -> nx.DiGraph:
    """
    Build attack graph from ARCS / VERTICES / VERTICES_MAPPED:
    - Nodes: P/E/C codes
    - Edges: (src_id -> dst_id) from arcs.csv
    Note: VERTICES_MAPPED must align with VERTICES by row order (desc may repeat)
    """
    arcs = pd.read_csv(arcs_csv, header=None, names=["src", "dst", "w"])
    verts = pd.read_csv(vertices_csv, header=None, names=["id", "desc", "node_type", "is_leaf"])
    mapped = pd.read_csv(vertices_mapped_csv, header=None, names=["code", "desc"])

    if len(verts) != len(mapped):
        raise ValueError(f"VERTICES row count ({len(verts)}) does not match VERTICES_MAPPED ({len(mapped)})")

    if not (verts["desc"].astype(str).reset_index(drop=True) == mapped["desc"].astype(str).reset_index(drop=True)).all():
        raise ValueError("VERTICES and VERTICES_MAPPED desc not aligned by row, cannot map by row")

    verts = verts.copy()
    verts["code"] = mapped["code"].astype(str)

    id2code = dict(zip(verts["id"].astype(int), verts["code"]))

    G = nx.DiGraph()
    for _, r in verts.iterrows():
        G.add_node(
            r["code"],
            kind="attack",
            node_type=str(r["node_type"]),
            desc=str(r["desc"]),
        )

    for _, r in arcs.iterrows():
        u = id2code[int(r["src"])]
        v = id2code[int(r["dst"])]
        G.add_edge(u, v, kind="attack")

    return G


def attach_defense_plan(G: nx.DiGraph, defense_plan: dict, alias: dict) -> nx.DiGraph:
    """
    Add defense nodes D*, and add edges Cxx -> Dyy according to mapping.
    """
    H = G.copy()
    for d, cs in defense_plan.items():
        d2 = alias.get(d, d)

        if d2 not in H:
            H.add_node(d2, kind="defense", node_type="DEF", desc=d2)

        for c in cs:
            if c not in H:
                H.add_node(c, kind="attack", node_type="LEAF", desc=c)

            H.add_edge(c, d2, kind="defense")

    return H


def layout_graph(G: nx.DiGraph):
    """
    Prefer graphviz(dot) layered layout; fall back to spring_layout on failure.
    """
    try:
        from networkx.drawing.nx_agraph import graphviz_layout
        pos = graphviz_layout(G, prog="dot")
        return pos
    except Exception:
        k = 1 / math.sqrt(max(1, G.number_of_nodes()))
        pos = nx.spring_layout(G, seed=42, k=k)
        return pos


def plot_graph(G: nx.DiGraph, out_png: str = "attack_defense_graph.png"):
    """
    - OR/AND/LEAF use different shapes
    - D* defense nodes placed on the right side
    - Attack edges: solid lines; defense edges: dashed
    """
    pos = layout_graph(G)

    def_nodes = [n for n, a in G.nodes(data=True) if a.get("kind") == "defense"]
    if def_nodes:
        max_x = max(x for x, y in pos.values())
        min_y = min(y for x, y in pos.values())
        max_y = max(y for x, y in pos.values())
        step = (max_y - min_y) / max(1, len(def_nodes) - 1)
        for i, n in enumerate(sorted(def_nodes)):
            pos[n] = (max_x + 300, max_y - i * step)

    attack_edges = [(u, v) for u, v, a in G.edges(data=True) if a.get("kind") == "attack"]
    defense_edges = [(u, v) for u, v, a in G.edges(data=True) if a.get("kind") == "defense"]

    or_nodes = [n for n, a in G.nodes(data=True) if a.get("kind") == "attack" and a.get("node_type") == "OR"]
    and_nodes = [n for n, a in G.nodes(data=True) if a.get("kind") == "attack" and a.get("node_type") == "AND"]
    leaf_nodes = [n for n, a in G.nodes(data=True) if a.get("kind") == "attack" and a.get("node_type") == "LEAF"]

    plt.figure(figsize=(22, 12))

    nx.draw_networkx_edges(
        G, pos, edgelist=attack_edges,
        arrows=True, alpha=0.25, arrowsize=10, width=1.0
    )
    nx.draw_networkx_edges(
        G, pos, edgelist=defense_edges,
        arrows=True, style="dashed", alpha=0.85, arrowsize=12, width=1.6
    )

    nx.draw_networkx_nodes(G, pos, nodelist=or_nodes, node_shape="o", node_size=650)
    nx.draw_networkx_nodes(G, pos, nodelist=and_nodes, node_shape="s", node_size=650)
    nx.draw_networkx_nodes(G, pos, nodelist=leaf_nodes, node_shape="^", node_size=650)
    nx.draw_networkx_nodes(G, pos, nodelist=def_nodes, node_shape="h", node_size=900)

    labels = {n: n for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)

    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    G_attack = load_attack_graph(ARCS_CSV, VERTICES_CSV, VERTICES_MAPPED_CSV)
    G_full = attach_defense_plan(G_attack, DEFENSE_PLAN, DEF_NODE_ALIAS)
    plot_graph(G_full, out_png="attack_defense_graph.png")
