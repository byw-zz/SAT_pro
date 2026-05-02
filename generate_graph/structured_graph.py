import random
import math
from collections import defaultdict, deque

import networkx as nx
import matplotlib.pyplot as plt


def generate_bn_from_root_and_reverse(
    nP: int,
    seed: int | None = None,
    # P subgraph (root DAG) parameters
    extra_p_edge_prob: float = 0.25,
    max_extra_p_out_per_node: int = 4,
    # E/C/D structure parameters
    c_parents_per_e_range=(1, 3),
    max_e_children_per_c: int = 5,
    max_c_children_per_d: int = 3,
    # Node counts (optional: auto-generated if not provided)
    nC: int | None = None,
    nD: int | None = None,
):
    """
    Generate directed graph with four node types P/E/C/D, then reverse all edges.

    Generation logic (before reversal):
    1) Generate P subgraph DAG with P0 as root, compute shortest path length level(Pi) from P0 to each Pi
    2) Insert E for each P->P edge: Pi -> Ek -> Pj
    3) Assign 1-3 C as parent nodes to each E: C -> E
        - Each C's child count (E) <= max_e_children_per_c
    4) Assign one unique D parent to each C: D -> C
        - Each D's child count (C) <= max_c_children_per_d
    5) Assign levels:
        - level(Ek) = level(Pi) (Ek inserted on Pi->Pj edge)
        - level(C) = min(level(E) for E connected to C) (if C connects to multiple E)
        - level(D) = min(level(C) for C as its child) (optional, for debugging/visualization)
    6) Reverse all edges to get final graph.

    Returns:
    dict: {
        "P","E","C","D","edges","node_type","level",
        "meta": { "p_edges_count", "e_insert_map", "c_to_e", "d_to_c" }
    }
    """
    if seed is not None:
        random.seed(seed)

    if nP < 2:
        raise ValueError("nP must be at least 2 (root node P0 and at least one more privilege node).")
    if not (0.0 <= extra_p_edge_prob <= 1.0):
        raise ValueError("extra_p_edge_prob must be in [0,1].")
    lo, hi = c_parents_per_e_range
    if lo < 1 or hi < lo:
        raise ValueError("c_parents_per_e_range must be like (1,3) with 1 <= lo <= hi.")
    if max_e_children_per_c < 1:
        raise ValueError("max_e_children_per_c must be >= 1.")
    if max_c_children_per_d < 1:
        raise ValueError("max_c_children_per_d must be >= 1.")

    # 1) Generate P nodes
    P_nodes = [f"P{i}" for i in range(nP)]
    root = "P0"

    # 2) Generate P subgraph DAG with P0 as root
    # 2.1 First generate directed "spanning tree": ensures all nodes reachable from P0 (also ensures DAG)
    p_edges = set()
    for i in range(1, nP):
        child = P_nodes[i]
        parent = random.choice(P_nodes[:i])
        p_edges.add((parent, child))

    # 2.2 Add extra P->P edges (still DAG: only allow i<j direction)
    extra_out_used = defaultdict(int)
    for i in range(nP):
        for j in range(i + 1, nP):
            if extra_out_used[P_nodes[i]] >= max_extra_p_out_per_node:
                continue
            if (P_nodes[i], P_nodes[j]) in p_edges:
                continue
            if random.random() <= extra_p_edge_prob:
                p_edges.add((P_nodes[i], P_nodes[j]))
                extra_out_used[P_nodes[i]] += 1

    # Build P subgraph and compute shortest path level
    GP = nx.DiGraph()
    GP.add_nodes_from(P_nodes)
    GP.add_edges_from(p_edges)

    if not nx.is_directed_acyclic_graph(GP):
        raise RuntimeError("Internal error: P subgraph should be DAG but cycle detected.")

    level_P = {}
    q = deque([root])
    level_P[root] = 0
    while q:
        u = q.popleft()
        for v in GP.successors(u):
            if v not in level_P:
                level_P[v] = level_P[u] + 1
                q.append(v)
    if len(level_P) != len(P_nodes):
        missing = sorted(set(P_nodes) - set(level_P))
        raise RuntimeError(f"P subgraph cannot reach all P nodes from root {root}, missing: {missing}")

    # 3) Insert E for each P->P edge: Pi -> Ek -> Pj
    p_edges_sorted = sorted(p_edges)
    E_nodes = [f"E{k}" for k in range(len(p_edges_sorted))]

    edges = set()
    node_type = {}
    for p in P_nodes:
        node_type[p] = "P"
    for e in E_nodes:
        node_type[e] = "E"

    e_insert_map = {}
    for k, (pi, pj) in enumerate(p_edges_sorted):
        ek = E_nodes[k]
        e_insert_map[ek] = (pi, pj)
        edges.add((pi, ek))
        edges.add((ek, pj))

    # 4) Assign 1-3 C as parent nodes to each E: C -> E
    need_c_for_e = {}
    total_c_links = 0
    for e in E_nodes:
        m = random.randint(lo, hi)
        need_c_for_e[e] = m
        total_c_links += m

    # Default nC: estimate by capacity lower bound plus some redundancy
    if nC is None:
        min_nC = max(2, math.ceil(total_c_links / max_e_children_per_c))
        nC = min_nC + 2

    C_nodes = [f"C{i}" for i in range(nC)]
    for c in C_nodes:
        node_type[c] = "C"

    c_child_count = defaultdict(int)
    c_to_e = defaultdict(list)

    def pick_distinct_c(m: int):
        available = [c for c in C_nodes if c_child_count[c] < max_e_children_per_c]
        if len(available) < m:
            return None
        available.sort(key=lambda x: c_child_count[x])
        pool = available[: max(m * 3, m)]
        return random.sample(pool, m)

    for e in E_nodes:
        m = need_c_for_e[e]
        chosen = pick_distinct_c(m)
        if chosen is None:
            raise ValueError(
                f"nC={nC} insufficient to satisfy all E's 1-3 C parent node requirements; "
                f"try increasing nC or max_e_children_per_c (current={max_e_children_per_c})."
            )
        for c in chosen:
            edges.add((c, e))
            c_child_count[c] += 1
            c_to_e[c].append(e)

    # 5) Add unique D parent to each C: D -> C
    if nD is None:
        nD = max(1, math.ceil(len(C_nodes) / max_c_children_per_d))

    D_nodes = [f"D{i}" for i in range(nD)]
    for d in D_nodes:
        node_type[d] = "D"

    d_child_count = defaultdict(int)
    d_to_c = defaultdict(list)

    for c in C_nodes:
        candidates = [d for d in D_nodes if d_child_count[d] < max_c_children_per_d]
        if not candidates:
            raise ValueError(
                f"nD={nD} insufficient to satisfy each C's unique D parent constraint; "
                f"increase nD or max_c_children_per_d (current={max_c_children_per_d})."
            )
        d = random.choice(candidates)
        edges.add((d, c))
        d_child_count[d] += 1
        d_to_c[d].append(c)

    # 6) Level assignment (based on P0 shortest path)
    level = {}

    # 6.1 P level
    for p in P_nodes:
        level[p] = level_P[p]

    # 6.2 E level: level of inserted edge's source Pi
    for e, (pi, _pj) in e_insert_map.items():
        level[e] = level[_pj]

    # 6.3 C level: minimum of connected E's level (C may connect to multiple E)
    for c in C_nodes:
        if c_to_e.get(c):
            level[c] = min(level[e] for e in c_to_e[c])
        else:
            level[c] = 0

    # 6.4 D level: minimum of its child C's level
    for d in D_nodes:
        if d_to_c.get(d):
            level[d] = min(level[c] for c in d_to_c[d])
        else:
            level[d] = 0

    # 7) Reverse all edges
    reversed_edges = [(v, u) for (u, v) in edges]

    return {
        "P": P_nodes,
        "E": E_nodes,
        "C": C_nodes,
        "D": D_nodes,
        "edges": reversed_edges,
        "node_type": node_type,
        "level": level,
        "meta": {
            "root": root,
            "p_edges_count": len(p_edges_sorted),
            "e_insert_map": dict(e_insert_map),
            "c_to_e": {k: v[:] for k, v in c_to_e.items()},
            "d_to_c": {k: v[:] for k, v in d_to_c.items()},
        },
    }


# Layered layout & visualization

def layered_layout(bn):
    G = nx.DiGraph()
    all_nodes = bn["P"] + bn["E"] + bn["C"] + bn["D"]
    for n in all_nodes:
        G.add_node(n, kind=bn["node_type"][n], level=bn["level"].get(n, None))
    G.add_edges_from(bn["edges"])

    types_order = ["P", "E", "C", "D"]
    nodes_by_type = {t: [] for t in types_order}
    for n in G.nodes():
        nodes_by_type[G.nodes[n]["kind"]].append(n)

    for t in types_order:
        nodes_by_type[t].sort(key=lambda x: (G.nodes[x].get("level", 0), x))

    pos = {}
    y_gap = 1.7
    x_gap = 2.1
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


def visualize_bn(bn, layout="layered", show_level=True):
    if layout == "layered":
        G, pos = layered_layout(bn)
    else:
        G = nx.DiGraph()
        all_nodes = bn["P"] + bn["E"] + bn["C"] + bn["D"]
        for n in all_nodes:
            G.add_node(n, kind=bn["node_type"][n], level=bn["level"].get(n, None))
        G.add_edges_from(bn["edges"])
        pos = nx.spring_layout(G, seed=0)

    color_map = {"P": "#1f77b4", "E": "#ff7f0e", "C": "#2ca02c", "D": "#d62728"}
    node_colors = [color_map[G.nodes[n]["kind"]] for n in G.nodes()]

    if show_level:
        labels = {}
        for n in G.nodes():
            lv = G.nodes[n].get("level", None)
            labels[n] = f"{n}\nL={lv}" if lv is not None else n
    else:
        labels = {n: n for n in G.nodes()}

    plt.figure(figsize=(11, 7))
    nx.draw(
        G, pos,
        labels=labels,
        node_color=node_colors,
        node_size=900,
        arrows=True,
        arrowstyle="->",
        arrowsize=14,
        font_size=8,
    )

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w',
            markerfacecolor=color_map["P"], label='P (Privilege)', markersize=10),
        Line2D([0], [0], marker='o', color='w',
            markerfacecolor=color_map["E"], label='E (Exploit)', markersize=10),
        Line2D([0], [0], marker='o', color='w',
            markerfacecolor=color_map["C"], label='C (Condition)', markersize=10),
        Line2D([0], [0], marker='o', color='w',
            markerfacecolor=color_map["D"], label='D (Defense)', markersize=10),
    ]
    plt.legend(handles=legend_elements, loc="best")
    plt.title("Generated Graph (All edges reversed at the end)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    bn = generate_bn_from_root_and_reverse(
        nP=30,
        seed=42,
        extra_p_edge_prob=0.30,
        max_extra_p_out_per_node=2,
        c_parents_per_e_range=(1, 3),
        max_e_children_per_c=5,
        max_c_children_per_d=3,
        nC=None,
        nD=None,
    )

    G = nx.DiGraph()
    G.add_edges_from(bn["edges"])
    print("Is DAG after reversal:", nx.is_directed_acyclic_graph(G))
    print("Counts:", {k: len(bn[k]) for k in ["P", "E", "C", "D"]})
    print("Edge count:", len(bn["edges"]))

    print("\n" + "=" * 60)
    print("All edge information (grouped by type):")
    print("=" * 60)

    edge_info = []
    for u, v in sorted(bn["edges"]):
        u_type = bn["node_type"].get(u, "?")
        v_type = bn["node_type"].get(v, "?")
        edge_info.append((u, v, u_type, v_type))
        print(f"  {u} ({u_type}) -> {v} ({v_type})")

    output_file = "edges_info.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("Bayesian Network Edge Information\n")
        f.write("=" * 60 + "\n\n")

        f.write("Node counts:\n")
        for k in ["P", "E", "C", "D"]:
            f.write(f"  {k}: {len(bn[k])} nodes\n")
        f.write(f"  Total edges: {len(bn['edges'])}\n\n")

        f.write("All edge list:\n")
        for u, v, u_type, v_type in edge_info:
            f.write(f"  {u} ({u_type}) -> {v} ({v_type})\n")

        f.write("\n\nNode levels:\n")
        for node in sorted(bn["level"].keys(), key=lambda x: (bn["node_type"][x], x)):
            f.write(f"  {node}: level={bn['level'][node]}\n")

    print(f"\nEdge info saved to: {output_file}")
