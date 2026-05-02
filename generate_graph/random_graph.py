import random
from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt


def generate_bn_dag_multi_pe(
    nP,
    nE,
    nC,
    nD,
    max_children=5,
    p_EP=0.5,
    seed=None,
):
    """
    Generate an attack Bayesian network skeleton (DAG) with the following constraints:

    Node types:
        P: Privilege
        E: Exploit
        C: Condition
        D: Defense

    Edge types:
        (P, E), (E, P), (E, C), (C, D)

    Constraints:
    - nP >= 2, nE >= 1, nC >= 2, nD >= 0
    - nE >= nP                      # Each P has at least 1 child E
    - nE <= nP * max_children       # P's out-degree capacity covers all E
    - Each E has exactly 1 parent P (only one (P,E) incoming edge)
    - Each P has at least 1 child E
    - Each E has >= 2 children from {P, C}, and at most 1 P child
    - At least one E has a P child (E,P)
    - If nD > 0, each C has at least 1 D child; if nD == 0, no (C,D)
    - Any node out-degree <= max_children
    - Graph is DAG: for each E, parent is P_in, any (E,P_out) must satisfy
      rank(P_out) > rank(P_in), so along P-E-P-E-P-E paths,
      P's rank strictly increases, no cycles.

    Additional:
    - p_EP ∈ [0,1] controls probability of each E adding an E->P edge (if valid P_out exists).
    """

    if seed is not None:
        random.seed(seed)

    if nP < 2:
        raise ValueError("At least 2 privilege nodes nP >= 2 required for (E,P) edges and DAG.")
    if nE < 1 or nC < 2:
        raise ValueError("nE >= 1 and nC >= 2 required. nD can be 0.")
    if nE < nP:
        raise ValueError("nE >= nP required so each P has at least one E child.")
    if max_children < 2:
        raise ValueError("max_children >= 2 required because each E has >= 2 children.")
    if nE > nP * max_children:
        raise ValueError("Too many E nodes: cannot be covered with max_children limit.")
    if not (0.0 <= p_EP <= 1.0):
        raise ValueError("p_EP must be in [0,1].")

    # 1. Define nodes
    P_nodes = [f"P{i}" for i in range(nP)]
    E_nodes = [f"E{i}" for i in range(nE)]
    C_nodes = [f"C{i}" for i in range(nC)]
    D_nodes = [f"D{i}" for i in range(nD)]

    node_type = {}
    for p in P_nodes:
        node_type[p] = "P"
    for e in E_nodes:
        node_type[e] = "E"
    for c in C_nodes:
        node_type[c] = "C"
    for d in D_nodes:
        node_type[d] = "D"

    edges = set()
    out_degree = defaultdict(int)

    def try_add_edge(u, v):
        """Try adding (u,v), check for no duplicates, no self-loops, out-degree limit."""
        if u == v:
            return False
        if (u, v) in edges:
            return False
        if out_degree[u] >= max_children:
            return False
        edges.add((u, v))
        out_degree[u] += 1
        return True

    # 2. Define rank for P nodes
    # Use rank to ensure DAG: only allow E->P pointing to higher rank P
    P_rank = {p: i for i, p in enumerate(P_nodes)}

    # 3. Assign unique parent P to each E, ensure each P has at least 1 child E
    parent_of_E = {}

    shuffled_E = E_nodes[:]
    random.shuffle(shuffled_E)

    # 3.1 First ensure "each P has at least 1 child E": first nP E nodes correspond to P nodes
    for p, e in zip(P_nodes, shuffled_E[:nP]):
        ok = try_add_edge(p, e)   # (P, E)
        if not ok:
            raise RuntimeError("Failed to add base (P,E) edge, check logic or max_children.")
        parent_of_E[e] = p

    # 3.2 Remaining E nodes randomly select a parent P
    for e in shuffled_E[nP:]:
        possible_parents = [p for p in P_nodes if out_degree[p] < max_children]
        if not possible_parents:
            raise RuntimeError("All P nodes are at max out-degree, cannot assign parent to remaining E.")
        p = random.choice(possible_parents)
        ok = try_add_edge(p, e)
        if not ok:
            raise RuntimeError("Failed to add (P,E) edge, check logic.")
        parent_of_E[e] = p

    assert len(parent_of_E) == len(E_nodes)

    # 4. Try adding (E,P) edges for multiple E (multi-layer P-E-P-E), keeping DAG
    has_any_EP = False
    all_candidates = []
    EP_used = set()

    # Pre-compute valid P_out candidates for each E
    for e in E_nodes:
        P_in = parent_of_E[e]
        r_in = P_rank[P_in]
        possible_P_out = [
            p for p in P_nodes
            if P_rank[p] > r_in and out_degree[p] < max_children
        ]
        if possible_P_out:
            all_candidates.append((e, possible_P_out))

    # Iterate over E, add E->P by probability p_EP
    random.shuffle(all_candidates)
    for e, possible_P_out in all_candidates:
        if e in EP_used:
            continue
        if out_degree[e] >= max_children:
            continue
        if random.random() <= p_EP:
            p_child = random.choice(possible_P_out)
            if try_add_edge(e, p_child):
                has_any_EP = True
                EP_used.add(e)

    # If no E->P in this round, force add one to ensure "at least one E has (E,P)"
    if not has_any_EP and all_candidates:
        e_force, possible_P_out = random.choice(all_candidates)
        p_child = random.choice(possible_P_out)
        ok = try_add_edge(e_force, p_child)
        if ok:
            EP_used.add(e_force)
            has_any_EP = True
        else:
            raise RuntimeError("Failed to add (E,P) edge in fallback, try increasing max_children or decreasing scale.")

    if not has_any_EP:
        raise RuntimeError(
            "No valid (E,P) edge candidates under DAG constraint. "
            "Try increasing nP or max_children."
        )

    # 5. Add (E,C) edges for each E, ensure out-degree >= 2
    for e in E_nodes:
        current_deg = out_degree[e]
        remaining_cap = max_children - current_deg
        if remaining_cap <= 0:
            if current_deg < 2:
                raise RuntimeError(f"{e} out-degree insufficient, cannot meet children >= 2.")
            continue

        need_more = max(2 - current_deg, 0)

        candidate_C = [c for c in C_nodes if (e, c) not in edges]
        max_possible = min(remaining_cap, len(candidate_C))
        if max_possible < need_more:
            raise RuntimeError(
                f"{e} cannot add enough C children, check nC or max_children settings."
            )

        kC = random.randint(need_more, max_possible)

        random.shuffle(candidate_C)
        for c_child in candidate_C[:kC]:
            try_add_edge(e, c_child)

        if out_degree[e] < 2:
            raise RuntimeError(f"{e} out-degree still < 2, check parameters or logic.")

    # 6. Add D child nodes for C
    # Constraints:
    # - Each C has exactly 1 D child
    # - Each D has at least 1 parent C
    # - Each D has at most 3 parent C

    if nD > 0:
        if nD > nC:
            raise ValueError("Each D must have at least 1 parent C, so nD <= nC required.")
        if nC > 3 * nD:
            raise ValueError("Each D has at most 3 parent C, so nC <= 3 * nD required.")

        for c in C_nodes:
            if out_degree[c] >= max_children:
                raise RuntimeError(f"{c} has no out-degree capacity for its unique D child.")

        d_parent_count = {d: 0 for d in D_nodes}

        shuffled_C = C_nodes[:]
        shuffled_D = D_nodes[:]
        random.shuffle(shuffled_C)
        random.shuffle(shuffled_D)

        assignments = {}

        # Step 1: Ensure each D has at least 1 parent C
        for c, d in zip(shuffled_C[:nD], shuffled_D):
            ok = try_add_edge(c, d)
            if not ok:
                raise RuntimeError(f"Failed to add base edge: {c} -> {d}")
            assignments[c] = d
            d_parent_count[d] += 1

        # Step 2: Assign remaining C to D with capacity (each D max 3 parents)
        for c in shuffled_C[nD:]:
            candidate_D = [d for d in D_nodes if d_parent_count[d] < 3]
            if not candidate_D:
                raise RuntimeError("No available D capacity for remaining C, check parameters.")

            d = random.choice(candidate_D)
            ok = try_add_edge(c, d)
            if not ok:
                raise RuntimeError(f"Failed to add edge: {c} -> {d}")
            assignments[c] = d
            d_parent_count[d] += 1

    return {
        "P": P_nodes,
        "E": E_nodes,
        "C": C_nodes,
        "D": D_nodes,
        "edges": list(edges),
        "node_type": node_type,
        "out_degree": dict(out_degree),
    }


# Layered layout & visualization

def layered_layout(bn):
    """Assign coordinates for P/E/C/D layers, only for visualization."""
    G = nx.DiGraph()
    for n in bn["P"] + bn["E"] + bn["C"] + bn["D"]:
        G.add_node(n, kind=bn["node_type"][n])
    G.add_edges_from(bn["edges"])

    types_order = ["P", "E", "C", "D"]
    nodes_by_type = {t: [] for t in types_order}
    for n in G.nodes():
        t = G.nodes[n]["kind"]
        nodes_by_type[t].append(n)

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


def visualize_bn(bn, layout="layered"):
    """
    Visualize the generated Bayesian network.
    layout:
        - "layered": P/E/C/D layers (recommended)
        - "spring": spring_layout
        - "kamada_kawai": kamada_kawai_layout
    """
    if layout == "layered":
        G, pos = layered_layout(bn)
    else:
        G = nx.DiGraph()
        for n in bn["P"] + bn["E"] + bn["C"] + bn["D"]:
            G.add_node(n, kind=bn["node_type"][n])
        G.add_edges_from(bn["edges"])

        if layout == "kamada_kawai":
            pos = nx.kamada_kawai_layout(G)
        else:
            pos = nx.spring_layout(G, seed=0)

    color_map = {
        "P": "#1f77b4",
        "E": "#ff7f0e",
        "C": "#2ca02c",
        "D": "#d62728",
    }
    node_colors = [color_map[G.nodes[n]["kind"]] for n in G.nodes()]

    plt.figure(figsize=(9, 6))
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_color=node_colors,
        node_size=700,
        arrows=True,
        arrowstyle="->",
        arrowsize=15,
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
    plt.title("Bayesian Network Structure (DAG, multi-layer P-E-P-E)")
    plt.tight_layout()
    plt.tight_layout()
    plt.savefig("bn_visualization.png", dpi=150, bbox_inches='tight')
    print("Image saved to bn_visualization.png")


if __name__ == "__main__":
    bn = generate_bn_dag_multi_pe(
        nP=1000,
        nE=2400,
        nC=3000,
        nD=1500,
        max_children=5,
        p_EP=0.25,
        seed=42,
    )

    G = nx.DiGraph()
    G.add_edges_from(bn["edges"])
    print("Is DAG:", nx.is_directed_acyclic_graph(G))

    print("Edges:")
    for u, v in sorted(bn["edges"]):
        print(f"{u} -> {v}")
