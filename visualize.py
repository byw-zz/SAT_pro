import networkx as nx
import matplotlib.pyplot as plt

def visualize_bn(bn, layout="spring"):
    """Visualize the generated BN graph with nodes colored by type."""
    G = nx.DiGraph()

    P_nodes = bn["P"]
    E_nodes = bn["E"]
    C_nodes = bn["C"]
    D_nodes = bn["D"]
    edges    = bn["edges"]
    node_type = bn["node_type"]

    for n in P_nodes + E_nodes + C_nodes + D_nodes:
        G.add_node(n, kind=node_type[n])

    G.add_edges_from(edges)

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

    plt.figure(figsize=(8, 6))
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
               markerfacecolor=color_map["P"], label='P (Privilege)',
               markersize=10),
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor=color_map["E"], label='E (Exploit)',
               markersize=10),
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor=color_map["C"], label='C (Condition)',
               markersize=10),
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor=color_map["D"], label='D (Defense)',
               markersize=10),
    ]
    plt.legend(handles=legend_elements, loc="best")
    plt.title("Bayesian Network Structure")
    plt.tight_layout()
    plt.show()
