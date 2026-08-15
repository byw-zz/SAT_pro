"""Structured P/E/C/D DAGs with a controlled final P->E out-degree.

The graph is built in attack direction first.  ``P0`` is the root and every
other privilege node receives a prescribed number of incoming P edges.  One E
node is inserted on every P edge, C/D layers are attached as in
``structured_graph.py``, and all edges are reversed for the stored Bayesian
network representation.
"""

from __future__ import annotations

import random
from collections import defaultdict

import networkx as nx
import numpy as np

from generate_graph.normal_outdegree_random_graph import (
    truncated_normal_degree_sequence,
)


def _assign_feasible_indegrees(degrees: list[int], rng: random.Random) -> list[int]:
    """Assign a degree to P1..Pn while respecting degree(Pj) <= j."""
    remaining = list(degrees)
    assigned = []
    max_degree = max(remaining)

    for node_index in range(1, len(degrees) + 1):
        if node_index < max_degree:
            candidates = [i for i, degree in enumerate(remaining) if degree <= node_index]
            if not candidates:
                raise RuntimeError("Degree sequence cannot form a root-ordered P DAG.")
            selected = rng.choice(candidates)
        else:
            selected = rng.randrange(len(remaining))
        assigned.append(remaining.pop(selected))

    return assigned


def generate_bn_structured_normal_p_outdegree(
    nP: int,
    nC: int,
    nD: int,
    max_children: int = 5,
    p_mean: float = 3.0,
    p_std: float = 0.8,
    seed: int | None = None,
    c_parents_per_e_range=(1, 3),
    max_e_children_per_c: int = 5,
    max_c_children_per_d: int = 3,
):
    """Generate a structured DAG with controlled final P->E out-degrees.

    ``P0`` has final P->E degree zero.  Degrees for ``P1`` through
    ``P(nP-1)`` approximate a discrete normal distribution truncated to
    ``[1, max_children]``.  Their realised maximum is exactly
    ``max_children``.
    """
    if nP < 2:
        raise ValueError("nP must be at least 2.")
    if nC < 1 or nD < 1:
        raise ValueError("nC and nD must be positive.")
    if max_children >= nP:
        raise ValueError("max_children must be smaller than nP.")
    lo, hi = c_parents_per_e_range
    if lo < 1 or hi < lo:
        raise ValueError("c_parents_per_e_range must satisfy 1 <= lo <= hi.")
    if max_e_children_per_c < 1 or max_c_children_per_d < 1:
        raise ValueError("C/E and D/C capacity limits must be positive.")
    if nC < hi:
        raise ValueError("nC must allow the maximum distinct C parents per E.")
    if nD * max_c_children_per_d < nC:
        raise ValueError("nD capacity is insufficient to give every C one D parent.")

    rng = random.Random(seed)
    sampled_degrees, degree_meta = truncated_normal_degree_sequence(
        nP=nP - 1,
        max_children=max_children,
        mean=p_mean,
        std=p_std,
        seed=seed,
    )
    if max(sampled_degrees) != max_children:
        raise ValueError(
            "The requested truncated-normal distribution does not realise "
            "max_children; increase nP or adjust mean/std."
        )
    p_indegrees = _assign_feasible_indegrees(sampled_degrees, rng)

    P_nodes = [f"P{i}" for i in range(nP)]
    root = P_nodes[0]
    p_edges = set()
    for child_index, degree in enumerate(p_indegrees, start=1):
        parents = rng.sample(P_nodes[:child_index], degree)
        p_edges.update((parent, P_nodes[child_index]) for parent in parents)

    p_graph = nx.DiGraph()
    p_graph.add_nodes_from(P_nodes)
    p_graph.add_edges_from(p_edges)
    if not nx.is_directed_acyclic_graph(p_graph):
        raise RuntimeError("Internal error: generated P graph is not a DAG.")
    if len(nx.descendants(p_graph, root)) != nP - 1:
        raise RuntimeError("Internal error: not every P node is reachable from P0.")

    level_p = nx.single_source_shortest_path_length(p_graph, root)
    p_edges_sorted = sorted(p_edges)
    E_nodes = [f"E{i}" for i in range(len(p_edges_sorted))]
    C_nodes = [f"C{i}" for i in range(nC)]
    D_nodes = [f"D{i}" for i in range(nD)]
    node_type = {
        **{node: "P" for node in P_nodes},
        **{node: "E" for node in E_nodes},
        **{node: "C" for node in C_nodes},
        **{node: "D" for node in D_nodes},
    }

    forward_edges = set()
    e_insert_map = {}
    for e, (parent, child) in zip(E_nodes, p_edges_sorted):
        e_insert_map[e] = (parent, child)
        forward_edges.add((parent, e))
        forward_edges.add((e, child))

    need_c_for_e = {
        e: rng.randint(lo, hi)
        for e in E_nodes
    }
    total_c_links = sum(need_c_for_e.values())
    if total_c_links > nC * max_e_children_per_c:
        raise ValueError(
            "nC capacity is insufficient for the sampled C-to-E links; "
            "increase nC or max_e_children_per_c."
        )

    c_child_count = defaultdict(int)
    c_to_e = defaultdict(list)
    for e in E_nodes:
        count = need_c_for_e[e]
        available = [
            c for c in C_nodes
            if c_child_count[c] < max_e_children_per_c
        ]
        if len(available) < count:
            raise ValueError("nC capacity is insufficient for distinct C parents.")
        available.sort(key=lambda c: (c_child_count[c], c))
        pool = available[:max(count * 3, count)]
        for c in rng.sample(pool, count):
            forward_edges.add((c, e))
            c_child_count[c] += 1
            c_to_e[c].append(e)

    d_child_count = defaultdict(int)
    d_to_c = defaultdict(list)
    for c in C_nodes:
        candidates = [
            d for d in D_nodes
            if d_child_count[d] < max_c_children_per_d
        ]
        if not candidates:
            raise ValueError("nD capacity is insufficient to give every C a D parent.")
        d = rng.choice(candidates)
        forward_edges.add((d, c))
        d_child_count[d] += 1
        d_to_c[d].append(c)

    level = dict(level_p)
    for e, (_, child) in e_insert_map.items():
        level[e] = level_p[child]
    for c in C_nodes:
        level[c] = min((level[e] for e in c_to_e.get(c, [])), default=0)
    for d in D_nodes:
        level[d] = min((level[c] for c in d_to_c.get(d, [])), default=0)

    edges = sorted((v, u) for u, v in forward_edges)
    graph = nx.DiGraph()
    all_nodes = P_nodes + E_nodes + C_nodes + D_nodes
    graph.add_nodes_from(all_nodes)
    graph.add_edges_from(edges)
    if not nx.is_directed_acyclic_graph(graph):
        raise RuntimeError("Internal error: reversed structured graph is not a DAG.")

    out_degree = {node: graph.out_degree(node) for node in all_nodes}
    realised_p_degrees = [out_degree[p] for p in P_nodes[1:]]
    if out_degree[root] != 0 or max(realised_p_degrees) != max_children:
        raise RuntimeError("Internal error: final P->E degree constraints were not met.")

    degree_counts = [realised_p_degrees.count(d) for d in degree_meta["degrees"]]
    degree_meta = {
        **degree_meta,
        "counts": degree_counts,
        "actual_mean": float(np.mean(realised_p_degrees)),
        "actual_std": float(np.std(realised_p_degrees)),
        "actual_min": min(realised_p_degrees),
        "actual_max": max(realised_p_degrees),
        "sample_size": nP - 1,
        "root": root,
        "root_degree": out_degree[root],
    }

    return {
        "P": P_nodes,
        "E": E_nodes,
        "C": C_nodes,
        "D": D_nodes,
        "edges": edges,
        "node_type": node_type,
        "level": level,
        "out_degree": out_degree,
        "meta": {
            "generator": "structured_normal_p_outdegree",
            "root": root,
            "max_children": max_children,
            "p_outdegree": degree_meta,
            "p_edges_count": len(p_edges_sorted),
            "e_insert_map": e_insert_map,
            "c_to_e": {key: value[:] for key, value in c_to_e.items()},
            "d_to_c": {key: value[:] for key, value in d_to_c.items()},
        },
    }
