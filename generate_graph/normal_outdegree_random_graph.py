"""Random attack-graph generator with normally distributed P out-degrees.

This module is intentionally separate from ``random_graph.py`` so experiments
can change P-node degree generation without affecting existing workflows.
"""

import random
from collections import defaultdict

import numpy as np


def truncated_normal_degree_sequence(
    nP: int,
    max_children: int = 5,
    mean: float = 3.0,
    std: float = 0.8,
    seed: int | None = None,
):
    """Build an exact-size discrete truncated-normal degree sequence.

    Degrees are restricted to ``[1, max_children]``. Counts approximate the
    discretized normal density, while the total degree is adjusted to exactly
    ``round(nP * mean)``.
    """
    if nP < 1:
        raise ValueError("nP must be positive.")
    if max_children < 1:
        raise ValueError("max_children must be positive.")
    if not 1 <= mean <= max_children:
        raise ValueError("mean must be within [1, max_children].")
    if std <= 0:
        raise ValueError("std must be positive.")

    degrees = np.arange(1, max_children + 1, dtype=int)
    weights = np.exp(-0.5 * ((degrees - mean) / std) ** 2)
    probabilities = weights / weights.sum()

    raw_counts = probabilities * nP
    counts = np.floor(raw_counts).astype(int)
    remaining = nP - int(counts.sum())
    if remaining:
        fractional_order = np.argsort(-(raw_counts - counts), kind="stable")
        counts[fractional_order[:remaining]] += 1

    target_degree_sum = int(round(nP * mean))
    current_degree_sum = int(np.dot(counts, degrees))
    delta = target_degree_sum - current_degree_sum

    # Move one node by one degree at a time. Choose the move that least
    # increases squared deviation from the ideal real-valued bin counts.
    while delta != 0:
        candidates = []
        if delta > 0:
            for i in range(len(counts) - 1):
                if counts[i] <= 0:
                    continue
                before = (counts[i] - raw_counts[i]) ** 2 + (counts[i + 1] - raw_counts[i + 1]) ** 2
                after = (counts[i] - 1 - raw_counts[i]) ** 2 + (counts[i + 1] + 1 - raw_counts[i + 1]) ** 2
                candidates.append((after - before, i, i + 1))
        else:
            for i in range(1, len(counts)):
                if counts[i] <= 0:
                    continue
                before = (counts[i] - raw_counts[i]) ** 2 + (counts[i - 1] - raw_counts[i - 1]) ** 2
                after = (counts[i] - 1 - raw_counts[i]) ** 2 + (counts[i - 1] + 1 - raw_counts[i - 1]) ** 2
                candidates.append((after - before, i, i - 1))

        if not candidates:
            raise RuntimeError("Cannot adjust degree sequence to requested mean.")
        _, src, dst = min(candidates)
        counts[src] -= 1
        counts[dst] += 1
        delta += -1 if delta > 0 else 1

    sequence = np.repeat(degrees, counts).astype(int)
    rng = np.random.default_rng(seed)
    rng.shuffle(sequence)
    return sequence.tolist(), {
        "degrees": degrees.tolist(),
        "probabilities": probabilities.tolist(),
        "counts": counts.tolist(),
        "target_mean": float(mean),
        "target_std": float(std),
        "actual_mean": float(np.mean(sequence)),
        "actual_std": float(np.std(sequence)),
    }


def generate_bn_normal_p_outdegree(
    nP: int,
    nC: int,
    nD: int,
    max_children: int = 5,
    p_mean: float = 3.0,
    p_std: float = 0.8,
    p_EP: float = 0.25,
    seed: int | None = None,
    non_p_max_children: int = 5,
):
    """Generate a P/E/C/D DAG with normally distributed P out-degrees.

    ``max_children`` is the experimental upper bound for P->E out-degree.
    ``non_p_max_children`` independently fixes the out-degree bound for E/C/D
    nodes so a k_max sweep does not also relax the rest of the graph.
    """
    if seed is not None:
        random.seed(seed)
    if nP < 2:
        raise ValueError("nP must be at least 2.")
    if nC < 2 or nD < 0:
        raise ValueError("nC must be at least 2 and nD cannot be negative.")
    if max_children < 2:
        raise ValueError("max_children must be at least 2.")
    if non_p_max_children < 2:
        raise ValueError("non_p_max_children must be at least 2.")
    if not 0 <= p_EP <= 1:
        raise ValueError("p_EP must be in [0,1].")

    p_degrees, degree_meta = truncated_normal_degree_sequence(
        nP=nP,
        max_children=max_children,
        mean=p_mean,
        std=p_std,
        seed=seed,
    )
    nE = sum(p_degrees)

    P_nodes = [f"P{i}" for i in range(nP)]
    E_nodes = [f"E{i}" for i in range(nE)]
    C_nodes = [f"C{i}" for i in range(nC)]
    D_nodes = [f"D{i}" for i in range(nD)]
    node_type = {
        **{p: "P" for p in P_nodes},
        **{e: "E" for e in E_nodes},
        **{c: "C" for c in C_nodes},
        **{d: "D" for d in D_nodes},
    }

    edges = set()
    out_degree = defaultdict(int)

    def try_add_edge(u, v):
        limit = max_children if node_type[u] == "P" else non_p_max_children
        if u == v or (u, v) in edges or out_degree[u] >= limit:
            return False
        edges.add((u, v))
        out_degree[u] += 1
        return True

    p_rank = {p: i for i, p in enumerate(P_nodes)}
    shuffled_E = E_nodes[:]
    random.shuffle(shuffled_E)
    parent_of_E = {}
    cursor = 0
    for p, degree in zip(P_nodes, p_degrees):
        for e in shuffled_E[cursor:cursor + degree]:
            if not try_add_edge(p, e):
                raise RuntimeError(f"Cannot assign requested out-degree to {p}.")
            parent_of_E[e] = p
        cursor += degree

    has_any_ep = False
    candidates = []
    for e in E_nodes:
        p_in = parent_of_E[e]
        possible_p_out = [
            p for p in P_nodes
            if p_rank[p] > p_rank[p_in] and out_degree[p] < max_children
        ]
        if possible_p_out:
            candidates.append((e, possible_p_out))

    random.shuffle(candidates)
    for e, possible_p_out in candidates:
        if random.random() <= p_EP and try_add_edge(e, random.choice(possible_p_out)):
            has_any_ep = True

    if not has_any_ep and candidates:
        e, possible_p_out = random.choice(candidates)
        has_any_ep = try_add_edge(e, random.choice(possible_p_out))
    if not has_any_ep:
        raise RuntimeError("No valid E->P edge could be generated.")

    for e in E_nodes:
        remaining = non_p_max_children - out_degree[e]
        needed = max(2 - out_degree[e], 0)
        candidate_c = C_nodes[:]
        random.shuffle(candidate_c)
        max_possible = min(remaining, len(candidate_c))
        if max_possible < needed:
            raise RuntimeError(f"Cannot give {e} at least two children.")
        count = random.randint(needed, max_possible)
        for c in candidate_c[:count]:
            try_add_edge(e, c)

    if nD > 0:
        if nD > nC or nC > 3 * nD:
            raise ValueError("D capacity requires nD <= nC <= 3*nD.")
        shuffled_c = C_nodes[:]
        shuffled_d = D_nodes[:]
        random.shuffle(shuffled_c)
        random.shuffle(shuffled_d)
        d_parent_count = {d: 0 for d in D_nodes}
        for c, d in zip(shuffled_c[:nD], shuffled_d):
            if not try_add_edge(c, d):
                raise RuntimeError(f"Cannot add base C->D edge for {c}.")
            d_parent_count[d] += 1
        for c in shuffled_c[nD:]:
            available = [d for d in D_nodes if d_parent_count[d] < 3]
            if not available:
                raise RuntimeError("No D node has remaining parent capacity.")
            d = random.choice(available)
            if not try_add_edge(c, d):
                raise RuntimeError(f"Cannot add C->D edge for {c}.")
            d_parent_count[d] += 1

    return {
        "P": P_nodes,
        "E": E_nodes,
        "C": C_nodes,
        "D": D_nodes,
        "edges": sorted(edges),
        "node_type": node_type,
        "out_degree": dict(out_degree),
        "meta": {
            "generator": "normal_p_outdegree",
            "max_children": max_children,
            "non_p_max_children": non_p_max_children,
            "p_outdegree": degree_meta,
        },
    }
