from collections import Counter

import networkx as nx
import pytest

from generate_graph.structured_normal_outdegree_graph import (
    generate_bn_structured_normal_p_outdegree,
)


CASES = [
    (5, 3.0, 0.8),
    (7, 4.0, 1.2),
    (10, 5.5, 1.8),
]


@pytest.mark.parametrize(("k_max", "p_mean", "p_std"), CASES)
def test_structured_generator_realises_requested_p_degrees(k_max, p_mean, p_std):
    bn = generate_bn_structured_normal_p_outdegree(
        nP=100,
        nC=300,
        nD=150,
        max_children=k_max,
        p_mean=p_mean,
        p_std=p_std,
        seed=43,
    )

    p_degrees = [bn["out_degree"][node] for node in bn["P"]]
    non_root_degrees = p_degrees[1:]
    meta = bn["meta"]["p_outdegree"]

    assert p_degrees[0] == 0
    assert min(non_root_degrees) >= 1
    assert max(non_root_degrees) == k_max
    assert sum(non_root_degrees) == round(99 * p_mean)
    assert len(bn["E"]) == sum(non_root_degrees)
    assert meta["actual_mean"] == pytest.approx(sum(non_root_degrees) / 99)
    assert meta["actual_max"] == k_max
    assert meta["root_degree"] == 0
    assert meta["counts"] == [
        Counter(non_root_degrees)[degree]
        for degree in meta["degrees"]
    ]


@pytest.mark.parametrize(("k_max", "p_mean", "p_std"), CASES)
def test_structured_generator_preserves_structure_and_capacity(k_max, p_mean, p_std):
    bn = generate_bn_structured_normal_p_outdegree(
        nP=100,
        nC=300,
        nD=150,
        max_children=k_max,
        p_mean=p_mean,
        p_std=p_std,
        seed=51,
    )

    graph = nx.DiGraph()
    all_nodes = bn["P"] + bn["E"] + bn["C"] + bn["D"]
    graph.add_nodes_from(all_nodes)
    graph.add_edges_from(bn["edges"])
    assert nx.is_directed_acyclic_graph(graph)
    assert bn["edges"] == sorted(bn["edges"])
    assert bn["out_degree"] == dict(graph.out_degree())

    node_type = bn["node_type"]
    edge_types = Counter((node_type[u], node_type[v]) for u, v in bn["edges"])
    assert edge_types[("P", "E")] == len(bn["E"])
    assert edge_types[("E", "P")] == len(bn["E"])
    assert edge_types[("C", "D")] == len(bn["C"])

    for e in bn["E"]:
        assert sum(node_type[v] == "P" for v in graph.successors(e)) == 1
        assert 1 <= sum(node_type[v] == "C" for v in graph.successors(e)) <= 3
        assert sum(node_type[u] == "P" for u in graph.predecessors(e)) == 1
    for c in bn["C"]:
        assert sum(node_type[u] == "E" for u in graph.predecessors(c)) <= 5
        assert sum(node_type[v] == "D" for v in graph.successors(c)) == 1
    for d in bn["D"]:
        assert sum(node_type[u] == "C" for u in graph.predecessors(d)) <= 3

    attack_p = nx.DiGraph()
    attack_p.add_nodes_from(bn["P"])
    attack_p.add_edges_from(bn["meta"]["e_insert_map"].values())
    assert nx.is_directed_acyclic_graph(attack_p)
    assert nx.descendants(attack_p, "P0") == set(bn["P"][1:])
    assert {
        node: attack_p.in_degree(node)
        for node in bn["P"]
    } == {
        node: bn["out_degree"][node]
        for node in bn["P"]
    }


def test_structured_generator_is_reproducible_and_seed_sensitive():
    kwargs = dict(
        nP=100,
        nC=300,
        nD=150,
        max_children=7,
        p_mean=4.0,
        p_std=1.2,
    )
    first = generate_bn_structured_normal_p_outdegree(seed=43, **kwargs)
    replay = generate_bn_structured_normal_p_outdegree(seed=43, **kwargs)
    other = generate_bn_structured_normal_p_outdegree(seed=44, **kwargs)

    assert first == replay
    assert first["edges"] != other["edges"]
