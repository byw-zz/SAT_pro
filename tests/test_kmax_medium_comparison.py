from comparison.kmax_medium_comparison import (
    CASE_SPECS,
    _compare as compare_random,
    summarize,
)
from comparison.kmax_structured_medium_comparison import (
    _compare as compare_structured,
)
from generate_graph.normal_outdegree_random_graph import generate_bn_normal_p_outdegree


def test_normal_generator_separates_p_and_non_p_outdegree_limits():
    bn = generate_bn_normal_p_outdegree(
        nP=40,
        nC=120,
        nD=60,
        max_children=10,
        non_p_max_children=5,
        p_mean=5.5,
        p_std=1.8,
        p_EP=0.25,
        seed=43,
    )

    p_degrees = [bn["out_degree"].get(node, 0) for node in bn["P"]]
    assert sum(p_degrees) == 40 * 5.5
    assert max(p_degrees) <= 10
    assert all(
        bn["out_degree"].get(node, 0) <= 5
        for kind in ("E", "C", "D")
        for node in bn[kind]
    )
    assert bn["edges"] == sorted(bn["edges"])


def test_case_specs_define_requested_sweep():
    assert CASE_SPECS == {
        5: {"p_mean": 3.0, "p_std": 0.8},
        7: {"p_mean": 4.0, "p_std": 1.2},
        10: {"p_mean": 5.5, "p_std": 1.8},
    }


def test_kmax_comparisons_require_exact_objective_equality_for_ties():
    for compare in (compare_random, compare_structured):
        assert compare(100.0, 100.0) == "tie"
        assert compare(100.000000000001, 100.0) == "win"
        assert compare(99.999999999999, 100.0) == "loss"


def test_summary_uses_baseline_viewpoint_against_maxsat():
    payload = {
        "cases": [
            {
                "k_max": 5,
                "graphs": [
                    {
                        "methods": {
                            "maxsat": {"status": "complete", "objective": 10.0, "wall_time_s": 1.0},
                            "ga": {"status": "complete", "objective": 11.0, "wall_time_s": 2.0},
                            "khouzani": {"status": "complete", "objective": 10.0, "wall_time_s": 3.0},
                            "zenitani": {"status": "complete", "objective": 8.0, "wall_time_s": 4.0},
                        }
                    }
                ],
            }
        ]
    }

    case = summarize(payload)["cases"][0]
    assert case["pairwise_vs_maxsat"]["ga"] == {"win": 1, "tie": 0, "loss": 0, "na": 0}
    assert case["pairwise_vs_maxsat"]["khouzani"] == {"win": 0, "tie": 1, "loss": 0, "na": 0}
    assert case["pairwise_vs_maxsat"]["zenitani"] == {"win": 0, "tie": 0, "loss": 1, "na": 0}
