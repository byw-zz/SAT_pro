from comparison.batch_comparison import _winner
from comparison.khouzani_vs_stored import summarize as summarize_khouzani
from comparison.plot_baselines_comparison import _wtl as legacy_wtl
from comparison.plot_baselines_comparison_ga_median import _wtl as median_wtl
from comparison.zenitani_vs_stored import summarize as summarize_zenitani


def test_final_baseline_wtl_uses_fixed_floating_point_tolerance():
    rows = [
        {"a": 100.0 + 5e-10, "b": 100.0},
        {"a": 100.0 + 2e-9, "b": 100.0},
        {"a": 99.8, "b": 100.0},
    ]
    assert median_wtl(rows, "a", "b") == [1, 1, 1]
    assert legacy_wtl(rows, "a", "b") == (1, 1, 1)


def test_raw_baseline_helpers_use_the_same_tolerance():
    assert _winner(100.0 + 5e-10, "A", 100.0, "B") == "TIE"
    assert _winner(100.0 + 2e-9, "A", 100.0, "B") == "A"

    kh_rows = [
        {"khouzani_obj": 100.0 + 5e-10, "maxsat_obj": 100.0,
         "ga_obj": 100.0, "khouzani_milp_ms": 1.0},
    ]
    zen_rows = [
        {"zenitani_obj": 100.0 + 5e-10, "maxsat_obj": 100.0,
         "ga_obj": 100.0, "zenitani_n_evals": 1,
         "zenitani_wall_ms": 1.0},
    ]
    assert summarize_khouzani(kh_rows)["vs_maxsat"]["tie"] == 1
    assert summarize_zenitani(zen_rows)["vs_maxsat"]["tie"] == 1
