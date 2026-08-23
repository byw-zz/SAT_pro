import copy
import math

import pytest

from comparison import parameter_sensitivity_core as core
from comparison.parameter_sensitivity_core import (
    InstanceKey,
    apply_node_perturbation,
    atomic_write_json,
    compare_vs_maxsat,
    ga_seed,
    iter_instance_keys,
    load_json,
    make_perturbation_vector,
    method_needs_run,
    missing_ga_runs,
    perturbation_seed,
    select_median_ga_run,
    summarize_instance_rows,
)
from comparison.parameter_sensitivity_medium_comparison import (
    _assignment_satisfies_hard_clauses,
    _build_config,
    _compact_evaluation,
    _exclusive_controller_lock,
    _load_method_result,
    _parse_args,
    _parse_maxhs_bound,
    _parse_maxhs_status,
    _next_schedulable_index,
    _structured_target_np,
    task_needs_execution,
)


@pytest.fixture
def values_table():
    return [
        {
            "node": "C2",
            "type": "C",
            "P_loss": None,
            "C_benefit": 20,
            "D_cost": None,
            "E_prob": None,
        },
        {
            "node": "P1",
            "type": "P",
            "P_loss": 200,
            "C_benefit": None,
            "D_cost": None,
            "E_prob": None,
        },
        {
            "node": "D2",
            "type": "D",
            "P_loss": None,
            "C_benefit": None,
            "D_cost": 80,
            "E_prob": None,
        },
        {
            "node": "E1",
            "type": "E",
            "P_loss": None,
            "C_benefit": None,
            "D_cost": None,
            "E_prob": 0.4,
        },
        {
            "node": "C1",
            "type": "C",
            "P_loss": None,
            "C_benefit": 10,
            "D_cost": None,
            "E_prob": None,
        },
        {
            "node": "P0",
            "type": "P",
            "P_loss": 0,
            "C_benefit": None,
            "D_cost": None,
            "E_prob": None,
        },
        {
            "node": "D1",
            "type": "D",
            "P_loss": None,
            "C_benefit": None,
            "D_cost": 50,
            "E_prob": None,
        },
    ]


def test_perturbation_is_reproducible_order_independent_and_non_mutating(values_table):
    original = copy.deepcopy(values_table)
    first = make_perturbation_vector(values_table, "C_benefit", 12345)
    replay = make_perturbation_vector(list(reversed(values_table)), "C_benefit", 12345)
    other = make_perturbation_vector(values_table, "C_benefit", 12346)

    assert first == replay
    assert first != other
    assert set(first) == {"C1", "C2"}
    assert all(-1.0 <= value <= 1.0 for value in first.values())
    assert len(set(first.values())) == 2

    perturbed = apply_node_perturbation(values_table, "C_benefit", 0.3, first)
    assert values_table == original
    assert perturbed is not values_table
    assert all(row is not source for row, source in zip(perturbed, values_table))


@pytest.mark.parametrize(
    ("parameter", "target_type", "field"),
    [
        ("C_benefit", "C", "C_benefit"),
        ("P_loss", "P", "P_loss"),
        ("D_cost", "D", "D_cost"),
    ],
)
def test_each_parameter_changes_only_its_own_node_field(
    values_table, parameter, target_type, field
):
    vector = make_perturbation_vector(values_table, parameter, 77)
    perturbed = apply_node_perturbation(values_table, parameter, 0.2, vector)
    by_node = {row["node"]: row for row in values_table}
    changed_by_node = {row["node"]: row for row in perturbed}

    for node, original in by_node.items():
        changed = changed_by_node[node]
        for key, value in original.items():
            if original["type"] == target_type and key == field and value is not None:
                expected = value * (1.0 + 0.2 * vector[node])
                assert changed[key] == pytest.approx(expected)
                assert value * 0.8 <= changed[key] <= value * 1.2
            else:
                assert changed[key] == value
    assert changed_by_node["E1"]["E_prob"] == 0.4


@pytest.mark.parametrize("parameter", ["E_prob", "D_cos"])
def test_unapproved_parameters_are_rejected(values_table, parameter):
    with pytest.raises(ValueError):
        make_perturbation_vector(values_table, parameter, 1)
    with pytest.raises(ValueError):
        apply_node_perturbation(values_table, parameter, 0.1, {})


def test_same_direction_vector_is_reused_across_delta(values_table):
    key10 = InstanceKey("random", 3, "P_loss", 10, 2)
    key20 = InstanceKey("random", 3, "P_loss", 20, 2)
    key30 = InstanceKey("random", 3, "P_loss", 30, 2)
    seeds = {perturbation_seed(42, key) for key in (key10, key20, key30)}
    assert len(seeds) == 1

    vector = make_perturbation_vector(values_table, "P_loss", seeds.pop())
    results = {
        delta: {
            row["node"]: row["P_loss"]
            for row in apply_node_perturbation(values_table, "P_loss", delta, vector)
            if row["type"] == "P"
        }
        for delta in (0.1, 0.2, 0.3)
    }
    for node in ("P0", "P1"):
        base = next(row["P_loss"] for row in values_table if row["node"] == node)
        if base == 0:
            assert all(results[delta][node] == 0 for delta in results)
            continue
        directions = [(results[delta][node] / base - 1.0) / delta for delta in results]
        assert directions[0] == pytest.approx(directions[1])
        assert directions[1] == pytest.approx(directions[2])


def test_full_and_pilot_matrix_and_seed_counts_are_exact():
    full = list(iter_instance_keys())
    pilot = list(iter_instance_keys(pilot=True))
    assert len(full) == 180
    assert len({key.instance_id for key in full}) == 180
    assert len(pilot) == 18
    assert len(full) * (len(core.METHODS) + 4) == 1440
    assert len(pilot) * (len(core.METHODS) + 4) == 144
    assert sum(key.parameter == "nominal" for key in full) == 0
    assert sum(key.parameter != "nominal" for key in full) == 180

    perturb_seeds = {
        perturbation_seed(42, key) for key in full if key.parameter != "nominal"
    }
    assert len(perturb_seeds) == 2 * 10 * 3

    ga_seeds = {ga_seed(42, key, run_index) for key in full for run_index in range(5)}
    assert len(ga_seeds) == 180 * 5


def _ga_run(run_index, objective, elapsed=10.0, status="complete"):
    return {
        "status": status,
        "run_index": run_index,
        "objective": objective,
        "complete_protocol_time_s": elapsed,
        "defended": [f"D{run_index}"],
        "C_benefit": objective + 2,
        "P_expected_loss": 1.0,
        "D_cost": 1.0,
    }


def test_ga_selects_third_objective_and_charges_all_five_runs():
    runs = [
        _ga_run(0, 50, 1),
        _ga_run(1, 10, 2),
        _ga_run(2, 40, 3),
        _ga_run(3, 30, 4),
        _ga_run(4, 20, 5),
    ]
    result = select_median_ga_run(runs)
    assert result["status"] == "complete"
    assert result["objective"] == 30
    assert result["selected_run_index"] == 3
    assert result["defended"] == ["D3"]
    assert result["complete_protocol_time_s"] == 15
    assert result["median_single_run_time_s"] == 3


def test_ga_partial_duplicate_and_nonfinite_runs_are_handled():
    assert select_median_ga_run([_ga_run(0, 1)])["status"] == "partial"
    with pytest.raises(ValueError, match="duplicate"):
        select_median_ga_run([_ga_run(0, 1), _ga_run(0, 2)])
    nonfinite = [
        _ga_run(index, value) for index, value in enumerate([1, 2, 3, 4, math.nan])
    ]
    assert select_median_ga_run(nonfinite)["status"] == "partial"


def _result(objective, status="complete", elapsed=1.0):
    return {
        "status": status,
        "objective": objective,
        "complete_protocol_time_s": elapsed,
    }


def test_win_tie_loss_uses_exact_objective_equality():
    maxsat = _result(100.0)
    assert compare_vs_maxsat(_result(100.0), maxsat) == "tie"
    assert compare_vs_maxsat(_result(100.000000000001), maxsat) == "win"
    assert compare_vs_maxsat(_result(99.999999999999), maxsat) == "loss"
    assert compare_vs_maxsat(_result(100.0, "timeout"), maxsat) == "na"
    assert compare_vs_maxsat(_result(100.0, "timeout_incumbent"), maxsat) == "na"
    assert compare_vs_maxsat(_result(100.0, "feasible_unverified"), maxsat) == "na"
    assert compare_vs_maxsat(_result(None), maxsat) == "na"

    # A nonzero tolerance remains available only when explicitly requested.
    assert compare_vs_maxsat(_result(100.5), maxsat, 0.5, 0.0) == "tie"


def test_summary_balances_perturbations_within_each_graph():
    rows = []
    for graph_id in (1, 2):
        for repetition in range(1, 6):
            ga = _result(100.0)
            if graph_id == 2:
                if repetition == 1:
                    ga = _result(90.0)
                elif repetition == 2:
                    ga = {
                        "status": "timeout",
                        "complete_protocol_time_s": 999.0,
                    }
                else:
                    ga = {"status": "missing"}
            rows.append(
                {
                    "family": "structured",
                    "graph_id": graph_id,
                    "parameter": "C_benefit",
                    "delta_pct": 10,
                    "perturbation_run": repetition,
                    "methods": {
                        "maxsat": _result(100.0),
                        "ga": ga,
                        "khouzani": {"status": "missing"},
                        "zenitani": {"status": "missing"},
                    },
                }
            )
    rows.append(
        {
            "family": "random",
            "graph_id": 1,
            "parameter": "P_loss",
            "delta_pct": 20,
            "perturbation_run": 1,
            "methods": {method: {"status": "missing"} for method in core.METHODS},
        }
    )

    summary = summarize_instance_rows(rows)
    group = next(
        item for item in summary["perturbed"] if item["family"] == "structured"
    )
    ga = group["methods"]["ga"]
    comparison = group["pairwise_vs_maxsat"]["ga"]
    assert ga["complete_count"] == 6
    assert ga["total_instances"] == 10
    assert ga["complete_fraction"] == pytest.approx(0.6)
    assert ga["complete_protocol_time_raw_s"]["n"] == 6
    assert ga["complete_protocol_time_raw_s"]["max"] == 1.0
    assert comparison["paired_count"] == 6
    assert comparison["graphs_with_pairs"] == 2
    assert comparison["absolute_gap_raw"]["mean"] == pytest.approx(10 / 6)
    assert comparison["absolute_gap_graph_balanced"]["mean"] == pytest.approx(5.0)
    assert comparison["wtl_raw"] == {"win": 0, "tie": 5, "loss": 1, "na": 4}
    assert comparison["wtl_fraction_graph_balanced"]["tie"]["mean"] == pytest.approx(
        0.5
    )
    assert comparison["wtl_paired_fraction_graph_balanced"]["tie"][
        "mean"
    ] == pytest.approx(0.5)
    assert len(summary["perturbed"]) == 2


def test_atomic_json_replace_and_failure_preserve_previous_file(tmp_path, monkeypatch):
    target = tmp_path / "state.json"
    atomic_write_json(target, {"version": 1})
    atomic_write_json(target, {"version": 2})
    assert load_json(target) == {"version": 2}
    assert not list(tmp_path.glob("*.tmp"))
    assert not list(tmp_path.glob(".*.tmp"))

    def fail_replace(_source, _target):
        raise OSError("simulated replace failure")

    monkeypatch.setattr(core.os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated"):
        atomic_write_json(target, {"version": 3})
    assert load_json(target) == {"version": 2}
    assert not list(tmp_path.glob(".*.tmp"))


def test_resume_helpers_only_schedule_missing_work():
    runs = [
        {"run_index": 0, "status": "complete"},
        {"run_index": 1, "status": "running"},
        {"run_index": 2, "status": "complete"},
    ]
    assert missing_ga_runs(runs, 5) == [1, 3, 4]
    assert not method_needs_run({"status": "complete"})
    assert not method_needs_run({"status": "timeout"})
    assert not method_needs_run({"status": "timeout_incumbent"})
    assert not method_needs_run({"status": "feasible_unverified"})
    assert method_needs_run({"status": "running"})
    assert not method_needs_run({"status": "error"})
    assert method_needs_run({"status": "error"}, retry_failed=True)


def test_task_resume_validates_provenance_and_terminal_states(tmp_path):
    result_path = tmp_path / "result.json"
    task = {
        "task_signature": "task-a",
        "instance_signature": "instance-a",
        "result_path": str(result_path),
    }
    assert task_needs_execution(task)

    for status, expected in (
        ("running", True),
        ("complete", False),
        ("timeout", False),
    ):
        atomic_write_json(
            result_path,
            {
                "task_signature": "task-a",
                "instance_signature": "instance-a",
                "status": status,
            },
        )
        assert task_needs_execution(task) is expected

    atomic_write_json(
        result_path,
        {
            "task_signature": "task-a",
            "instance_signature": "instance-a",
            "status": "error",
        },
    )
    assert not task_needs_execution(task)
    assert task_needs_execution(task, retry_failed=True)

    atomic_write_json(
        result_path,
        {
            "task_signature": "stale-task",
            "instance_signature": "instance-a",
            "status": "complete",
        },
    )
    with pytest.raises(RuntimeError, match="provenance"):
        task_needs_execution(task)


def test_summary_loader_rejects_stale_method_result(tmp_path):
    instance_dir = tmp_path / "instance"
    instance = {
        "config_fingerprint": "config-a",
        "instance_signature": "instance-a",
    }
    task_path = instance_dir / "tasks" / "maxsat.json"
    result_path = instance_dir / "results" / "maxsat.json"
    atomic_write_json(
        task_path,
        {"task_signature": "task-a", "instance_signature": "instance-a"},
    )
    atomic_write_json(
        result_path,
        {
            "config_fingerprint": "config-a",
            "instance_signature": "instance-a",
            "task_signature": "task-a",
            "status": "complete",
        },
    )
    assert _load_method_result(instance_dir, instance, "maxsat")["status"] == "complete"

    stale = load_json(result_path)
    stale["task_signature"] = "stale-task"
    atomic_write_json(result_path, stale)
    with pytest.raises(RuntimeError, match="task provenance"):
        _load_method_result(instance_dir, instance, "maxsat")


def test_output_controller_lock_is_exclusive(tmp_path):
    with _exclusive_controller_lock(tmp_path):
        with pytest.raises(RuntimeError, match="another controller"):
            with _exclusive_controller_lock(tmp_path):
                pass


def test_scheduler_never_runs_two_methods_of_one_instance_concurrently():
    pending = [
        ("a-maxsat", {"instance_id": "a"}),
        ("a-ga0", {"instance_id": "a"}),
        ("b-maxsat", {"instance_id": "b"}),
    ]
    active = {"a-maxsat": {"task": {"instance_id": "a"}}}
    assert _next_schedulable_index(pending, active) == 2
    active["b-maxsat"] = {"task": {"instance_id": "b"}}
    assert _next_schedulable_index(pending, active) is None


@pytest.mark.parametrize(
    "extra_args",
    [
        ["--methods", "ga", "ga"],
        ["--families", "structured", "random", "random"],
        ["--parameters", "C_benefit", "P_loss", "D_cost", "D_cost"],
    ],
)
def test_cli_rejects_duplicate_matrix_or_method_items(extra_args):
    with pytest.raises(SystemExit):
        _parse_args(["--generate-only", *extra_args])


def test_frozen_config_and_structured_size_sequence():
    args = _parse_args(["--generate-only"])
    config = _build_config(args)
    assert config["parameters"] == ["C_benefit", "P_loss", "D_cost"]
    assert config["deltas_pct"] == [10, 20, 30]
    assert config["perturbation_runs"] == 1
    assert config["include_nominal"] is False
    assert config["algorithms"]["ga"]["runs"] == 5
    assert config["algorithms"]["ga"]["population_size"] == 100
    assert config["algorithms"]["ga"]["generations"] == 50
    assert config["algorithms"]["ga"]["crossover"] == "single_point"
    assert config["algorithms"]["khouzani"]["n_budget"] == 10
    assert config["algorithms"]["khouzani"]["milp_time_limit_s"] == 60
    assert config["algorithms"]["zenitani"]["iterations"] == 5
    assert config["algorithms"]["zenitani"]["neighbor_sample"] == 5
    assert config["final_evaluator"] == {
        "name": "Loopy BP",
        "bp_max_iters": 100,
        "bp_damping": 0.2,
        "bp_tol": 1e-6,
        "bp_fast": True,
    }
    assert config["execution"]["wall_timeouts_s"] == {
        "maxsat": 630,
        "ga": 7200,
        "khouzani": 900,
        "zenitani": 7200,
    }
    assert config["execution"]["workers"] == 8
    assert config["comparison"]["tie_rule"] == (
        "exact equality of final objective values"
    )
    assert config["comparison"]["tie_abs"] == 0.0
    assert config["comparison"]["tie_rel"] == 0.0
    assert config["source_fingerprints"]
    assert len(config["value_ranges"]["E_prob"]["values"]) == 24
    assert config["datasets"]["random"]["legacy_rng_compat"] is False
    assert config["datasets"]["structured"]["legacy_rng_compat"] is False
    assert [
        config["datasets"]["random"][name] for name in ("nP", "nE", "nC", "nD")
    ] == [100, 240, 300, 150]
    assert [_structured_target_np(config, graph_id) for graph_id in range(1, 11)] == [
        191,
        158,
        152,
        198,
        168,
        166,
        165,
        159,
        198,
        157,
    ]


def test_solver_status_bound_and_bp_status_parsing():
    output = "\n".join(
        [
            "c Best LB Found: 123",
            "c Best UB Found: 130",
            "s UNKNOWN",
        ]
    )
    assert _parse_maxhs_status(output) == "UNKNOWN"
    assert _parse_maxhs_bound(output, "LB") == 123
    assert _parse_maxhs_bound(output, "UB") == 130
    assert _assignment_satisfies_hard_clauses(
        {1: True, 2: False, 3: True}, [[1, 2], [-2, 3]]
    )
    assert not _assignment_satisfies_hard_clauses(
        {1: False, 2: False, 3: True}, [[1, 2], [-2, 3]]
    )

    compact = _compact_evaluation(
        {
            "objective": 1.0,
            "C_benefit": 4.0,
            "P_expected_loss": 2.0,
            "D_cost": 1.0,
            "_bp_converged": False,
            "_bp_iters": 100,
            "_bp_max_delta": 2.5e-5,
            "_bp_time_ms": 12.5,
        }
    )
    assert compact["bp_converged"] is False
    assert compact["bp_iters"] == 100
    assert compact["bp_max_delta"] == 2.5e-5
    assert compact["bp_time_ms"] == 12.5
