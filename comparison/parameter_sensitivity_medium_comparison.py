#!/usr/bin/env python3
"""Four-method comparison under node-wise parameter perturbations.

The runner freezes the two standard medium graph families, creates paired
node-wise perturbations of C_benefit/P_loss/D_cost, and evaluates MaxSAT,
GA-BP, Khouzani, and Zenitani on exactly the same parameter tables.
Nominal tables are retained only as immutable source data and are not solved.

No expensive work is started implicitly.  Use ``--generate-only`` to prepare
the immutable inputs or ``--execute`` to launch solver workers.  ``--pilot``
selects the confirmed reusable subset (graph 1, perturbation repetition 1).
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import re
import signal
import subprocess
import sys
import time
import traceback

import networkx as nx


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from common import parse_maxhs_solution_str  # noqa: E402
from comparison.bp_core import (  # noqa: E402
    MAXHS_BIN,
    compute_objective_bp_style,
    find_best_defense_bp,
    get_cnf_converter,
    interpret_solution,
)
from comparison.khouzani_baseline import find_best_defense_khouzani  # noqa: E402
from comparison.parameter_sensitivity_core import (  # noqa: E402
    FAMILIES,
    METHODS,
    NOMINAL_PARAMETER,
    PARAM_FIELDS,
    InstanceKey,
    apply_node_perturbation,
    atomic_write_json,
    canonical_hash,
    compare_vs_maxsat,
    config_fingerprint,
    ga_seed,
    iter_instance_keys,
    load_json,
    make_perturbation_vector,
    method_needs_run,
    method_seed,
    pairwise_gap,
    perturbation_seed,
    select_median_ga_run,
    summarize_instance_rows,
)
from comparison.zenitani_baseline import find_best_defense_zenitani  # noqa: E402
from generate_graph.config import FIXED_E_PROBS  # noqa: E402
from generate_graph.number_generation import (  # noqa: E402
    generate_node_values as generate_structured_values,
)
from generate_graph.numerical_generation import (  # noqa: E402
    generate_node_values as generate_random_values,
)
from generate_graph.random_graph import generate_bn_dag_multi_pe  # noqa: E402
from generate_graph.structured_graph import generate_bn_from_root_and_reverse  # noqa: E402
from graph2sat.graph2sat import export_to_wcnf  # noqa: E402


SCHEMA_VERSION = 1
DEFAULT_OUTPUT_DIR = "comparison/parameter_sensitivity_medium_results"
FROZEN_SOURCE_FILES = (
    "comparison/parameter_sensitivity_medium_comparison.py",
    "comparison/parameter_sensitivity_core.py",
    "comparison/bp_core.py",
    "comparison/khouzani_baseline.py",
    "comparison/zenitani_baseline.py",
    "common.py",
    "generate_graph/random_graph.py",
    "generate_graph/structured_graph.py",
    "generate_graph/numerical_generation.py",
    "generate_graph/number_generation.py",
    "generate_graph/config.py",
    "graph2sat/graph2sat.py",
)
THREAD_LIMIT_ENV = {
    "PYTHONHASHSEED": "0",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}
KHOUZANI_BP = {
    "bp_max_iters": 50,
    "bp_damping": 0.5,
    "bp_tol": 1e-6,
    "bp_fast": False,
}
ZENITANI_BP = {
    "bp_max_iters": 50,
    "bp_damping": 0.5,
    "bp_tol": 1e-6,
    "bp_fast": True,
}


def _now():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _resolve_path(path):
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def _file_sha256(path):
    path = Path(path)
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def _source_fingerprints():
    return {relative: _file_sha256(ROOT / relative) for relative in FROZEN_SOURCE_FILES}


def _canonicalize_graph(bn):
    out = dict(bn)
    out["edges"] = sorted([tuple(edge) for edge in bn["edges"]])
    return out


def _graph_signature(bn):
    canonical = dict(bn)
    canonical["edges"] = [
        list(edge) for edge in sorted(tuple(edge) for edge in bn["edges"])
    ]
    return canonical_hash(canonical)


def _values_signature(values_table):
    return canonical_hash(sorted(values_table, key=lambda row: row["node"]))


def _graph_stats(bn):
    return {
        "nP": len(bn["P"]),
        "nE": len(bn["E"]),
        "nC": len(bn["C"]),
        "nD": len(bn["D"]),
        "total": sum(len(bn[kind]) for kind in ("P", "E", "C", "D")),
        "edges": len(bn["edges"]),
        "max_level": max(bn.get("level", {}).values(), default=None),
    }


def _validate_graph(bn):
    nodes = sum((list(bn[kind]) for kind in ("P", "E", "C", "D")), [])
    if set(nodes) != set(bn["node_type"]):
        raise RuntimeError("node lists and node_type map disagree")
    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(bn["edges"])
    if not nx.is_directed_acyclic_graph(graph):
        raise RuntimeError("generated medium graph is not a DAG")


def _structured_target_np(config, graph_id):
    rng = random.Random(config["seed"])
    value = None
    for _ in range(graph_id):
        value = rng.randint(
            config["datasets"]["structured"]["nP_min"],
            config["datasets"]["structured"]["nP_max"],
        )
    return value


def generate_base_graph(config, family, graph_id):
    """Serial-only graph/value generation used to create immutable snapshots."""
    graph_seed = config["seed"] + graph_id
    value_ranges = config["value_ranges"]
    if family == "random":
        spec = config["datasets"]["random"]
        bn = generate_bn_dag_multi_pe(
            nP=spec["nP"],
            nE=spec["nE"],
            nC=spec["nC"],
            nD=spec["nD"],
            max_children=spec["max_children"],
            p_EP=spec["p_ep"],
            seed=graph_seed,
        )
        values_table = generate_random_values(
            bn,
            seed=graph_seed,
            fixed_e_probs=FIXED_E_PROBS,
            p_loss_range=tuple(value_ranges["P_loss"]),
            c_benefit_range=tuple(value_ranges["C_benefit"]),
            d_cost_range=tuple(value_ranges["D_cost"]),
            legacy_rng_compat=spec["legacy_rng_compat"],
        )
        target_nP = spec["nP"]
    elif family == "structured":
        spec = config["datasets"]["structured"]
        target_nP = _structured_target_np(config, graph_id)
        bn = generate_bn_from_root_and_reverse(
            nP=target_nP,
            seed=graph_seed,
            extra_p_edge_prob=spec["extra_p_edge_prob"],
            max_extra_p_out_per_node=spec["max_extra_p_out_per_node"],
            c_parents_per_e_range=tuple(spec["c_parents_per_e_range"]),
            max_e_children_per_c=spec["max_e_children_per_c"],
            max_c_children_per_d=spec["max_c_children_per_d"],
        )
        values_table = generate_structured_values(
            bn,
            seed=graph_seed,
            fixed_e_probs=FIXED_E_PROBS,
            p_loss_range=tuple(value_ranges["P_loss"]),
            c_benefit_range=tuple(value_ranges["C_benefit"]),
            d_cost_range=tuple(value_ranges["D_cost"]),
            use_level_scaling=spec["use_level_scaling"],
            p_alpha_max=spec["p_alpha_max"],
            cd_beta_max=spec["cd_beta_max"],
            cd_floor=spec["cd_floor"],
            legacy_rng_compat=spec["legacy_rng_compat"],
        )
    else:
        raise ValueError(f"unsupported graph family: {family}")

    bn = _canonicalize_graph(bn)
    _validate_graph(bn)
    return {
        "schema_version": SCHEMA_VERSION,
        "config_fingerprint": config_fingerprint(config),
        "family": family,
        "graph_id": graph_id,
        "graph_seed": graph_seed,
        "value_seed": graph_seed,
        "target_nP": target_nP,
        "graph_signature": _graph_signature(bn),
        "base_values_signature": _values_signature(values_table),
        "graph_stats": _graph_stats(bn),
        "bn": bn,
        "values_table": values_table,
    }


def _base_snapshot_path(output_dir, family, graph_id):
    return output_dir / "dataset" / f"{family}_g{graph_id:03d}.json"


def prepare_base_snapshot(config, output_dir, family, graph_id):
    path = _base_snapshot_path(output_dir, family, graph_id)
    if path.exists():
        snapshot = load_json(path)
        if snapshot.get("config_fingerprint") != config_fingerprint(config):
            raise RuntimeError(f"base snapshot config mismatch: {path}")
        if snapshot.get("family") != family or snapshot.get("graph_id") != graph_id:
            raise RuntimeError(f"base snapshot identity mismatch: {path}")
        if snapshot.get("graph_signature") != _graph_signature(snapshot["bn"]):
            raise RuntimeError(f"base snapshot graph signature mismatch: {path}")
        if snapshot.get("base_values_signature") != _values_signature(
            snapshot["values_table"]
        ):
            raise RuntimeError(f"base snapshot value signature mismatch: {path}")
        return path, snapshot
    snapshot = generate_base_graph(config, family, graph_id)
    atomic_write_json(path, snapshot)
    return path, snapshot


def _instance_dir(output_dir, key):
    return output_dir / "instances" / key.instance_id


def _instance_signature(instance):
    return canonical_hash(
        {
            "key": instance["key"],
            "base_graph_signature": instance["base_graph_signature"],
            "values_signature": instance["values_signature"],
            "perturbation": instance["perturbation"],
        }
    )


def prepare_instance(config, output_dir, key, base_path, base_snapshot):
    instance_dir = _instance_dir(output_dir, key)
    path = instance_dir / "instance.json"
    if path.exists():
        instance = load_json(path)
        if instance.get("config_fingerprint") != config_fingerprint(config):
            raise RuntimeError(f"instance config mismatch: {path}")
        if instance.get("key") != key.to_dict():
            raise RuntimeError(f"instance key mismatch: {path}")
        if instance.get("base_graph_signature") != base_snapshot["graph_signature"]:
            raise RuntimeError(f"instance base graph mismatch: {path}")
        if (
            instance.get("base_values_signature")
            != base_snapshot["base_values_signature"]
        ):
            raise RuntimeError(f"instance base values mismatch: {path}")
        if instance.get("values_signature") != _values_signature(
            instance["values_table"]
        ):
            raise RuntimeError(f"instance values signature mismatch: {path}")
        if key.parameter == NOMINAL_PARAMETER:
            expected_perturbation = {"seed": None, "vector": None}
            expected_values = base_snapshot["values_table"]
        else:
            expected_seed = perturbation_seed(config["seed"], key)
            expected_vector = make_perturbation_vector(
                base_snapshot["values_table"], key.parameter, expected_seed
            )
            expected_perturbation = {
                "seed": expected_seed,
                "vector": expected_vector,
            }
            expected_values = apply_node_perturbation(
                base_snapshot["values_table"],
                key.parameter,
                key.delta_pct / 100.0,
                expected_vector,
            )
        stored_perturbation = instance.get("perturbation", {})
        if any(
            stored_perturbation.get(field) != expected
            for field, expected in expected_perturbation.items()
        ):
            raise RuntimeError(f"instance perturbation mismatch: {path}")
        if instance.get("values_signature") != _values_signature(expected_values):
            raise RuntimeError(f"instance derived values mismatch: {path}")
        if instance.get("instance_signature") != _instance_signature(instance):
            raise RuntimeError(f"instance signature mismatch: {path}")
        return path, instance

    base_values = base_snapshot["values_table"]
    if key.parameter == NOMINAL_PARAMETER:
        seed = None
        vector = None
        values_table = [dict(row) for row in base_values]
    else:
        seed = perturbation_seed(config["seed"], key)
        vector = make_perturbation_vector(base_values, key.parameter, seed)
        values_table = apply_node_perturbation(
            base_values,
            key.parameter,
            key.delta_pct / 100.0,
            vector,
        )
    instance = {
        "schema_version": SCHEMA_VERSION,
        "config_fingerprint": config_fingerprint(config),
        "key": key.to_dict(),
        "instance_id": key.instance_id,
        "base_snapshot": os.path.relpath(base_path, output_dir),
        "base_graph_signature": base_snapshot["graph_signature"],
        "base_values_signature": base_snapshot["base_values_signature"],
        "perturbation": {
            "distribution": "independent uniform[-1,1] multiplicative direction",
            "seed": seed,
            "vector": vector,
        },
        "values_signature": _values_signature(values_table),
        "values_table": values_table,
    }
    instance["instance_signature"] = _instance_signature(instance)
    atomic_write_json(path, instance)
    return path, instance


def _compact_evaluation(evaluation):
    return {
        "objective": evaluation.get("objective"),
        "C_benefit": evaluation.get("C_benefit"),
        "P_expected_loss": evaluation.get("P_expected_loss"),
        "D_cost": evaluation.get("D_cost"),
        "bp_converged": evaluation.get("_bp_converged"),
        "bp_iters": evaluation.get("_bp_iters"),
        "bp_max_delta": evaluation.get("_bp_max_delta"),
        "bp_time_ms": evaluation.get("_bp_time_ms"),
    }


def _final_evaluate(config, bn, values_table, state):
    final_bp = config["final_evaluator"]
    return compute_objective_bp_style(
        bn,
        values_table,
        state,
        bp_max_iters=final_bp["bp_max_iters"],
        bp_damping=final_bp["bp_damping"],
        bp_tol=final_bp["bp_tol"],
        fast=final_bp["bp_fast"],
    )


def _defended(state):
    return sorted(node for node, enabled in state.items() if enabled)


def _parse_maxhs_status(output):
    statuses = [
        line[2:].strip() for line in output.splitlines() if line.startswith("s ")
    ]
    return statuses[-1] if statuses else None


def _parse_maxhs_bound(output, name):
    matches = re.findall(
        rf"^c Best {name} Found:\s*(-?\d+)", output, flags=re.MULTILINE
    )
    return int(matches[-1]) if matches else None


def _assignment_satisfies_hard_clauses(assignment, hard_clauses):
    return all(
        any(assignment.get(abs(literal)) is (literal > 0) for literal in clause)
        for clause in hard_clauses
    )


def _write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{os.getpid()}.tmp"
    try:
        with open(temporary, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


@contextmanager
def _exclusive_controller_lock(output_dir):
    lock_path = output_dir / ".controller.lock"
    handle = open(lock_path, "a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                f"another controller is using this output directory: {output_dir}"
            ) from error
        handle.seek(0)
        handle.truncate()
        json.dump({"pid": os.getpid(), "locked_at": _now()}, handle)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        yield
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def _run_maxsat(config, bn, values_table, task):
    algorithm = config["algorithms"]["maxsat"]
    wcnf_path = Path(task["wcnf_path"])
    solver_log_path = Path(task["solver_log_path"])
    wcnf_path.parent.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    converter = get_cnf_converter("pro")
    cnf_data = converter(bn, values_table, initial_true_nodes=None)
    export_to_wcnf(cnf_data, str(wcnf_path))
    cmd = [
        config["maxhs_bin"],
        "-printSoln",
        "-printBstSoln",
        "-verb=0",
        f"-cpu-lim={algorithm['cpu_limit_s']}",
        str(wcnf_path),
    ]
    timed_out_inside = False
    return_code = None
    try:
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=algorithm["subprocess_timeout_s"],
            check=False,
        )
        output = (completed.stdout or "") + (completed.stderr or "")
        return_code = completed.returncode
    except subprocess.TimeoutExpired as error:
        timed_out_inside = True
        stdout = (
            error.stdout.decode()
            if isinstance(error.stdout, bytes)
            else (error.stdout or "")
        )
        stderr = (
            error.stderr.decode()
            if isinstance(error.stderr, bytes)
            else (error.stderr or "")
        )
        output = stdout + stderr
    _write_text(solver_log_path, output)
    solver_status = _parse_maxhs_status(output)
    assignment = parse_maxhs_solution_str(output, len(cnf_data["var_map"]))
    search_time = time.perf_counter() - started
    base = {
        "search_wall_time_s": search_time,
        "solver_status": solver_status,
        "solver_return_code": return_code,
        "solver_internal_timeout": timed_out_inside,
        "lower_bound": _parse_maxhs_bound(output, "LB"),
        "upper_bound": _parse_maxhs_bound(output, "UB"),
        "solver_log": task["solver_log_path"],
        "wcnf_path": task["wcnf_path"],
        "assignment_variables": len(assignment),
        "expected_variables": len(cnf_data["var_map"]),
    }
    assignment_complete = set(assignment) == set(range(1, len(cnf_data["var_map"]) + 1))
    hard_clauses_satisfied = assignment_complete and _assignment_satisfies_hard_clauses(
        assignment, cnf_data["hard_clauses"]
    )
    base["assignment_complete"] = assignment_complete
    base["hard_clauses_satisfied"] = hard_clauses_satisfied
    if not assignment_complete or not hard_clauses_satisfied:
        status = (
            "timeout" if timed_out_inside or solver_status == "UNKNOWN" else "error"
        )
        return {"status": status, "complete_protocol_time_s": search_time, **base}

    _, by_type = interpret_solution(bn, cnf_data, assignment)
    state = {node: by_type["D"].get(node, False) for node in bn["D"]}
    eval_started = time.perf_counter()
    final_evaluation = _compact_evaluation(
        _final_evaluate(config, bn, values_table, state)
    )
    evaluation_time = time.perf_counter() - eval_started
    if solver_status == "OPTIMUM FOUND":
        status = "complete"
    elif solver_status == "UNKNOWN" or timed_out_inside:
        status = "timeout_incumbent"
    else:
        status = "feasible_unverified"
    return {
        "status": status,
        "final_evaluation_time_s": evaluation_time,
        "complete_protocol_time_s": search_time + evaluation_time,
        "defended": _defended(state),
        "n_defended": len(_defended(state)),
        "final_evaluation": final_evaluation,
        **final_evaluation,
        **base,
    }


def _run_ga(config, bn, values_table, key, run_index):
    algorithm = config["algorithms"]["ga"]
    run_seed = ga_seed(config["seed"], key, run_index)
    started = time.perf_counter()
    result = find_best_defense_bp(
        bn,
        values_table,
        population_size=algorithm["population_size"],
        genmax=algorithm["generations"],
        seed=run_seed,
        bp_max_iters=algorithm["search_bp"]["bp_max_iters"],
        bp_damping=algorithm["search_bp"]["bp_damping"],
        bp_tol=algorithm["search_bp"]["bp_tol"],
        bp_fast=algorithm["search_bp"]["bp_fast"],
        top_n=algorithm["top_n"],
        crossover_kind="single_point",
        crossover_prob=algorithm["crossover_probability"],
        mutation_prob=algorithm["individual_mutation_probability"],
        mutation_prob_var=algorithm["per_bit_mutation_probability"],
    )
    search_time = time.perf_counter() - started
    state = result["best_D_state"]
    eval_started = time.perf_counter()
    final_evaluation = _compact_evaluation(
        _final_evaluate(config, bn, values_table, state)
    )
    evaluation_time = time.perf_counter() - eval_started
    return {
        "status": "complete",
        "run_index": run_index,
        "seed": run_seed,
        "search_wall_time_s": search_time,
        "final_evaluation_time_s": evaluation_time,
        "complete_protocol_time_s": search_time + evaluation_time,
        "unique_evaluated": result["unique_evaluated"],
        "search_evaluation": _compact_evaluation(result["best_result"]),
        "final_evaluation": final_evaluation,
        "defended": _defended(state),
        "n_defended": len(_defended(state)),
        **final_evaluation,
    }


def _run_khouzani(config, bn, values_table):
    algorithm = config["algorithms"]["khouzani"]
    started = time.perf_counter()
    result = find_best_defense_khouzani(
        bn,
        values_table,
        n_budget=algorithm["n_budget"],
        eval_mode="bp",
        method=algorithm["method"],
        milp_time_limit=algorithm["milp_time_limit_s"],
        threads=algorithm["threads"],
        bp_max_iters=algorithm["search_bp"]["bp_max_iters"],
        bp_damping=algorithm["search_bp"]["bp_damping"],
        bp_tol=algorithm["search_bp"]["bp_tol"],
        bp_fast=algorithm["search_bp"]["bp_fast"],
    )
    search_time = time.perf_counter() - started
    state = result["best_D_state"]
    eval_started = time.perf_counter()
    final_evaluation = _compact_evaluation(
        _final_evaluate(config, bn, values_table, state)
    )
    evaluation_time = time.perf_counter() - eval_started
    return {
        "status": "complete",
        "search_wall_time_s": search_time,
        "final_evaluation_time_s": evaluation_time,
        "complete_protocol_time_s": search_time + evaluation_time,
        "milp_time_s": result["milp_time_s"],
        "n_milp_solves": result["n_milp_solves"],
        "n_milp_optimal": result["n_milp_optimal"],
        "all_milp_optimal": result["n_milp_optimal"] == result["n_milp_solves"],
        "best_budget": result["best_budget"],
        "search_evaluation": _compact_evaluation(result["best_eval"]),
        "final_evaluation": final_evaluation,
        "defended": _defended(state),
        "n_defended": len(_defended(state)),
        **final_evaluation,
    }


def _run_zenitani(config, bn, values_table, key):
    algorithm = config["algorithms"]["zenitani"]
    seed = method_seed(config["seed"], "zenitani", key)
    started = time.perf_counter()
    result = find_best_defense_zenitani(
        bn,
        values_table,
        eval_mode="bp",
        n_iter=algorithm["iterations"],
        sample=algorithm["neighbor_sample"],
        max_k=algorithm["max_controls"],
        seed=seed,
        bp_max_iters=algorithm["search_bp"]["bp_max_iters"],
        bp_damping=algorithm["search_bp"]["bp_damping"],
        bp_tol=algorithm["search_bp"]["bp_tol"],
        bp_fast=algorithm["search_bp"]["bp_fast"],
    )
    search_time = time.perf_counter() - started
    state = result["best_D_state"]
    eval_started = time.perf_counter()
    final_evaluation = _compact_evaluation(
        _final_evaluate(config, bn, values_table, state)
    )
    evaluation_time = time.perf_counter() - eval_started
    return {
        "status": "complete",
        "seed": seed,
        "search_wall_time_s": search_time,
        "final_evaluation_time_s": evaluation_time,
        "complete_protocol_time_s": search_time + evaluation_time,
        "n_evals": result["n_evals"],
        "hit_max_controls": result["hit_max_k"],
        "search_evaluation": _compact_evaluation(result["best_eval"]),
        "final_evaluation": final_evaluation,
        "defended": _defended(state),
        "n_defended": len(_defended(state)),
        **final_evaluation,
    }


def _task_result_path(instance_dir, method, run_index=None):
    if method == "ga":
        return instance_dir / "results" / f"ga_run_{run_index:02d}.json"
    return instance_dir / "results" / f"{method}.json"


def _task_descriptor_path(instance_dir, method, run_index=None):
    suffix = f"ga_run_{run_index:02d}" if method == "ga" else method
    return instance_dir / "tasks" / f"{suffix}.json"


def make_task(config_path, output_dir, instance_path, instance, method, run_index=None):
    instance_dir = instance_path.parent
    result_path = _task_result_path(instance_dir, method, run_index)
    suffix = f"ga_run_{run_index:02d}" if method == "ga" else method
    task_id = f"{instance['instance_id']}__{suffix}"
    task = {
        "schema_version": SCHEMA_VERSION,
        "task_id": task_id,
        "instance_id": instance["instance_id"],
        "config_path": str(config_path),
        "config_fingerprint": instance["config_fingerprint"],
        "instance_path": str(instance_path),
        "instance_signature": instance["instance_signature"],
        "base_snapshot_path": str(output_dir / instance["base_snapshot"]),
        "method": method,
        "run_index": run_index,
        "result_path": str(result_path),
        "worker_log_path": str(instance_dir / "logs" / f"{suffix}.worker.log"),
        "wcnf_path": str(instance_dir / "maxsat" / "input.wcnf")
        if method == "maxsat"
        else None,
        "solver_log_path": str(instance_dir / "maxsat" / "maxhs.log")
        if method == "maxsat"
        else None,
    }
    task["task_signature"] = canonical_hash(
        {key: value for key, value in task.items() if key != "task_signature"}
    )
    task_path = _task_descriptor_path(instance_dir, method, run_index)
    atomic_write_json(task_path, task)
    return task_path, task


def _validate_worker_inputs(task, config, instance, base_snapshot):
    task_content = {
        key: value for key, value in task.items() if key != "task_signature"
    }
    if task.get("task_signature") != canonical_hash(task_content):
        raise RuntimeError("worker task content signature mismatch")
    if task["config_fingerprint"] != config_fingerprint(config):
        raise RuntimeError("worker config fingerprint mismatch")
    if config.get("source_fingerprints") != _source_fingerprints():
        raise RuntimeError("worker source fingerprint mismatch")
    if task["method"] == "maxsat" and config.get("maxhs_binary_sha256") != _file_sha256(
        config["maxhs_bin"]
    ):
        raise RuntimeError("worker MaxHS binary fingerprint mismatch")
    if task["instance_signature"] != instance["instance_signature"]:
        raise RuntimeError("worker instance signature mismatch")
    if instance["instance_signature"] != _instance_signature(instance):
        raise RuntimeError("worker instance content signature mismatch")
    if instance["base_graph_signature"] != base_snapshot["graph_signature"]:
        raise RuntimeError("worker base graph signature mismatch")
    if instance["base_values_signature"] != base_snapshot["base_values_signature"]:
        raise RuntimeError("worker base values signature mismatch")
    if base_snapshot["graph_signature"] != _graph_signature(base_snapshot["bn"]):
        raise RuntimeError("worker base graph content signature mismatch")
    if base_snapshot["base_values_signature"] != _values_signature(
        base_snapshot["values_table"]
    ):
        raise RuntimeError("worker base values content signature mismatch")
    if instance["values_signature"] != _values_signature(instance["values_table"]):
        raise RuntimeError("worker perturbed values signature mismatch")


def run_worker(task_path):
    task = load_json(task_path)
    config = load_json(task["config_path"])
    instance = load_json(task["instance_path"])
    base_snapshot = load_json(task["base_snapshot_path"])
    result_path = Path(task["result_path"])
    envelope = {
        "schema_version": SCHEMA_VERSION,
        "task_id": task["task_id"],
        "task_signature": task["task_signature"],
        "config_fingerprint": task["config_fingerprint"],
        "instance_signature": task["instance_signature"],
        "method": task["method"],
        "run_index": task.get("run_index"),
        "worker_pid": os.getpid(),
        "started_at": _now(),
    }
    try:
        _validate_worker_inputs(task, config, instance, base_snapshot)
        bn = base_snapshot["bn"]
        values_table = instance["values_table"]
        key = InstanceKey(**instance["key"])
        if task["method"] == "maxsat":
            result = _run_maxsat(config, bn, values_table, task)
        elif task["method"] == "ga":
            result = _run_ga(config, bn, values_table, key, int(task["run_index"]))
        elif task["method"] == "khouzani":
            result = _run_khouzani(config, bn, values_table)
        elif task["method"] == "zenitani":
            result = _run_zenitani(config, bn, values_table, key)
        else:
            raise ValueError(f"unknown method: {task['method']}")
        envelope.update(result)
        envelope["finished_at"] = _now()
        atomic_write_json(result_path, envelope)
        return 0
    except Exception as error:
        envelope.update(
            {
                "status": "error",
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "finished_at": _now(),
            }
        )
        atomic_write_json(result_path, envelope)
        traceback.print_exc()
        return 1


def _task_timeout(config, task):
    return config["execution"]["wall_timeouts_s"][task["method"]]


def _result_matches_task(record, task):
    return bool(
        record.get("task_signature") == task["task_signature"]
        and record.get("instance_signature") == task["instance_signature"]
    )


def task_needs_execution(task, retry_failed=False):
    result_path = Path(task["result_path"])
    if not result_path.exists():
        return True
    record = load_json(result_path)
    if not _result_matches_task(record, task):
        raise RuntimeError(f"task result provenance mismatch: {result_path}")
    return method_needs_run(record, retry_failed=retry_failed)


def _terminate_process_group(process):
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=5)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    process.wait(timeout=5)


def _timeout_result(task, elapsed, timeout_s):
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": task["task_id"],
        "task_signature": task["task_signature"],
        "config_fingerprint": task["config_fingerprint"],
        "instance_signature": task["instance_signature"],
        "method": task["method"],
        "run_index": task.get("run_index"),
        "status": "timeout",
        "outer_wall_timeout_s": timeout_s,
        "elapsed_until_timeout_s": elapsed,
        "finished_at": _now(),
    }


def _error_result(task, error_message, return_code=None):
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": task["task_id"],
        "task_signature": task["task_signature"],
        "config_fingerprint": task["config_fingerprint"],
        "instance_signature": task["instance_signature"],
        "method": task["method"],
        "run_index": task.get("run_index"),
        "status": "error",
        "worker_return_code": return_code,
        "error": error_message,
        "finished_at": _now(),
    }


def _write_progress(output_dir, total, completed, active, pending):
    atomic_write_json(
        output_dir / "progress.json",
        {
            "updated_at": _now(),
            "total_selected_tasks": total,
            "completed_this_invocation": completed,
            "active": sorted(active),
            "pending": pending,
        },
    )


def _next_schedulable_index(pending, active):
    active_instances = {item["task"]["instance_id"] for item in active.values()}
    return next(
        (
            index
            for index, (_path, candidate) in enumerate(pending)
            if candidate["instance_id"] not in active_instances
        ),
        None,
    )


def execute_tasks(config, output_dir, tasks, workers, retry_failed=False):
    pending = [
        (task_path, task)
        for task_path, task in tasks
        if task_needs_execution(task, retry_failed=retry_failed)
    ]
    total = len(pending)
    active = {}
    completed_count = 0
    worker_env = os.environ.copy()
    worker_env.update(THREAD_LIMIT_ENV)

    try:
        while pending or active:
            while pending and len(active) < workers:
                next_index = _next_schedulable_index(pending, active)
                if next_index is None:
                    break
                task_path, task = pending.pop(next_index)
                running = {
                    "schema_version": SCHEMA_VERSION,
                    "task_id": task["task_id"],
                    "task_signature": task["task_signature"],
                    "config_fingerprint": task["config_fingerprint"],
                    "instance_signature": task["instance_signature"],
                    "method": task["method"],
                    "run_index": task.get("run_index"),
                    "status": "running",
                    "started_at": _now(),
                }
                atomic_write_json(task["result_path"], running)
                log_path = Path(task["worker_log_path"])
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log_handle = open(log_path, "a", encoding="utf-8")
                log_handle.write(f"\n[{_now()}] starting {task['task_id']}\n")
                log_handle.flush()
                process = subprocess.Popen(
                    [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--worker-task",
                        str(task_path),
                    ],
                    cwd=ROOT,
                    env=worker_env,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                active[task["task_id"]] = {
                    "process": process,
                    "task": task,
                    "started": time.monotonic(),
                    "timeout": _task_timeout(config, task),
                    "log_handle": log_handle,
                }

            changed = False
            for task_id, item in list(active.items()):
                process = item["process"]
                task = item["task"]
                elapsed = time.monotonic() - item["started"]
                return_code = process.poll()
                if return_code is None and elapsed <= item["timeout"]:
                    continue
                if return_code is None:
                    _terminate_process_group(process)
                    record = (
                        load_json(task["result_path"])
                        if Path(task["result_path"]).exists()
                        else None
                    )
                    if not record or record.get("status") == "running":
                        atomic_write_json(
                            task["result_path"],
                            _timeout_result(task, elapsed, item["timeout"]),
                        )
                else:
                    item["log_handle"].flush()
                    record = (
                        load_json(task["result_path"])
                        if Path(task["result_path"]).exists()
                        else None
                    )
                    if not record or record.get("status") == "running":
                        atomic_write_json(
                            task["result_path"],
                            _error_result(
                                task,
                                "worker exited without a terminal result",
                                return_code,
                            ),
                        )
                item["log_handle"].close()
                del active[task_id]
                completed_count += 1
                changed = True
            if changed or not active:
                _write_progress(
                    output_dir, total, completed_count, active, len(pending)
                )
            if active:
                time.sleep(0.5)
    except BaseException:
        for item in active.values():
            _terminate_process_group(item["process"])
            item["log_handle"].close()
        _write_progress(output_dir, total, completed_count, {}, len(pending))
        raise


def _load_method_result(instance_dir, instance, method, run_index=None):
    path = _task_result_path(instance_dir, method, run_index)
    if not path.exists():
        return {"status": "missing"}
    record = load_json(path)
    if record.get("config_fingerprint") != instance["config_fingerprint"]:
        raise RuntimeError(f"result config provenance mismatch: {path}")
    if record.get("instance_signature") != instance["instance_signature"]:
        raise RuntimeError(f"result instance provenance mismatch: {path}")
    task_path = _task_descriptor_path(instance_dir, method, run_index)
    if not task_path.exists():
        raise RuntimeError(f"result task descriptor missing: {task_path}")
    task = load_json(task_path)
    if not _result_matches_task(record, task):
        raise RuntimeError(f"result task provenance mismatch: {path}")
    return record


def aggregate_instance(config, instance_path):
    instance = load_json(instance_path)
    if instance.get("config_fingerprint") != config_fingerprint(config):
        raise RuntimeError(f"aggregate instance config mismatch: {instance_path}")
    if instance.get("instance_signature") != _instance_signature(instance):
        raise RuntimeError(f"aggregate instance signature mismatch: {instance_path}")
    if instance.get("values_signature") != _values_signature(instance["values_table"]):
        raise RuntimeError(f"aggregate instance values mismatch: {instance_path}")
    instance_dir = instance_path.parent
    methods = {
        "maxsat": _load_method_result(instance_dir, instance, "maxsat"),
        "khouzani": _load_method_result(instance_dir, instance, "khouzani"),
        "zenitani": _load_method_result(instance_dir, instance, "zenitani"),
    }
    ga_runs = []
    for run_index in range(config["algorithms"]["ga"]["runs"]):
        path = _task_result_path(instance_dir, "ga", run_index)
        if path.exists():
            record = _load_method_result(instance_dir, instance, "ga", run_index)
            record["result_file"] = str(path)
            ga_runs.append(record)
    methods["ga"] = select_median_ga_run(
        ga_runs,
        expected_runs=config["algorithms"]["ga"]["runs"],
    )
    key = instance["key"]
    row = {
        **key,
        "instance_id": instance["instance_id"],
        "instance_signature": instance["instance_signature"],
        "methods": methods,
    }
    pairwise = {}
    for method in ("ga", "khouzani", "zenitani"):
        pairwise[method] = {
            "outcome": compare_vs_maxsat(methods[method], methods["maxsat"]),
            "gap": pairwise_gap(methods[method], methods["maxsat"]),
        }
    aggregate = {**row, "pairwise_vs_maxsat": pairwise, "updated_at": _now()}
    atomic_write_json(instance_dir / "aggregate.json", aggregate)
    return row


def build_summary(config, output_dir):
    rows = []
    for instance_path in sorted((output_dir / "instances").glob("*/instance.json")):
        instance = load_json(instance_path)
        if instance.get("config_fingerprint") != config_fingerprint(config):
            raise RuntimeError(f"summary encountered foreign instance: {instance_path}")
        rows.append(aggregate_instance(config, instance_path))
    status_counts = {method: Counter() for method in METHODS}
    for row in rows:
        for method in METHODS:
            status_counts[method][row["methods"][method].get("status", "missing")] += 1
    summary = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _now(),
        "config_fingerprint": config_fingerprint(config),
        "num_instance_records": len(rows),
        "status_counts": {
            method: dict(sorted(counts.items()))
            for method, counts in status_counts.items()
        },
        "results": summarize_instance_rows(rows),
    }
    atomic_write_json(output_dir / "summary.json", summary)
    return summary


def _build_config(args):
    maxhs_bin = _resolve_path(args.maxhs_bin)
    final_bp = {
        "name": "Loopy BP",
        "bp_max_iters": 100,
        "bp_damping": 0.2,
        "bp_tol": 1e-6,
        "bp_fast": True,
    }
    config = {
        "schema_version": SCHEMA_VERSION,
        "experiment": "medium_parameter_sensitivity_four_method_comparison",
        "source_fingerprints": _source_fingerprints(),
        "dataset_provenance": (
            "new immutable nominal snapshots generated by the current generators; "
            "historical comparison result tables are not reused"
        ),
        "seed": args.seed,
        "families": list(args.families),
        "num_graphs": args.num_graphs,
        "parameters": list(args.parameters),
        "deltas_pct": list(args.deltas_pct),
        "perturbation_runs": args.perturbation_runs,
        "include_nominal": False,
        "perturbation": {
            "scope": "one parameter class at a time",
            "node_model": "independent z_i ~ uniform[-1,1]",
            "formula": "x_i(delta)=x_i*(1+delta*z_i)",
            "reuse_direction_across_deltas": True,
            "E_prob_perturbed": False,
        },
        "datasets": {
            "random": {
                "generator": "generate_bn_dag_multi_pe",
                "nP": 100,
                "nE": 240,
                "nC": 300,
                "nD": 150,
                "max_children": 5,
                "p_ep": 0.25,
                "legacy_rng_compat": False,
            },
            "structured": {
                "generator": "generate_bn_from_root_and_reverse",
                "nP_min": 151,
                "nP_max": 199,
                "extra_p_edge_prob": 0.25,
                "max_extra_p_out_per_node": 2,
                "c_parents_per_e_range": [1, 3],
                "max_e_children_per_c": 5,
                "max_c_children_per_d": 3,
                "use_level_scaling": True,
                "p_alpha_max": 0.8,
                "cd_beta_max": 0.5,
                "cd_floor": 0.2,
                "legacy_rng_compat": False,
            },
        },
        "value_ranges": {
            "P_loss": [50, 500],
            "C_benefit": [10, 50],
            "D_cost": [50, 100],
            "E_prob": {
                "source": "generate_graph.config.FIXED_E_PROBS",
                "values": list(FIXED_E_PROBS),
            },
        },
        "maxhs_bin": str(maxhs_bin),
        "maxhs_binary_sha256": _file_sha256(maxhs_bin),
        "algorithms": {
            "maxsat": {
                "cpu_limit_s": args.maxhs_cpu_limit,
                "subprocess_timeout_s": args.maxsat_wall_timeout - 5,
            },
            "ga": {
                "population_size": 100,
                "generations": 50,
                "runs": 5,
                "top_n": 10,
                "crossover": "single_point",
                "crossover_probability": 0.8,
                "mutation": "bitflip",
                "individual_mutation_probability": 1.0,
                "per_bit_mutation_probability": 0.01,
                "search_bp": dict(final_bp),
                "aggregation": "third objective after sorting five final BP scores",
            },
            "khouzani": {
                "method": "bigm",
                "n_budget": 10,
                "milp_time_limit_s": 60,
                "threads": 1,
                "search_bp": dict(KHOUZANI_BP),
            },
            "zenitani": {
                "iterations": 5,
                "neighbor_sample": 5,
                "max_controls": 40,
                "search_bp": dict(ZENITANI_BP),
            },
        },
        "final_evaluator": dict(final_bp),
        "comparison": {
            "reference": "maxsat",
            "tie_rule": "exact equality of final objective values",
            "tie_abs": 0.0,
            "tie_rel": 0.0,
            "normalized_gap_denominator_floor": 1.0,
            "primary_unit": "base graph",
        },
        "execution": {
            "workers": args.workers,
            "single_thread_environment": dict(THREAD_LIMIT_ENV),
            "wall_timeouts_s": {
                "maxsat": args.maxsat_wall_timeout,
                "ga": args.ga_wall_timeout,
                "khouzani": args.khouzani_wall_timeout,
                "zenitani": args.zenitani_wall_timeout,
            },
            "time_boundary": "frozen in-memory graph/values to final common BP score",
            "atomic_result_per_method_run": True,
        },
    }
    return config


def _validate_args(parser, args):
    if args.num_graphs < 1 or args.perturbation_runs < 1 or args.workers < 1:
        parser.error(
            "--num-graphs, --perturbation-runs, and --workers must be positive"
        )
    if args.perturbation_runs != 1:
        parser.error("the frozen protocol requires exactly 1 perturbation repetition")
    if sorted(args.deltas_pct) != [10, 20, 30]:
        parser.error("the frozen protocol requires --deltas-pct 10 20 30")
    if len(args.parameters) != len(set(args.parameters)):
        parser.error("--parameters must not contain duplicates")
    if set(args.parameters) != set(PARAM_FIELDS):
        parser.error("the frozen protocol requires C_benefit P_loss D_cost")
    if len(args.families) != len(set(args.families)):
        parser.error("--families must not contain duplicates")
    if set(args.families) != set(FAMILIES):
        parser.error("the frozen protocol requires structured and random families")
    if len(args.methods) != len(set(args.methods)):
        parser.error("--methods must not contain duplicates")
    if args.num_graphs != 10:
        parser.error("the frozen protocol requires exactly 10 graphs per family")
    if args.seed != 42 or args.workers != 8:
        parser.error("the frozen protocol requires --seed 42 and --workers 8")
    frozen_timeouts = (600, 630, 7200, 900, 7200)
    actual_timeouts = (
        args.maxhs_cpu_limit,
        args.maxsat_wall_timeout,
        args.ga_wall_timeout,
        args.khouzani_wall_timeout,
        args.zenitani_wall_timeout,
    )
    if actual_timeouts != frozen_timeouts:
        parser.error(
            "the frozen protocol requires MaxSAT CPU/wall=600/630, "
            "GA=7200, Khouzani=900, and Zenitani=7200 seconds"
        )


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--generate-only", action="store_true")
    action.add_argument("--execute", action="store_true")
    action.add_argument("--summarize-only", action="store_true")
    action.add_argument("--preflight-only", action="store_true")
    action.add_argument("--worker-task", help=argparse.SUPPRESS)

    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument(
        "--families", nargs="+", choices=FAMILIES, default=list(FAMILIES)
    )
    parser.add_argument("--num-graphs", type=int, default=10)
    parser.add_argument(
        "--parameters",
        nargs="+",
        choices=list(PARAM_FIELDS),
        default=list(PARAM_FIELDS),
    )
    parser.add_argument("--deltas-pct", type=int, nargs="+", default=[10, 20, 30])
    parser.add_argument("--perturbation-runs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--maxhs-bin", default=MAXHS_BIN)
    parser.add_argument("--maxhs-cpu-limit", type=int, default=600)
    parser.add_argument("--maxsat-wall-timeout", type=int, default=630)
    parser.add_argument("--ga-wall-timeout", type=int, default=7200)
    parser.add_argument("--khouzani-wall-timeout", type=int, default=900)
    parser.add_argument("--zenitani-wall-timeout", type=int, default=7200)
    args = parser.parse_args(argv)
    if args.worker_task:
        return args
    _validate_args(parser, args)
    return args


def _load_or_create_config(args, output_dir):
    config_path = output_dir / "config.json"
    current = _build_config(args)
    if config_path.exists():
        if not args.resume and not args.summarize_only and not args.preflight_only:
            raise RuntimeError(f"output already exists; use --resume: {config_path}")
        saved = load_json(config_path)
        if saved != current:
            raise RuntimeError(
                "resume configuration does not match the frozen experiment config"
            )
        return config_path, saved
    if args.summarize_only:
        raise RuntimeError(f"missing config: {config_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(config_path, current)
    return config_path, current


def _process_conflicts():
    completed = subprocess.run(
        ["ps", "-eo", "pid=,ppid=,stat=,comm=,args="],
        capture_output=True,
        text=True,
        check=True,
    )
    own_pid = os.getpid()
    conflicts = []
    for raw in completed.stdout.splitlines():
        parts = raw.strip().split(None, 4)
        if len(parts) < 5:
            continue
        pid, _ppid, stat, command, arguments = parts
        if int(pid) == own_pid:
            continue
        is_worker = Path(__file__).name in arguments and "--worker-task" in arguments
        is_solver = command.lower() in {"maxhs", "cbc"}
        is_zombie = stat.startswith("Z")
        if is_worker or is_solver or is_zombie:
            conflicts.append(
                {
                    "pid": int(pid),
                    "status": stat,
                    "command": command,
                    "arguments": arguments,
                }
            )
    return conflicts


def _prepare_selected_instances(config, output_dir, pilot):
    selected = list(
        iter_instance_keys(
            families=config["families"],
            num_graphs=config["num_graphs"],
            parameters=config["parameters"],
            deltas_pct=config["deltas_pct"],
            perturbation_runs=config["perturbation_runs"],
            include_nominal=config["include_nominal"],
            pilot=pilot,
        )
    )
    # Freeze the complete 20-graph nominal dataset before either a pilot or the
    # full matrix.  A later resume therefore cannot mix generator versions.
    base_cache = {}
    for family in config["families"]:
        for graph_id in range(1, config["num_graphs"] + 1):
            base_cache[(family, graph_id)] = prepare_base_snapshot(
                config,
                output_dir,
                family,
                graph_id,
            )
    atomic_write_json(
        output_dir / "dataset" / "manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "config_fingerprint": config_fingerprint(config),
            "dataset_provenance": config["dataset_provenance"],
            "base_graphs": [
                {
                    "family": family,
                    "graph_id": graph_id,
                    "path": os.path.relpath(path, output_dir),
                    "graph_signature": snapshot["graph_signature"],
                    "base_values_signature": snapshot["base_values_signature"],
                    "graph_stats": snapshot["graph_stats"],
                }
                for (family, graph_id), (path, snapshot) in sorted(base_cache.items())
            ],
        },
    )
    prepared = []
    for key in selected:
        base_key = (key.family, key.graph_id)
        base_path, base_snapshot = base_cache[base_key]
        prepared.append(
            prepare_instance(config, output_dir, key, base_path, base_snapshot)
        )
    return prepared


def _make_selected_tasks(config_path, config, output_dir, prepared, methods):
    tasks = []
    for instance_path, instance in prepared:
        for method in methods:
            if method == "ga":
                for run_index in range(config["algorithms"]["ga"]["runs"]):
                    tasks.append(
                        make_task(
                            config_path,
                            output_dir,
                            instance_path,
                            instance,
                            method,
                            run_index,
                        )
                    )
            else:
                tasks.append(
                    make_task(config_path, output_dir, instance_path, instance, method)
                )
    return tasks


def _run_locked_controller(args, output_dir):
    config_path, config = _load_or_create_config(args, output_dir)
    if args.summarize_only:
        build_summary(config, output_dir)
        print(f"saved: {output_dir / 'summary.json'}")
        return 0

    prepared = _prepare_selected_instances(config, output_dir, args.pilot)
    tasks = _make_selected_tasks(
        config_path,
        config,
        output_dir,
        prepared,
        [] if args.generate_only else args.methods,
    )
    atomic_write_json(
        output_dir / "selection.json",
        {
            "updated_at": _now(),
            "pilot": args.pilot,
            "instances": len(prepared),
            "tasks": len(tasks),
            "methods": [] if args.generate_only else args.methods,
        },
    )

    if args.generate_only:
        build_summary(config, output_dir)
        print(f"prepared {len(prepared)} instances; no algorithms executed")
        return 0

    if not Path(config["maxhs_bin"]).is_file() or not os.access(
        config["maxhs_bin"], os.X_OK
    ):
        raise RuntimeError(
            f"MaxHS binary missing or not executable: {config['maxhs_bin']}"
        )
    if not args.skip_preflight:
        conflicts = _process_conflicts()
        if conflicts:
            raise RuntimeError(
                "residual/zombie processes found; inspect with --preflight-only: "
                + ", ".join(str(item["pid"]) for item in conflicts)
            )
    try:
        execute_tasks(
            config,
            output_dir,
            tasks,
            workers=config["execution"]["workers"],
            retry_failed=args.retry_failed,
        )
    finally:
        build_summary(config, output_dir)
    print(f"saved: {output_dir / 'summary.json'}")
    return 0


def main(argv=None):
    args = _parse_args(argv)
    if args.worker_task:
        return run_worker(_resolve_path(args.worker_task))

    output_dir = _resolve_path(args.output_dir)
    if args.preflight_only:
        conflicts = _process_conflicts()
        print(json.dumps({"conflicts": conflicts}, ensure_ascii=False, indent=2))
        return 1 if conflicts else 0
    if args.summarize_only and not output_dir.is_dir():
        raise RuntimeError(f"missing output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    with _exclusive_controller_lock(output_dir):
        return _run_locked_controller(args, output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
