#!/usr/bin/env python3
"""Compare MaxSAT with three baselines across medium normal-P-degree graphs.

The experiment varies the P->E out-degree distribution through k_max while
keeping the P/C/D counts, non-P out-degree cap, numerical ranges, evaluator,
and graph-specific seeds fixed. Results are saved atomically after every
completed method (and after every GA repetition) so long runs can resume.
"""

import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

import networkx as nx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from comparison.bp_core import (  # noqa: E402
    MAXHS_BIN,
    compute_objective_bp_style,
    extract_D_state_from_solution,
    find_best_defense_bp,
    find_best_defense_maxsat,
)
from comparison.khouzani_baseline import find_best_defense_khouzani  # noqa: E402
from comparison.zenitani_baseline import find_best_defense_zenitani  # noqa: E402
from generate_graph.config import FIXED_E_PROBS  # noqa: E402
from generate_graph.normal_outdegree_random_graph import (  # noqa: E402
    generate_bn_normal_p_outdegree,
)
from generate_graph.numerical_generation import generate_node_values  # noqa: E402


CASE_SPECS = {
    5: {"p_mean": 3.0, "p_std": 0.8},
    7: {"p_mean": 4.0, "p_std": 1.2},
    10: {"p_mean": 5.5, "p_std": 1.8},
}
METHODS = ("maxsat", "ga", "khouzani", "zenitani")
GA_SEED_STRIDE = 100_000
KHOUZANI_BP_CONFIG = {
    "bp_max_iters": 50,
    "bp_damping": 0.5,
    "bp_tol": 1e-6,
    "bp_fast": False,
}
ZENITANI_BP_CONFIG = {
    "bp_max_iters": 50,
    "bp_damping": 0.5,
    "bp_tol": 1e-6,
    "bp_fast": True,
}


def _defended(state):
    return sorted(node for node, enabled in state.items() if enabled)


def _mean(values):
    return sum(values) / len(values) if values else None


def _atomic_save(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload["summary"] = summarize(payload)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    temporary.replace(path)


def _graph_signature(bn, values_table):
    canonical = {
        "nodes": {kind: bn.get(kind, []) for kind in ("P", "E", "C", "D")},
        "edges": sorted([list(edge) for edge in bn["edges"]]),
        "values": sorted(values_table, key=lambda row: row["node"]),
    }
    raw = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _validate_graph(bn, spec, non_p_max_children):
    p_degrees = [bn["out_degree"].get(node, 0) for node in bn["P"]]
    if min(p_degrees) < 1 or max(p_degrees) > spec["k_max"]:
        raise RuntimeError("P out-degree falls outside the requested range")
    if sum(p_degrees) != round(len(bn["P"]) * spec["p_mean"]):
        raise RuntimeError("P out-degree sum does not match the requested mean")

    for kind in ("E", "C", "D"):
        if any(bn["out_degree"].get(node, 0) > non_p_max_children for node in bn[kind]):
            raise RuntimeError(f"{kind} out-degree exceeds non-P limit")

    graph = nx.DiGraph()
    graph.add_nodes_from(sum((bn[kind] for kind in ("P", "E", "C", "D")), []))
    graph.add_edges_from(bn["edges"])
    if not nx.is_directed_acyclic_graph(graph):
        raise RuntimeError("generated graph is not a DAG")


def generate_case_graph(config, spec, graph_id):
    graph_seed = config["seed"] + graph_id
    bn = generate_bn_normal_p_outdegree(
        nP=config["nP"],
        nC=config["nC"],
        nD=config["nD"],
        max_children=spec["k_max"],
        non_p_max_children=config["non_p_max_children"],
        p_mean=spec["p_mean"],
        p_std=spec["p_std"],
        p_EP=config["p_ep"],
        seed=graph_seed,
    )
    values_table = generate_node_values(
        bn,
        seed=graph_seed,
        fixed_e_probs=FIXED_E_PROBS,
        p_loss_range=(50, 500),
        c_benefit_range=(10, 50),
        d_cost_range=(50, 100),
    )
    _validate_graph(bn, spec, config["non_p_max_children"])
    return bn, values_table, graph_seed


def _graph_record(bn, values_table, graph_id, graph_seed):
    degree_meta = bn["meta"]["p_outdegree"]
    return {
        "graph_id": graph_id,
        "graph_seed": graph_seed,
        "graph_signature": _graph_signature(bn, values_table),
        "graph_stats": {
            "nP": len(bn["P"]),
            "nE": len(bn["E"]),
            "nC": len(bn["C"]),
            "nD": len(bn["D"]),
            "total": sum(len(bn[kind]) for kind in ("P", "E", "C", "D")),
            "edges": len(bn["edges"]),
        },
        "p_outdegree": degree_meta,
        "methods": {},
    }


def _evaluate(config, bn, values_table, state):
    return compute_objective_bp_style(
        bn,
        values_table,
        state,
        bp_max_iters=config["bp_max_iters"],
        bp_damping=config["bp_damping"],
        bp_tol=config["bp_tol"],
        fast=config["bp_fast"],
    )


def _compact_eval(evaluation):
    return {
        key: evaluation.get(key)
        for key in ("objective", "C_benefit", "P_expected_loss", "D_cost")
    }


def _run_maxsat(config, bn, values_table, wcnf_path):
    started = time.perf_counter()
    node_state, node_state_by_type, _, _ = find_best_defense_maxsat(
        bn,
        values_table,
        wcnf_path,
        MAXHS_BIN,
        timeout=config["maxhs_timeout"],
    )
    wall_time = time.perf_counter() - started
    if node_state is None:
        return {"status": "timeout", "wall_time_s": wall_time, "objective": None}
    state = extract_D_state_from_solution(bn, node_state_by_type)
    eval_started = time.perf_counter()
    evaluation = _evaluate(config, bn, values_table, state)
    return {
        "status": "complete",
        "wall_time_s": wall_time,
        "evaluation_time_s": time.perf_counter() - eval_started,
        "defended": _defended(state),
        **_compact_eval(evaluation),
    }


def _run_ga_once(config, bn, values_table, graph_id, run_index):
    run_seed = config["seed"] + graph_id + run_index * GA_SEED_STRIDE
    started = time.perf_counter()
    result = find_best_defense_bp(
        bn,
        values_table,
        population_size=config["population_size"],
        genmax=config["genmax"],
        seed=run_seed,
        bp_max_iters=config["bp_max_iters"],
        bp_damping=config["bp_damping"],
        bp_tol=config["bp_tol"],
        top_n=config["top_n"],
        bp_fast=config["bp_fast"],
        crossover_kind="single_point",
        crossover_prob=0.8,
        mutation_prob=1.0,
        mutation_prob_var=0.01,
    )
    wall_time = time.perf_counter() - started
    evaluation = _evaluate(config, bn, values_table, result["best_D_state"])
    return {
        "run_index": run_index,
        "seed": run_seed,
        "wall_time_s": wall_time,
        "defended": _defended(result["best_D_state"]),
        "unique_evaluated": result["unique_evaluated"],
        **_compact_eval(evaluation),
    }


def _finalize_ga(method, expected_runs):
    runs = method.get("runs", [])
    if len(runs) != expected_runs:
        method["status"] = "partial"
        return
    ordered = sorted(runs, key=lambda run: (run["objective"], run["run_index"]))
    median_run = ordered[len(ordered) // 2]
    method.update({
        "status": "complete",
        "aggregation": "median of independent runs",
        "objective": median_run["objective"],
        "wall_time_s": statistics.median(run["wall_time_s"] for run in runs),
        "selected_run_index": median_run["run_index"],
        "defended": median_run["defended"],
        "C_benefit": median_run["C_benefit"],
        "P_expected_loss": median_run["P_expected_loss"],
        "D_cost": median_run["D_cost"],
    })


def _run_khouzani(config, bn, values_table):
    started = time.perf_counter()
    bp_config = config["khouzani_bp"]
    result = find_best_defense_khouzani(
        bn,
        values_table,
        n_budget=config["khouzani_n_budget"],
        eval_mode="bp",
        method=config["khouzani_method"],
        milp_time_limit=config["khouzani_milp_time_limit"],
        threads=1,
        bp_max_iters=bp_config["bp_max_iters"],
        bp_damping=bp_config["bp_damping"],
        bp_tol=bp_config["bp_tol"],
        bp_fast=bp_config["bp_fast"],
    )
    return {
        "status": "complete",
        "wall_time_s": time.perf_counter() - started,
        "milp_time_s": result["milp_time_s"],
        "n_milp_solves": result["n_milp_solves"],
        "n_milp_optimal": result["n_milp_optimal"],
        "defended": _defended(result["best_D_state"]),
        **_compact_eval(result["best_eval"]),
    }


def _run_zenitani(config, bn, values_table, graph_seed):
    started = time.perf_counter()
    bp_config = config["zenitani_bp"]
    result = find_best_defense_zenitani(
        bn,
        values_table,
        eval_mode="bp",
        n_iter=config["zenitani_n_iter"],
        sample=config["zenitani_sample"],
        max_k=config["zenitani_max_controls"],
        seed=graph_seed,
        bp_fast=bp_config["bp_fast"],
        bp_max_iters=bp_config["bp_max_iters"],
        bp_damping=bp_config["bp_damping"],
        bp_tol=bp_config["bp_tol"],
    )
    return {
        "status": "complete",
        "wall_time_s": time.perf_counter() - started,
        "n_evals": result["n_evals"],
        "hit_max_k": result["hit_max_k"],
        "defended": _defended(result["best_D_state"]),
        **_compact_eval(result["best_eval"]),
    }


def _compare(baseline, maxsat, tol_abs=0.5, tol_rel=1e-3):
    if baseline is None or maxsat is None:
        return None
    tolerance = max(tol_abs, tol_rel * abs(maxsat))
    if baseline > maxsat + tolerance:
        return "win"
    if baseline < maxsat - tolerance:
        return "loss"
    return "tie"


def summarize(payload):
    summary = {"wtl_order": ["baseline_win", "tie", "baseline_loss"], "cases": []}
    for case in payload.get("cases", []):
        rows = case.get("graphs", [])
        item = {
            "k_max": case["k_max"],
            "num_graphs": len(rows),
            "completed": {},
            "pairwise_vs_maxsat": {},
            "mean_wall_time_s": {},
            "mean_objective": {},
        }
        for method_name in METHODS:
            completed = [
                row["methods"][method_name]
                for row in rows
                if row.get("methods", {}).get(method_name, {}).get("status") == "complete"
            ]
            item["completed"][method_name] = len(completed)
            item["mean_wall_time_s"][method_name] = _mean(
                [method["wall_time_s"] for method in completed]
            )
            item["mean_objective"][method_name] = _mean(
                [method["objective"] for method in completed]
            )

        for baseline in ("ga", "khouzani", "zenitani"):
            counts = {"win": 0, "tie": 0, "loss": 0, "na": 0}
            for row in rows:
                methods = row.get("methods", {})
                comparison = _compare(
                    methods.get(baseline, {}).get("objective"),
                    methods.get("maxsat", {}).get("objective"),
                )
                counts[comparison if comparison is not None else "na"] += 1
            item["pairwise_vs_maxsat"][baseline] = counts
        summary["cases"].append(item)
    return summary


def _config(args):
    return {
        "experiment": "medium_normal_p_outdegree_four_method_kmax_sweep",
        "k_values": args.k_values,
        "num_graphs": args.num_graphs,
        "seed": args.seed,
        "nP": args.nP,
        "nC": args.nC,
        "nD": args.nD,
        "non_p_max_children": args.non_p_max_children,
        "p_ep": args.p_ep,
        "maxhs_timeout": args.maxhs_timeout,
        "population_size": args.population_size,
        "genmax": args.genmax,
        "ga_runs": args.ga_runs,
        "top_n": args.top_n,
        "bp_max_iters": args.bp_max_iters,
        "bp_damping": args.bp_damping,
        "bp_tol": args.bp_tol,
        "bp_fast": args.bp_fast,
        "khouzani_n_budget": args.khouzani_n_budget,
        "khouzani_method": args.khouzani_method,
        "khouzani_milp_time_limit": args.khouzani_milp_time_limit,
        "khouzani_bp": dict(KHOUZANI_BP_CONFIG),
        "zenitani_n_iter": args.zenitani_n_iter,
        "zenitani_sample": args.zenitani_sample,
        "zenitani_max_controls": args.zenitani_max_controls,
        "zenitani_bp": dict(ZENITANI_BP_CONFIG),
        "value_ranges": {
            "P_loss": [50, 500],
            "C_benefit": [10, 50],
            "D_cost": [50, 100],
            "E_prob": "FIXED_E_PROBS",
        },
        "final_evaluator": "Loopy BP",
        "ga_aggregation": f"median of {args.ga_runs} independent runs",
        "time_boundary": "in-memory graph and values to selected defense output",
    }


def _baseline_config_migration_compatible(payload, config):
    """Preserve saved MaxSAT/GA work while correcting unused baseline config."""
    if not all(
        not ({"khouzani", "zenitani"} & set(row.get("methods", {})))
        for case in payload.get("cases", [])
        for row in case.get("graphs", [])
    ):
        return False
    stored_config = dict(payload.get("config", {}))
    expected_config = dict(config)
    baseline_keys = {
        "khouzani_method",
        "khouzani_milp_time_limit",
        "khouzani_bp",
        "zenitani_bp",
    }
    for key in baseline_keys:
        stored_config.pop(key, None)
        expected_config.pop(key, None)
    return stored_config == expected_config


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k-values", type=int, nargs="+", default=[5, 7, 10])
    parser.add_argument("--num-graphs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--nP", type=int, default=100)
    parser.add_argument("--nC", type=int, default=300)
    parser.add_argument("--nD", type=int, default=150)
    parser.add_argument("--non-p-max-children", type=int, default=5)
    parser.add_argument("--p-ep", type=float, default=0.25)
    parser.add_argument("--output-dir", default="comparison/kmax_medium_results")
    parser.add_argument("--result-json", default=None)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--generate-only", action="store_true")

    parser.add_argument("--maxhs-timeout", type=int, default=600)
    parser.add_argument("--population-size", type=int, default=100)
    parser.add_argument("--genmax", type=int, default=50)
    parser.add_argument("--ga-runs", type=int, default=5)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--bp-max-iters", type=int, default=100)
    parser.add_argument("--bp-damping", type=float, default=0.2)
    parser.add_argument("--bp-tol", type=float, default=1e-6)
    parser.add_argument("--no-bp-fast", dest="bp_fast", action="store_false")
    parser.set_defaults(bp_fast=True)

    parser.add_argument("--khouzani-n-budget", type=int, default=10)
    parser.add_argument("--khouzani-method", choices=["rowgen", "bigm"], default="bigm")
    parser.add_argument("--khouzani-milp-time-limit", type=int, default=60)
    parser.add_argument("--zenitani-n-iter", type=int, default=5)
    parser.add_argument("--zenitani-sample", type=int, default=5)
    parser.add_argument("--zenitani-max-controls", type=int, default=40)
    args = parser.parse_args()

    unknown = sorted(set(args.k_values) - set(CASE_SPECS))
    if unknown:
        parser.error(f"unsupported k values: {unknown}; supported: {sorted(CASE_SPECS)}")
    if args.num_graphs < 1 or args.ga_runs < 1:
        parser.error("--num-graphs and --ga-runs must be positive")
    return args


def main():
    args = _parse_args()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    result_path = Path(args.result_json) if args.result_json else output_dir / "kmax_medium_comparison.json"
    if not result_path.is_absolute():
        result_path = ROOT / result_path

    config = _config(args)
    if result_path.exists():
        if not args.resume:
            raise RuntimeError(f"result already exists; use --resume: {result_path}")
        with open(result_path, encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("config") != config:
            if not _baseline_config_migration_compatible(payload, config):
                raise RuntimeError("resume configuration does not match existing result")
            payload["config"] = config
    else:
        payload = {"config": config, "cases": []}

    cases_by_k = {case["k_max"]: case for case in payload["cases"]}
    requested_methods = [] if args.generate_only else args.methods
    for k_max in args.k_values:
        spec = {"k_max": k_max, **CASE_SPECS[k_max]}
        case = cases_by_k.get(k_max)
        if case is None:
            case = {**spec, "graphs": []}
            payload["cases"].append(case)
            cases_by_k[k_max] = case
        rows_by_id = {row["graph_id"]: row for row in case["graphs"]}

        for graph_id in range(1, args.num_graphs + 1):
            bn, values_table, graph_seed = generate_case_graph(config, spec, graph_id)
            current = _graph_record(bn, values_table, graph_id, graph_seed)
            row = rows_by_id.get(graph_id)
            if row is None:
                row = current
                case["graphs"].append(row)
                rows_by_id[graph_id] = row
                _atomic_save(result_path, payload)
            elif row["graph_signature"] != current["graph_signature"]:
                raise RuntimeError(f"k={k_max} graph={graph_id}: regeneration mismatch")

            methods = row["methods"]
            stem = f"k{k_max}_graph_{graph_id:03d}"
            print(f"[k={k_max} graph={graph_id}/{args.num_graphs}]", flush=True)

            if "maxsat" in requested_methods and "maxsat" not in methods:
                wcnf_path = output_dir / "wcnf" / f"{stem}.wcnf"
                wcnf_path.parent.mkdir(parents=True, exist_ok=True)
                methods["maxsat"] = _run_maxsat(config, bn, values_table, wcnf_path)
                _atomic_save(result_path, payload)

            if "ga" in requested_methods:
                ga = methods.setdefault("ga", {"status": "partial", "runs": []})
                completed = {run["run_index"] for run in ga["runs"]}
                for run_index in range(config["ga_runs"]):
                    if run_index in completed:
                        continue
                    ga["runs"].append(
                        _run_ga_once(config, bn, values_table, graph_id, run_index)
                    )
                    ga["runs"].sort(key=lambda run: run["run_index"])
                    _finalize_ga(ga, config["ga_runs"])
                    _atomic_save(result_path, payload)

            if "khouzani" in requested_methods and "khouzani" not in methods:
                methods["khouzani"] = _run_khouzani(config, bn, values_table)
                _atomic_save(result_path, payload)

            if "zenitani" in requested_methods and "zenitani" not in methods:
                methods["zenitani"] = _run_zenitani(
                    config, bn, values_table, graph_seed
                )
                _atomic_save(result_path, payload)

    payload["cases"].sort(key=lambda case: case["k_max"])
    for case in payload["cases"]:
        case["graphs"].sort(key=lambda row: row["graph_id"])
    _atomic_save(result_path, payload)
    print(f"saved: {result_path}")


if __name__ == "__main__":
    main()
