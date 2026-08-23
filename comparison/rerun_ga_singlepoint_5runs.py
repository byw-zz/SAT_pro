#!/usr/bin/env python3
"""Rerun BP+GA five times per stored graph with Poolsappasit-style operators.

Each run uses single-point crossover (0.8) and per-bit mutation (0.01). Results
are atomically saved after every run, so an interrupted job can resume without
discarding completed repetitions. The best objective among the five runs is the
per-graph quality result; the composite method time is the sum of all five runs.
"""

import argparse
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent.parent))

from comparison.batch_comparison import (
    generate_random_graph,
    generate_structured_graph_with_np_range,
)
from comparison.bp_core import compute_objective_bp_style, find_best_defense_bp
from result_analysis.exact_analysis import run_exact_analysis


SEED_STRIDE = 100_000


def _defended_state(nodes, defended):
    selected = set(defended or [])
    return {d: d in selected for d in nodes}


def _defended_list(state):
    return sorted(d for d, enabled in state.items() if enabled)


def _evaluate(bn, values_table, d_state, use_ve, args, bp_fast):
    if use_ve:
        return run_exact_analysis(bn, values_table, D_state=d_state)
    return compute_objective_bp_style(
        bn,
        values_table,
        d_state,
        bp_max_iters=args.bp_max_iters,
        bp_damping=args.bp_damping,
        bp_tol=args.bp_tol,
        fast=bp_fast,
    )


def _close_enough(actual, expected):
    tolerance = max(1e-6, 1e-8 * max(1.0, abs(expected)))
    return abs(actual - expected) <= tolerance, tolerance


def _top_results(items, top_n):
    return [
        {
            "rank": rank,
            "objective": objective,
            "defended": _defended_list(d_state),
            "C_benefit": detail.get("C_benefit", 0.0),
            "P_expected_loss": detail.get("P_expected_loss", 0.0),
            "D_cost": detail.get("D_cost", 0.0),
        }
        for rank, (objective, d_state, detail) in enumerate(items[:top_n], start=1)
    ]


def _run_seed(base_seed, graph_id, run_index):
    return base_seed + graph_id + run_index * SEED_STRIDE


def _config(source_args, bp_fast, n_runs):
    return {
        "population_size": source_args["population_size"],
        "genmax": source_args["genmax"],
        "crossover": "single_point",
        "crossover_prob": 0.8,
        "mutation": "bitflip",
        "mutation_individual_prob": 1.0,
        "mutation_per_bit_prob": 0.01,
        "n_runs": n_runs,
        "selection": "best_objective",
        "seed_formula": "base_seed + graph_id + run_index * 100000",
        "bp_fast": bp_fast,
        "bp_max_iters": source_args["bp_max_iters"],
        "bp_damping": source_args["bp_damping"],
        "bp_tol": source_args["bp_tol"],
    }


def _complete_rows(rows, n_runs):
    return [row for row in rows if len(row.get("runs", [])) == n_runs]


def _summary(rows, n_runs):
    complete = _complete_rows(rows, n_runs)
    all_runs = [run for row in complete for run in row["runs"]]
    total_times = [row["total_5run_time_s"] for row in complete]
    return {
        "num_graphs_complete": len(complete),
        "num_graphs_started": len(rows),
        "n_runs_per_graph": n_runs,
        "selection": "best_objective",
        "num_same_strategy_as_stored": sum(
            row["strategy_same_as_stored"] for row in complete
        ),
        "num_different_strategy_from_stored": sum(
            not row["strategy_same_as_stored"] for row in complete
        ),
        "avg_single_run_time_s": (
            sum(run["ga_time_s"] for run in all_runs) / len(all_runs)
            if all_runs else None
        ),
        "avg_total_5run_time_s": (
            sum(total_times) / len(total_times) if total_times else None
        ),
        # Because quality is best-of-five, the fair composite-method runtime is
        # the five-run total.
        "avg_new_ga_time_s": (
            sum(total_times) / len(total_times) if total_times else None
        ),
        "avg_objective_delta_new_minus_old": (
            sum(row["objective_delta_new_minus_old"] for row in complete)
            / len(complete)
            if complete else None
        ),
    }


def _finalize_row(row, stored, n_runs):
    if len(row["runs"]) != n_runs:
        return
    best = max(row["runs"], key=lambda run: (run["objective"], -run["run_index"]))
    total_time = sum(run["ga_time_s"] for run in row["runs"])
    row.update({
        "complete": True,
        "best_run_index": best["run_index"],
        "best_run_seed": best["seed"],
        "best_run_time_s": best["ga_time_s"],
        "mean_single_run_time_s": total_time / n_runs,
        "total_5run_time_s": total_time,
        "new_ga_time_s": total_time,
        "new_ga_objective": best["objective"],
        "objective_delta_new_minus_old": best["objective"] - stored["bp_objective"],
        "new_ga_defended": best["defended"],
        "new_ga_evaluated": best["unique_evaluated"],
        "ga_top_n": best["ga_top_n"],
        "strategy_same_as_stored": best["defended"] == stored["bp_defended"],
    })


def _save(path, source, source_args, config, rows, n_runs):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "source_dataset": str(source),
        "source_args": source_args,
        "ga_config": config,
        "legacy_rng_compat": True,
        "summary": _summary(rows, n_runs),
        "results": rows,
    }
    temp = path.with_suffix(path.suffix + ".tmp")
    with open(temp, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(
        description="Rerun BP+GA with single-point crossover, five runs per graph"
    )
    parser.add_argument("dataset_json")
    parser.add_argument("--out", required=True)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--bp-fast", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args_cli = parser.parse_args()
    if args_cli.runs < 1:
        parser.error("--runs must be at least 1")

    source = Path(args_cli.dataset_json)
    output = Path(args_cli.out)
    with open(source, encoding="utf-8") as handle:
        stored_data = json.load(handle)

    source_args = stored_data["args"]
    args = SimpleNamespace(**source_args)
    args.legacy_rng_compat = True
    stored_rows = stored_data["results"]
    config = _config(source_args, args_cli.bp_fast, args_cli.runs)

    rows = []
    if args_cli.resume and output.exists():
        with open(output, encoding="utf-8") as handle:
            previous = json.load(handle)
        if previous.get("source_dataset") != str(source):
            raise RuntimeError("resume output belongs to a different source dataset")
        if previous.get("ga_config") != config:
            raise RuntimeError("resume output uses a different GA configuration")
        rows = previous.get("results", [])

    rows_by_id = {row["graph_id"]: row for row in rows}
    rng = random.Random(args.seed)

    for stored in stored_rows:
        graph_id = stored["graph_id"]
        if args.graph_type == "structured":
            bn, values_table = generate_structured_graph_with_np_range(
                args, rng, graph_id
            )
        else:
            bn, values_table = generate_random_graph(args, graph_id)

        complete_count = len(_complete_rows(rows, args_cli.runs))
        if args_cli.limit is not None and complete_count >= args_cli.limit:
            break

        use_ve = stored.get(
            "use_ve", len(bn["P"]) + len(bn["C"]) + len(bn["E"]) < 100
        )
        old_state = _defended_state(bn["D"], stored["bp_defended"])
        check = _evaluate(
            bn, values_table, old_state, use_ve, args, args_cli.bp_fast
        )
        ok, tolerance = _close_enough(check["objective"], stored["bp_objective"])
        if not ok:
            raise RuntimeError(
                f"graph {graph_id}: regenerated data mismatch: "
                f"stored objective={stored['bp_objective']}, "
                f"re-evaluated={check['objective']}, tolerance={tolerance}"
            )

        row = rows_by_id.get(graph_id)
        if row is None:
            row = {
                "graph_id": graph_id,
                "graph_stats": stored.get("graph_stats"),
                "use_ve": use_ve,
                "data_validation_objective": check["objective"],
                "old_ga_time_s": stored["bp_time_s"],
                "old_ga_objective": stored["bp_objective"],
                "old_ga_defended": stored["bp_defended"],
                "complete": False,
                "runs": [],
            }
            rows.append(row)
            rows_by_id[graph_id] = row

        completed_runs = {run["run_index"] for run in row["runs"]}
        for run_index in range(args_cli.runs):
            if run_index in completed_runs:
                continue
            seed = _run_seed(args.seed, graph_id, run_index)
            ga = find_best_defense_bp(
                bn,
                values_table,
                population_size=args.population_size,
                genmax=args.genmax,
                seed=seed,
                bp_max_iters=args.bp_max_iters,
                bp_damping=args.bp_damping,
                bp_tol=args.bp_tol,
                top_n=args.top_n,
                return_all=True,
                bp_fast=args_cli.bp_fast,
                crossover_kind="single_point",
                crossover_prob=0.8,
                mutation_prob=1.0,
                mutation_prob_var=0.01,
            )
            final_eval = _evaluate(
                bn, values_table, ga["best_D_state"], use_ve, args, args_cli.bp_fast
            )
            run = {
                "run_index": run_index,
                "seed": seed,
                "ga_time_s": ga["total_time_s"],
                "objective": final_eval["objective"],
                "defended": _defended_list(ga["best_D_state"]),
                "unique_evaluated": ga["unique_evaluated"],
                "C_benefit": final_eval.get("C_benefit", 0.0),
                "P_expected_loss": final_eval.get("P_expected_loss", 0.0),
                "D_cost": final_eval.get("D_cost", 0.0),
                "ga_top_n": _top_results(ga["all_results"], args.top_n),
            }
            row["runs"].append(run)
            row["runs"].sort(key=lambda item: item["run_index"])
            _finalize_row(row, stored, args_cli.runs)
            _save(output, source, source_args, config, rows, args_cli.runs)
            print(
                f"[{graph_id:>3}/{len(stored_rows)} run {run_index + 1}/{args_cli.runs}] "
                f"seed={seed} time={ga['total_time_s']:.3f}s "
                f"objective={final_eval['objective']:.6f}",
                flush=True,
            )

        _finalize_row(row, stored, args_cli.runs)
        _save(output, source, source_args, config, rows, args_cli.runs)

    _save(output, source, source_args, config, rows, args_cli.runs)
    print(json.dumps(_summary(rows, args_cli.runs), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
