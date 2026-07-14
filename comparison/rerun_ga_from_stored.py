#!/usr/bin/env python3
"""Replay historical datasets and rerun only the BP+GA method.

The source JSON keeps the original graph-generation and GA parameters.  Graphs
and node values are regenerated deterministically, with the legacy random draw
that used to create P_benefit consumed but discarded.  Before GA starts on each
graph, the stored GA strategy is re-evaluated; a mismatch aborts the run so old
and new datasets cannot be mixed accidentally.
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


def _summary(rows):
    if not rows:
        return {"num_graphs": 0}
    times = [row["new_ga_time_s"] for row in rows]
    unchanged = sum(row["strategy_same_as_stored"] for row in rows)
    return {
        "num_graphs": len(rows),
        "num_same_strategy_as_stored": unchanged,
        "num_different_strategy_from_stored": len(rows) - unchanged,
        "avg_old_ga_time_s": sum(row["old_ga_time_s"] for row in rows) / len(rows),
        "avg_new_ga_time_s": sum(times) / len(times),
        "speedup": (
            sum(row["old_ga_time_s"] for row in rows)
            / sum(times)
            if sum(times) > 0
            else None
        ),
        "avg_objective_delta_new_minus_old": (
            sum(row["new_ga_objective"] - row["old_ga_objective"] for row in rows)
            / len(rows)
        ),
    }


def _save(path, source, source_args, bp_fast, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "source_dataset": str(source),
        "source_args": source_args,
        "bp_fast": bp_fast,
        "legacy_rng_compat": True,
        "summary": _summary(rows),
        "results": rows,
    }
    temp = path.with_suffix(path.suffix + ".tmp")
    with open(temp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate a stored dataset and rerun only BP+GA"
    )
    parser.add_argument("dataset_json")
    parser.add_argument("--out", required=True)
    parser.add_argument("--bp-fast", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args_cli = parser.parse_args()

    source = Path(args_cli.dataset_json)
    output = Path(args_cli.out)
    with open(source, encoding="utf-8") as f:
        stored_data = json.load(f)

    source_args = stored_data["args"]
    args = SimpleNamespace(**source_args)
    args.legacy_rng_compat = True
    stored_rows = stored_data["results"]

    rows = []
    if args_cli.resume and output.exists():
        with open(output, encoding="utf-8") as f:
            previous = json.load(f)
        if previous.get("source_dataset") != str(source):
            raise RuntimeError("resume output belongs to a different source dataset")
        if previous.get("bp_fast") != args_cli.bp_fast:
            raise RuntimeError("resume output uses a different BP implementation")
        rows = previous.get("results", [])

    completed = {row["graph_id"] for row in rows}
    rng = random.Random(args.seed)

    for stored in stored_rows:
        graph_id = stored["graph_id"]
        if args.graph_type == "structured":
            bn, values_table = generate_structured_graph_with_np_range(
                args, rng, graph_id
            )
        else:
            bn, values_table = generate_random_graph(args, graph_id)

        if graph_id in completed:
            continue
        if args_cli.limit is not None and len(rows) >= args_cli.limit:
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

        ga = find_best_defense_bp(
            bn,
            values_table,
            population_size=args.population_size,
            genmax=args.genmax,
            seed=args.seed + graph_id,
            bp_max_iters=args.bp_max_iters,
            bp_damping=args.bp_damping,
            bp_tol=args.bp_tol,
            top_n=args.top_n,
            return_all=True,
            bp_fast=args_cli.bp_fast,
        )
        final_eval = _evaluate(
            bn,
            values_table,
            ga["best_D_state"],
            use_ve,
            args,
            args_cli.bp_fast,
        )
        new_defended = _defended_list(ga["best_D_state"])
        row = {
            "graph_id": graph_id,
            "graph_stats": stored.get("graph_stats"),
            "use_ve": use_ve,
            "data_validation_objective": check["objective"],
            "old_ga_time_s": stored["bp_time_s"],
            "new_ga_time_s": ga["total_time_s"],
            "old_ga_objective": stored["bp_objective"],
            "new_ga_objective": final_eval["objective"],
            "objective_delta_new_minus_old": (
                final_eval["objective"] - stored["bp_objective"]
            ),
            "old_ga_defended": stored["bp_defended"],
            "new_ga_defended": new_defended,
            "strategy_same_as_stored": new_defended == stored["bp_defended"],
            "new_ga_evaluated": ga["unique_evaluated"],
            "ga_top_n": _top_results(ga["all_results"], args.top_n),
        }
        rows.append(row)
        _save(output, source, source_args, args_cli.bp_fast, rows)
        print(
            f"[{graph_id:>3}/{len(stored_rows)}] "
            f"validated={check['objective']:.6f} "
            f"old={stored['bp_time_s']:.3f}s new={ga['total_time_s']:.3f}s "
            f"speedup={stored['bp_time_s']/ga['total_time_s']:.2f}x "
            f"objective={final_eval['objective']:.6f}",
            flush=True,
        )

    _save(output, source, source_args, args_cli.bp_fast, rows)
    print(json.dumps(_summary(rows), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
