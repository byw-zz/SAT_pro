"""Pure helpers for the medium-graph parameter-sensitivity experiment.

This module deliberately contains no solver or graph-generator imports.  The
experiment runner can therefore be tested without starting MaxHS, GA, CBC, or
Zenitani searches.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import uuid


FAMILIES = ("structured", "random")
METHODS = ("maxsat", "ga", "khouzani", "zenitani")
PARAM_FIELDS = {
    "C_benefit": ("C", "C_benefit"),
    "P_loss": ("P", "P_loss"),
    "D_cost": ("D", "D_cost"),
}
NOMINAL_PARAMETER = "nominal"
QUALITY_STATUSES = {"complete"}
TERMINAL_STATUSES = QUALITY_STATUSES | {
    "timeout_incumbent",
    "feasible_unverified",
    "timeout",
    "error",
}


@dataclass(frozen=True, order=True)
class InstanceKey:
    """Stable identifier for one nominal or perturbed parameter table."""

    family: str
    graph_id: int
    parameter: str
    delta_pct: int
    perturbation_run: int

    def __post_init__(self):
        if self.family not in FAMILIES:
            raise ValueError(f"unsupported graph family: {self.family}")
        if self.graph_id < 1:
            raise ValueError("graph_id must be positive")
        if self.parameter == NOMINAL_PARAMETER:
            if self.delta_pct != 0 or self.perturbation_run != 0:
                raise ValueError(
                    "nominal instances require delta_pct=0 and perturbation_run=0"
                )
            return
        if self.parameter not in PARAM_FIELDS:
            raise ValueError(f"unsupported perturbed parameter: {self.parameter}")
        if not 0 < self.delta_pct < 100:
            raise ValueError("perturbed instances require 0 < delta_pct < 100")
        if self.perturbation_run < 1:
            raise ValueError("perturbation_run must be positive")

    @property
    def instance_id(self):
        if self.parameter == NOMINAL_PARAMETER:
            return f"{self.family}_g{self.graph_id:03d}_nominal"
        return (
            f"{self.family}_g{self.graph_id:03d}_{self.parameter}"
            f"_d{self.delta_pct:02d}_r{self.perturbation_run:02d}"
        )

    def to_dict(self):
        return asdict(self)


def iter_instance_keys(
    families=FAMILIES,
    num_graphs=10,
    parameters=tuple(PARAM_FIELDS),
    deltas_pct=(10, 20, 30),
    perturbation_runs=1,
    include_nominal=False,
    pilot=False,
):
    """Enumerate nominal and perturbed instances in a deterministic order."""
    graph_ids = range(1, 2 if pilot else num_graphs + 1)
    run_ids = range(1, (1 if pilot else perturbation_runs) + 1)
    for family in families:
        for graph_id in graph_ids:
            if include_nominal:
                yield InstanceKey(family, graph_id, NOMINAL_PARAMETER, 0, 0)
            for parameter in parameters:
                for perturbation_run in run_ids:
                    for delta_pct in deltas_pct:
                        yield InstanceKey(
                            family,
                            graph_id,
                            parameter,
                            int(delta_pct),
                            perturbation_run,
                        )


def canonical_hash(value):
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def derive_seed(base_seed, namespace, *parts):
    """Derive a cross-process-stable 63-bit seed without Python ``hash``."""
    digest = hashlib.sha256(
        json.dumps(
            [int(base_seed), str(namespace), *parts],
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def perturbation_seed(base_seed, key):
    if key.parameter == NOMINAL_PARAMETER:
        raise ValueError("nominal instances have no perturbation seed")
    # delta is intentionally excluded: one random direction vector is reused
    # at 10/20/30 percent within the same graph/parameter/repetition.
    return derive_seed(
        base_seed,
        "perturbation",
        key.family,
        key.graph_id,
        key.parameter,
        key.perturbation_run,
    )


def ga_seed(base_seed, key, run_index):
    if run_index < 0:
        raise ValueError("run_index must be non-negative")
    return derive_seed(
        base_seed,
        "ga",
        key.family,
        key.graph_id,
        key.parameter,
        key.delta_pct,
        key.perturbation_run,
        run_index,
    )


def method_seed(base_seed, method, key):
    return derive_seed(
        base_seed,
        method,
        key.family,
        key.graph_id,
        key.parameter,
        key.delta_pct,
        key.perturbation_run,
    )


def _target_rows(values_table, parameter):
    if parameter not in PARAM_FIELDS:
        raise ValueError(f"unsupported perturbed parameter: {parameter}")
    node_type, field = PARAM_FIELDS[parameter]
    return sorted(
        (
            row
            for row in values_table
            if row.get("type") == node_type and row.get(field) is not None
        ),
        key=lambda row: row["node"],
    )


def make_perturbation_vector(values_table, parameter, seed):
    """Draw one independent U[-1,1] value per target node."""
    rng = random.Random(int(seed))
    return {
        row["node"]: rng.uniform(-1.0, 1.0)
        for row in _target_rows(values_table, parameter)
    }


def apply_node_perturbation(values_table, parameter, delta, vector):
    """Return a deep row copy with only the requested field perturbed."""
    if parameter not in PARAM_FIELDS:
        raise ValueError(f"unsupported perturbed parameter: {parameter}")
    if not 0.0 <= float(delta) < 1.0:
        raise ValueError("delta must satisfy 0 <= delta < 1")
    node_type, field = PARAM_FIELDS[parameter]
    expected_nodes = {row["node"] for row in _target_rows(values_table, parameter)}
    if set(vector) != expected_nodes:
        missing = sorted(expected_nodes - set(vector))
        extra = sorted(set(vector) - expected_nodes)
        raise ValueError(
            f"perturbation vector nodes mismatch; missing={missing}, extra={extra}"
        )

    out = []
    for source in values_table:
        row = dict(source)
        if row.get("type") == node_type and row.get(field) is not None:
            z_value = float(vector[row["node"]])
            if not -1.0 <= z_value <= 1.0:
                raise ValueError(f"perturbation value outside [-1,1]: {row['node']}")
            row[field] = float(row[field]) * (1.0 + float(delta) * z_value)
        out.append(row)
    return out


def _finite_number(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def is_quality_usable(record):
    return bool(
        record
        and record.get("status") in QUALITY_STATUSES
        and _finite_number(record.get("objective"))
    )


def select_median_ga_run(runs, expected_runs=5):
    """Select the objective median while charging the full multi-run protocol."""
    if expected_runs < 1 or expected_runs % 2 == 0:
        raise ValueError("expected_runs must be a positive odd number")
    indexed = {}
    for run in runs:
        run_index = run.get("run_index")
        if not isinstance(run_index, int) or run_index < 0:
            raise ValueError("each GA run requires a non-negative integer run_index")
        if run_index in indexed:
            raise ValueError(f"duplicate GA run_index: {run_index}")
        indexed[run_index] = run

    complete = [
        run
        for run in indexed.values()
        if run.get("status") == "complete" and _finite_number(run.get("objective"))
    ]
    if len(complete) != expected_runs:
        return {
            "status": "partial",
            "expected_runs": expected_runs,
            "completed_runs": len(complete),
            "status_counts": dict(
                Counter(run.get("status", "missing") for run in runs)
            ),
        }

    for run in complete:
        if not _finite_number(run.get("complete_protocol_time_s")):
            raise ValueError("complete GA runs require finite complete_protocol_time_s")
    ordered = sorted(complete, key=lambda run: (run["objective"], run["run_index"]))
    selected = ordered[len(ordered) // 2]
    result = {
        "status": "complete",
        "aggregation": "median final objective of independent runs",
        "expected_runs": expected_runs,
        "completed_runs": expected_runs,
        "selected_run_index": selected["run_index"],
        "median_single_run_time_s": statistics.median(
            run["complete_protocol_time_s"] for run in complete
        ),
        "complete_protocol_time_s": sum(
            run["complete_protocol_time_s"] for run in complete
        ),
        "run_result_files": [
            run.get("result_file")
            for run in sorted(complete, key=lambda x: x["run_index"])
        ],
    }
    for field in (
        "objective",
        "C_benefit",
        "P_expected_loss",
        "D_cost",
        "defended",
        "n_defended",
        "final_evaluation",
        "bp_converged",
    ):
        if field in selected:
            result[field] = selected[field]
    return result


def compare_vs_maxsat(method_record, maxsat_record, tol_abs=0.0, tol_rel=0.0):
    """Compare final objectives; by default, only exact equality is a tie."""
    if not is_quality_usable(method_record) or not is_quality_usable(maxsat_record):
        return "na"
    method_value = method_record["objective"]
    maxsat_value = maxsat_record["objective"]
    tolerance = max(float(tol_abs), float(tol_rel) * abs(maxsat_value))
    if method_value > maxsat_value + tolerance:
        return "win"
    if method_value < maxsat_value - tolerance:
        return "loss"
    return "tie"


def pairwise_gap(method_record, maxsat_record):
    if not is_quality_usable(method_record) or not is_quality_usable(maxsat_record):
        return None
    absolute = maxsat_record["objective"] - method_record["objective"]
    return {
        "absolute_maxsat_minus_method": absolute,
        "normalized_maxsat_minus_method": absolute
        / max(abs(maxsat_record["objective"]), 1.0),
    }


def descriptive_stats(values):
    values = [float(value) for value in values if _finite_number(value)]
    if not values:
        return {
            "n": 0,
            "mean": None,
            "median": None,
            "stdev": None,
            "min": None,
            "max": None,
        }
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def _group_summary(rows, methods):
    graph_ids = sorted({row["graph_id"] for row in rows})
    method_summary = {}
    for method_name in methods:
        status_counts = Counter()
        status_by_graph = defaultdict(Counter)
        objectives = []
        times = []
        objectives_by_graph = defaultdict(list)
        times_by_graph = defaultdict(list)
        for row in rows:
            record = row.get("methods", {}).get(method_name) or {"status": "missing"}
            status_counts[record.get("status", "missing")] += 1
            status_by_graph[row["graph_id"]][record.get("status", "missing")] += 1
            if is_quality_usable(record):
                objectives.append(record["objective"])
                objectives_by_graph[row["graph_id"]].append(record["objective"])
            if is_quality_usable(record) and _finite_number(
                record.get("complete_protocol_time_s")
            ):
                times.append(record["complete_protocol_time_s"])
                times_by_graph[row["graph_id"]].append(
                    record["complete_protocol_time_s"]
                )
        graph_objective_means = [
            statistics.fmean(values) for values in objectives_by_graph.values()
        ]
        graph_time_means = [
            statistics.fmean(values) for values in times_by_graph.values()
        ]
        method_summary[method_name] = {
            "status_counts": dict(sorted(status_counts.items())),
            "complete_count": status_counts["complete"],
            "quality_usable_count": len(objectives),
            "total_instances": len(rows),
            "complete_fraction": status_counts["complete"] / len(rows)
            if rows
            else None,
            "complete_fraction_graph_balanced": descriptive_stats(
                counts["complete"] / sum(counts.values())
                for counts in status_by_graph.values()
            ),
            "objective_raw": descriptive_stats(objectives),
            "objective_graph_balanced": descriptive_stats(graph_objective_means),
            "complete_protocol_time_raw_s": descriptive_stats(times),
            "complete_protocol_time_graph_balanced_s": descriptive_stats(
                graph_time_means
            ),
        }

    comparisons = {}
    for method_name in methods:
        if method_name == "maxsat":
            continue
        wtl = Counter()
        wtl_by_graph = defaultdict(Counter)
        absolute = []
        normalized = []
        absolute_by_graph = defaultdict(list)
        normalized_by_graph = defaultdict(list)
        for row in rows:
            records = row.get("methods", {})
            method_record = records.get(method_name)
            maxsat_record = records.get("maxsat")
            outcome = compare_vs_maxsat(method_record, maxsat_record)
            wtl[outcome] += 1
            wtl_by_graph[row["graph_id"]][outcome] += 1
            gap = pairwise_gap(method_record, maxsat_record)
            if gap is None:
                continue
            graph_id = row["graph_id"]
            absolute.append(gap["absolute_maxsat_minus_method"])
            normalized.append(gap["normalized_maxsat_minus_method"])
            absolute_by_graph[graph_id].append(gap["absolute_maxsat_minus_method"])
            normalized_by_graph[graph_id].append(gap["normalized_maxsat_minus_method"])
        graph_absolute = [
            statistics.fmean(values) for values in absolute_by_graph.values()
        ]
        graph_normalized = [
            statistics.fmean(values) for values in normalized_by_graph.values()
        ]
        comparisons[method_name] = {
            "wtl_raw": {name: wtl[name] for name in ("win", "tie", "loss", "na")},
            "wtl_fraction_graph_balanced": {
                name: descriptive_stats(
                    counts[name] / sum(counts.values())
                    for counts in wtl_by_graph.values()
                )
                for name in ("win", "tie", "loss", "na")
            },
            "wtl_paired_fraction_graph_balanced": {
                name: descriptive_stats(
                    counts[name]
                    / sum(counts[outcome] for outcome in ("win", "tie", "loss"))
                    for counts in wtl_by_graph.values()
                    if sum(counts[outcome] for outcome in ("win", "tie", "loss"))
                )
                for name in ("win", "tie", "loss")
            },
            "paired_count": len(absolute),
            "graphs_with_pairs": len(graph_absolute),
            "absolute_gap_raw": descriptive_stats(absolute),
            "absolute_gap_graph_balanced": descriptive_stats(graph_absolute),
            "normalized_gap_raw": descriptive_stats(normalized),
            "normalized_gap_graph_balanced": descriptive_stats(graph_normalized),
        }

    return {
        "num_instance_records": len(rows),
        "num_base_graphs": len(graph_ids),
        "graph_ids": graph_ids,
        "methods": method_summary,
        "pairwise_vs_maxsat": comparisons,
    }


def summarize_instance_rows(rows, methods=METHODS):
    """Aggregate perturbations within graph before aggregating across graphs."""
    nominal = defaultdict(list)
    perturbed = defaultdict(list)
    for row in rows:
        if row["parameter"] == NOMINAL_PARAMETER:
            nominal[row["family"]].append(row)
        else:
            perturbed[(row["family"], row["parameter"], row["delta_pct"])].append(row)

    return {
        "aggregation": {
            "primary_unit": "base graph",
            "within_graph": "mean over perturbation repetitions",
            "across_graphs": "descriptive statistics over graph-level means",
            "wtl": (
                "exact-objective win/tie/loss counts plus graph-balanced "
                "outcome fractions"
            ),
        },
        "nominal": [
            {"family": family, **_group_summary(group_rows, methods)}
            for family, group_rows in sorted(nominal.items())
        ],
        "perturbed": [
            {
                "family": family,
                "parameter": parameter,
                "delta_pct": delta_pct,
                **_group_summary(group_rows, methods),
            }
            for (family, parameter, delta_pct), group_rows in sorted(perturbed.items())
        ],
    }


def missing_ga_runs(run_records, expected_runs=5):
    complete = {
        record.get("run_index")
        for record in run_records
        if record.get("status") == "complete"
    }
    return [
        run_index for run_index in range(expected_runs) if run_index not in complete
    ]


def method_needs_run(record, retry_failed=False):
    if not record:
        return True
    status = record.get("status")
    if status in TERMINAL_STATUSES - {"error"}:
        return False
    if status == "error":
        return bool(retry_failed)
    return True


def config_fingerprint(config):
    return canonical_hash(config)


def atomic_write_json(path, payload):
    """Durably replace one JSON file without exposing partial contents."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    try:
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)
