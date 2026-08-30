"""LLM이 제출한 최종 pairing 해를 수정 없이 직접 평가함."""

from __future__ import annotations

import math
from collections import Counter
from typing import Dict, List, Sequence

from evaluation.llm_scoring import score_submitted_pairing
from evaluation.validator import VALIDATOR_VERSION, constraint_hash

try:
    import config
except ModuleNotFoundError:
    from RL import config


DECLARED_BASE_MISMATCH = "DECLARED_BASE_MISMATCH"


def _pairing_constraint(
    record: Dict,
    flights: Dict[int, Dict],
    constraint_template: Dict,
    base_ids: Sequence[int],
) -> Dict:
    legs = record.get("legs", [])
    if legs and legs[0] in flights and flights[legs[0]]["origin"] in set(base_ids):
        base_airport = flights[legs[0]]["origin"]
    else:
        base_airport = base_ids[0]
    return {
        **constraint_template,
        "base_airport": base_airport,
        "base_ids": list(base_ids),
    }


def _invalid_entry(record: Dict, validation: Dict, extra_codes=()) -> Dict:
    violation_codes = list(validation["violation_codes"])
    for code in extra_codes:
        if code not in violation_codes:
            violation_codes.append(code)
    return {
        "pairing_number": record.get("pairing_number"),
        "declared_base": record.get("declared_base"),
        "legs": list(record.get("legs", [])),
        "violation_codes": violation_codes,
        "invalid_flight_ids": validation["invalid_flight_ids"],
        "duplicate_flight_ids": validation["duplicate_flight_ids"],
        "validator_version": validation["validator_version"],
        "constraint_hash": validation["constraint_hash"],
    }


def aggregate_direct_metrics(scored_pairings: Sequence[Dict]) -> Dict:
    """개별적으로 legal한 제출 pairing의 진단 지표만 합산함."""
    total_fly = sum(pairing["fly"] for pairing in scored_pairings)
    total_intra_gap = sum(pairing["intra_duty_gap"] for pairing in scored_pairings)
    total_raw_dead = sum(pairing["dead_time"] for pairing in scored_pairings)
    total_legs = sum(pairing["n_legs"] for pairing in scored_pairings)
    total_duties = sum(pairing["n_duties"] for pairing in scored_pairings)
    n_pairings = len(scored_pairings)
    return {
        "n_pairings": n_pairings,
        "total_cost": sum(pairing["cost"] for pairing in scored_pairings),
        "man_days": sum(math.ceil(pairing["elapsed"] / 24.0) for pairing in scored_pairings),
        "total_flying_time": total_fly,
        "total_dead_time_within_duty": total_intra_gap,
        "raw_dead_time_for_cost": total_raw_dead,
        "ftc_pct": total_intra_gap / total_fly * 100.0 if total_fly else 0.0,
        "avg_legs_per_pairing": total_legs / n_pairings if n_pairings else 0.0,
        "avg_duties_per_pairing": total_duties / n_pairings if n_pairings else 0.0,
    }


def build_legacy_forced_100(direct_result: Dict, flights: Dict[int, Dict]) -> Dict:
    """기존 forced-singleton 수치를 primary direct 결과와 분리해 계산함."""
    universe = set(flights)
    legal_pairings = list(direct_result["legal_pairings"])
    legal_counts = Counter(
        flight_id for pairing in legal_pairings for flight_id in pairing["legs"]
    )
    duplicate_legal_ids = {
        flight_id for flight_id, count in legal_counts.items() if count > 1
    }
    # 기존 evaluator와 같이 중복 flight가 든 legal pairing을 forced 계산에서 모두 제외함.
    legacy_valid_pairings = [
        pairing
        for pairing in legal_pairings
        if not duplicate_legal_ids.intersection(pairing["legs"])
    ]
    legacy_covered = {
        flight_id
        for pairing in legacy_valid_pairings
        for flight_id in pairing["legs"]
        if flight_id in universe
    }
    forced_flight_ids = sorted(universe - legacy_covered)
    valid_metrics = aggregate_direct_metrics(legacy_valid_pairings)
    n_forced = len(forced_flight_ids)
    n_pairings = len(legacy_valid_pairings) + n_forced
    forced_fly = sum(
        flights[flight_id]["arr_time"] - flights[flight_id]["dep_time"]
        for flight_id in forced_flight_ids
    )
    total_fly = valid_metrics["total_flying_time"] + forced_fly
    total_dead = valid_metrics["total_dead_time_within_duty"]
    valid_legs = sum(pairing["n_legs"] for pairing in legacy_valid_pairings)
    valid_duties = sum(pairing["n_duties"] for pairing in legacy_valid_pairings)
    forced_unit_cost = config.IP_DEADHEAD_PENALTY + config.IP_PAIRING_FIXED_COST

    return {
        "schema_version": "legacy-forced-100-1.0",
        "synthetic": n_forced > 0,
        "is_legal_solution": bool(direct_result["solution_feasible"] and n_forced == 0),
        "use_as_primary_result": False,
        "duplicate_policy": "invalidate-all-overlapping-legal-pairings",
        "legacy_valid_pairing_count": len(legacy_valid_pairings),
        "duplicate_legal_flight_ids": sorted(duplicate_legal_ids),
        "forced_flight_ids": forced_flight_ids,
        "n_forced_pairings": n_forced,
        "synthetic_completion_coverage": 1.0,
        "n_pairings": n_pairings,
        "n_deadheads": n_forced,
        "man_days": valid_metrics["man_days"] + n_forced,
        "total_flying_time": total_fly,
        "total_dead_time_within_duty": total_dead,
        "raw_dead_time_for_cost": valid_metrics["raw_dead_time_for_cost"],
        "ftc_pct": total_dead / total_fly * 100.0 if total_fly else 0.0,
        "total_cost": valid_metrics["total_cost"] + n_forced * forced_unit_cost,
        "avg_legs_per_pairing": (
            (valid_legs + n_forced) / n_pairings if n_pairings else 0.0
        ),
        "avg_duties_per_pairing": (
            (valid_duties + n_forced) / n_pairings if n_pairings else 0.0
        ),
        "forced_unit_cost": forced_unit_cost,
    }


def evaluate_direct_solution(
    pairing_records: Sequence[Dict],
    declared_uncovered: Sequence[int],
    flights: Dict[int, Dict],
    constraint_template: Dict,
    base_ids: Sequence[int],
    airport_code_to_id: Dict[str, int],
) -> Dict:
    """LLM의 제출 해를 선택·보완하지 않고 feasibility와 지표를 계산함."""
    if not base_ids:
        raise ValueError("base_ids가 비어 있음")

    universe = set(flights)
    all_submitted_legs = [
        flight_id
        for record in pairing_records
        for flight_id in record.get("legs", [])
    ]
    submitted_counts = Counter(all_submitted_legs)
    submitted_set = set(all_submitted_legs)
    uncovered_counts = Counter(declared_uncovered)
    uncovered_set = set(declared_uncovered)

    duplicate_across_pairings = sorted(
        flight_id for flight_id, count in submitted_counts.items() if count > 1
    )
    duplicate_uncovered = sorted(
        flight_id for flight_id, count in uncovered_counts.items() if count > 1
    )
    pairing_uncovered_overlap = sorted(submitted_set & uncovered_set)
    unknown_flight_ids = sorted((submitted_set | uncovered_set) - universe)
    undeclared_flight_ids = sorted(universe - submitted_set - uncovered_set)

    legal_pairings: List[Dict] = []
    invalid_pairings: List[Dict] = []
    for record in pairing_records:
        constraint = _pairing_constraint(record, flights, constraint_template, base_ids)
        scored, validation = score_submitted_pairing(record, flights, constraint)

        declared_base = record.get("declared_base")
        base_mismatch = False
        if declared_base is not None:
            declared_base_id = airport_code_to_id.get(str(declared_base).upper())
            legs = record.get("legs", [])
            actual_start = flights[legs[0]]["origin"] if legs and legs[0] in flights else None
            base_mismatch = declared_base_id is None or declared_base_id != actual_start

        if scored is None or base_mismatch:
            invalid_pairings.append(
                _invalid_entry(
                    record,
                    validation,
                    [DECLARED_BASE_MISMATCH] if base_mismatch else [],
                )
            )
        else:
            legal_pairings.append(scored)

    legal_covered = {
        flight_id
        for pairing in legal_pairings
        for flight_id in pairing["legs"]
        if flight_id in universe
    }
    conflict_free_legal_covered = {
        flight_id
        for flight_id in legal_covered
        if submitted_counts[flight_id] == 1 and flight_id not in uncovered_set
    }
    legally_uncovered = sorted(universe - legal_covered)

    solution_feasible = not any((
        invalid_pairings,
        duplicate_across_pairings,
        duplicate_uncovered,
        pairing_uncovered_overlap,
        unknown_flight_ids,
        undeclared_flight_ids,
        uncovered_set,
        universe - legal_covered,
    ))
    diagnostic_metrics = aggregate_direct_metrics(legal_pairings)

    return {
        "schema_version": "llm-direct-eval-1.0",
        "evaluation_mode": "direct",
        "solution_feasible": solution_feasible,
        "n_total_flights": len(universe),
        "n_submitted_pairings": len(pairing_records),
        "n_legal_pairings": len(legal_pairings),
        "n_invalid_pairings": len(invalid_pairings),
        "declared_uncovered_flight_ids": sorted(uncovered_set & universe),
        "undeclared_flight_ids": undeclared_flight_ids,
        "unknown_flight_ids": unknown_flight_ids,
        "duplicate_flight_ids": duplicate_across_pairings,
        "duplicate_uncovered_flight_ids": duplicate_uncovered,
        "pairing_uncovered_overlap_flight_ids": pairing_uncovered_overlap,
        "legally_uncovered_flight_ids": legally_uncovered,
        "legal_union_coverage": len(legal_covered) / len(universe) if universe else 1.0,
        "conflict_free_legal_coverage": (
            len(conflict_free_legal_covered) / len(universe) if universe else 1.0
        ),
        "legal_pairings": legal_pairings,
        "invalid_pairings": invalid_pairings,
        "legal_pairing_diagnostics": diagnostic_metrics,
        "solution_metrics": diagnostic_metrics if solution_feasible else None,
        "validator_version": VALIDATOR_VERSION,
        "constraint_hash": constraint_hash(constraint_template),
        "optimizer_applied": False,
        "rescue_applied": False,
        "completion_applied": False,
    }
