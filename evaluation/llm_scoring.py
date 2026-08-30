"""LLM이 제출한 pairing을 수정하지 않고 공통 기준으로 채점함."""

from __future__ import annotations

from typing import Dict, Tuple

from evaluation.validator import _split_into_duties, validate_pairing

try:
    import config
except ModuleNotFoundError:
    from RL import config


def score_submitted_pairing(
    pairing_record: Dict,
    flights: Dict[int, Dict],
    constraint: Dict,
) -> Tuple[Dict | None, Dict]:
    """제출된 leg 순서를 그대로 검증하고 합법인 경우 지표를 계산함."""
    validation = validate_pairing(pairing_record, flights, constraint)
    if not validation["is_valid"]:
        return None, validation

    legs = list(pairing_record["legs"])
    min_rest = float(constraint["min_rest"])
    duties = _split_into_duties(
        legs, flights, min_rest, pairing_record.get("duty_break_indices")
    )
    fly = sum(
        flights[flight_id]["arr_time"] - flights[flight_id]["dep_time"]
        for flight_id in legs
    )
    elapsed = flights[legs[-1]]["arr_time"] - flights[legs[0]]["dep_time"]
    n_rest = max(len(duties) - 1, 0)
    dead_time = max(elapsed - fly - min_rest * n_rest, 0.0)

    duty_start_indices = set()
    offset = 0
    for duty in duties[:-1]:
        offset += len(duty)
        duty_start_indices.add(offset)
    intra_gap = 0.0
    inter_excess = 0.0
    inferred_breaks = []
    for index, (previous_id, current_id) in enumerate(zip(legs, legs[1:]), start=1):
        gap = flights[current_id]["dep_time"] - flights[previous_id]["arr_time"]
        if index in duty_start_indices:
            inferred_breaks.append(index)
            inter_excess += max(gap - min_rest, 0.0)
        else:
            intra_gap += max(gap, 0.0)

    cost = max(
        dead_time
        - config.IP_LEG_BONUS * max(len(legs) - 1, 0)
        + config.IP_PAIRING_FIXED_COST,
        0.0,
    )
    start_airport = flights[legs[0]]["origin"]
    end_airport = flights[legs[-1]]["dest"]
    scored = {
        **pairing_record,
        "is_legal": True,
        "cost": cost,
        "fly": fly,
        "elapsed": elapsed,
        "dead_time": dead_time,
        "n_legs": len(legs),
        "n_duties": len(duties),
        "intra_duty_gap": intra_gap,
        "inter_duty_excess": inter_excess,
        "duty_break_indices": inferred_breaks,
        "start_airport": start_airport,
        "end_airport": end_airport,
        "validator_version": validation["validator_version"],
        "constraint_hash": validation["constraint_hash"],
    }
    return scored, validation
