"""
completion/candidate_schema.py -- rescue candidate column 스키마 (F3/V2 rescue generator)

evaluation/full_flight_master.py::validate_master_inputs()와 evaluation/
completion_runner.py::merge_rescue_columns()가 실제로 요구하는 필드를 그대로 따른다
(혜린 코드를 직접 읽고 확인함, journal/experiment-plan/v2-chanju.md §3 참고) -- 여기서
만든 candidate가 그 계약을 만족하지 않으면 evaluate_ip.py --full-flight-master
--rescue-pool-path 단계에서 바로 거부된다. 이 모듈은 그 계약을 미리 로컬에서
확인해서, 문제가 있으면 evaluate_ip.py까지 안 가고 여기서 바로 알 수 있게 한다.
"""

import math
from typing import Dict, List, Optional


class RescueCandidateError(ValueError):
    """rescue candidate가 full_flight_master의 입력 계약을 만족하지 않음."""


def make_rescue_candidate(
    legs: List[int],
    repair_target_flights: List[int],
    cost: float,
    validator_version: str,
    constraint_hash: str,
    column_id: Optional[str] = None,
) -> Dict:
    """evaluation/full_flight_master.py가 요구하는 rescue column 스키마로 dict를 구성.

    validator_version/constraint_hash는 rescue_generator가 evaluation/validator.py::
    validate_pairing()을 호출한 결과에서 그대로 가져와서 넣어야 한다(여기서 새로
    계산하지 않음) -- 같은 값을 두 번 만들 이유가 없고, "이 candidate가 실제로 그
    validate_pairing() 호출을 통과했다"는 provenance를 그대로 전달하는 의미도 있다.
    """
    if not legs:
        raise RescueCandidateError("legs가 비어 있음")
    if len(set(legs)) != len(legs):
        raise RescueCandidateError("legs 내부에 중복 flight가 있음")
    if not repair_target_flights:
        raise RescueCandidateError("repair_target_flights가 비어 있음")
    if not set(repair_target_flights).issubset(set(legs)):
        raise RescueCandidateError("repair_target_flights가 legs의 부분집합이 아님")

    try:
        cost = float(cost)
    except (TypeError, ValueError) as exc:
        raise RescueCandidateError("cost가 유효한 숫자가 아님") from exc
    if not math.isfinite(cost) or cost < 0:
        raise RescueCandidateError("cost는 0 이상의 유한값이어야 함")

    if not validator_version:
        raise RescueCandidateError("validator_version이 비어 있음")
    if not constraint_hash:
        raise RescueCandidateError("constraint_hash가 비어 있음")

    candidate = {
        "legs": list(legs),
        "source_type": "rescue",
        "is_legal": True,
        "cost": cost,
        "repair_target_flights": list(repair_target_flights),
        "validator_version": validator_version,
        "constraint_hash": constraint_hash,
    }
    if column_id is not None:
        candidate["column_id"] = column_id
    return candidate
