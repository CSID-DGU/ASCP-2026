"""
evaluation/validation_report.py -- source_type별 결과 분리 집계 (F1/V1 C2)

policy가 정상 생성한 pairing과 salvage/repair/forced로 보완된 pairing을 섞어서 보고하면
"generator가 진짜로 얼마나 잘 만들었는지"가 왜곡된다 -- 이 모듈은 pairing_record의
source_type(policy|salvage|repair|forced, v1.md §2 스키마)별로 나눠서 각각 따로 집계한다.
`policy_direct`만 실제 generator direct coverage로 쓰고, 나머지는 별도 completion
결과로만 쓴다.

Deadhead/ManDays/FTC는 pairing_record가 생성 쪽(RL/rollout.py)에서 채워주는
cost/dead_time 값을 그대로 믿지 않고, flights 데이터로부터 독립적으로 다시 계산한다
-- C1의 "생성 코드와 독립적으로 검증한다" 원칙을 집계 지표에도 동일하게 적용함.
"""

from typing import Dict, List, Optional, Tuple

try:
    from validator import (
        validate_pairing,
        find_cross_pairing_duplicates,
        _split_into_duties,
        VALIDATOR_VERSION,
    )
except ModuleNotFoundError:
    # evaluate_ip.py 등 evaluation/ 패키지를 sys.path에 직접 추가하지 않고
    # `from evaluation.xxx import ...` 식으로 부르는 호출부를 위한 fallback
    from evaluation.validator import (
        validate_pairing,
        find_cross_pairing_duplicates,
        _split_into_duties,
        VALIDATOR_VERSION,
    )


# pairing_record.source_type 값 -> 이 report의 집계 bucket 이름
_SOURCE_TO_BUCKET = {
    "policy":  "policy_direct",
    "salvage": "salvage",
    "repair":  "repair",
    "forced":  "forced",
}


def _pairing_time_metrics(
    legs: List[int], flights: Dict[int, Dict], min_rest: float,
    duty_break_indices: Optional[List[int]] = None,
):
    """flying_time, dead_time(휴식 제외, duty별 elapsed-flying 합), pairing_days를
    flights 데이터로부터 독립 계산 (pairing_record의 cost/dead_time 필드는 안 씀).
    """
    duties = _split_into_duties(legs, flights, min_rest, duty_break_indices)
    total_fly = 0.0
    total_dead = 0.0
    for duty in duties:
        fly = sum(flights[fid]["arr_time"] - flights[fid]["dep_time"] for fid in duty)
        elapsed = flights[duty[-1]]["arr_time"] - flights[duty[0]]["dep_time"]
        total_fly += fly
        total_dead += max(elapsed - fly, 0.0)
    pairing_days = (flights[legs[-1]]["arr_time"] - flights[legs[0]]["dep_time"]) / 24.0
    return total_fly, total_dead, pairing_days


def _aggregate(
    pairing_constraint_pairs: List[Tuple[Dict, Optional[Dict]]],
    flights: Dict[int, Dict],
    n_total_flights: Optional[int],
    min_rest: float,
) -> Dict[str, Dict]:
    """실제 집계 로직 -- (pairing_record, 그 pairing을 검증할 constraint) 쌍의 리스트를
    받는다. constraint가 pairing마다 달라도(chunk별 base_airport 등) 각자 자기
    constraint로 검증되므로 정확함. aggregate_by_source()/aggregate_by_source_per_chunk()
    둘 다 이 함수를 감싼 얇은 wrapper임.
    """
    buckets: Dict[str, List[Tuple[Dict, Optional[Dict]]]] = {name: [] for name in _SOURCE_TO_BUCKET.values()}
    for p, c in pairing_constraint_pairs:
        source = p.get("source_type", "policy")
        bucket_name = _SOURCE_TO_BUCKET.get(source, source)
        buckets.setdefault(bucket_name, []).append((p, c))

    denom = n_total_flights if n_total_flights is not None else len(flights)

    report = {}
    for bucket_name, bucket_pairs in buckets.items():
        covered = set()
        invalid_count = 0
        any_constraint_given = False
        deadhead_count = 0
        total_fly = 0.0
        total_dead = 0.0
        total_man_days = 0.0
        bucket_pairings_only = [p for p, _ in bucket_pairs]

        for p, c in bucket_pairs:
            legs = p.get("legs", [])
            covered.update(legs)

            if c is not None:
                any_constraint_given = True
                result = validate_pairing(p, flights, c)
                if not result["is_valid"]:
                    invalid_count += 1

            if p.get("is_deadhead"):
                deadhead_count += 1

            if legs and all(fid in flights for fid in legs):
                fly, dead, days = _pairing_time_metrics(
                    legs, flights, min_rest, p.get("duty_break_indices")
                )
                total_fly += fly
                total_dead += dead
                total_man_days += days
            # legs가 비어있거나 unknown flight를 포함하면(이미 invalid로 잡힘)
            # 시간 지표 계산은 건너뜀 -- flights[fid] 접근이 안전하지 않으므로.

        ftc_pct = (total_dead / total_fly * 100) if total_fly > 0 else None

        report[bucket_name] = {
            "pairing_count":                 len(bucket_pairs),
            "covered_flights":               len(covered),
            "invalid_count":                 invalid_count if any_constraint_given else None,
            "internal_duplicate_flight_ids": find_cross_pairing_duplicates(bucket_pairings_only),
            "direct_coverage_ratio":         (len(covered) / denom) if denom else 0.0,
            "deadhead_count":                deadhead_count,
            "total_flying_time":             total_fly,
            "total_dead_time":               total_dead,
            "man_days":                      total_man_days,
            "ftc_pct":                       ftc_pct,
        }

    all_pairings = [p for p, _ in pairing_constraint_pairs]
    # 최종 selection 기준(모든 bucket 합산) duplicate -- policy가 커버한 flight를
    # salvage/repair/forced가 또 커버한 경우도 여기서 잡힘.
    report["cross_bucket_duplicate_flight_ids"] = find_cross_pairing_duplicates(all_pairings)
    # policy_direct만 진짜 generator coverage로 쓴다는 원칙을 결과에도 명시.
    report["_direct_coverage_source"] = "policy_direct"
    # C3 provenance 요구사항 -- 이 report가 어느 validator 버전으로 만들어졌는지.
    # (constraint_hash는 pairing마다 다를 수 있어 여기(전체 report)엔 안 두고,
    # 필요하면 validate_pairing() 개별 호출 결과의 constraint_hash를 참고.)
    report["_validator_version"] = VALIDATOR_VERSION
    return report


def aggregate_by_source(
    pairings: List[Dict],
    flights: Dict[int, Dict],
    constraint: Optional[Dict] = None,
    n_total_flights: Optional[int] = None,
    min_rest: float = 10.0,
) -> Dict[str, Dict]:
    """pairing들을 source_type별로 나눠서 각각 집계 -- 배치 전체가 같은 constraint
    하나를 쓸 때 사용(예: 단일 chunk, 또는 constraint가 정말 동일한 경우).

    여러 chunk(서로 다른 base_airport 등)를 합쳐서 봐야 하면
    aggregate_by_source_per_chunk()를 쓸 것 -- evaluation/evaluate_ip.py가 chunk마다
    base_id = random.choice(base_ids)로 constraint를 다시 뽑는 걸 확인했으므로, 여러
    chunk의 pairing을 이 함수 하나에 몰아넣으면 일부 pairing이 자기 생성 시점과 다른
    constraint로 검증될 수 있음.

    duplicate flights는 두 층위로 나눠서 본다 (스펙에 bucket별인지 전체인지 명시가
    없어서, 둘 다 보여주고 어느 걸 "duplicate flights"로 볼지는 사용하는 쪽에서 고르게 함):
      - bucket별 "internal_duplicate_flight_ids": 그 bucket 안에서만 중복
      - 최상위 "cross_bucket_duplicate_flight_ids": 전체 선택(모든 bucket 합산)
        기준 중복 -- 최종 solution의 진짜 duplicate assignment는 이쪽이 맞음.

    반환: {
        "policy_direct": {...}, "salvage": {...}, "repair": {...}, "forced": {...},
        "cross_bucket_duplicate_flight_ids": [...],
        "_direct_coverage_source": "policy_direct",
    }
    각 bucket: pairing_count, covered_flights, invalid_count,
    internal_duplicate_flight_ids, direct_coverage_ratio,
    deadhead_count, total_flying_time, total_dead_time, man_days, ftc_pct
    """
    pairs = [(p, constraint) for p in pairings]
    return _aggregate(pairs, flights, n_total_flights, min_rest)


def aggregate_by_source_per_chunk(
    chunks: List[Tuple[List[Dict], Optional[Dict]]],
    flights: Dict[int, Dict],
    n_total_flights: Optional[int] = None,
    min_rest: float = 10.0,
) -> Dict[str, Dict]:
    """여러 chunk(각자 자기 constraint를 가짐)의 pairing들을 하나의 report로 합쳐서 집계.

    chunks: [(이 chunk의 pairing_record 리스트, 이 chunk에서 쓰인 constraint), ...]
    -- evaluate_ip.py가 chunk마다 base_id를 다시 뽑는 것과 맞춰, pairing마다 자기가
    생성될 때 쓰인 constraint로 검증되도록 함(§3 TODO에서 언급한 옵션 1).

    # TODO(확인 필요, 대안): 지금은 "호출하는 쪽이 chunk별로 pairing과 constraint를
    # 짝지어 넘겨준다"고 가정함 -- 만약 나중에 pairing_record 자체가 자기 constraint(최소
    # base_airport)를 필드로 들고 있는 형태로 바뀐다면, 이 함수 대신 pairing_record에서
    # 직접 constraint를 꺼내 쓰는 방식으로 더 간단해질 수 있음. 실제 evaluate_ip.py 연결
    # (C3)할 때 어느 쪽이 더 자연스러운지 다시 판단.
    """
    pairs: List[Tuple[Dict, Optional[Dict]]] = []
    for chunk_pairings, chunk_constraint in chunks:
        pairs.extend((p, chunk_constraint) for p in chunk_pairings)
    return _aggregate(pairs, flights, n_total_flights, min_rest)
