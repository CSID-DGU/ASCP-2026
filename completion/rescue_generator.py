"""
completion/rescue_generator.py -- uncovered flight을 위한 rescue candidate 생성 (F3/V2, H-V2-1~3)

v2-chanju.md §4에서 확정한 대로 prefix(허용 base -> target 직전) -> target 강제 포함 ->
suffix(target 직후 -> 허용 base) 3단계로 legal한 pairing 후보를 예산 제한 BFS로 찾는다.
탐색 자체는 heuristic이라 실제로 legal한지 보장 안 함 -- 최종적으로 만들어진 leg
시퀀스는 반드시 evaluation/validator.py::validate_pairing()을 통과한 것만 candidate로
채택한다(H-V2-2). 통과 못 하면 실패 사유 코드를 기록한다(H-V2-3).

flight/constraint dict 포맷은 evaluation/validator.py와 동일함.

# TODO(추후 조정): search_budget_per_target/max_prefix_legs/max_suffix_legs는 작은 값으로
# 시작함 -- 실제 데이터셋에서 rescue 성공률 보고 조정 필요 (v2-chanju.md §4)
"""

from collections import Counter, deque
from typing import Dict, List, Optional, Set

import config as _rl_config  # RL/ 이 sys.path에 있다고 가정 (evaluation/validator.py와 동일 관례)

from evaluation.validator import validate_pairing
from completion.candidate_schema import make_rescue_candidate


# ── 실패 사유 코드 (v2-chanju.md §5 H-V2-3) ──────────────────────
NO_BASE_PREFIX          = "NO_BASE_PREFIX"
NO_BASE_SUFFIX          = "NO_BASE_SUFFIX"
CONNECTION_INFEASIBLE   = "CONNECTION_INFEASIBLE"
DUTY_LIMIT              = "DUTY_LIMIT"
REST_LIMIT              = "REST_LIMIT"
PAIRING_DURATION_LIMIT  = "PAIRING_DURATION_LIMIT"
WINDOW_BOUNDARY         = "WINDOW_BOUNDARY"
NO_ALLOWED_BASE         = "NO_ALLOWED_BASE"
SEARCH_BUDGET_EXHAUSTED = "SEARCH_BUDGET_EXHAUSTED"

# validate_pairing()이 반환하는 violation code -> rescue 실패 사유 코드 매핑.
# prefix/suffix 후보는 있었지만 결합한 pairing이 legal하지 않았던 경우에 사용.
_VIOLATION_TO_FAILURE_REASON = {
    "MAX_DUTY_FAILURE":        DUTY_LIMIT,
    "MAX_LEGS_FAILURE":        DUTY_LIMIT,
    "MAX_DUTIES_FAILURE":      DUTY_LIMIT,
    "MIN_REST_FAILURE":        REST_LIMIT,
    "MAX_PAIRING_DAYS_FAILURE": PAIRING_DURATION_LIMIT,
    "MIN_CONNECTION_FAILURE":  CONNECTION_INFEASIBLE,
    "MAX_CONNECTION_FAILURE":  CONNECTION_INFEASIBLE,
    "AIRPORT_DISCONTINUITY":   CONNECTION_INFEASIBLE,
    "MIN_PAIRING_LEGS_FAILURE": CONNECTION_INFEASIBLE,
    "TIME_ORDER_FAILURE":      CONNECTION_INFEASIBLE,
    "INVALID_BASE_START":      NO_BASE_PREFIX,
    "BASE_RETURN_FAILURE":     NO_BASE_SUFFIX,
}


def _valid_gap(gap: float, min_conn: float, max_conn: float, min_rest: float) -> bool:
    """같은 duty 안 connection(min_conn~max_conn) 또는 새 duty 시작(rest >= min_rest)만 유효.
    그 사이(dead zone)는 어느 쪽으로도 legal하지 않음 -- validator.py와 동일 기준."""
    if gap < 0:
        return False
    return (min_conn <= gap <= max_conn) or (gap >= min_rest)


def _index_by_origin(flights: Dict[int, Dict], ids) -> Dict[int, List[int]]:
    by_origin: Dict[int, List[int]] = {}
    for fid in ids:
        by_origin.setdefault(flights[fid]["origin"], []).append(fid)
    for lst in by_origin.values():
        lst.sort(key=lambda fid: flights[fid]["dep_time"])
    return by_origin


def _search_chains(start_ids, is_goal, successors_fn, max_depth, budget, stats):
    """budget 제한 BFS: start_ids 각각을 길이 1 chain으로 시작해서 successors_fn으로
    확장하며 is_goal(chain 마지막 flight)이 참인 chain을 순서대로 yield한다.

    stats["budget_exhausted"]는 budget을 다 써서 중간에 멈췄는지(True) 아니면
    탐색 공간을 다 뒤졌는데도 못 찾은 건지(False)를 호출부가 구분하기 위한 out 파라미터.
    """
    queue = deque([[fid] for fid in start_ids])
    visited = 0
    while queue:
        if visited >= budget:
            stats["budget_exhausted"] = True
            return
        chain = queue.popleft()
        visited += 1
        last = chain[-1]
        if is_goal(last):
            yield list(chain)
        if len(chain) >= max_depth:
            continue
        for nxt in successors_fn(last):
            if nxt in chain:
                continue
            queue.append(chain + [nxt])
    stats["budget_exhausted"] = False


def _classify_combination_failure(violation_codes: List[str]) -> str:
    if not violation_codes:
        return SEARCH_BUDGET_EXHAUSTED
    most_common_code, _ = Counter(violation_codes).most_common(1)[0]
    return _VIOLATION_TO_FAILURE_REASON.get(most_common_code, CONNECTION_INFEASIBLE)


def _compute_cost(legs: List[int], flights: Dict[int, Dict], min_rest: float) -> float:
    """RL/rollout.py::emit_prefix()와 동일한 cost 공식 -- salvage/forced pairing과
    같은 취급(강제로 만든 completion 수단이라 IP_DEADHEAD_PENALTY를 항상 더함)."""
    first, last = flights[legs[0]], flights[legs[-1]]
    fly = sum(flights[fid]["arr_time"] - flights[fid]["dep_time"] for fid in legs)
    elapsed = last["arr_time"] - first["dep_time"]
    n_rest = sum(
        1 for prev, curr in zip(legs, legs[1:])
        if flights[curr]["dep_time"] - flights[prev]["arr_time"] >= min_rest
    )
    rest = min_rest * n_rest
    dead_time = max(elapsed - fly - rest, 0.0)
    n_legs = len(legs)
    return (dead_time
            - _rl_config.IP_LEG_BONUS * max(n_legs - 1, 0)
            + _rl_config.IP_DEADHEAD_PENALTY
            + _rl_config.IP_PAIRING_FIXED_COST)


def generate_rescue_candidates(
    flights: Dict[int, Dict],
    constraint: Dict,
    uncovered_flight_ids: List[int],
    *,
    max_candidates_per_target: int = 3,
    max_prefix_legs: int = 4,
    max_suffix_legs: int = 4,
    search_budget_per_target: int = 2000,
) -> Dict:
    """각 uncovered flight에 대해 base->target->base legal pairing 후보를 찾는다(H-V2-1~2).

    반환: {"candidates": [rescue candidate dict, ...], "failures": {flight_id: 실패사유코드}}
    (v2-chanju.md §7 완료 조건: "실패 target 누락 0" -- 모든 uncovered_flight_ids는
    candidates 또는 failures 둘 중 하나에 반드시 나타남)
    """
    min_conn = constraint.get("min_conn", _rl_config.DEFAULT_CONSTRAINTS["min_conn"])
    max_conn = constraint.get("max_conn", _rl_config.DEFAULT_CONSTRAINTS["max_conn"])
    min_rest = constraint.get("min_rest", _rl_config.DEFAULT_CONSTRAINTS["min_rest"])
    max_pairing_days = constraint.get(
        "max_pairing_days", _rl_config.DEFAULT_CONSTRAINTS["max_pairing_days"]
    )
    start_base = constraint.get("base_airport")

    # _check_base()와 동일한 규칙: 시작 base는 항상 단일 base_airport, 복귀만
    # allow_cross_base_return일 때 base_ids 중 아무 곳이나 허용됨(Turkish).
    allowed_return_bases = constraint.get("allowed_return_bases")
    if allowed_return_bases is None and constraint.get("allow_cross_base_return"):
        allowed_return_bases = constraint.get("base_ids")
    if allowed_return_bases:
        allowed_return_bases = set(allowed_return_bases)
    elif start_base is not None:
        allowed_return_bases = {start_base}
    else:
        allowed_return_bases = set()

    candidates: List[Dict] = []
    failures: Dict[int, str] = {}

    if start_base is None or not allowed_return_bases:
        for fid in uncovered_flight_ids:
            failures[fid] = NO_ALLOWED_BASE
        return {"candidates": candidates, "failures": failures}

    by_origin = _index_by_origin(flights, flights.keys())

    def successors(fid):
        f = flights[fid]
        for g_id in by_origin.get(f["dest"], []):
            if g_id == fid:
                continue
            gap = flights[g_id]["dep_time"] - f["arr_time"]
            if _valid_gap(gap, min_conn, max_conn, min_rest):
                yield g_id

    window_hours = max_pairing_days * 24.0
    seen_leg_sequences: Set[tuple] = set()

    for target_id in uncovered_flight_ids:
        if target_id not in flights:
            failures[target_id] = CONNECTION_INFEASIBLE
            continue
        target = flights[target_id]

        # ── prefix 후보 ──
        prefix_options: List[List[int]] = []
        prefix_stats = {"budget_exhausted": False}
        prefix_start_ids: List[int] = []
        if target["origin"] == start_base:
            prefix_options.append([])
        else:
            prefix_start_ids = [
                fid for fid in by_origin.get(start_base, [])
                if target["dep_time"] - window_hours <= flights[fid]["dep_time"] <= target["dep_time"]
            ]

            def is_prefix_goal(fid, _target=target):
                g = flights[fid]
                if g["dest"] != _target["origin"]:
                    return False
                return _valid_gap(_target["dep_time"] - g["arr_time"], min_conn, max_conn, min_rest)

            for chain in _search_chains(
                prefix_start_ids, is_prefix_goal, successors,
                max_prefix_legs, search_budget_per_target, prefix_stats,
            ):
                prefix_options.append(chain)
                if len(prefix_options) >= max_candidates_per_target:
                    break

        if not prefix_options:
            if prefix_stats["budget_exhausted"]:
                failures[target_id] = SEARCH_BUDGET_EXHAUSTED
            elif not prefix_start_ids:
                # window(max_pairing_days) 안에서는 base 출발 flight가 없음 -- window
                # 밖에는 있는지로 "그냥 없음(스케줄 경계, WINDOW_BOUNDARY)"과 "max_pairing_days
                # 제약 때문에 잘림(PAIRING_DURATION_LIMIT)"을 구분한다.
                any_before = any(
                    flights[fid]["dep_time"] <= target["dep_time"]
                    for fid in by_origin.get(start_base, [])
                )
                failures[target_id] = PAIRING_DURATION_LIMIT if any_before else WINDOW_BOUNDARY
            else:
                failures[target_id] = NO_BASE_PREFIX
            continue

        # ── suffix 후보 ──
        suffix_options: List[List[int]] = []
        suffix_stats = {"budget_exhausted": False}
        suffix_start_ids: List[int] = []
        if target["dest"] in allowed_return_bases:
            suffix_options.append([])
        else:
            suffix_start_ids = list(successors(target_id))

            def is_suffix_goal(fid):
                return flights[fid]["dest"] in allowed_return_bases

            for chain in _search_chains(
                suffix_start_ids, is_suffix_goal, successors,
                max_suffix_legs, search_budget_per_target, suffix_stats,
            ):
                suffix_options.append(chain)
                if len(suffix_options) >= max_candidates_per_target:
                    break

        if not suffix_options:
            # suffix_start_ids는 window로 걸러지지 않으므로(target 직후 successors 전체),
            # 비어 있으면 스케줄 경계가 아니라 진짜로 이어지는 flight가 없다는 뜻.
            if suffix_stats["budget_exhausted"]:
                failures[target_id] = SEARCH_BUDGET_EXHAUSTED
            else:
                failures[target_id] = NO_BASE_SUFFIX
            continue

        # ── prefix + target + suffix 결합 후 최종 검증(H-V2-2) ──
        found = 0
        attempted_violations: List[str] = []
        for prefix in prefix_options:
            for suffix in suffix_options:
                legs = prefix + [target_id] + suffix
                if len(set(legs)) != len(legs):
                    continue
                key = tuple(legs)
                if key in seen_leg_sequences:
                    continue
                result = validate_pairing({"legs": legs}, flights, constraint)
                if not result["is_valid"]:
                    attempted_violations.extend(result["violation_codes"])
                    continue
                cost = _compute_cost(legs, flights, min_rest)
                candidate = make_rescue_candidate(
                    legs=legs,
                    repair_target_flights=[target_id],
                    cost=cost,
                    validator_version=result["validator_version"],
                    constraint_hash=result["constraint_hash"],
                )
                candidates.append(candidate)
                seen_leg_sequences.add(key)
                found += 1
                if found >= max_candidates_per_target:
                    break
            if found >= max_candidates_per_target:
                break

        if found == 0:
            failures[target_id] = _classify_combination_failure(attempted_violations)

    return {"candidates": candidates, "failures": failures}
