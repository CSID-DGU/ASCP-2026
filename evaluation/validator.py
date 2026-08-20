"""
evaluation/validator.py -- independent pairing legality validator

이 모듈은 RL/environment.py::get_mask()/step()을 재사용하지 않는다 --
-> 생성 쪽(mask)과 같은 구현을 쓰면 같은 버그를 검출할 수 없기 때문임 
여기서 각 제약을 완전히 새로 계산해서, policy가 만든 pairing이 실제로
legal한지 독립적으로 재확인한다.

flight dict 포맷은 RL/loader.py와 동일함: {"id", "origin", "dest", "dep_time", "arr_time"}
(origin/dest는 정수 airport ID, dep_time/arr_time은 시간(hour) 단위 절대값)
constraint dict 포맷은 RL/config.py::DEFAULT_CONSTRAINTS와 동일한 키를 씀

# TODO(추후 코드 확인 필요): violation code enum을 evaluation/validator.py 안에 두는 걸로
# 우선 진행함 -- RL/ 쪽(mask)에서도 이 코드가 필요해지면 공통 모듈로 옮기는 게 나을 수 있음
"""

from typing import Dict, List, Optional

import config as _rl_config  # RL/ 이 sys.path에 있다고 가정 (evaluate_ip.py와 동일 관례)


# ── Violation codes (공통 violation code 14개 그대로) ──────────
UNKNOWN_FLIGHT             = "UNKNOWN_FLIGHT"
DUPLICATE_FLIGHT           = "DUPLICATE_FLIGHT"
INVALID_BASE_START         = "INVALID_BASE_START"
BASE_RETURN_FAILURE        = "BASE_RETURN_FAILURE"
AIRPORT_DISCONTINUITY      = "AIRPORT_DISCONTINUITY"
MIN_CONNECTION_FAILURE     = "MIN_CONNECTION_FAILURE"
MAX_CONNECTION_FAILURE     = "MAX_CONNECTION_FAILURE"
MIN_REST_FAILURE           = "MIN_REST_FAILURE"
MAX_DUTY_FAILURE           = "MAX_DUTY_FAILURE"
MAX_LEGS_FAILURE           = "MAX_LEGS_FAILURE"
MAX_DUTIES_FAILURE         = "MAX_DUTIES_FAILURE"
MAX_PAIRING_DAYS_FAILURE   = "MAX_PAIRING_DAYS_FAILURE"
MIN_PAIRING_LEGS_FAILURE   = "MIN_PAIRING_LEGS_FAILURE"
TIME_ORDER_FAILURE         = "TIME_ORDER_FAILURE"


def _split_into_duties(legs: List[int], flights: Dict[int, Dict], min_rest: float):
    """gap >= min_rest인 지점을 duty 경계(overnight rest)로 보고 분리

    min_conn <= gap <= max_conn 이면 같은 duty 안의 connection, gap >= min_rest면
    새 duty 시작 -- 그 사이(min_rest 미만이지만 max_conn 초과)는 어느 쪽으로도
    유효하지 않은 "dead zone"이며, 호출부(_check_connections_and_rest)에서
    별도로 위반 처리한다. 여기서는 min_rest 기준으로만 1차 분리
    """
    if not legs:
        return []
    duties = [[legs[0]]]
    for i in range(1, len(legs)):
        prev = flights[legs[i - 1]]
        curr = flights[legs[i]]
        gap = curr["dep_time"] - prev["arr_time"]
        if gap >= min_rest:
            duties.append([curr["id"]])
        else:
            duties[-1].append(curr["id"])
    return duties


def _check_time_order_and_unknown(legs, flights, violations):
    """UNKNOWN_FLIGHT, TIME_ORDER_FAILURE. 이후 체크가 의존하는 전제 조건이라 가장 먼저 실행."""
    for fid in legs:
        if fid not in flights:
            violations.append(UNKNOWN_FLIGHT)
    if any(v == UNKNOWN_FLIGHT for v in violations):
        return False  # 이후 체크는 flights[fid] 접근이 안전하지 않으므로 중단
    for i in range(1, len(legs)):
        if flights[legs[i]]["dep_time"] < flights[legs[i - 1]]["arr_time"]:
            violations.append(TIME_ORDER_FAILURE)
    return True


def _check_duplicate_within(legs, violations):
    if len(set(legs)) != len(legs):
        violations.append(DUPLICATE_FLIGHT)


def _check_base(legs, flights, constraint, violations):
    # 기본 규칙(Delta/Alaska/JetBlue): 출발 base == 도착 base == 배정된 base_airport
    # 두 체크를 서로 독립적으로 본다 -- first==last(출발지로 그대로 복귀)만 보면, 애초에
    # 엉뚱한 곳에서 출발한 pairing(INVALID_BASE_START)이 그 엉뚱한 곳으로 되돌아왔을 때
    # BASE_RETURN_FAILURE를 놓친다.
    #
    # 예외: Turkish는 HB1->HB2, HB2->HB1처럼 서로 다른 base로 돌아와도 유효함
    # (evaluation/evaluate_ip.py의 _pairing_valid()가 airline=="turkish"일 때 별도로 이렇게
    # 처리하는 걸 확인함). 이 검증기에서는 constraint에 명시적으로
    # `allowed_return_bases`(이 base들 중 아무거나로 복귀해도 됨, 서로 같을 필요 없음)가
    # 주어졌을 때만 이 예외를 적용한다 -- base_ids만으로는(Delta도 여러 base를 갖고
    # 있어서) 항공사를 구분할 수 없으므로 반드시 별도 필드로 명시적으로 opt-in해야 함
    #
    # "Turkish HB1/HB2를 allowed-return-base 규칙으로 표현"하라고 이미
    # 합의돼 있음 -- 이 접근 자체는 확정. TODO로 남는 건 두 가지:
    # (1) 실제 필드명이 `allowed_return_bases`가 맞는지, (2) Turkish constraint
    # 모듈(RL/turkish/constraints_turkish.py 등)이 실제로 이 필드를 채워줄지.
    base = constraint.get("base_airport")
    allowed_return_bases = constraint.get("allowed_return_bases")
    first, last = flights[legs[0]], flights[legs[-1]]

    if base is not None and first["origin"] != base:
        violations.append(INVALID_BASE_START)

    if allowed_return_bases:
        if last["dest"] not in allowed_return_bases:
            violations.append(BASE_RETURN_FAILURE)
    elif base is not None:
        if last["dest"] != base:
            violations.append(BASE_RETURN_FAILURE)
    elif first["origin"] != last["dest"]:
        # base_airport 자체가 안 주어진 경우(예외적) -- 최소한 출발==도착이라도 확인
        violations.append(BASE_RETURN_FAILURE)


def _check_connections_and_rest(duties, flights, constraint, violations):
    min_conn = constraint.get("min_conn", _rl_config.DEFAULT_CONSTRAINTS["min_conn"])
    max_conn = constraint.get("max_conn", _rl_config.DEFAULT_CONSTRAINTS["max_conn"])
    min_rest = constraint.get("min_rest", _rl_config.DEFAULT_CONSTRAINTS["min_rest"])

    for duty in duties:
        # duty 내부 공항 연속성 + connection 시간
        for i in range(1, len(duty)):
            prev, curr = flights[duty[i - 1]], flights[duty[i]]
            if prev["dest"] != curr["origin"]:
                violations.append(AIRPORT_DISCONTINUITY)
            gap = curr["dep_time"] - prev["arr_time"]
            if gap < min_conn:
                violations.append(MIN_CONNECTION_FAILURE)
            if gap > max_conn:
                violations.append(MAX_CONNECTION_FAILURE)

    # duty 간 (overnight) 공항 연속성 + rest 시간
    #
    # TODO(확인 필요): 지금 구조에서 MIN_REST_FAILURE는 사실상 발생할 수 없는 죽은
    # 코드임 -- _split_into_duties()가 "gap >= min_rest"인 지점에서만 duty를 나누기
    # 때문에 여기서 구한 duty 간 rest는 나누는 조건 자체가 이미 min_rest 이상이라서
    # 항상 min_rest를 만족함 max_conn을 넘었지만 min_rest에는 못 미치는 "dead zone"
    # gap은 같은 duty로 묶여서 MAX_CONNECTION_FAILURE로만 잡힌다(의미상 맞을 수도, 아닐
    # 수도 있음). flat leg 목록만으로는 "duty 내 connection이었는지 실패한 rest였는지"를
    # 구조적으로 구분할 방법이 없어서 -- pairing_record가 duty 경계를 명시적으로 주는
    # 형태가 되면 그때 구현 지금은 TODO로 남기고 fixture도 안 만듦.
    for i in range(1, len(duties)):
        prev_last = flights[duties[i - 1][-1]]
        curr_first = flights[duties[i][0]]
        if prev_last["dest"] != curr_first["origin"]:
            violations.append(AIRPORT_DISCONTINUITY)
        rest = curr_first["dep_time"] - prev_last["arr_time"]
        if rest < min_rest:
            violations.append(MIN_REST_FAILURE)


def _check_duty_and_pairing_limits(legs, duties, flights, constraint, violations):
    max_duty         = constraint.get("max_duty", _rl_config.DEFAULT_CONSTRAINTS["max_duty"])
    max_legs         = constraint.get("max_legs", _rl_config.DEFAULT_CONSTRAINTS["max_legs"])
    max_duty_periods = constraint.get("max_duty_periods", _rl_config.DEFAULT_CONSTRAINTS["max_duty_periods"])
    max_pairing_days = constraint.get("max_pairing_days", _rl_config.DEFAULT_CONSTRAINTS["max_pairing_days"])
    min_pairing_legs = constraint.get("min_pairing_legs", _rl_config.DEFAULT_CONSTRAINTS["min_pairing_legs"])

    for duty in duties:
        if len(duty) > max_legs:
            violations.append(MAX_LEGS_FAILURE)
        elapsed = flights[duty[-1]]["arr_time"] - flights[duty[0]]["dep_time"]
        if elapsed > max_duty:
            violations.append(MAX_DUTY_FAILURE)

    # max_duty_periods는 "duty 수"가 아니라 "overnight 횟수" 기준 (RL/environment.py의
    # duty_period < max_duty_periods 게이트와 동일 의미) -- duties가 n개면 overnight은 n-1개
    n_overnights = len(duties) - 1
    if n_overnights > max_duty_periods:
        violations.append(MAX_DUTIES_FAILURE)

    pairing_days = (flights[legs[-1]]["arr_time"] - flights[legs[0]]["dep_time"]) / 24.0
    if pairing_days > max_pairing_days:
        violations.append(MAX_PAIRING_DAYS_FAILURE)

    if len(legs) < min_pairing_legs:
        violations.append(MIN_PAIRING_LEGS_FAILURE)


def validate_pairing(pairing_record: Dict, flights: Dict[int, Dict], constraint: Dict) -> Dict:
    """pairing_record(최소 {"legs": [...]})를 완전히 독립적으로 재검증

    반환: {"is_valid", "violation_codes", "invalid_flight_ids", "duplicate_flight_ids",
           "start_base", "end_airport", "n_duties"}  (v1.md §2 "Validator 결과 최소 필드")
    """
    legs = pairing_record.get("legs", [])
    violations: List[str] = []

    if not legs:
        return {
            "is_valid": False, "violation_codes": [UNKNOWN_FLIGHT],
            "invalid_flight_ids": [], "duplicate_flight_ids": [],
            "start_base": None, "end_airport": None, "n_duties": 0,
        }

    ok = _check_time_order_and_unknown(legs, flights, violations)
    invalid_flight_ids = [fid for fid in legs if fid not in flights]
    duplicate_flight_ids = [fid for fid in set(legs) if legs.count(fid) > 1]
    _check_duplicate_within(legs, violations)

    if ok:  # flights[fid] 접근이 안전할 때만 나머지 체크 진행
        _check_base(legs, flights, constraint, violations)
        min_rest = constraint.get("min_rest", _rl_config.DEFAULT_CONSTRAINTS["min_rest"])
        duties = _split_into_duties(legs, flights, min_rest)
        _check_connections_and_rest(duties, flights, constraint, violations)
        _check_duty_and_pairing_limits(legs, duties, flights, constraint, violations)
        start_base  = flights[legs[0]]["origin"]
        end_airport = flights[legs[-1]]["dest"]
        n_duties    = len(duties)
    else:
        start_base, end_airport, n_duties = None, None, 0

    return {
        "is_valid":            len(violations) == 0,
        "violation_codes":     violations,
        "invalid_flight_ids":  invalid_flight_ids,
        "duplicate_flight_ids": duplicate_flight_ids,
        "start_base":          start_base,
        "end_airport":         end_airport,
        "n_duties":            n_duties,
    }


def find_cross_pairing_duplicates(pairings: List[Dict]) -> List[int]:
    """Selected solution 전체에서 같은 flight가 2개 이상 pairing에 중복 배정됐는지 확인
    (v1.md C1 "Selected solution 전체 duplicate conflict"). 개별 pairing 내부 중복은
    validate_pairing()의 duplicate_flight_ids가 이미 잡음 -- 이건 pairing 간 중복 전용
    """
    seen: Dict[int, int] = {}
    dupes = []
    for p in pairings:
        for fid in p.get("legs", []):
            seen[fid] = seen.get(fid, 0) + 1
    for fid, count in seen.items():
        if count > 1:
            dupes.append(fid)
    return dupes
