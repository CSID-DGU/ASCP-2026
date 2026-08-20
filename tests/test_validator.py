"""
tests/test_validator.py -- evaluation/validator.py 스모크 테스트 (첫 골격 단계)

C5(항목별 violation fixture 전부)는 별도 커밋에서 채운다 -- 여기서는 validator가
정상 pairing/base-미복귀 pairing을 각각 올바르게 판정하는지만 우선 확인
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "RL"))
sys.path.insert(0, os.path.join(REPO_ROOT, "evaluation"))

from validator import (  # noqa: E402
    validate_pairing,
    UNKNOWN_FLIGHT,
    DUPLICATE_FLIGHT,
    INVALID_BASE_START,
    BASE_RETURN_FAILURE,
    AIRPORT_DISCONTINUITY,
    MIN_CONNECTION_FAILURE,
    MAX_CONNECTION_FAILURE,
    MAX_DUTY_FAILURE,
    MAX_LEGS_FAILURE,
    MAX_DUTIES_FAILURE,
    MAX_PAIRING_DAYS_FAILURE,
    MIN_PAIRING_LEGS_FAILURE,
    TIME_ORDER_FAILURE,
)


BASE = 0
OTHER = 1
THIRD = 2

# base(0) -> other(1) -> base(0), 정상적인 connection/휴식 없이 한 duty로 끝나는 pairing
FLIGHTS_VALID = {
    0: {"id": 0, "origin": BASE,  "dest": OTHER, "dep_time": 0.0, "arr_time": 2.0},
    1: {"id": 1, "origin": OTHER, "dest": BASE,  "dep_time": 3.0, "arr_time": 5.0},
}
CONSTRAINT = {
    "base_airport": BASE,
    "min_conn": 0.5, "max_conn": 9.0,
    "min_rest": 10.0,
    "max_duty": 13.0, "max_legs": 8,
    "max_duty_periods": 2, "max_pairing_days": 5,
    "min_pairing_legs": 2,
}


def test_valid_pairing_passes():
    result = validate_pairing({"legs": [0, 1]}, FLIGHTS_VALID, CONSTRAINT)
    assert result["is_valid"], result["violation_codes"]
    assert result["violation_codes"] == []
    assert result["start_base"] == BASE
    assert result["end_airport"] == BASE
    assert result["n_duties"] == 1


def test_non_base_return_is_caught():
    # 두 번째 leg를 지워서 base로 안 돌아오는(도착지가 OTHER인) pairing으로 만듦
    result = validate_pairing({"legs": [0]}, FLIGHTS_VALID, CONSTRAINT)
    assert not result["is_valid"]
    assert BASE_RETURN_FAILURE in result["violation_codes"]


# Turkish HB1/HB2 비대칭 복귀: HB1(base=BASE)에서 출발해서 HB2(=OTHER)로 끝나는 pairing.
FLIGHTS_CROSS_BASE = {
    0: {"id": 0, "origin": BASE,  "dest": OTHER, "dep_time": 0.0, "arr_time": 2.0},
}


def test_cross_base_return_fails_by_default():
    # allowed_return_bases 없이(Delta/Alaska/JetBlue 기본 규칙)는 HB1->HB2도
    # 그냥 base 미복귀로 잡혀야 함.
    result = validate_pairing({"legs": [0]}, FLIGHTS_CROSS_BASE, CONSTRAINT)
    assert not result["is_valid"]
    assert BASE_RETURN_FAILURE in result["violation_codes"]


def test_cross_base_return_allowed_when_opted_in():
    # allowed_return_bases = {BASE, OTHER}로 명시하면(Turkish HB1/HB2 케이스) 서로
    # 다른 base로 끝나도 BASE_RETURN_FAILURE가 나면 안 됨.
    turkish_constraint = {**CONSTRAINT, "allowed_return_bases": {BASE, OTHER}}
    result = validate_pairing({"legs": [0]}, FLIGHTS_CROSS_BASE, turkish_constraint)
    assert BASE_RETURN_FAILURE not in result["violation_codes"], result["violation_codes"]


def test_unknown_flight_is_caught():
    result = validate_pairing({"legs": [999]}, {}, CONSTRAINT)
    assert not result["is_valid"]
    assert UNKNOWN_FLIGHT in result["violation_codes"]


def test_duplicate_within_pairing_is_caught():
    # 같은 flight(0)를 두 번 넣음
    result = validate_pairing({"legs": [0, 0]}, FLIGHTS_VALID, CONSTRAINT)
    assert not result["is_valid"]
    assert DUPLICATE_FLIGHT in result["violation_codes"]
    assert result["duplicate_flight_ids"] == [0]


def test_invalid_base_start_is_caught():
    # OTHER(1)에서 출발해서 BASE(0)로 끝남 -- 도착은 base라 BASE_RETURN_FAILURE는 안 나야 함
    flights = {0: {"id": 0, "origin": OTHER, "dest": BASE, "dep_time": 0.0, "arr_time": 2.0}}
    result = validate_pairing({"legs": [0]}, flights, CONSTRAINT)
    assert not result["is_valid"]
    assert INVALID_BASE_START in result["violation_codes"]
    assert BASE_RETURN_FAILURE not in result["violation_codes"]


def test_airport_discontinuity_is_caught():
    # leg0 도착지(1)와 leg1 출발지(THIRD=2)가 안 맞음 -- 같은 duty(연결시간 정상 범위)
    flights = {
        0: {"id": 0, "origin": BASE,  "dest": OTHER, "dep_time": 0.0, "arr_time": 2.0},
        1: {"id": 1, "origin": THIRD, "dest": BASE,  "dep_time": 3.0, "arr_time": 5.0},
    }
    result = validate_pairing({"legs": [0, 1]}, flights, CONSTRAINT)
    assert not result["is_valid"]
    assert AIRPORT_DISCONTINUITY in result["violation_codes"]


def test_min_connection_failure_is_caught():
    # gap = 0.1h < min_conn(0.5h)
    flights = {
        0: {"id": 0, "origin": BASE,  "dest": OTHER, "dep_time": 0.0, "arr_time": 2.0},
        1: {"id": 1, "origin": OTHER, "dest": BASE,  "dep_time": 2.1, "arr_time": 4.0},
    }
    result = validate_pairing({"legs": [0, 1]}, flights, CONSTRAINT)
    assert not result["is_valid"]
    assert MIN_CONNECTION_FAILURE in result["violation_codes"]


def test_max_connection_failure_is_caught():
    # gap = 9.1h: max_conn(9.0h) 초과, min_rest(10h) 미만이라 duty는 안 나뉨 -- "dead zone"
    flights = {
        0: {"id": 0, "origin": BASE,  "dest": OTHER, "dep_time": 0.0, "arr_time": 0.5},
        1: {"id": 1, "origin": OTHER, "dest": BASE,  "dep_time": 9.6, "arr_time": 10.1},
    }
    result = validate_pairing({"legs": [0, 1]}, flights, CONSTRAINT)
    assert not result["is_valid"]
    assert MAX_CONNECTION_FAILURE in result["violation_codes"]


def test_max_duty_failure_is_caught():
    # duty 경과시간 14.1h > max_duty(13h), connection은 정상 범위(0.6h)
    flights = {
        0: {"id": 0, "origin": BASE,  "dest": OTHER, "dep_time": 0.0, "arr_time": 7.0},
        1: {"id": 1, "origin": OTHER, "dest": BASE,  "dep_time": 7.6, "arr_time": 14.1},
    }
    result = validate_pairing({"legs": [0, 1]}, flights, CONSTRAINT)
    assert not result["is_valid"]
    assert MAX_DUTY_FAILURE in result["violation_codes"]


def test_max_legs_failure_is_caught():
    # max_legs를 1로 낮춰서, 정상 2-leg 1-duty pairing도 leg 수 초과로 잡히는지 확인
    constraint = {**CONSTRAINT, "max_legs": 1}
    result = validate_pairing({"legs": [0, 1]}, FLIGHTS_VALID, constraint)
    assert not result["is_valid"]
    assert MAX_LEGS_FAILURE in result["violation_codes"]


def test_max_duties_failure_is_caught():
    # max_duty_periods를 0(overnight 0번, 즉 duty 1개만 허용)으로 낮춘 뒤 2-duty pairing 검증
    flights = {
        0: {"id": 0, "origin": BASE,  "dest": OTHER, "dep_time": 0.0,  "arr_time": 2.0},
        1: {"id": 1, "origin": OTHER, "dest": BASE,  "dep_time": 15.0, "arr_time": 17.0},  # gap 13h >= min_rest
    }
    constraint = {**CONSTRAINT, "max_duty_periods": 0}
    result = validate_pairing({"legs": [0, 1]}, flights, constraint)
    assert not result["is_valid"]
    assert MAX_DUTIES_FAILURE in result["violation_codes"]
    assert result["n_duties"] == 2


def test_max_pairing_days_failure_is_caught():
    # 기존 정상 pairing(약 0.2일)에 max_pairing_days만 아주 작게(0.05일) 낮춰서 위반 유도
    constraint = {**CONSTRAINT, "max_pairing_days": 0.05}
    result = validate_pairing({"legs": [0, 1]}, FLIGHTS_VALID, constraint)
    assert not result["is_valid"]
    assert MAX_PAIRING_DAYS_FAILURE in result["violation_codes"]


def test_min_pairing_legs_failure_is_caught():
    # 기존 정상 2-leg pairing에 min_pairing_legs만 3으로 올려서 위반 유도
    constraint = {**CONSTRAINT, "min_pairing_legs": 3}
    result = validate_pairing({"legs": [0, 1]}, FLIGHTS_VALID, constraint)
    assert not result["is_valid"]
    assert MIN_PAIRING_LEGS_FAILURE in result["violation_codes"]


def test_time_order_failure_is_caught():
    # leg1의 출발(3.0)이 leg0의 도착(7.0)보다 이름 -- 시간 역순
    flights = {
        0: {"id": 0, "origin": BASE,  "dest": OTHER, "dep_time": 5.0, "arr_time": 7.0},
        1: {"id": 1, "origin": OTHER, "dest": BASE,  "dep_time": 3.0, "arr_time": 9.0},
    }
    result = validate_pairing({"legs": [0, 1]}, flights, CONSTRAINT)
    assert not result["is_valid"]
    assert TIME_ORDER_FAILURE in result["violation_codes"]


if __name__ == "__main__":
    test_fns = [
        test_valid_pairing_passes,
        test_non_base_return_is_caught,
        test_cross_base_return_fails_by_default,
        test_cross_base_return_allowed_when_opted_in,
        test_unknown_flight_is_caught,
        test_duplicate_within_pairing_is_caught,
        test_invalid_base_start_is_caught,
        test_airport_discontinuity_is_caught,
        test_min_connection_failure_is_caught,
        test_max_connection_failure_is_caught,
        test_max_duty_failure_is_caught,
        test_max_legs_failure_is_caught,
        test_max_duties_failure_is_caught,
        test_max_pairing_days_failure_is_caught,
        test_min_pairing_legs_failure_is_caught,
        test_time_order_failure_is_caught,
    ]
    for fn in test_fns:
        fn()
    print(f"OK: {len(test_fns)}개 테스트 통과 (MIN_REST_FAILURE는 TODO -- validator.py 참고)")
