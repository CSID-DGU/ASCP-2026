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

from validator import validate_pairing, BASE_RETURN_FAILURE  # noqa: E402


BASE = 0
OTHER = 1

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


if __name__ == "__main__":
    test_valid_pairing_passes()
    test_non_base_return_is_caught()
    test_cross_base_return_fails_by_default()
    test_cross_base_return_allowed_when_opted_in()
    print("OK: 4개 스모크 테스트 통과")
