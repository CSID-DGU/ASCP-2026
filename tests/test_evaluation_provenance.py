"""
tests/test_evaluation_provenance.py -- validator_version/constraint_hash provenance가
V1 파이프라인 전 구간(validator.py -> validation_report.py -> evaluate_ip.py)에서
일관되게 기록되는지 확인
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "RL"))
sys.path.insert(0, os.path.join(REPO_ROOT, "evaluation"))

from validator import validate_pairing, constraint_hash, VALIDATOR_VERSION  # noqa: E402
from validation_report import aggregate_by_source, aggregate_by_source_per_chunk  # noqa: E402
from evaluation.evaluate_ip import validate_selected_pairings  # noqa: E402


BASE, OTHER = 0, 1
FLIGHTS = {
    0: {"id": 0, "origin": BASE,  "dest": OTHER, "dep_time": 0.0, "arr_time": 2.0},
    1: {"id": 1, "origin": OTHER, "dest": BASE,  "dep_time": 3.0, "arr_time": 5.0},
}
CONSTRAINT = {
    "base_airport": BASE,
    "min_conn": 0.5, "max_conn": 9.0, "min_rest": 10.0,
    "max_duty": 13.0, "max_legs": 8,
    "max_duty_periods": 2, "max_pairing_days": 5,
    "min_pairing_legs": 2,
}


def test_validate_pairing_carries_version_and_hash():
    result = validate_pairing({"legs": [0, 1]}, FLIGHTS, CONSTRAINT)
    assert result["validator_version"] == VALIDATOR_VERSION
    assert result["constraint_hash"] == constraint_hash(CONSTRAINT)


def test_constraint_hash_is_order_independent_and_content_sensitive():
    # 같은 내용이면(딕셔너리 순서가 달라도) 같은 해시
    reordered = {"min_pairing_legs": 2, **CONSTRAINT}
    assert constraint_hash(CONSTRAINT) == constraint_hash(reordered)
    # 내용이 다르면 다른 해시
    different = {**CONSTRAINT, "max_duty": 10.0}
    assert constraint_hash(CONSTRAINT) != constraint_hash(different)
    # set(allowed_return_bases)이 섞여 있어도 순서 무관하게 안정적
    with_set = {**CONSTRAINT, "allowed_return_bases": {BASE, OTHER}}
    assert constraint_hash(with_set) == constraint_hash({**CONSTRAINT, "allowed_return_bases": {OTHER, BASE}})


def test_aggregate_by_source_report_carries_version():
    report = aggregate_by_source([{"legs": [0, 1], "source_type": "policy"}], FLIGHTS,
                                  constraint=CONSTRAINT, n_total_flights=2)
    assert report["validator_version"] == VALIDATOR_VERSION


def test_aggregate_by_source_per_chunk_report_carries_version():
    chunks = [([{"legs": [0, 1], "source_type": "policy"}], CONSTRAINT)]
    report = aggregate_by_source_per_chunk(chunks, FLIGHTS, n_total_flights=2)
    assert report["validator_version"] == VALIDATOR_VERSION


def test_evaluate_ip_validate_selected_pairings_report_carries_version():
    # C3: evaluate_ip.py의 최종 selected pairing 재검증 결과에도 같은 provenance가 남는지
    selected = [{"legs": [0, 1], "source_type": "policy", "_gen_base_airport": BASE}]
    report = validate_selected_pairings(selected, FLIGHTS, CONSTRAINT, [BASE], n_total_flights=2)
    assert report["validator_version"] == VALIDATOR_VERSION


if __name__ == "__main__":
    test_validate_pairing_carries_version_and_hash()
    test_constraint_hash_is_order_independent_and_content_sensitive()
    test_aggregate_by_source_report_carries_version()
    test_aggregate_by_source_per_chunk_report_carries_version()
    test_evaluate_ip_validate_selected_pairings_report_carries_version()
    print("OK: 5개 테스트 통과")
