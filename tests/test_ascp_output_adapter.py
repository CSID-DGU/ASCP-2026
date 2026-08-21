"""
tests/test_ascp_output_adapter.py -- evaluation/ascp_output_adapter.py 테스트 (F1 산출물 5번)

evaluate_ip.py::save_result_json()으로 저장한 파일을 ascp_output_adapter로 다시 읽어서
validate_pairing()/aggregate_by_source()에 재투입하는 전체 흐름(저장 -> 로드 -> 재검증)을
확인한다.
"""

import os
import sys
import tempfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "RL"))
sys.path.insert(0, os.path.join(REPO_ROOT, "evaluation"))

from evaluation.evaluate_ip import save_result_json, validate_selected_pairings  # noqa: E402
from evaluation.ascp_output_adapter import load_ascp_output, ascp_output_to_pairing_records  # noqa: E402
from validator import validate_pairing  # noqa: E402
from validation_report import aggregate_by_source  # noqa: E402


BASE, OTHER = 0, 1
FLIGHTS = {
    100: {"id": 100, "global_id": 100, "origin": BASE,  "dest": OTHER, "dep_time": 0.0, "arr_time": 2.0},
    101: {"id": 101, "global_id": 101, "origin": OTHER, "dest": BASE,  "dep_time": 3.0, "arr_time": 5.0},
    200: {"id": 200, "global_id": 200, "origin": OTHER, "dest": BASE,  "dep_time": 0.0, "arr_time": 2.0},
}
CONSTRAINT = {
    "base_airport": BASE,
    "min_conn": 0.5, "max_conn": 9.0, "min_rest": 10.0,
    "max_duty": 13.0, "max_legs": 8,
    "max_duty_periods": 2, "max_pairing_days": 5,
    "min_pairing_legs": 2,
}


def _save_sample(path, selected):
    validation_report = validate_selected_pairings(selected, FLIGHTS, CONSTRAINT, [BASE], n_total_flights=3)
    result = {
        "selected": selected, "n_pairings": len(selected), "coverage": 1.0,
        "uncoverable": 0, "deadhead_count": 0, "mip_obj": 1.0, "status": "Optimal",
        "validation_report": validation_report,
    }
    save_result_json(path, result, "checkpoints/foo/phase2_best.pt", "delta", "strict")


def test_load_ascp_output_returns_full_payload():
    selected = [{"legs": [100, 101], "source_type": "policy", "_gen_base_airport": BASE}]
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "result.json")
        _save_sample(path, selected)
        payload = load_ascp_output(path)
    assert payload["airline"] == "delta"
    assert payload["eval_mode"] == "strict"
    assert len(payload["pairings"]) == 1


def test_ascp_output_to_pairing_records_accepts_path_or_dict():
    selected = [{"legs": [100, 101], "source_type": "policy", "_gen_base_airport": BASE}]
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "result.json")
        _save_sample(path, selected)

        records_from_path = ascp_output_to_pairing_records(path)
        records_from_dict = ascp_output_to_pairing_records(load_ascp_output(path))

    assert records_from_path == records_from_dict
    assert records_from_path[0]["legs"] == [100, 101]
    assert records_from_path[0]["source_type"] == "policy"


def test_reloaded_pairing_can_be_re_validated_end_to_end():
    # 저장 -> 로드 -> validate_pairing()/aggregate_by_source() 재투입까지 전체 흐름 확인.
    # flight 200 pairing은 base 미복귀라서 재채점해도 여전히 invalid로 나와야 함.
    selected = [
        {"legs": [100, 101], "source_type": "policy", "_gen_base_airport": BASE},
        {"legs": [200],      "source_type": "salvage", "_gen_base_airport": BASE},
    ]
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "result.json")
        _save_sample(path, selected)
        records = ascp_output_to_pairing_records(path)

    result0 = validate_pairing(records[0], FLIGHTS, CONSTRAINT)
    result1 = validate_pairing(records[1], FLIGHTS, CONSTRAINT)
    assert result0["is_valid"]
    assert not result1["is_valid"]

    report = aggregate_by_source(records, FLIGHTS, constraint=CONSTRAINT, n_total_flights=3)
    assert report["policy_direct"]["invalid_count"] == 0
    assert report["salvage"]["invalid_count"] == 1


if __name__ == "__main__":
    test_load_ascp_output_returns_full_payload()
    test_ascp_output_to_pairing_records_accepts_path_or_dict()
    test_reloaded_pairing_can_be_re_validated_end_to_end()
    print("OK: 3개 테스트 통과")
