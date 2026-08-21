"""
tests/test_evaluate_ip_validation.py -- evaluate_ip.py::validate_selected_pairings()/
save_result_json()/default_save_json_path() 단위 테스트 (F1 C3)

evaluate_full()을 실제로 끝까지 돌리는 통합 테스트는 지금 안 됨(V2 full-flight master
완성 전까진 coverage<1.0 RuntimeError로 막힘) -- 그래서 각 함수를 목(mock)
pairing/flight 데이터로 독립적으로 테스트한다.
"""

import json
import os
import sys
import tempfile
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "RL"))

from evaluation.evaluate_ip import (  # noqa: E402
    validate_selected_pairings,
    save_result_json,
    default_save_json_path,
)


BASE, OTHER = 0, 1

FLIGHTS_BY_ID = {
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
BASE_IDS = [BASE]


class ValidateSelectedPairingsTests(unittest.TestCase):
    def test_all_valid_selected_pairings_report_zero_invalid(self):
        selected = [
            {"legs": [100, 101], "source_type": "policy", "_gen_base_airport": BASE},
        ]
        report = validate_selected_pairings(selected, FLIGHTS_BY_ID, CONSTRAINT, BASE_IDS, n_total_flights=3)
        self.assertEqual(report["n_invalid_selected"], 0)
        self.assertEqual(report["invalid_selected"], [])
        self.assertEqual(report["policy_direct"]["pairing_count"], 1)

    def test_invalid_selected_pairing_is_reported(self):
        # flight 200은 OTHER에서 출발해서(base 아님) BASE로 오는, INVALID_BASE_START짜리 pairing
        selected = [
            {"legs": [100, 101], "source_type": "policy", "_gen_base_airport": BASE},
            {"legs": [200],      "source_type": "salvage", "_gen_base_airport": BASE},
        ]
        report = validate_selected_pairings(selected, FLIGHTS_BY_ID, CONSTRAINT, BASE_IDS, n_total_flights=3)
        self.assertEqual(report["n_invalid_selected"], 1)
        self.assertEqual(report["invalid_selected"][0]["legs"], [200])
        self.assertIn("INVALID_BASE_START", report["invalid_selected"][0]["violation_codes"])
        self.assertEqual(report["policy_direct"]["invalid_count"], 0)
        self.assertEqual(report["salvage"]["invalid_count"], 1)

    def test_strict_mode_raises_on_invalid(self):
        selected = [{"legs": [200], "source_type": "policy", "_gen_base_airport": BASE}]
        with self.assertRaises(RuntimeError) as ctx:
            validate_selected_pairings(selected, FLIGHTS_BY_ID, CONSTRAINT, BASE_IDS,
                                        n_total_flights=3, strict=True)
        self.assertIn("strict-validation", str(ctx.exception))

    def test_pairings_grouped_by_their_own_generation_base(self):
        # chunk1은 base=BASE로 생성된 pairing, chunk2는 base=OTHER로 생성된 pairing --
        # 서로 다른 constraint로 검증돼야 둘 다 valid로 나옴 (evaluate_ip.py의 chunk별
        # base_id 랜덤 선택과 동일한 상황).
        flights = {
            **FLIGHTS_BY_ID,
            300: {"id": 300, "global_id": 300, "origin": OTHER, "dest": BASE, "dep_time": 0.0, "arr_time": 1.0},
            301: {"id": 301, "global_id": 301, "origin": BASE,  "dest": OTHER, "dep_time": 2.0, "arr_time": 3.0},
        }
        selected = [
            {"legs": [100, 101], "source_type": "policy", "_gen_base_airport": BASE},
            {"legs": [300, 301], "source_type": "policy", "_gen_base_airport": OTHER},
        ]
        report = validate_selected_pairings(selected, flights, CONSTRAINT, [BASE, OTHER], n_total_flights=5)
        self.assertEqual(report["n_invalid_selected"], 0)
        self.assertEqual(report["policy_direct"]["pairing_count"], 2)


class SaveResultJsonTests(unittest.TestCase):
    def test_round_trips_pairings_and_validation_report(self):
        selected = [{"legs": [100, 101], "source_type": "policy", "_gen_base_airport": BASE,
                     "duty_break_indices": []}]
        validation_report = validate_selected_pairings(selected, FLIGHTS_BY_ID, CONSTRAINT,
                                                        BASE_IDS, n_total_flights=3)
        result = {
            "selected": selected,
            "n_pairings": 1,
            "coverage": 1.0,
            "uncoverable": 0,
            "deadhead_count": 0,
            "mip_obj": 12.3,
            "status": "Optimal",
            "validation_report": validation_report,
        }

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "sub", "result.json")
            save_result_json(path, result, "checkpoints/foo/phase2_best.pt", "delta", "strict")

            with open(path) as f:
                saved = json.load(f)

        self.assertEqual(saved["checkpoint"], "checkpoints/foo/phase2_best.pt")
        self.assertEqual(saved["airline"], "delta")
        self.assertEqual(saved["eval_mode"], "strict")
        self.assertEqual(saved["n_pairings"], 1)
        self.assertEqual(saved["pairings"], [
            {"legs": [100, 101], "source_type": "policy", "duty_break_indices": [], "_gen_base_airport": BASE}
        ])
        self.assertEqual(saved["validation_report"]["n_invalid_selected"], 0)
        self.assertEqual(saved["validation_report"]["validator_version"], validation_report["validator_version"])


class DefaultSaveJsonPathTests(unittest.TestCase):
    def test_distinguishes_eval_mode_and_airline(self):
        strict_path = default_save_json_path("checkpoints/foo/phase2_best.pt", "delta", "strict")
        legacy_path = default_save_json_path("checkpoints/foo/phase2_best.pt", "delta", "legacy")
        other_airline_path = default_save_json_path("checkpoints/foo/phase2_best.pt", "alaska", "strict")

        self.assertTrue(strict_path.startswith("log/eval_json/"))
        self.assertTrue(strict_path.endswith(".json"))
        # 같은 checkpoint라도 mode/airline이 다르면 서로 덮어쓰지 않게 경로가 달라야 함
        self.assertNotEqual(strict_path, legacy_path)
        self.assertNotEqual(strict_path, other_airline_path)


if __name__ == "__main__":
    unittest.main()
