"""
tests/test_candidate_schema.py -- completion/candidate_schema.py 단위 테스트 (F3/V2)

로컬 스키마 검증뿐 아니라, evaluation/full_flight_master.py::validate_master_inputs()와
evaluation/completion_runner.py::merge_rescue_columns()가 실제로 이 candidate를
받아주는지까지 확인한다(계약을 문서로만 확인하지 않고 실제 코드로 재확인).
"""

import os
import sys
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "RL"))

from completion.candidate_schema import make_rescue_candidate, RescueCandidateError  # noqa: E402
from evaluation.full_flight_master import validate_master_inputs, FullFlightInputError  # noqa: E402
from evaluation.completion_runner import merge_rescue_columns  # noqa: E402


class MakeRescueCandidateTests(unittest.TestCase):
    def test_builds_expected_fields(self):
        c = make_rescue_candidate(
            legs=[100, 101], repair_target_flights=[101], cost=3.5,
            validator_version="0.1.0", constraint_hash="abc123",
        )
        self.assertEqual(c["legs"], [100, 101])
        self.assertEqual(c["source_type"], "rescue")
        self.assertTrue(c["is_legal"])
        self.assertEqual(c["cost"], 3.5)
        self.assertEqual(c["repair_target_flights"], [101])
        self.assertEqual(c["validator_version"], "0.1.0")
        self.assertEqual(c["constraint_hash"], "abc123")
        self.assertNotIn("column_id", c)

    def test_optional_column_id(self):
        c = make_rescue_candidate(
            legs=[100], repair_target_flights=[100], cost=1.0,
            validator_version="0.1.0", constraint_hash="abc123",
            column_id="rescue-0",
        )
        self.assertEqual(c["column_id"], "rescue-0")

    def test_rejects_empty_legs(self):
        with self.assertRaises(RescueCandidateError):
            make_rescue_candidate([], [1], 1.0, "0.1.0", "abc")

    def test_rejects_duplicate_legs(self):
        with self.assertRaises(RescueCandidateError):
            make_rescue_candidate([1, 1], [1], 1.0, "0.1.0", "abc")

    def test_rejects_empty_repair_targets(self):
        with self.assertRaises(RescueCandidateError):
            make_rescue_candidate([1, 2], [], 1.0, "0.1.0", "abc")

    def test_rejects_target_not_subset_of_legs(self):
        with self.assertRaises(RescueCandidateError):
            make_rescue_candidate([1, 2], [3], 1.0, "0.1.0", "abc")

    def test_rejects_negative_cost(self):
        with self.assertRaises(RescueCandidateError):
            make_rescue_candidate([1, 2], [1], -1.0, "0.1.0", "abc")

    def test_rejects_infinite_cost(self):
        with self.assertRaises(RescueCandidateError):
            make_rescue_candidate([1, 2], [1], float("inf"), "0.1.0", "abc")

    def test_rejects_missing_validator_version(self):
        with self.assertRaises(RescueCandidateError):
            make_rescue_candidate([1, 2], [1], 1.0, "", "abc")

    def test_rejects_missing_constraint_hash(self):
        with self.assertRaises(RescueCandidateError):
            make_rescue_candidate([1, 2], [1], 1.0, "0.1.0", "")


class HerinContractTests(unittest.TestCase):
    """evaluation/full_flight_master.py + completion_runner.py 실제 코드로 재확인."""

    def test_passes_validate_master_inputs(self):
        candidate = make_rescue_candidate(
            legs=[100, 101], repair_target_flights=[101], cost=3.5,
            validator_version="0.1.0", constraint_hash="abc123",
        )
        # 예외 없이 통과해야 함
        validate_master_inputs([candidate], all_flight_ids=[100, 101])

    def test_merge_rescue_columns_accepts_candidate(self):
        existing = [{
            "column_id": "policy-0", "legs": [100, 101], "source_type": "policy",
            "is_legal": True, "cost": 1.0,
        }]
        rescue = make_rescue_candidate(
            legs=[200], repair_target_flights=[200], cost=5.0,
            validator_version="0.1.0", constraint_hash="def456",
        )
        merged = merge_rescue_columns(existing, [rescue], all_flight_ids=[100, 101, 200])
        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[1]["column_id"], "rescue-0")

    def test_merge_rejects_candidate_missing_target_subset(self):
        # merge_rescue_columns 쪽 방어도 재확인 -- candidate_schema를 우회해서
        # 억지로 만든 불량 dict를 넣었을 때도 혜린 코드가 거부하는지 확인.
        bad = {
            "column_id": "rescue-bad", "legs": [100], "source_type": "rescue",
            "is_legal": True, "cost": 1.0, "repair_target_flights": [999],
            "validator_version": "0.1.0", "constraint_hash": "abc",
        }
        with self.assertRaises(FullFlightInputError):
            merge_rescue_columns([], [bad], all_flight_ids=[100, 999])


if __name__ == "__main__":
    unittest.main()
