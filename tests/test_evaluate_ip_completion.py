import json
import tempfile
import unittest
from pathlib import Path

from evaluation.evaluate_ip import solve_pool_completion


class EvaluateIpCompletionTests(unittest.TestCase):
    def test_pool_completion_returns_legacy_compatible_and_v2_fields(self):
        pool = [{
            "column_id": "policy-0", "legs": [0], "cost": 2.0,
            "source_type": "policy", "is_legal": True,
        }]
        result = solve_pool_completion(pool, 2, artificial_penalty=100)
        self.assertEqual(result["coverage"], 0.5)
        self.assertEqual(result["completion_coverage"], 1.0)
        self.assertEqual(result["artificial_flight_ids"], [1])
        self.assertEqual(result["uncoverable"], 0)
        self.assertEqual(result["mip_obj"], result["mip_objective"])

    def test_report_is_saved_under_requested_result_path(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result" / "completion.json"
            solve_pool_completion([], 2, artificial_penalty=7, report_path=output)
            loaded = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(loaded["artificial_count"], 2)
        self.assertEqual(loaded["completion_coverage"], 1.0)



    def test_rescue_columns_are_merged_before_staged_solve(self):
        policy = {"column_id": "p", "legs": [0], "cost": 2, "source_type": "policy", "is_legal": True}
        rescue = {
            "column_id": "r", "legs": [1], "cost": 3,
            "source_type": "rescue", "is_legal": True,
            "repair_target_flights": [1], "validator_version": "0.1.0", "constraint_hash": "fixture",
        }
        result = solve_pool_completion([policy], 2, rescue_columns=[rescue], artificial_penalty=100)
        self.assertEqual(result["completion_report"]["post_rescue_candidate_coverage"], 1.0)
        self.assertEqual(result["artificial_count"], 0)


if __name__ == "__main__":
    unittest.main()
