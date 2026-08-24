import json
import tempfile
import unittest
from pathlib import Path

from evaluation.evaluate_ip import solve_pool_completion, validate_rescue_columns_current_run
from evaluation.validator import constraint_hash, VALIDATOR_VERSION


class EvaluateIpCompletionTests(unittest.TestCase):
    def test_pool_completion_returns_legacy_compatible_and_v2_fields(self):
        pool = [{
            "column_id": "policy-0", "legs": [0], "cost": 2.0,
            "source_type": "policy", "is_legal": True,
        }]
        result = solve_pool_completion(pool, 2, artificial_penalty=100)
        self.assertEqual(result["coverage"], 0.5)
        self.assertEqual(result["completion_coverage"], 1.0)
        self.assertEqual(result["reposition_flight_ids"], [])
        self.assertEqual(result["artificial_flight_ids"], [1])
        self.assertEqual(result["uncoverable"], 0)
        self.assertEqual(result["mip_obj"], result["mip_objective"])
        operational = result["completion_report"]["stages"][3]
        self.assertFalse(operational["operational_stage_has_inputs"])
        self.assertTrue(operational["solve_reused"])

    def test_auto_operational_is_explicit_opt_in(self):
        pool = [{
            "column_id": "policy-0", "legs": [0], "cost": 2.0,
            "source_type": "policy", "is_legal": True,
        }]
        result = solve_pool_completion(
            pool, 2, artificial_penalty=100, auto_operational=True,
        )
        self.assertEqual(result["reposition_flight_ids"], [1])
        self.assertEqual(result["artificial_flight_ids"], [])
        operational = result["completion_report"]["stages"][3]
        self.assertTrue(operational["operational_stage_has_inputs"])
        self.assertTrue(operational["auto_operational"])

    def test_report_is_saved_under_requested_result_path(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result" / "completion.json"
            solve_pool_completion([], 2, artificial_penalty=7, report_path=output)
            loaded = json.loads(output.read_text(encoding="utf-8"))
        operational = loaded["stages"][3]
        self.assertFalse(operational["operational_stage_has_inputs"])
        self.assertTrue(operational["solve_reused"])
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


class RescueCurrentRunValidationTests(unittest.TestCase):
    def setUp(self):
        self.flights = {
            0: {"id": 0, "origin": 10, "dest": 20, "dep_time": 1.0, "arr_time": 2.0},
            1: {"id": 1, "origin": 20, "dest": 10, "dep_time": 3.0, "arr_time": 4.0},
        }
        self.constraint = {
            "base_airport": 10, "min_conn": 0.5, "max_conn": 3.0,
            "min_rest": 10.0, "max_duty": 13.0, "max_legs": 8,
            "max_duty_periods": 2, "max_pairing_days": 5,
            "min_pairing_legs": 2,
        }

    def candidate(self):
        return {
            "legs": [0, 1], "source_type": "rescue", "is_legal": True,
            "cost": 1.0, "repair_target_flights": [1],
            "validator_version": VALIDATOR_VERSION,
            "constraint_hash": constraint_hash(self.constraint),
        }

    def test_valid_rescue_is_rechecked(self):
        result = validate_rescue_columns_current_run(
            [self.candidate()], self.flights, self.constraint, [10]
        )
        self.assertEqual(result[0]["_gen_base_airport"], 10)

    def test_stale_constraint_hash_is_rejected(self):
        candidate = self.candidate()
        candidate["constraint_hash"] = "stale"
        with self.assertRaisesRegex(ValueError, "constraint_hash"):
            validate_rescue_columns_current_run(
                [candidate], self.flights, self.constraint, [10]
            )

    def test_non_crew_base_rescue_is_rejected(self):
        flights = {
            0: {"id": 0, "origin": 20, "dest": 30, "dep_time": 1.0, "arr_time": 2.0},
            1: {"id": 1, "origin": 30, "dest": 20, "dep_time": 3.0, "arr_time": 4.0},
        }
        candidate = self.candidate()
        candidate["constraint_hash"] = constraint_hash(
            {**self.constraint, "base_airport": 20}
        )
        with self.assertRaisesRegex(ValueError, "configured crew base"):
            validate_rescue_columns_current_run(
                [candidate], flights, self.constraint, [10]
            )


if __name__ == "__main__":
    unittest.main()
