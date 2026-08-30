import os
import sys
import unittest


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "RL"))

from evaluation.llm_scoring import score_submitted_pairing  # noqa: E402


CONSTRAINT = {
    "base_airport": 0,
    "min_conn": 0.65,
    "max_conn": 12.0,
    "max_legs": 8,
    "min_rest": 10.0,
    "max_duty_periods": 2,
    "max_pairing_days": 5,
    "min_pairing_legs": 3,
    "max_duty": 13.0,
}


class LlmScoringTests(unittest.TestCase):
    def test_uses_hour_scale_ip_cost(self):
        flights = {
            1: {"id": 1, "origin": 0, "dest": 1, "dep_time": 0.0, "arr_time": 1.0},
            2: {"id": 2, "origin": 1, "dest": 2, "dep_time": 2.0, "arr_time": 3.0},
            3: {"id": 3, "origin": 2, "dest": 0, "dep_time": 4.0, "arr_time": 5.0},
        }
        scored, validation = score_submitted_pairing(
            {"legs": [1, 2, 3], "source_type": "policy"},
            flights,
            CONSTRAINT,
        )
        self.assertTrue(validation["is_valid"])
        self.assertAlmostEqual(scored["dead_time"], 2.0)
        self.assertAlmostEqual(scored["cost"], 3.0)

    def test_cost_includes_rest_time_above_minimum(self):
        flights = {
            1: {"id": 1, "origin": 0, "dest": 1, "dep_time": 0.0, "arr_time": 1.0},
            2: {"id": 2, "origin": 1, "dest": 2, "dep_time": 2.0, "arr_time": 3.0},
            3: {"id": 3, "origin": 2, "dest": 0, "dep_time": 18.0, "arr_time": 19.0},
        }
        scored, validation = score_submitted_pairing(
            {"legs": [1, 2, 3], "source_type": "policy"},
            flights,
            CONSTRAINT,
        )
        self.assertTrue(validation["is_valid"])
        self.assertAlmostEqual(scored["dead_time"], 6.0)
        self.assertAlmostEqual(scored["intra_duty_gap"], 1.0)
        self.assertAlmostEqual(scored["inter_duty_excess"], 5.0)
        self.assertAlmostEqual(scored["cost"], 7.0)

    def test_returns_no_column_for_illegal_pairing(self):
        flights = {
            1: {"id": 1, "origin": 0, "dest": 1, "dep_time": 0.0, "arr_time": 1.0},
            2: {"id": 2, "origin": 1, "dest": 2, "dep_time": 2.0, "arr_time": 3.0},
            3: {"id": 3, "origin": 2, "dest": 3, "dep_time": 4.0, "arr_time": 5.0},
        }
        scored, validation = score_submitted_pairing(
            {"legs": [1, 2, 3], "source_type": "policy"},
            flights,
            CONSTRAINT,
        )
        self.assertIsNone(scored)
        self.assertIn("BASE_RETURN_FAILURE", validation["violation_codes"])


if __name__ == "__main__":
    unittest.main()
