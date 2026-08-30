import os
import sys
import unittest


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "RL"))

from evaluation.llm_direct import (  # noqa: E402
    build_legacy_forced_100,
    evaluate_direct_solution,
)


CONSTRAINT = {
    "base_airport": 0,
    "min_conn": 0.5,
    "max_conn": 5.0,
    "max_legs": 8,
    "min_rest": 10.0,
    "max_duty_periods": 2,
    "max_pairing_days": 5,
    "min_pairing_legs": 3,
    "max_duty": 13.0,
}
FLIGHTS = {
    1: {"id": 1, "origin": 0, "dest": 1, "dep_time": 0.0, "arr_time": 1.0},
    2: {"id": 2, "origin": 1, "dest": 2, "dep_time": 2.0, "arr_time": 3.0},
    3: {"id": 3, "origin": 2, "dest": 0, "dep_time": 4.0, "arr_time": 5.0},
    4: {"id": 4, "origin": 0, "dest": 2, "dep_time": 6.0, "arr_time": 7.0},
    5: {"id": 5, "origin": 2, "dest": 1, "dep_time": 8.0, "arr_time": 9.0},
    6: {"id": 6, "origin": 1, "dest": 0, "dep_time": 10.0, "arr_time": 11.0},
}
AIRPORT_MAP = {"ATL": 0, "A": 1, "B": 2}


def record(number, legs, base="ATL"):
    return {
        "pairing_number": number,
        "declared_base": base,
        "legs": legs,
        "source_type": "policy",
    }


class DirectEvaluationTests(unittest.TestCase):
    def evaluate(self, records, uncovered):
        return evaluate_direct_solution(
            records,
            uncovered,
            FLIGHTS,
            CONSTRAINT,
            [0],
            AIRPORT_MAP,
        )

    def test_accepts_exact_legal_partition_without_rewriting_it(self):
        submitted = [record(1, [1, 2, 3]), record(2, [4, 5, 6])]
        result = self.evaluate(submitted, [])
        self.assertTrue(result["solution_feasible"])
        self.assertEqual(
            [pairing["legs"] for pairing in result["legal_pairings"]],
            [[1, 2, 3], [4, 5, 6]],
        )
        self.assertEqual(result["legal_union_coverage"], 1.0)
        self.assertIsNotNone(result["solution_metrics"])
        self.assertFalse(result["optimizer_applied"])
        self.assertFalse(result["rescue_applied"])

    def test_keeps_declared_uncovered_uncovered(self):
        result = self.evaluate([record(1, [1, 2, 3])], [4, 5, 6])
        self.assertFalse(result["solution_feasible"])
        self.assertEqual(result["legal_union_coverage"], 0.5)
        self.assertEqual(result["legally_uncovered_flight_ids"], [4, 5, 6])
        self.assertEqual(len(result["legal_pairings"]), 1)
        self.assertIsNone(result["solution_metrics"])

    def test_reports_duplicate_without_selecting_a_better_subset(self):
        submitted = [record(1, [1, 2, 3]), record(2, [1, 2, 3])]
        result = self.evaluate(submitted, [4, 5, 6])
        self.assertFalse(result["solution_feasible"])
        self.assertEqual(result["duplicate_flight_ids"], [1, 2, 3])
        self.assertEqual(result["conflict_free_legal_coverage"], 0.0)
        self.assertEqual(len(result["legal_pairings"]), 2)

    def test_invalid_pairing_is_not_repaired_or_counted_as_legal(self):
        result = self.evaluate([record(1, [1, 2, 4])], [3, 5, 6])
        self.assertFalse(result["solution_feasible"])
        self.assertEqual(result["n_invalid_pairings"], 1)
        self.assertEqual(result["legal_union_coverage"], 0.0)
        self.assertEqual(result["legally_uncovered_flight_ids"], [1, 2, 3, 4, 5, 6])

    def test_reports_undeclared_and_unknown_flights(self):
        result = self.evaluate([record(1, [1, 2, 999])], [])
        self.assertFalse(result["solution_feasible"])
        self.assertEqual(result["unknown_flight_ids"], [999])
        self.assertEqual(result["undeclared_flight_ids"], [3, 4, 5, 6])

    def test_declared_base_mismatch_invalidates_pairing(self):
        result = self.evaluate([record(1, [1, 2, 3], base="A")], [4, 5, 6])
        self.assertEqual(result["n_invalid_pairings"], 1)
        self.assertIn(
            "DECLARED_BASE_MISMATCH",
            result["invalid_pairings"][0]["violation_codes"],
        )

    def test_forced_100_is_always_a_separate_non_primary_view(self):
        direct = self.evaluate([record(1, [1, 2, 3])], [4, 5, 6])
        forced = build_legacy_forced_100(direct, FLIGHTS)
        self.assertFalse(direct["solution_feasible"])
        self.assertEqual(direct["legal_union_coverage"], 0.5)
        self.assertEqual(forced["synthetic_completion_coverage"], 1.0)
        self.assertEqual(forced["forced_flight_ids"], [4, 5, 6])
        self.assertFalse(forced["is_legal_solution"])
        self.assertFalse(forced["use_as_primary_result"])

    def test_forced_100_keeps_previous_duplicate_policy(self):
        direct = self.evaluate(
            [record(1, [1, 2, 3]), record(2, [1, 2, 3])],
            [4, 5, 6],
        )
        forced = build_legacy_forced_100(direct, FLIGHTS)
        self.assertEqual(forced["duplicate_legal_flight_ids"], [1, 2, 3])
        self.assertEqual(forced["legacy_valid_pairing_count"], 0)
        self.assertEqual(forced["forced_flight_ids"], [1, 2, 3, 4, 5, 6])


if __name__ == "__main__":
    unittest.main()
