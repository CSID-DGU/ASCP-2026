"""V2 full-flight master 단위 테스트."""

import unittest

from evaluation.full_flight_master import FullFlightInputError, solve_full_flight_master, validate_master_inputs


def _column(column_id="p0", legs=None, **overrides):
    value = {
        "column_id": column_id,
        "legs": [10, 20] if legs is None else legs,
        "cost": 3.0,
        "source_type": "policy",
        "is_legal": True,
    }
    value.update(overrides)
    return value


class InputContractTests(unittest.TestCase):
    def test_accepts_non_contiguous_global_ids(self):
        columns, universe = validate_master_inputs([_column()], [10, 20, 30])
        self.assertEqual(universe, (10, 20, 30))
        self.assertEqual(columns[0]["column_id"], "p0")

    def test_rejects_invalid_columns(self):
        cases = [
            (_column(legs=[10, 10]), [10], "중복 flight"),
            (_column(legs=[10, 99]), [10], "universe 밖"),
            (_column(source_type="unknown"), [10, 20], "source_type"),
            (_column(is_legal=False), [10, 20], "is_legal"),
            (_column(cost=float("inf")), [10, 20], "유한값"),
        ]
        for column, universe, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(FullFlightInputError, message):
                    validate_master_inputs([column], universe)

    def test_rejects_duplicate_universe_and_column_ids(self):
        with self.assertRaisesRegex(FullFlightInputError, "중복 ID"):
            validate_master_inputs([], [10, 10])
        with self.assertRaisesRegex(FullFlightInputError, "중복 column_id"):
            validate_master_inputs([_column(), _column()], [10, 20])




class FullUniverseConstraintTests(unittest.TestCase):
    def test_known_optimum_covers_non_contiguous_universe(self):
        columns = [
            _column("p0", [10, 20], cost=3),
            _column("p1", [30], cost=2),
            _column("p2", [10, 20, 30], cost=10),
        ]
        result = solve_full_flight_master(columns, [10, 20, 30])
        self.assertEqual(result["status"], "Optimal")
        self.assertEqual(result["selected_column_ids"], ["p0", "p1"])
        self.assertEqual(result["coverage"], 1.0)
        self.assertEqual(result["mip_objective"], 5.0)

    def test_candidate_missing_flight_is_infeasible_not_silently_deleted(self):
        result = solve_full_flight_master([_column(legs=[10])], [10, 20])
        self.assertEqual(result["status"], "Infeasible")
        self.assertFalse(result["is_feasible"])
        self.assertEqual(result["uncovered_flight_ids"], [10, 20])

    def test_empty_universe_is_trivially_feasible(self):
        result = solve_full_flight_master([], [])
        self.assertEqual(result["status"], "Empty")
        self.assertEqual(result["coverage"], 1.0)


class ArtificialSlackTests(unittest.TestCase):
    def test_artificial_completes_only_missing_flight(self):
        result = solve_full_flight_master(
            [_column(legs=[10])], [10, 20],
            allow_artificial=True, artificial_penalty=100,
        )
        self.assertEqual(result["status"], "Optimal")
        self.assertEqual(result["covered_flight_ids"], [10])
        self.assertEqual(result["artificial_flight_ids"], [20])
        self.assertEqual(result["coverage"], 0.5)
        self.assertEqual(result["completion_coverage"], 1.0)
        self.assertEqual(result["artificial_cost"], 100)
        self.assertEqual(result["mip_objective"], 103)

    def test_empty_pool_uses_one_artificial_per_flight(self):
        result = solve_full_flight_master(
            [], [10, 20], allow_artificial=True, artificial_penalty=7,
        )
        self.assertEqual(result["artificial_flight_ids"], [10, 20])
        self.assertEqual(result["artificial_count"], 2)
        self.assertEqual(result["mip_objective"], 14)

    def test_legal_column_wins_when_cheaper_than_artificial(self):
        result = solve_full_flight_master(
            [_column(legs=[10], cost=5)], [10],
            allow_artificial=True, artificial_penalty=100,
        )
        self.assertEqual(result["selected_column_ids"], ["p0"])
        self.assertEqual(result["artificial_count"], 0)




class OperationalCompletionTests(unittest.TestCase):
    def test_reposition_precedes_reserve_and_artificial_by_cost(self):
        result = solve_full_flight_master(
            [], [10], allow_reposition=True, allow_reserve=True,
            allow_artificial=True, reposition_penalty=10,
            reserve_penalty=20, artificial_penalty=100,
        )
        self.assertEqual(result["reposition_flight_ids"], [10])
        self.assertEqual(result["reserve_flight_ids"], [])
        self.assertEqual(result["artificial_flight_ids"], [])
        self.assertEqual(result["operational_completion_coverage"], 1.0)

    def test_target_restriction_forces_reserve(self):
        result = solve_full_flight_master(
            [], [10, 20], allow_reposition=True, allow_reserve=True,
            reposition_flight_ids=[10], reserve_flight_ids=[20],
            reposition_penalty=10, reserve_penalty=20,
        )
        self.assertEqual(result["reposition_flight_ids"], [10])
        self.assertEqual(result["reserve_flight_ids"], [20])
        self.assertEqual(result["completion_coverage"], 1.0)

    def test_operational_column_is_disabled_without_flag(self):
        operational = _column("r0", [10], cost=1, source_type="reposition", is_legal=False)
        disabled = solve_full_flight_master([operational], [10])
        enabled = solve_full_flight_master([operational], [10], allow_reposition=True)
        self.assertFalse(disabled["is_feasible"])
        self.assertEqual(enabled["selected_column_ids"], ["r0"])
        self.assertEqual(enabled["coverage"], 0.0)
        self.assertEqual(enabled["operational_completion_coverage"], 1.0)


if __name__ == "__main__":
    unittest.main()
