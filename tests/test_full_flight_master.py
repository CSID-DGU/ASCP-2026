"""V2 full-flight master 단위 테스트."""

import unittest
from unittest.mock import patch

from evaluation.full_flight_master import FullFlightInputError, calibrate_completion_penalties, solve_full_flight_master, validate_master_inputs
from evaluation.completion_runner import merge_rescue_columns, solve_completion_stages


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
            reposition_flight_ids=[10], reserve_flight_ids=[10],
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




class SourceAwareObjectiveTests(unittest.TestCase):
    def test_rescue_is_reported_separately_and_beats_artificial(self):
        rescue = _column("rescue-10", [10], cost=12, source_type="rescue")
        result = solve_full_flight_master(
            [rescue], [10], allow_artificial=True, artificial_penalty=100,
        )
        self.assertEqual(result["selected_count_by_source"]["rescue"], 1)
        self.assertEqual(result["selected_count_by_source"]["policy"], 0)
        self.assertEqual(result["selected_cost_by_source"]["rescue"], 12)
        self.assertEqual(result["artificial_count"], 0)

    def test_objective_breakdown_matches_solver_objective(self):
        columns = [
            _column("p0", [10, 20], cost=3),
            _column("p1", [20, 30], cost=4),
        ]
        result = solve_full_flight_master(columns, [10, 20, 30], lambda_excess=5)
        self.assertAlmostEqual(sum(result["objective_breakdown"].values()), result["mip_objective"])
        self.assertEqual(result["excess_flight_ids"], [20])




class CompletionStageTests(unittest.TestCase):
    def test_stage_candidate_coverage_is_monotonic(self):
        columns = [
            _column("policy", [10], source_type="policy"),
            _column("salvage", [20], source_type="salvage"),
            _column("rescue", [30], source_type="rescue"),
        ]
        stages = solve_completion_stages(
            columns, [10, 20, 30, 40],
            reposition_flight_ids=[], reserve_flight_ids=[], artificial_penalty=100,
        )
        self.assertEqual([stage["stage"] for stage in stages], ["policy", "salvage", "rescue", "operational", "artificial"])
        self.assertEqual([stage["candidate_coverage"] for stage in stages], [0.25, 0.5, 0.75, 0.75, 0.75])
        self.assertEqual(stages[-1]["completion_coverage"], 1.0)
        self.assertEqual(stages[-1]["artificial_flight_ids"], [40])

    def test_rescue_is_not_visible_before_rescue_stage(self):
        columns = [
            _column("policy", [10], source_type="policy"),
            _column("rescue", [20], source_type="rescue"),
        ]
        stages = solve_completion_stages(columns, [10, 20], artificial_penalty=100)
        self.assertEqual(stages[0]["candidate_uncovered_flight_ids"], [20])
        self.assertEqual(stages[2]["candidate_uncovered_flight_ids"], [])




class RescueInterfaceTests(unittest.TestCase):
    def test_rescue_contract_merges_and_deduplicates(self):
        policy = _column("p", [10], source_type="policy")
        rescue = _column(
            "r", [20], source_type="rescue", repair_target_flights=[20],
            validator_version="0.1.0", constraint_hash="abc",
        )
        duplicate = dict(rescue, column_id="r-duplicate")
        merged = merge_rescue_columns([policy], [rescue, duplicate], [10, 20])
        self.assertEqual([column["column_id"] for column in merged], ["p", "r"])

    def test_rescue_requires_validator_provenance(self):
        rescue = _column("r", [20], source_type="rescue", repair_target_flights=[20])
        with self.assertRaisesRegex(FullFlightInputError, "validator_version"):
            merge_rescue_columns([], [rescue], [20])

    def test_rescue_target_must_be_present_in_legs(self):
        rescue = _column("r", [20], source_type="rescue", repair_target_flights=[30])
        with self.assertRaisesRegex(FullFlightInputError, "target flight"):
            merge_rescue_columns([], [rescue], [20, 30])




class PenaltyCalibrationTests(unittest.TestCase):
    def test_penalties_follow_pool_cost_scale_and_strict_order(self):
        penalties = calibrate_completion_penalties([_column(cost=25)])
        self.assertEqual(penalties["reposition_penalty"], 50)
        self.assertEqual(penalties["reserve_penalty"], 100)
        self.assertEqual(penalties["artificial_penalty"], 200)

    def test_auto_penalties_are_recorded_in_result(self):
        result = solve_full_flight_master([], [10], allow_artificial=True)
        self.assertEqual(result["penalties"]["artificial_penalty"], 8)
        self.assertEqual(result["artificial_cost"], 8)



class SolverPerformanceTests(unittest.TestCase):
    def test_structural_infeasibility_skips_model_construction(self):
        with patch("evaluation.full_flight_master.pulp.LpProblem") as problem_mock:
            result = solve_full_flight_master([_column(legs=[10])], [10, 20])
        problem_mock.assert_not_called()
        self.assertTrue(result["solve_skipped"])
        self.assertEqual(result["structural_infeasible_flight_ids"], [20])

    def test_time_limited_incumbent_is_feasible_not_optimal(self):
        def fake_solve(problem, _solver):
            problem.status = 1
            problem.sol_status = 2
            return 1

        with patch("evaluation.full_flight_master.pulp.LpProblem.solve", new=fake_solve):
            result = solve_full_flight_master(
                [_column(legs=[10])], [10], allow_artificial=True
            )
        self.assertEqual(result["status"], "Feasible")
        self.assertTrue(result["is_feasible"])
        self.assertFalse(result["is_optimal"])
        self.assertEqual(result["pulp_status"], "Optimal")
        self.assertEqual(result["pulp_solution_status"], "Solution Found")

    def test_threads_are_forwarded_to_cbc(self):
        with patch("evaluation.full_flight_master.pulp.PULP_CBC_CMD") as solver_mock:
            from evaluation.full_flight_master import _solver
            _solver(60, False, False, threads=8)
        solver_mock.assert_called_once_with(
            timeLimit=60, threads=8, warmStart=True, msg=0
        )

    def test_threads_and_warm_start_are_forwarded_to_gurobi(self):
        with patch("evaluation.full_flight_master.pulp.GUROBI") as solver_mock:
            from evaluation.full_flight_master import _solver
            _solver(60, True, False, threads=8)
        solver_mock.assert_called_once_with(
            timeLimit=60, threads=8, warmStart=True, msg=0
        )

    def test_artificial_stage_has_trivial_warm_start(self):
        captured = {}

        def fake_solve(problem, _solver):
            captured.update({variable.name: variable.varValue for variable in problem.variables()})
            problem.status = 1
            problem.sol_status = 1
            return 1

        with patch("evaluation.full_flight_master.pulp.LpProblem.solve", new=fake_solve):
            solve_full_flight_master(
                [_column(legs=[10])], [10], allow_artificial=True
            )
        self.assertEqual(captured["x_0"], 0)
        self.assertEqual(captured["artificial_10"], 1)
        self.assertEqual(captured["excess_10"], 0)

    def test_gurobi_time_limit_with_incumbent_is_feasible(self):
        class NativeModel:
            Status = 9
            SolCount = 1

        def fake_solve(problem, _solver):
            problem.status = 0
            problem.sol_status = 0
            problem.solverModel = NativeModel()
            return 0

        with patch("evaluation.full_flight_master.pulp.LpProblem.solve", new=fake_solve):
            result = solve_full_flight_master(
                [_column(legs=[10])], [10], allow_artificial=True, use_gurobi=True
            )
        self.assertEqual(result["status"], "Feasible")
        self.assertTrue(result["is_feasible"])
        self.assertFalse(result["is_optimal"])
        self.assertEqual(result["gurobi_status"], 9)
        self.assertEqual(result["gurobi_solution_count"], 1)

    def test_rejects_non_positive_threads(self):
        with self.assertRaisesRegex(FullFlightInputError, "threads"):
            solve_full_flight_master([_column(legs=[10])], [10], threads=0)


if __name__ == "__main__":
    unittest.main()
