import sys
import unittest
from unittest.mock import patch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "RL"))

from experiments import train


def rule(**updates):
    value = {
        "base_airport": 0, "min_conn": 0.5, "max_conn": 4.0,
        "min_rest": 8.0, "max_duty": 14.0, "max_legs": 4,
        "max_duty_periods": 2, "max_pairing_days": 2,
        "min_pairing_legs": 2, "pairing_cost": 5.0,
        "base_penalty": 500.0, "uncovered_penalty": 10.0,
    }
    value.update(updates)
    return value


class NeverCalledDecoder:
    def __call__(self, *args, **kwargs):
        raise AssertionError("all-zero strict 상태에서 decoder가 호출되면 안 됨")


class StrictTrainingTest(unittest.TestCase):
    def setUp(self):
        self.flights = [
            {"id": 0, "origin": 0, "dest": 1, "dep_time": 1.0, "arr_time": 2.0}
        ]

    def test_training_constraint_always_builds_cpp_reachability(self):
        prepared = train._prepare_cpp_constraint(self.flights, rule())
        self.assertIn("_base_reach", prepared)
        self.assertEqual(prepared["_base_reach_base"], 0)

    def test_prepared_constraint_reuses_reachability(self):
        prepared = train._prepare_cpp_constraint(self.flights, rule())
        with patch.object(train, "build_base_reaches", side_effect=AssertionError("rebuild")):
            reused = train._prepare_cpp_constraint(self.flights, prepared)
        self.assertIs(reused["_base_reach"], prepared["_base_reach"])

    def test_legacy_flag_cannot_disable_cpp_training(self):
        legacy_flag = rule(require_base_return=False, strict_base_start=False)
        prepared = train._prepare_cpp_constraint(self.flights, legacy_flag)
        self.assertIn("_base_reach", prepared)
        _, _, _, metrics = train.run_episode(
            self.flights, legacy_flag, None, NeverCalledDecoder(), None, greedy=True
        )
        self.assertEqual(metrics["n_zero_mask"], 1)
        self.assertEqual(metrics["n_uncovered"], 1)

    def test_stage_episode_stops_instead_of_arbitrary_restart(self):
        _, _, _, metrics = train.run_episode(
            self.flights, rule(), None, NeverCalledDecoder(), None, greedy=True
        )
        self.assertEqual(metrics["n_zero_mask"], 1)
        self.assertEqual(metrics["n_uncovered"], 1)
        self.assertEqual(metrics["n_deadheads"], 0)

    def test_dual_episode_uses_same_strict_stop(self):
        _, _, _, metrics = train.run_episode_with_dual(
            self.flights, rule(), None, NeverCalledDecoder(), None, {}, greedy=True
        )
        self.assertEqual(metrics["n_zero_mask"], 1)
        self.assertEqual(metrics["n_uncovered"], 1)

    def test_phase2_pool_drops_doomed_partial_pairing(self):
        pairings = train._rollout_with_pairings(
            self.flights, rule(), None, NeverCalledDecoder(), None, greedy=True
        )
        self.assertEqual(pairings, [])


    def test_turkish_constraint_builds_reachability_for_both_home_bases(self):
        flights = [
            {"id": 0, "origin": 0, "dest": 2, "dep_time": 1.0, "arr_time": 2.0},
            {"id": 1, "origin": 2, "dest": 1, "dep_time": 3.0, "arr_time": 4.0},
        ]
        prepared = train._prepare_cpp_constraint(
            flights,
            rule(
                base_airport=0, base_ids=[0, 1],
                allow_cross_base_return=True,
            ),
        )
        self.assertEqual(set(prepared["_base_reaches"]), {0, 1})
        self.assertIs(prepared["_base_reach"], prepared["_base_reaches"][0])

if __name__ == "__main__":
    unittest.main()
