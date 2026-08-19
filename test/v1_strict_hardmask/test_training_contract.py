import sys
import unittest
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

    def test_training_constraint_enables_strict_by_default(self):
        prepared = train._prepare_training_constraint(self.flights, rule())
        self.assertTrue(prepared["require_base_return"])
        self.assertTrue(prepared["strict_base_start"])
        self.assertIn("_base_reach", prepared)

    def test_explicit_legacy_mode_is_preserved(self):
        prepared = train._prepare_training_constraint(
            self.flights, rule(require_base_return=False)
        )
        self.assertFalse(prepared["require_base_return"])
        self.assertNotIn("_base_reach", prepared)

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


if __name__ == "__main__":
    unittest.main()
