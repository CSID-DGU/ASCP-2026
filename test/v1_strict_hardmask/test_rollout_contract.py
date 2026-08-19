import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "RL"))

import rollout
from base_reach import build_base_reach


class DummyLayer:
    weight = torch.zeros((1, 78))


class GreedyLegalDecoder:
    state_mlp = [DummyLayer()]

    def __call__(self, encoded, state_vec, mask, gap_bias=None):
        probs = torch.zeros_like(mask)
        flight_count = len(mask) - 2
        feasible_flights = torch.nonzero(mask[:flight_count], as_tuple=False)
        if len(feasible_flights):
            probs[feasible_flights[0].item()] = 1.0
        elif mask[-1] > 0:
            probs[-1] = 1.0
        else:
            probs[-2] = 1.0
        return probs


def strict_fixture():
    flights = [
        {"id": 0, "origin": 0, "dest": 1, "dep_time": 1.0, "arr_time": 2.0},
        {"id": 1, "origin": 1, "dest": 0, "dep_time": 3.0, "arr_time": 4.0},
    ]
    rule = {
        "base_airport": 0, "base_ids": [0],
        "min_conn": 0.5, "max_conn": 4.0, "min_rest": 8.0,
        "max_duty": 14.0, "max_legs": 4, "max_duty_periods": 2,
        "max_pairing_days": 2, "min_pairing_legs": 2,
    }
    rule["_base_reach"] = build_base_reach(flights, 0, rule)
    return flights, rule


class StrictRolloutTest(unittest.TestCase):
    def test_single_rollout_returns_only_base_to_base_pairing(self):
        flights, rule = strict_fixture()
        old_state_to_vec = rollout.state_to_vec
        old_gap_bias = rollout.flight_gap_bias
        rollout.state_to_vec = lambda *args, **kwargs: torch.zeros(78)
        rollout.flight_gap_bias = lambda *args, **kwargs: torch.zeros(len(flights) + 2)
        try:
            with patch.object(rollout, "build_base_reach", side_effect=AssertionError("cache miss")):
                pairings = rollout.rollout_with_pairings(
                    flights, rule, None, GreedyLegalDecoder(), None, greedy=True
                )
        finally:
            rollout.state_to_vec = old_state_to_vec
            rollout.flight_gap_bias = old_gap_bias

        self.assertEqual(len(pairings), 1)
        self.assertEqual(pairings[0]["legs"], [0, 1])
        self.assertTrue(pairings[0]["ends_at_base"])
        self.assertEqual(pairings[0]["true_start_airport"], 0)

    def test_batch_rollout_preserves_cpp_contract(self):
        flights, rule = strict_fixture()
        old_state_to_vec = rollout.state_to_vec
        old_gap_bias = rollout.flight_gap_bias
        rollout.state_to_vec = lambda *args, **kwargs: torch.zeros(78)
        rollout.flight_gap_bias = lambda *args, **kwargs: torch.zeros(len(flights) + 2)
        try:
            results = rollout.rollout_batch(
                flights, rule, None, GreedyLegalDecoder(), None, B=2, greedy=True
            )
        finally:
            rollout.state_to_vec = old_state_to_vec
            rollout.flight_gap_bias = old_gap_bias
        self.assertEqual(len(results), 2)
        self.assertTrue(all(len(items) == 1 for items in results))
        self.assertTrue(all(items[0]["ends_at_base"] for items in results))


    def test_all_zero_at_base_does_not_emit_short_pairing(self):
        flights = [
            {"id": 0, "origin": 0, "dest": 0, "dep_time": 1.0, "arr_time": 2.0},
        ]
        rule = {
            "base_airport": 0, "base_ids": [0],
            "min_conn": 0.5, "max_conn": 4.0, "min_rest": 8.0,
            "max_duty": 14.0, "max_legs": 4, "max_duty_periods": 0,
            "max_pairing_days": 2, "min_pairing_legs": 2,
        }
        rule["_base_reach"] = build_base_reach(flights, 0, rule)
        pairings = rollout.rollout_with_pairings(
            flights, rule, None, GreedyLegalDecoder(), None, greedy=True
        )
        self.assertEqual(pairings, [])

if __name__ == "__main__":
    unittest.main()
