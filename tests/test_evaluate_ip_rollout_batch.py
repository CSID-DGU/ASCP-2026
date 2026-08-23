"""
tests/test_evaluate_ip_rollout_batch.py -- evaluation/evaluate_ip.py::
rollout_subset_global_batch() 단위 테스트 (Phase 5b, experiment/rollout-batch-vectorization)

rollout_subset_global()을 B번 순차 호출한 것과 rollout_subset_global_batch(B=B)가
동일한 global-ID pairing을 내는지(local->global 변환까지 포함) 확인한다. 진짜
FlightEncoder/PointerDecoder(랜덤 초기화)를 써서, encoder를 한 번만 부르는 최적화가
실제 신경망 앞에서도 문제없이 동작하는지까지 확인.
"""

import os
import sys
import unittest
from unittest.mock import patch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "RL"))

import torch  # noqa: E402
from model import FlightEncoder, PointerDecoder  # noqa: E402
from base_reach import build_base_reaches  # noqa: E402
from evaluation.evaluate_ip import (  # noqa: E402
    rollout_subset_global, rollout_subset_global_batch, validate_window_days,
    constraint_for_pairing_base, collect_pool_full, _attach_window_lookahead,
)


BASE, A, C = 0, 1, 2
MAX_TIME = 5 * 24.0

SUBSET = [
    {"id": 0, "local_id": 0, "global_id": 100, "origin": BASE, "dest": A,
     "dep_time": 1.0, "arr_time": 2.0},
    {"id": 1, "local_id": 1, "global_id": 101, "origin": A, "dest": C,
     "dep_time": 2.5, "arr_time": 3.5},
    {"id": 2, "local_id": 2, "global_id": 102, "origin": C, "dest": BASE,
     "dep_time": 4.0, "arr_time": 5.0},
]

CONSTRAINT = {
    "base_airport": BASE,
    "min_conn": 0.3, "max_conn": 3.0, "min_rest": 8.0,
    "max_duty": 13.0, "max_legs": 4, "max_duty_periods": 2,
    "max_pairing_days": 3, "min_pairing_legs": 1,
}


def _prepared_constraint():
    c = dict(CONSTRAINT)
    local_flights = [{**f, "id": f["local_id"]} for f in SUBSET]
    c["_base_reaches"] = build_base_reaches(local_flights, [BASE], c)
    c["_base_reach"] = c["_base_reaches"][BASE]
    return c


class RolloutSubsetGlobalBatchTests(unittest.TestCase):
    def _make_models(self, seed):
        torch.manual_seed(seed)
        encoder = FlightEncoder(n_airports=8)
        decoder = PointerDecoder()
        return encoder, decoder

    def test_batch_of_one_matches_sequential_with_global_ids(self):
        rule = _prepared_constraint()
        encoder, decoder = self._make_models(seed=0)

        torch.manual_seed(42)
        sequential = rollout_subset_global(SUBSET, rule, encoder, decoder, MAX_TIME, greedy=True)

        torch.manual_seed(42)
        batched = rollout_subset_global_batch(
            SUBSET, rule, encoder, decoder, MAX_TIME, B=1, greedy=True
        )

        self.assertEqual(len(batched), 1)
        self.assertEqual(batched[0], sequential)
        # global ID로 제대로 변환됐는지도 같이 확인 (local 0/1/2가 아니라 100/101/102)
        for pairing in batched[0]:
            self.assertTrue(all(leg >= 100 for leg in pairing["legs"]))

    def test_batch_greater_than_one_all_valid_global_ids(self):
        rule = _prepared_constraint()
        encoder, decoder = self._make_models(seed=1)

        torch.manual_seed(7)
        results = rollout_subset_global_batch(
            SUBSET, rule, encoder, decoder, MAX_TIME, B=4, greedy=False
        )
        self.assertEqual(len(results), 4)
        valid_global_ids = {100, 101, 102}
        for episode_pairings in results:
            for pairing in episode_pairings:
                self.assertTrue(set(pairing["legs"]).issubset(valid_global_ids))
                self.assertTrue(pairing["is_legal"])


class WindowContractTests(unittest.TestCase):
    def test_lookahead_preserves_unique_universe_and_cross_boundary_time(self):
        core_windows = [
            [{"id": 0, "origin": BASE, "dest": A, "dep_time": 119.0, "arr_time": 120.0}],
            [{"id": 0, "origin": A, "dest": BASE, "dep_time": 1.0, "arr_time": 2.0}],
        ]
        windows, n_total = _attach_window_lookahead(
            core_windows, window_days=5, lookahead_days=1,
        )
        self.assertEqual(n_total, 2)
        self.assertEqual([f["global_id"] for f in windows[0]], [0, 1])
        self.assertEqual(windows[0][1]["dep_time"], 121.0)
        self.assertFalse(windows[0][1]["_is_core"])
        self.assertEqual([f["global_id"] for f in windows[1]], [1])
        self.assertTrue(windows[1][0]["_is_core"])

    def test_jetblue_short_window_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "max_pairing_days"):
            validate_window_days(5, {"max_pairing_days": 7}, "jetblue")

    def test_window_equal_to_pairing_limit_is_allowed(self):
        self.assertEqual(
            validate_window_days(7, {"max_pairing_days": 7}, "jetblue"), 7
        )

    @patch("evaluation.evaluate_ip.rollout_subset_global_batch")
    def test_pool_uses_checkpoint_model_max_time(self, rollout_batch):
        rollout_batch.return_value = [[]]
        flights = [[{**SUBSET[0], "global_id": 0}]]
        collect_pool_full(
            flights, [BASE], CONSTRAINT, object(), object(),
            n_rollouts_per_chunk=1, subset_size=10,
            window_days=5, model_max_time=192,
        )
        self.assertTrue(rollout_batch.called)
        self.assertTrue(all(call.args[4] == 192.0 for call in rollout_batch.call_args_list))


class PairingBaseContractTests(unittest.TestCase):
    def test_rotated_configured_base_is_used_for_validation(self):
        constraint = {**CONSTRAINT, "base_ids": [0, 3]}
        resolved = constraint_for_pairing_base(
            {"true_start_airport": 3}, constraint
        )
        self.assertEqual(resolved["base_airport"], 3)

    def test_unconfigured_start_base_is_rejected(self):
        constraint = {**CONSTRAINT, "base_ids": [0, 3]}
        self.assertIsNone(
            constraint_for_pairing_base({"true_start_airport": 7}, constraint)
        )


if __name__ == "__main__":
    unittest.main()
