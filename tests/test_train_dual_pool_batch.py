"""
tests/test_train_dual_pool_batch.py -- experiments/train.py::_rollout_batch_dual_pool()
단위 테스트 (Phase 5, experiment/rollout-batch-vectorization)

train.py의 dual pool 수집 로직(_rollout_with_pairings(), salvage_doomed 없는 단순
버전)을 배치화한 _rollout_batch_dual_pool()이 예전 순차 버전과 정확히 같은 pairing을
내는지, _collect_pool()이 여전히 정상 동작하는지 확인한다.
"""

import os
import sys
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "RL"))
sys.path.insert(0, os.path.join(REPO_ROOT, "experiments"))
sys.path.insert(0, os.path.join(REPO_ROOT, "evaluation"))

import torch  # noqa: E402
import experiments.train as train  # noqa: E402
from base_reach import build_base_reaches  # noqa: E402
from validator import validate_pairing  # noqa: E402


class Phase2DualNormalizationTests(unittest.TestCase):
    def test_net_signal_is_bounded_and_preserves_relative_values(self):
        signal = train.normalize_phase2_dual_signal(
            {0: 1000.0, 1: 500.0}, {0: 100.0, 1: 600.0}, mode="net"
        )
        self.assertEqual(signal[0], 1.0)
        self.assertLess(signal[1], 0.0)
        self.assertTrue(all(-1.0 <= value <= 1.0 for value in signal.values()))

    def test_coverage_only_ignores_excess_dual(self):
        signal = train.normalize_phase2_dual_signal(
            {0: 1000.0, 1: 250.0}, {0: 999.0, 1: 999.0},
            mode="coverage_only",
        )
        self.assertEqual(signal, {0: 1.0, 1: 0.25})

    def test_zero_keeps_universe_and_removes_signal(self):
        signal = train.normalize_phase2_dual_signal(
            {0: 10.0, 1: 20.0}, {0: 1.0}, mode="zero"
        )
        self.assertEqual(signal, {0: 0.0, 1: 0.0})

    def test_uncovered_only_marks_only_artificial_flights(self):
        signal = train.normalize_phase2_dual_signal(
            {0: 10.0, 1: 20.0, 2: 30.0}, mode="uncovered-only",
            uncovered_flight_ids=[1],
        )
        self.assertEqual(signal, {0: 0.0, 1: 1.0, 2: 0.0})

    def test_shuffled_preserves_values_but_changes_mapping(self):
        real = train.normalize_phase2_dual_signal(
            {0: 10.0, 1: 5.0, 2: 1.0}, mode="real"
        )
        shuffled = train.normalize_phase2_dual_signal(
            {0: 10.0, 1: 5.0, 2: 1.0}, mode="shuffled", shuffle_seed=1
        )
        self.assertEqual(sorted(real.values()), sorted(shuffled.values()))
        self.assertNotEqual(real, shuffled)


class _DummyLayer:
    weight = torch.zeros((1, 78))


class _GreedyLegalDecoder:
    """train.py의 _rollout_with_pairings()(비배치, scalar mask)용 결정론적 더미."""
    state_mlp = [_DummyLayer()]

    def __call__(self, encoded, state_vec, mask, gap_bias=None):
        probs = torch.zeros_like(mask)
        flight_count = len(mask) - 2
        feasible = torch.nonzero(mask[:flight_count], as_tuple=False)
        if len(feasible):
            probs[feasible[0].item()] = 1.0
        elif mask[-1] > 0:
            probs[-1] = 1.0
        else:
            probs[-2] = 1.0
        return probs


class _BatchGreedyLegalDecoder:
    """_rollout_batch_dual_pool()(배치, 2D mask)용 결정론적 더미."""
    state_mlp = [_DummyLayer()]

    def __call__(self, encoded, state_vec, mask, gap_bias=None):
        B, total = mask.shape
        flight_count = total - 2
        probs = torch.zeros_like(mask)
        for b in range(B):
            row = mask[b]
            feasible = torch.nonzero(row[:flight_count], as_tuple=False)
            if len(feasible):
                probs[b, feasible[0].item()] = 1.0
            elif row[-1] > 0:
                probs[b, -1] = 1.0
            else:
                probs[b, -2] = 1.0
        return probs


BASE, A, C = 0, 1, 2

FLIGHTS = [
    {"id": 0, "origin": BASE, "dest": A,    "dep_time": 1.0,  "arr_time": 2.0},
    {"id": 1, "origin": A,    "dest": C,    "dep_time": 2.5,  "arr_time": 3.5},
    {"id": 2, "origin": C,    "dest": BASE, "dep_time": 4.0,  "arr_time": 5.0},
    {"id": 3, "origin": BASE, "dest": A,    "dep_time": 20.0, "arr_time": 21.0},
    {"id": 4, "origin": A,    "dest": BASE, "dep_time": 22.0, "arr_time": 23.0},
    {"id": 5, "origin": BASE, "dest": C,    "dep_time": 30.0, "arr_time": 31.5},
    {"id": 6, "origin": C,    "dest": A,    "dep_time": 32.0, "arr_time": 33.0},
    {"id": 7, "origin": A,    "dest": BASE, "dep_time": 33.5, "arr_time": 34.5},
]

CONSTRAINT = {
    "base_airport": BASE,
    "min_conn": 0.3, "max_conn": 3.0, "min_rest": 8.0,
    "max_duty": 13.0, "max_legs": 4, "max_duty_periods": 2,
    "max_pairing_days": 3, "min_pairing_legs": 1,
}


def _prepared_constraint():
    c = dict(CONSTRAINT)
    c["_base_reaches"] = build_base_reaches(FLIGHTS, [BASE], c)
    c["_base_reach"] = c["_base_reaches"][BASE]
    return c


class DualPoolBatchMatchesSequentialTests(unittest.TestCase):
    def test_greedy_batch_of_one_matches_sequential(self):
        rule = _prepared_constraint()

        old_stv, old_gb = train.state_to_vec, train.flight_gap_bias
        train.state_to_vec = lambda *a, **k: torch.zeros(78)
        train.flight_gap_bias = lambda *a, **k: torch.zeros(len(FLIGHTS) + 2)
        try:
            sequential = train._rollout_with_pairings(
                FLIGHTS, rule, None, _GreedyLegalDecoder(), None, greedy=True
            )
        finally:
            train.state_to_vec, train.flight_gap_bias = old_stv, old_gb

        old_stv_b, old_gb_b = train.state_to_vec_batch, train.flight_gap_bias_batch
        train.state_to_vec_batch = lambda states, *a, **k: torch.zeros(len(states), 78)
        train.flight_gap_bias_batch = lambda states, flights, *a, **k: torch.zeros(
            len(states), len(flights) + 2
        )
        try:
            batched = train._rollout_batch_dual_pool(
                FLIGHTS, rule, None, _BatchGreedyLegalDecoder(), None, B=1, greedy=True
            )
        finally:
            train.state_to_vec_batch, train.flight_gap_bias_batch = old_stv_b, old_gb_b

        self.assertEqual(len(batched), 1)
        self.assertEqual(batched[0], sequential)

    def test_stochastic_batch_pairings_all_valid(self):
        rule = _prepared_constraint()
        flights_by_id = {f["id"]: f for f in FLIGHTS}

        old_stv_b, old_gb_b = train.state_to_vec_batch, train.flight_gap_bias_batch
        train.state_to_vec_batch = lambda states, *a, **k: torch.zeros(len(states), 78)
        train.flight_gap_bias_batch = lambda states, flights, *a, **k: torch.zeros(
            len(states), len(flights) + 2
        )
        torch.manual_seed(0)

        class _RandomLegalDecoder:
            state_mlp = [_DummyLayer()]

            def __call__(self, encoded, state_vec, mask, gap_bias=None):
                probs = mask.clone().float()
                return probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-9)

        try:
            results = train._rollout_batch_dual_pool(
                FLIGHTS, rule, None, _RandomLegalDecoder(), None, B=10, greedy=False
            )
        finally:
            train.state_to_vec_batch, train.flight_gap_bias_batch = old_stv_b, old_gb_b

        self.assertEqual(len(results), 10)
        n_pairings = 0
        for episode in results:
            for p in episode:
                n_pairings += 1
                self.assertTrue(p["ends_at_base"])
                check = validate_pairing({"legs": p["legs"]}, flights_by_id, rule)
                self.assertTrue(
                    check["is_valid"],
                    f"invalid pairing {p['legs']}: {check['violation_codes']}",
                )
        self.assertGreater(n_pairings, 0)


class CollectPoolStillWorksTests(unittest.TestCase):
    def test_collect_pool_produces_deduplicated_legal_pool(self):
        rule = dict(CONSTRAINT)  # _collect_pool()이 자체적으로 _prepare_cpp_constraint 호출
        flights_by_id = {f["id"]: f for f in FLIGHTS}

        old_stv_b, old_gb_b = train.state_to_vec_batch, train.flight_gap_bias_batch
        train.state_to_vec_batch = lambda states, *a, **k: torch.zeros(len(states), 78)
        train.flight_gap_bias_batch = lambda states, flights, *a, **k: torch.zeros(
            len(states), len(flights) + 2
        )
        torch.manual_seed(1)

        class _RandomLegalDecoder:
            state_mlp = [_DummyLayer()]

            def __call__(self, encoded, state_vec, mask, gap_bias=None):
                probs = mask.clone().float()
                return probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-9)

        try:
            pool = train._collect_pool(FLIGHTS, rule, None, _RandomLegalDecoder(), None, n_rollouts=8)
        finally:
            train.state_to_vec_batch, train.flight_gap_bias_batch = old_stv_b, old_gb_b

        self.assertGreater(len(pool), 0)
        seen_keys = set()
        for p in pool:
            key = tuple(sorted(p["legs"]))
            self.assertNotIn(key, seen_keys)  # 중복 제거됐는지
            seen_keys.add(key)
            check = validate_pairing({"legs": p["legs"]}, flights_by_id, {**rule, **{
                "_base_reaches": build_base_reaches(FLIGHTS, [BASE], rule),
            }})
            self.assertTrue(check["is_valid"], f"invalid pooled pairing {p['legs']}")


if __name__ == "__main__":
    unittest.main()
