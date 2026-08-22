"""
tests/test_rollout_batch_real.py -- RL/rollout.py::rollout_batch()의 실제 배치
구현 단위 테스트 (Phase 4, experiment/rollout-batch-vectorization)

핵심 검증: B=1(greedy)일 때 Phase 0 기준선(tests/test_rollout_batch_regression.py)과
정확히 같은 pairing이 나오는가, B>1일 때 나온 모든 pairing이 validate_pairing()을
통과하는가, Turkish HB1/HB2 교차 base 회전이 배치 안에서 섞여도 올바른가.
"""

import os
import sys
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "RL"))
sys.path.insert(0, os.path.join(REPO_ROOT, "evaluation"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402
import rollout  # noqa: E402
from base_reach import build_base_reach, build_base_reaches  # noqa: E402
from validator import validate_pairing  # noqa: E402

from test_rollout_batch_regression import (  # noqa: E402
    FLIGHTS as BASELINE_FLIGHTS,
    CONSTRAINT as BASELINE_CONSTRAINT,
    EXPECTED_BASELINE,
)


class _DummyLayer:
    weight = torch.zeros((1, 78))


class _BatchGreedyLegalDecoder:
    """rollout_batch()용 결정론적 더미 decoder -- 각 행(episode)마다 mask에서
    첫 번째로 가능한 flight를 고르고, 없으면 EndDuty, 그것도 없으면 EndPairing
    (test/v1_strict_hardmask/test_rollout_contract.py의 GreedyLegalDecoder를
    배치(2D mask) 버전으로 확장한 것)."""

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


def _patch_batch_vec_fns(n_flights):
    old_stv = rollout.state_to_vec_batch
    old_gb = rollout.flight_gap_bias_batch
    rollout.state_to_vec_batch = lambda states, *a, **k: torch.zeros(len(states), 78)
    rollout.flight_gap_bias_batch = lambda states, flights, *a, **k: torch.zeros(
        len(states), n_flights + 2
    )
    return old_stv, old_gb


def _unpatch_batch_vec_fns(old_stv, old_gb):
    rollout.state_to_vec_batch = old_stv
    rollout.flight_gap_bias_batch = old_gb


class RolloutBatchMatchesBaselineTests(unittest.TestCase):
    def test_b1_greedy_matches_phase0_baseline_exactly(self):
        rule = dict(BASELINE_CONSTRAINT)
        rule["_base_reach"] = build_base_reach(BASELINE_FLIGHTS, rule["base_airport"], rule)

        old_stv, old_gb = _patch_batch_vec_fns(len(BASELINE_FLIGHTS))
        try:
            results = rollout.rollout_batch(
                BASELINE_FLIGHTS, rule, None, _BatchGreedyLegalDecoder(), None,
                B=1, greedy=True,
            )
        finally:
            _unpatch_batch_vec_fns(old_stv, old_gb)

        self.assertEqual(len(results), 1)
        actual = [
            {"legs": p["legs"], "cost": p["cost"],
             "source_type": p["source_type"], "n_duties": p["n_duties"]}
            for p in results[0]
        ]
        self.assertEqual(actual, EXPECTED_BASELINE)


BASE, A, C = 0, 1, 2

RICH_FLIGHTS = [
    {"id": 0, "origin": BASE, "dest": A,    "dep_time": 1.0,  "arr_time": 2.0},
    {"id": 1, "origin": A,    "dest": C,    "dep_time": 2.5,  "arr_time": 3.5},
    {"id": 2, "origin": C,    "dest": BASE, "dep_time": 4.0,  "arr_time": 5.0},
    {"id": 3, "origin": BASE, "dest": A,    "dep_time": 20.0, "arr_time": 21.0},
    {"id": 4, "origin": A,    "dest": BASE, "dep_time": 22.0, "arr_time": 23.0},
    {"id": 5, "origin": BASE, "dest": C,    "dep_time": 30.0, "arr_time": 31.5},
    {"id": 6, "origin": C,    "dest": A,    "dep_time": 32.0, "arr_time": 33.0},
    {"id": 7, "origin": A,    "dest": BASE, "dep_time": 33.5, "arr_time": 34.5},
]

RICH_CONSTRAINT = {
    "base_airport": BASE,
    "min_conn": 0.3, "max_conn": 3.0, "min_rest": 8.0,
    "max_duty": 13.0, "max_legs": 4, "max_duty_periods": 2,
    "max_pairing_days": 3, "min_pairing_legs": 1,
}


class RolloutBatchStochasticLegalityTests(unittest.TestCase):
    def test_all_pairings_pass_independent_validator(self):
        rule = dict(RICH_CONSTRAINT)
        rule["_base_reach"] = build_base_reach(RICH_FLIGHTS, BASE, rule)
        flights_by_id = {f["id"]: f for f in RICH_FLIGHTS}

        old_stv, old_gb = _patch_batch_vec_fns(len(RICH_FLIGHTS))
        try:
            results = rollout.rollout_batch(
                RICH_FLIGHTS, rule, None, _BatchGreedyLegalDecoder(), None,
                B=5, greedy=True,
            )
        finally:
            _unpatch_batch_vec_fns(old_stv, old_gb)

        self.assertEqual(len(results), 5)
        for episode_pairings in results:
            for p in episode_pairings:
                if not p.get("ends_at_base", True):
                    continue
                check = validate_pairing({"legs": p["legs"]}, flights_by_id, rule)
                self.assertTrue(
                    check["is_valid"],
                    f"invalid pairing {p['legs']}: {check['violation_codes']}",
                )


HB1, HB2 = 0, 1

TURKISH_FLIGHTS = [
    {"id": 0, "origin": HB1, "dest": A,   "dep_time": 1.0,  "arr_time": 2.0},
    {"id": 1, "origin": A,   "dest": HB2, "dep_time": 2.5,  "arr_time": 3.5},
    {"id": 2, "origin": HB2, "dest": A,   "dep_time": 4.0,  "arr_time": 5.0},
    {"id": 3, "origin": A,   "dest": HB1, "dep_time": 5.5,  "arr_time": 6.5},
    {"id": 4, "origin": HB2, "dest": A,   "dep_time": 20.0, "arr_time": 21.0},
    {"id": 5, "origin": A,   "dest": HB1, "dep_time": 21.5, "arr_time": 22.5},
]


class RolloutBatchTurkishCrossBaseTests(unittest.TestCase):
    def test_mixed_episode_bases_all_legal(self):
        # HB1과 HB2 둘 다에서 출발하는 flight가 섞여 있어서, 배치 안 episode들이
        # 서로 다른 episode_base로 시작/회전할 수 있는 시나리오.
        rule = {
            "base_airport": HB1, "base_ids": [HB1, HB2], "allow_cross_base_return": True,
            "min_conn": 0.3, "max_conn": 3.0, "min_rest": 8.0,
            "max_legs": 4, "max_duty_periods": 2,
            "max_pairing_days": 3, "min_pairing_legs": 1,
        }
        rule["_base_reaches"] = build_base_reaches(TURKISH_FLIGHTS, [HB1, HB2], rule)
        rule["_base_reach"] = rule["_base_reaches"][HB1]
        flights_by_id = {f["id"]: f for f in TURKISH_FLIGHTS}

        rollout.set_environment("turkish")
        old_stv, old_gb = _patch_batch_vec_fns(len(TURKISH_FLIGHTS))
        try:
            results = rollout.rollout_batch(
                TURKISH_FLIGHTS, rule, None, _BatchGreedyLegalDecoder(), None,
                B=4, greedy=True,
            )
        finally:
            _unpatch_batch_vec_fns(old_stv, old_gb)
            rollout.set_environment("delta")

        self.assertEqual(len(results), 4)
        for episode_pairings in results:
            for p in episode_pairings:
                if not p.get("ends_at_base", True):
                    continue
                # Turkish는 pairing마다 실제 시작 base(true_start_airport)가 다를 수
                # 있음(HB1 또는 HB2로 회전) -- 실제 파이프라인(evaluate_ip.py::
                # validate_selected_pairings())도 pairing이 생성된 base로 constraint를
                # 바꿔서 검증하므로 여기서도 동일하게 맞춰줌.
                pairing_rule = {**rule, "base_airport": p["true_start_airport"]}
                check = validate_pairing({"legs": p["legs"]}, flights_by_id, pairing_rule)
                self.assertTrue(
                    check["is_valid"],
                    f"invalid pairing {p['legs']} (start={p['true_start_airport']}): "
                    f"{check['violation_codes']}",
                )


if __name__ == "__main__":
    unittest.main()
