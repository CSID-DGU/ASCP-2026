"""
tests/test_rollout_batch_regression.py -- Phase 0 기준선 (experiment/rollout-batch-vectorization)

RL/rollout.py::rollout_with_pairings()를 실제 학습된 모델 없이 결정론적으로 돌리기
위해, test/v1_strict_hardmask/test_rollout_contract.py의 GreedyLegalDecoder 패턴을
그대로 재사용한다(항상 mask에서 첫 번째로 가능한 flight를 고르는 더미 decoder --
state_to_vec/flight_gap_bias는 0벡터로 patch해서 신경망 없이도 완전히 결정론적임).

이 파일의 목적은 "지금(벡터화 전) 순차 rollout이 이 시나리오에서 정확히 어떤
pairing을 만드는가"를 고정해두는 것 -- Phase 1~5(state_to_vec/get_mask/step/
rollout 메인 루프 배치화)를 진행하면서 이 테스트가 계속 통과해야, CPP 판정이나
pairing 조립 로직이 조용히 달라지지 않았다는 걸 확인할 수 있다.
"""

import os
import sys
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "RL"))

import torch  # noqa: E402
import rollout  # noqa: E402
from base_reach import build_base_reach  # noqa: E402


class _DummyLayer:
    weight = torch.zeros((1, 78))


class _GreedyLegalDecoder:
    """mask에서 첫 번째로 가능한 flight를 고르고, 없으면 EndDuty, 그것도 없으면
    EndPairing을 고르는 완전 결정론적 더미 decoder (신경망 학습 상태 불필요)."""

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


BASE, A, C = 0, 1, 2

FLIGHTS = [
    {"id": 0, "origin": BASE, "dest": A,    "dep_time": 1.0,  "arr_time": 2.0},
    {"id": 1, "origin": A,    "dest": C,    "dep_time": 2.5,  "arr_time": 3.5},
    {"id": 2, "origin": C,    "dest": BASE, "dep_time": 4.0,  "arr_time": 5.0},
    {"id": 3, "origin": BASE, "dest": A,    "dep_time": 20.0, "arr_time": 21.0},
    {"id": 4, "origin": A,    "dest": BASE, "dep_time": 22.0, "arr_time": 23.0},
]

CONSTRAINT = {
    "base_airport": BASE, "base_ids": [BASE],
    "min_conn": 0.3, "max_conn": 3.0, "min_rest": 8.0,
    "max_duty": 14.0, "max_legs": 6, "max_duty_periods": 2,
    "max_pairing_days": 3, "min_pairing_legs": 1,
}

# Phase 0 기준선 -- 순차(현재) rollout_with_pairings()가 위 시나리오에서 실제로
# 만들어낸 결과를 그대로 고정. (legs, cost, source_type, n_duties)만 비교 대상으로 삼음
# -- 그 외 필드(fly/elapsed/duty_break_indices 등)는 legs/n_duties가 같으면 항상
# 같이 결정되므로 굳이 다 박아둘 필요 없음.
EXPECTED_BASELINE = [
    {"legs": [0, 1, 2], "cost": 2.0, "source_type": "policy", "n_duties": 1},
    {"legs": [3, 4],    "cost": 3.5, "source_type": "policy", "n_duties": 1},
]


def _run_deterministic_rollout():
    rule = dict(CONSTRAINT)
    rule["_base_reach"] = build_base_reach(FLIGHTS, BASE, rule)

    old_state_to_vec = rollout.state_to_vec
    old_gap_bias = rollout.flight_gap_bias
    rollout.state_to_vec = lambda *a, **k: torch.zeros(78)
    rollout.flight_gap_bias = lambda *a, **k: torch.zeros(len(FLIGHTS) + 2)
    try:
        return rollout.rollout_with_pairings(
            FLIGHTS, rule, None, _GreedyLegalDecoder(), None, greedy=True
        )
    finally:
        rollout.state_to_vec = old_state_to_vec
        rollout.flight_gap_bias = old_gap_bias


class Phase0BaselineRegressionTests(unittest.TestCase):
    def test_deterministic_rollout_matches_frozen_baseline(self):
        pairings = _run_deterministic_rollout()
        actual = [
            {"legs": p["legs"], "cost": p["cost"],
             "source_type": p["source_type"], "n_duties": p["n_duties"]}
            for p in pairings
        ]
        self.assertEqual(actual, EXPECTED_BASELINE)


if __name__ == "__main__":
    unittest.main()
