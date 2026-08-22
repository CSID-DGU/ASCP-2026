"""
tests/test_utils_batch.py -- RL/utils.py::state_to_vec_batch()/flight_gap_bias_batch()
단위 테스트 (Phase 1, experiment/rollout-batch-vectorization)

두 함수 모두 "B개 state를 한 번에 처리한 결과"가 "state 하나씩 개별 호출해서 쌓은
결과"와 정확히 같아야 한다 -- 이게 다르면 배치화 과정에서 의미가 바뀐 것이므로,
값 하나라도 어긋나면 실패해야 하는 엄격한 동등성 테스트.
"""

import os
import sys
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "RL"))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from utils import state_to_vec, state_to_vec_batch, flight_gap_bias, flight_gap_bias_batch  # noqa: E402


class _DummyEncoder:
    def __init__(self, n_airports=8, emb_dim=32):
        torch.manual_seed(0)
        self.airport_emb = nn.Embedding(n_airports, emb_dim)


CONSTRAINT = {
    "base_airport": 0,
    "max_duty": 13.0, "min_conn": 0.65, "max_conn": 9.0, "max_legs": 8,
    "min_rest": 10.0, "max_duty_periods": 2, "max_pairing_days": 5,
    "min_pairing_legs": 2,
}

FLIGHTS = [
    {"id": 0, "origin": 0, "dest": 1, "dep_time": 1.0, "arr_time": 2.0},
    {"id": 1, "origin": 1, "dest": 2, "dep_time": 3.0, "arr_time": 4.0},
    {"id": 2, "origin": 2, "dest": 0, "dep_time": 6.0, "arr_time": 7.0},
]

STATES = [
    {"current_airport": 0, "current_time": 1.0, "legs": 0, "duty_period": 0,
     "pairing_start": True, "is_resting": False, "total_legs": 0},
    {"current_airport": 1, "current_time": 4.5, "legs": 2, "duty_period": 0,
     "pairing_start": False, "is_resting": False, "duty_start_time": 1.0, "total_legs": 2},
    {"current_airport": 2, "current_time": 20.0, "legs": 0, "duty_period": 1,
     "pairing_start": False, "is_resting": True, "rest_end_time": 24.0, "total_legs": 3},
    {"current_airport": 0, "current_time": 30.0, "legs": 1, "duty_period": 1,
     "pairing_start": False, "is_resting": False, "duty_start_time": 28.0, "total_legs": 4},
]


class StateToVecBatchTests(unittest.TestCase):
    def test_matches_individual_calls_stacked(self):
        encoder = _DummyEncoder()
        expected = torch.stack([state_to_vec(s, encoder, CONSTRAINT) for s in STATES])
        actual = state_to_vec_batch(STATES, encoder, CONSTRAINT)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6))

    def test_batch_of_one_matches_single_call(self):
        encoder = _DummyEncoder()
        expected = state_to_vec(STATES[0], encoder, CONSTRAINT).unsqueeze(0)
        actual = state_to_vec_batch([STATES[0]], encoder, CONSTRAINT)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6))

    def test_include_total_legs_false_matches(self):
        encoder = _DummyEncoder()
        expected = torch.stack([
            state_to_vec(s, encoder, CONSTRAINT, include_total_legs=False) for s in STATES
        ])
        actual = state_to_vec_batch(STATES, encoder, CONSTRAINT, include_total_legs=False)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6))


class FlightGapBiasBatchTests(unittest.TestCase):
    def test_matches_individual_calls_stacked(self):
        expected = torch.stack([flight_gap_bias(s, FLIGHTS, CONSTRAINT) for s in STATES])
        actual = flight_gap_bias_batch(STATES, FLIGHTS, CONSTRAINT)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6))

    def test_batch_of_one_matches_single_call(self):
        expected = flight_gap_bias(STATES[1], FLIGHTS, CONSTRAINT).unsqueeze(0)
        actual = flight_gap_bias_batch([STATES[1]], FLIGHTS, CONSTRAINT)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6))

    def test_empty_batch_returns_correct_shape(self):
        actual = flight_gap_bias_batch([], FLIGHTS, CONSTRAINT)
        self.assertEqual(tuple(actual.shape), (0, len(FLIGHTS) + 2))

    def test_resting_and_pairing_start_rows_are_zero(self):
        actual = flight_gap_bias_batch(STATES, FLIGHTS, CONSTRAINT)
        self.assertTrue(torch.all(actual[0] == 0.0))  # pairing_start
        self.assertTrue(torch.all(actual[2] == 0.0))  # is_resting


if __name__ == "__main__":
    unittest.main()
