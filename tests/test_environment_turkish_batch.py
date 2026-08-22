"""
tests/test_environment_turkish_batch.py -- RL/turkish/environment_turkish.py::
get_mask_batch() 단위 테스트 (Phase 2b, experiment/rollout-batch-vectorization)

tests/test_environment_batch.py와 같은 방식(무작위 (state, assigned) 조합 전수
비교)이지만, HB1/HB2 교차 base 복귀(allow_cross_base_return)까지 커버함 --
시작(pairing_start)은 단일 base_ap로 고정, 복귀(EndPairing)는 base_ids
중 아무 곳이나 가능해야 한다.
"""

import os
import random
import sys
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "RL"))

from turkish.environment_turkish import get_mask, get_mask_batch  # noqa: E402
from base_reach import build_base_reaches  # noqa: E402


HB1, HB2, A, C = 0, 1, 2, 3

FLIGHTS = [
    {"id": 0, "origin": HB1, "dest": A,   "dep_time": 1.0,  "arr_time": 2.0},
    {"id": 1, "origin": A,   "dest": HB2, "dep_time": 2.5,  "arr_time": 3.5},
    {"id": 2, "origin": HB2, "dest": HB1, "dep_time": 4.0,  "arr_time": 5.0},
    {"id": 3, "origin": HB1, "dest": C,   "dep_time": 20.0, "arr_time": 21.0},
    {"id": 4, "origin": C,   "dest": HB2, "dep_time": 22.0, "arr_time": 23.0},
    {"id": 5, "origin": HB2, "dest": A,   "dep_time": 30.0, "arr_time": 31.5},
    {"id": 6, "origin": A,   "dest": HB1, "dep_time": 32.0, "arr_time": 33.0},
    {"id": 7, "origin": HB1, "dest": HB2, "dep_time": 33.5, "arr_time": 34.5},
]


def _make_constraint(max_duty=None):
    c = {
        "base_airport": HB1, "base_ids": [HB1, HB2], "allow_cross_base_return": True,
        "min_conn": 0.3, "max_conn": 3.0, "min_rest": 8.0,
        "max_legs": 4, "max_duty_periods": 2,
        "max_pairing_days": 3, "min_pairing_legs": 1,
    }
    if max_duty is not None:
        c["max_duty"] = max_duty
    c["_base_reaches"] = build_base_reaches(FLIGHTS, c["base_ids"], c)
    return c


def _random_state(rng):
    pairing_start = rng.random() < 0.3
    is_resting = (not pairing_start) and rng.random() < 0.3
    current_time = rng.uniform(0.0, 36.0)
    return {
        "current_airport": rng.choice([HB1, HB2, A, C]),
        "current_time": current_time,
        "pairing_start": pairing_start,
        "is_resting": is_resting,
        "rest_end_time": current_time + rng.uniform(0.0, 5.0) if is_resting else None,
        "duty_period": rng.choice([0, 1, 2]),
        "duty_start_time": current_time - rng.uniform(0.0, 5.0),
        "pairing_start_time": current_time - rng.uniform(0.0, 10.0),
        "legs": rng.choice([0, 1, 2, 3]),
        "total_legs": rng.choice([0, 1, 2, 3, 4]),
    }


def _random_assigned(rng):
    return {f["id"]: rng.random() < 0.4 for f in FLIGHTS}


class GetMaskBatchTurkishEquivalenceTests(unittest.TestCase):
    def _check(self, constraint, stage, n_samples=200, seed=0):
        rng = random.Random(seed)
        states = [_random_state(rng) for _ in range(n_samples)]
        assigneds = [_random_assigned(rng) for _ in range(n_samples)]

        expected = [
            get_mask(s, FLIGHTS, a, constraint, stage=stage)
            for s, a in zip(states, assigneds)
        ]
        actual = get_mask_batch(states, FLIGHTS, assigneds, constraint, stage=stage)
        self.assertEqual(actual, expected)

    def test_stage3_custom_max_duty(self):
        self._check(_make_constraint(max_duty=13.0), stage=3)

    def test_stage1_disallows_end_duty(self):
        self._check(_make_constraint(max_duty=13.0), stage=1)

    def test_faa_duty_table_fallback_when_max_duty_missing(self):
        self._check(_make_constraint(max_duty=None), stage=3)

    def test_batch_of_one_matches_single_call(self):
        constraint = _make_constraint(max_duty=13.0)
        rng = random.Random(42)
        state = _random_state(rng)
        assigned = _random_assigned(rng)
        expected = [get_mask(state, FLIGHTS, assigned, constraint, stage=3)]
        actual = get_mask_batch([state], FLIGHTS, [assigned], constraint, stage=3)
        self.assertEqual(actual, expected)

    def test_empty_batch_returns_empty_list(self):
        constraint = _make_constraint(max_duty=13.0)
        actual = get_mask_batch([], FLIGHTS, [], constraint, stage=3)
        self.assertEqual(actual, [])

    def test_single_base_fallback_from_base_reach(self):
        # allow_cross_base_return 없이(단일 base) _base_reach만 있는 경우도
        # get_mask_batch()가 _base_reaches로 정상 변환해서 처리하는지 확인
        constraint = {
            "base_airport": HB1,
            "min_conn": 0.3, "max_conn": 3.0, "min_rest": 8.0,
            "max_legs": 4, "max_duty_periods": 2,
            "max_pairing_days": 3, "min_pairing_legs": 1, "max_duty": 13.0,
        }
        from base_reach import build_base_reach
        constraint["_base_reach"] = build_base_reach(FLIGHTS, HB1, constraint)
        self._check(constraint, stage=3)


if __name__ == "__main__":
    unittest.main()
