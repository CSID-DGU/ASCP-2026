"""
tests/test_environment_batch.py -- RL/environment.py::get_mask_batch() 단위 테스트

get_mask_batch()가 기존 scalar get_mask()를 episode 하나씩 호출한 것과 정확히
같은 mask를 내는지, 무작위로 생성한 대량의 (state, assigned) 조합에 대해 전수
비교한다. 이건 CPP 판정 로직 자체를 재구현한 것이라 여기서 하나라도 어긋나면
F1에서 맞춘 "학습 mask == 독립 validator" 일치가 깨질 수 있으므로,
값 하나라도 다르면 실패
"""

import os
import random
import sys
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "RL"))

from environment import get_mask, get_mask_batch  # noqa: E402
from base_reach import build_base_reach, can_reach_base  # noqa: E402


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


def _make_constraint(max_duty=None):
    c = {
        "base_airport": BASE,
        "min_conn": 0.3, "max_conn": 3.0, "min_rest": 8.0,
        "max_legs": 4, "max_duty_periods": 2,
        "max_pairing_days": 3, "min_pairing_legs": 1,
    }
    if max_duty is not None:
        c["max_duty"] = max_duty
    c["_base_reach"] = build_base_reach(FLIGHTS, BASE, c)
    return c


def _random_state(rng):
    pairing_start = rng.random() < 0.3
    is_resting = (not pairing_start) and rng.random() < 0.3
    current_time = rng.uniform(0.0, 36.0)
    return {
        "current_airport": rng.choice([BASE, A, C]),
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


class GetMaskBatchEquivalenceTests(unittest.TestCase):
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
        # max_duty를 안 준 경우 -- get_max_duty()가 FAA_DUTY_TABLE로 legs_after별 lookup함
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


class JointReachabilityTests(unittest.TestCase):
    def test_time_and_duty_bounds_must_come_from_same_path(self):
        flights = [
            {"id": 0, "origin": 9, "dest": 1, "dep_time": 0.0, "arr_time": 1.0},
            {"id": 1, "origin": 1, "dest": 0, "dep_time": 11.0, "arr_time": 12.0},
            {"id": 2, "origin": 1, "dest": 2, "dep_time": 2.0, "arr_time": 7.0},
            {"id": 3, "origin": 2, "dest": 0, "dep_time": 8.0, "arr_time": 13.0},
        ]
        constraint = {"min_conn": 0.5, "max_conn": 2.0, "min_rest": 10.0}
        reach = build_base_reach(flights, 0, constraint)
        self.assertFalse(can_reach_base(
            reach, flights[0], 0.0, 0.52, duty_period=0, max_duty_periods=0
        ))


if __name__ == "__main__":
    unittest.main()
