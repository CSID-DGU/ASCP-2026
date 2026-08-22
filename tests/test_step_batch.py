"""
tests/test_step_batch.py -- RL/environment.py::step_batch() /
RL/turkish/environment_turkish.py::step_batch() 단위 테스트
(Phase 3, experiment/rollout-batch-vectorization)

step_batch()는 기존 검증된 step()을 그대로 B번 호출하는 얇은 wrapper라
"개별 호출 결과와 정확히 일치"만 확인하면 충분함 -- flight 선택/EndDuty/
EndPairing 세 종류 액션이 배치 안에 섞여 있어도 각자 올바르게 처리되는지 확인.
"""

import copy
import os
import sys
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "RL"))

import environment as delta_env  # noqa: E402
import turkish.environment_turkish as turkish_env  # noqa: E402
from base_reach import build_base_reach, build_base_reaches  # noqa: E402


BASE, A = 0, 1

FLIGHTS = [
    {"id": 0, "origin": BASE, "dest": A,    "dep_time": 1.0, "arr_time": 2.0},
    {"id": 1, "origin": A,    "dest": BASE, "dep_time": 3.0, "arr_time": 4.0},
]

CONSTRAINT = {
    "base_airport": BASE,
    "min_conn": 0.3, "max_conn": 3.0, "min_rest": 8.0,
    "max_duty": 13.0, "max_legs": 4, "max_duty_periods": 2,
    "max_pairing_days": 3, "min_pairing_legs": 1,
}


def _base_state():
    return {
        "current_airport": BASE, "current_time": 1.0,
        "pairing_start": True, "is_resting": False,
        "rest_end_time": None, "duty_period": 0,
        "duty_start_time": 1.0, "pairing_start_time": 1.0,
        "legs": 0, "total_legs": 0, "duty_time": 0.0, "remaining": 2,
    }


def _mid_pairing_state():
    # flight 0을 이미 선택한 직후 상태 -- EndDuty 액션 테스트용
    return {
        "current_airport": A, "current_time": 2.0,
        "pairing_start": False, "is_resting": False,
        "rest_end_time": None, "duty_period": 0,
        "duty_start_time": 1.0, "pairing_start_time": 1.0,
        "legs": 1, "total_legs": 1, "duty_time": 1.0, "remaining": 1,
    }


class DeltaStepBatchTests(unittest.TestCase):
    def setUp(self):
        c = dict(CONSTRAINT)
        c["_base_reach"] = build_base_reach(FLIGHTS, BASE, c)
        self.constraint = c

    def test_matches_individual_calls_for_mixed_actions(self):
        # episode 0: flight 선택(action=0), episode 1: EndDuty(action=N)
        states = [_base_state(), _mid_pairing_state()]
        actions = [0, len(FLIGHTS)]
        assigneds = [{0: False, 1: False}, {0: True, 1: False}]

        expected = []
        expected_assigneds = copy.deepcopy(assigneds)
        for state, action, assigned in zip(states, actions, expected_assigneds):
            expected.append(delta_env.step(state, action, FLIGHTS, assigned, self.constraint))

        actual_assigneds = copy.deepcopy(assigneds)
        actual = delta_env.step_batch(states, actions, FLIGHTS, actual_assigneds, self.constraint)
        actual_states, actual_rewards, actual_dones = actual

        for i in range(2):
            self.assertEqual(actual_states[i], expected[i][0])
            self.assertEqual(actual_rewards[i], expected[i][1])
            self.assertEqual(actual_dones[i], expected[i][2])
        self.assertEqual(actual_assigneds, expected_assigneds)

    def test_end_pairing_action_matches(self):
        state = {
            "current_airport": BASE, "current_time": 4.0,
            "pairing_start": False, "is_resting": False,
            "rest_end_time": None, "duty_period": 0,
            "duty_start_time": 1.0, "pairing_start_time": 1.0,
            "legs": 2, "total_legs": 2, "duty_time": 2.0, "remaining": 0,
        }
        assigned_a = {0: True, 1: True}
        assigned_b = {0: True, 1: True}
        expected = delta_env.step(state, len(FLIGHTS) + 1, FLIGHTS, assigned_a, self.constraint)
        actual_states, actual_rewards, actual_dones = delta_env.step_batch(
            [state], [len(FLIGHTS) + 1], FLIGHTS, [assigned_b], self.constraint
        )
        self.assertEqual(actual_states[0], expected[0])
        self.assertEqual(actual_rewards[0], expected[1])
        self.assertEqual(actual_dones[0], expected[2])


class TurkishStepBatchTests(unittest.TestCase):
    def setUp(self):
        c = dict(CONSTRAINT)
        c["base_ids"] = [BASE]
        c["_base_reaches"] = build_base_reaches(FLIGHTS, [BASE], c)
        self.constraint = c

    def test_matches_individual_calls_for_mixed_actions(self):
        states = [_base_state(), _mid_pairing_state()]
        actions = [0, len(FLIGHTS)]
        assigneds = [{0: False, 1: False}, {0: True, 1: False}]

        expected = []
        expected_assigneds = copy.deepcopy(assigneds)
        for state, action, assigned in zip(states, actions, expected_assigneds):
            expected.append(turkish_env.step(state, action, FLIGHTS, assigned, self.constraint))

        actual_assigneds = copy.deepcopy(assigneds)
        actual_states, actual_rewards, actual_dones = turkish_env.step_batch(
            states, actions, FLIGHTS, actual_assigneds, self.constraint
        )

        for i in range(2):
            self.assertEqual(actual_states[i], expected[i][0])
            self.assertEqual(actual_rewards[i], expected[i][1])
            self.assertEqual(actual_dones[i], expected[i][2])
        self.assertEqual(actual_assigneds, expected_assigneds)


if __name__ == "__main__":
    unittest.main()
