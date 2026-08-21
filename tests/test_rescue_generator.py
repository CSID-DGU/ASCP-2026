"""
tests/test_rescue_generator.py -- completion/rescue_generator.py 단위 테스트 (F3/V2, H-V2-1~3)

각 케이스는 작은 flight graph를 만들어서: (1) 성공적으로 rescue candidate가 나오는
경우, (2) 실패 사유 코드가 정확히 나오는 경우, (3) 만들어진 candidate가 실제로
validate_pairing()과 evaluation/full_flight_master.py 계약을 통과하는지를 확인한다.
"""

import os
import sys
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "RL"))
sys.path.insert(0, os.path.join(REPO_ROOT, "evaluation"))

from completion.rescue_generator import (  # noqa: E402
    generate_rescue_candidates,
    NO_BASE_PREFIX,
    NO_BASE_SUFFIX,
    NO_ALLOWED_BASE,
    WINDOW_BOUNDARY,
    PAIRING_DURATION_LIMIT,
)
from validator import validate_pairing  # noqa: E402
from evaluation.full_flight_master import validate_master_inputs  # noqa: E402


BASE, A, B = 0, 1, 2

CONSTRAINT = {
    "base_airport": BASE,
    "min_conn": 0.5, "max_conn": 9.0, "min_rest": 10.0,
    "max_duty": 13.0, "max_legs": 8,
    "max_duty_periods": 2, "max_pairing_days": 5,
    "min_pairing_legs": 1,
}


def _f(fid, origin, dest, dep, arr):
    return {"id": fid, "origin": origin, "dest": dest, "dep_time": dep, "arr_time": arr}


class SuccessCaseTests(unittest.TestCase):
    def test_target_already_base_to_base(self):
        # target 자체가 base->base 왕복이면 prefix/suffix 없이 바로 채택돼야 함
        flights = {10: _f(10, BASE, BASE, 0.0, 2.0)}
        result = generate_rescue_candidates(flights, CONSTRAINT, [10])
        self.assertEqual(result["failures"], {})
        self.assertEqual(len(result["candidates"]), 1)
        c = result["candidates"][0]
        self.assertEqual(c["legs"], [10])
        self.assertEqual(c["repair_target_flights"], [10])
        self.assertEqual(c["source_type"], "rescue")

    def test_needs_prefix_and_suffix(self):
        # base -> A(prefix) -> target(A->B) -> B -> base(suffix)
        flights = {
            1: _f(1, BASE, A, 0.0, 2.0),
            2: _f(2, A, B, 3.0, 5.0),     # target
            3: _f(3, B, BASE, 6.0, 8.0),
        }
        result = generate_rescue_candidates(flights, CONSTRAINT, [2])
        self.assertEqual(result["failures"], {})
        self.assertEqual(len(result["candidates"]), 1)
        self.assertEqual(result["candidates"][0]["legs"], [1, 2, 3])
        self.assertEqual(result["candidates"][0]["repair_target_flights"], [2])

    def test_candidate_passes_independent_validate_pairing(self):
        flights = {
            1: _f(1, BASE, A, 0.0, 2.0),
            2: _f(2, A, B, 3.0, 5.0),
            3: _f(3, B, BASE, 6.0, 8.0),
        }
        result = generate_rescue_candidates(flights, CONSTRAINT, [2])
        candidate = result["candidates"][0]
        check = validate_pairing({"legs": candidate["legs"]}, flights, CONSTRAINT)
        self.assertTrue(check["is_valid"])

    def test_candidate_passes_full_flight_master_contract(self):
        flights = {10: _f(10, BASE, BASE, 0.0, 2.0)}
        result = generate_rescue_candidates(flights, CONSTRAINT, [10])
        validate_master_inputs(result["candidates"], all_flight_ids=list(flights.keys()))

    def test_respects_max_candidates_per_target_budget(self):
        # base -> A로 가는 flight가 2개(둘 다 legal prefix) -> target -> base
        flights = {
            1: _f(1, BASE, A, 0.0, 2.0),
            2: _f(2, BASE, A, 0.2, 2.2),
            3: _f(3, A, BASE, 3.0, 5.0),  # target
        }
        result = generate_rescue_candidates(flights, CONSTRAINT, [3], max_candidates_per_target=1)
        self.assertEqual(len(result["candidates"]), 1)


class FailureCaseTests(unittest.TestCase):
    def test_no_allowed_base_when_base_airport_missing(self):
        flights = {10: _f(10, BASE, BASE, 0.0, 2.0)}
        constraint = {k: v for k, v in CONSTRAINT.items() if k != "base_airport"}
        result = generate_rescue_candidates(flights, constraint, [10])
        self.assertEqual(result["failures"], {10: NO_ALLOWED_BASE})
        self.assertEqual(result["candidates"], [])

    def test_no_base_prefix_when_unreachable(self):
        # base에서 A로 가는 flight가 있지만 target 이전 시간대에는 없음(늦게 출발)
        flights = {
            1: _f(1, BASE, A, 100.0, 102.0),  # target보다 훨씬 늦게 출발 -> window 밖
            2: _f(2, A, BASE, 3.0, 5.0),       # target
        }
        result = generate_rescue_candidates(flights, CONSTRAINT, [2])
        self.assertEqual(result["failures"], {2: WINDOW_BOUNDARY})

    def test_no_base_prefix_when_base_departs_but_cannot_reach_target(self):
        # base에서 target 이전 시간대에 출발하는 flight는 있지만, target origin(A)이
        # 아니라 엉뚱한 곳(B)으로만 감 -- 진짜 "연결이 안 되는" NO_BASE_PREFIX
        flights = {
            1: _f(1, BASE, B, 0.0, 2.0),   # base -> B (target origin인 A로 못 감)
            2: _f(2, A, BASE, 3.0, 5.0),   # target: A -> base
        }
        result = generate_rescue_candidates(flights, CONSTRAINT, [2])
        self.assertEqual(result["failures"], {2: NO_BASE_PREFIX})

    def test_pairing_duration_limit_when_base_departure_outside_window(self):
        # base 출발 flight는 있지만 max_pairing_days window보다 훨씬 이전이라 제외됨
        constraint = dict(CONSTRAINT, max_pairing_days=1)  # window = 24h
        flights = {
            1: _f(1, BASE, A, 0.0, 2.0),      # target보다 100h 이전 -> window 밖
            2: _f(2, A, BASE, 100.0, 102.0),  # target
        }
        result = generate_rescue_candidates(flights, constraint, [2])
        self.assertEqual(result["failures"], {2: PAIRING_DURATION_LIMIT})

    def test_no_base_suffix_when_no_return_path(self):
        flights = {
            1: _f(1, BASE, A, 0.0, 2.0),   # target
            # A에서 나가는 flight가 전혀 없음 -> suffix 불가능
        }
        result = generate_rescue_candidates(flights, CONSTRAINT, [1])
        self.assertEqual(result["failures"], {1: NO_BASE_SUFFIX})

    def test_missing_flight_id_reported_as_failure(self):
        flights = {10: _f(10, BASE, BASE, 0.0, 2.0)}
        result = generate_rescue_candidates(flights, CONSTRAINT, [999])
        self.assertIn(999, result["failures"])
        self.assertEqual(result["candidates"], [])

    def test_every_uncovered_id_appears_in_candidates_or_failures(self):
        flights = {
            10: _f(10, BASE, BASE, 0.0, 2.0),
            1: _f(1, BASE, A, 0.0, 2.0),  # no suffix -> failure
        }
        result = generate_rescue_candidates(flights, CONSTRAINT, [10, 1])
        covered_targets = {c["repair_target_flights"][0] for c in result["candidates"]}
        self.assertEqual(covered_targets | set(result["failures"].keys()), {10, 1})


if __name__ == "__main__":
    unittest.main()
