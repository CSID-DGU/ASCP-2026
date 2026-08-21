"""
tests/test_validation_report.py -- evaluation/validation_report.py 단위 테스트 (F1 C2)
"""

import os
import sys
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "RL"))
sys.path.insert(0, os.path.join(REPO_ROOT, "evaluation"))

from validation_report import aggregate_by_source, aggregate_by_source_per_chunk  # noqa: E402


BASE, OTHER, THIRD = 0, 1, 2

# flight 0->1: 2h 비행. flight 1(다시 base로)은 3h 뒤 출발, 2h 비행 -> 그 사이 1h는 dead time.
FLIGHTS = {
    0: {"id": 0, "origin": BASE,  "dest": OTHER, "dep_time": 0.0, "arr_time": 2.0},
    1: {"id": 1, "origin": OTHER, "dest": BASE,  "dep_time": 3.0, "arr_time": 5.0},
    2: {"id": 2, "origin": BASE,  "dest": THIRD, "dep_time": 0.0, "arr_time": 1.0},
}
CONSTRAINT = {
    "base_airport": BASE,
    "min_conn": 0.5, "max_conn": 9.0, "min_rest": 10.0,
    "max_duty": 13.0, "max_legs": 8,
    "max_duty_periods": 2, "max_pairing_days": 5,
    "min_pairing_legs": 2,
}


class AggregateBySourceTests(unittest.TestCase):
    def test_buckets_split_by_source_type_and_time_metrics(self):
        pairings = [
            {"legs": [0, 1], "source_type": "policy"},
            {"legs": [2],    "source_type": "salvage", "is_truncated": True},  # base 미복귀 -> invalid
        ]
        report = aggregate_by_source(pairings, FLIGHTS, constraint=CONSTRAINT, n_total_flights=3)

        pd = report["policy_direct"]
        self.assertEqual(pd["pairing_count"], 1)
        self.assertEqual(pd["covered_flights"], 2)
        self.assertEqual(pd["invalid_count"], 0)
        # flying = (2-0) + (5-3) = 4h, elapsed = 5-0 = 5h, dead = 5-4 = 1h
        self.assertEqual(pd["total_flying_time"], 4.0)
        self.assertEqual(pd["total_dead_time"], 1.0)
        self.assertEqual(pd["ftc_pct"], 25.0)  # 1/4 * 100
        self.assertEqual(pd["man_days"], 5.0 / 24.0)

        sv = report["salvage"]
        self.assertEqual(sv["pairing_count"], 1)
        self.assertEqual(sv["covered_flights"], 1)
        self.assertEqual(sv["invalid_count"], 1)  # THIRD(2)로 끝나서 base 미복귀

        self.assertEqual(report["repair"]["pairing_count"], 0)
        self.assertEqual(report["forced"]["pairing_count"], 0)
        self.assertEqual(report["_direct_coverage_source"], "policy_direct")
        self.assertEqual(report["cross_bucket_duplicate_flight_ids"], [])

    def test_deadhead_count_and_cross_bucket_duplicate(self):
        pairings = [
            {"legs": [0], "source_type": "policy"},
            {"legs": [0], "source_type": "forced", "is_deadhead": True},  # flight 0을 policy와 중복 커버
        ]
        report = aggregate_by_source(pairings, FLIGHTS, n_total_flights=3)

        self.assertEqual(report["forced"]["deadhead_count"], 1)
        self.assertEqual(report["policy_direct"]["deadhead_count"], 0)
        # bucket 내부에는 중복 없음(각 bucket에 pairing 1개씩)
        self.assertEqual(report["policy_direct"]["internal_duplicate_flight_ids"], [])
        # 근데 전체(policy+forced)로 보면 flight 0이 두 번 커버됨
        self.assertEqual(report["cross_bucket_duplicate_flight_ids"], [0])

    def test_unknown_source_type_falls_back_to_own_name(self):
        pairings = [{"legs": [0, 1], "source_type": "weird"}]
        report = aggregate_by_source(pairings, FLIGHTS, n_total_flights=3)
        self.assertEqual(report["weird"]["pairing_count"], 1)
        self.assertIsNone(report["weird"]["invalid_count"])  # constraint 안 줬으니 검증 안 함


class AggregateBySourcePerChunkTests(unittest.TestCase):
    def test_per_chunk_uses_each_chunks_own_constraint(self):
        # chunk1은 base=BASE, chunk2는 base=OTHER -- evaluate_ip.py가 chunk마다 base_id를
        # 다시 뽑는 것과 동일한 상황. 각 pairing이 "자기" chunk의 base 기준으로는 정상이어야 함.
        flights = {
            **FLIGHTS,
            3: {"id": 3, "origin": OTHER, "dest": THIRD, "dep_time": 0.0, "arr_time": 1.0},
            4: {"id": 4, "origin": THIRD, "dest": OTHER, "dep_time": 2.0, "arr_time": 3.0},
        }
        constraint_other_base = {**CONSTRAINT, "base_airport": OTHER}

        chunks = [
            ([{"legs": [0, 1], "source_type": "policy"}], CONSTRAINT),           # base=BASE
            ([{"legs": [3, 4], "source_type": "policy"}], constraint_other_base),  # base=OTHER
        ]
        report = aggregate_by_source_per_chunk(chunks, flights, n_total_flights=5)

        pd = report["policy_direct"]
        self.assertEqual(pd["pairing_count"], 2)
        self.assertEqual(pd["covered_flights"], 4)  # 0,1,3,4 -- 겹치는 flight 없이 정확히 합산됨
        self.assertEqual(pd["invalid_count"], 0)  # 각자 자기 chunk의 base 기준으로는 둘 다 valid

        # 대조: 이걸 chunk 구분 없이 CONSTRAINT(base=BASE) 하나로만 검증했다면 chunk2
        # pairing(3,4)은 base=OTHER라서 잘못 invalid로 잡혔을 것 -- per-chunk가 왜 필요한지 확인.
        wrong = aggregate_by_source([{"legs": [3, 4], "source_type": "policy"}], flights,
                                     constraint=CONSTRAINT, n_total_flights=5)
        self.assertEqual(wrong["policy_direct"]["invalid_count"], 1)


if __name__ == "__main__":
    unittest.main()
