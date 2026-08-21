"""
tests/test_llm_adapter.py -- evaluation/llm_adapter.py 단위 테스트 (F1 C4)
"""

import os
import sys
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "RL"))
sys.path.insert(0, os.path.join(REPO_ROOT, "evaluation"))

from llm_adapter import (  # noqa: E402
    parse_llm_output,
    to_pairing_records,
    forced_singleton_records,
    llm_output_to_pairing_records,
)
from validation_report import aggregate_by_source  # noqa: E402


SAMPLE_LLM_OUTPUT = """
Here is my crew pairing solution:

Pairing 1 (base=ATL): [1, 23, 45]
Pairing 2 (base=SLC): [7, 88]
Pairing 3: [99]

Uncovered: [12, 34, 56]
"""


class ParseLlmOutputTests(unittest.TestCase):
    def test_extracts_pairings_and_uncovered(self):
        pairings, uncovered = parse_llm_output(SAMPLE_LLM_OUTPUT)
        self.assertEqual(pairings, [[1, 23, 45], [7, 88], [99]])
        self.assertEqual(uncovered, [12, 34, 56])

    def test_handles_empty_text(self):
        pairings, uncovered = parse_llm_output("no pairings here")
        self.assertEqual(pairings, [])
        self.assertEqual(uncovered, [])


class RecordConversionTests(unittest.TestCase):
    def test_to_pairing_records_tags_as_policy(self):
        records = to_pairing_records([[1, 23, 45], [7, 88]])
        self.assertEqual(records, [
            {"legs": [1, 23, 45], "source_type": "policy"},
            {"legs": [7, 88], "source_type": "policy"},
        ])

    def test_forced_singleton_records_tags_as_forced(self):
        records = forced_singleton_records([12, 34])
        self.assertEqual(records, [
            {"legs": [12], "source_type": "forced"},
            {"legs": [34], "source_type": "forced"},
        ])

    def test_llm_output_to_pairing_records_without_forced_completion(self):
        records = llm_output_to_pairing_records(SAMPLE_LLM_OUTPUT)
        self.assertEqual(len(records), 3)
        self.assertTrue(all(r["source_type"] == "policy" for r in records))

    def test_llm_output_to_pairing_records_with_forced_completion(self):
        records = llm_output_to_pairing_records(SAMPLE_LLM_OUTPUT, include_forced_completion=True)
        self.assertEqual(len(records), 3 + 3)  # pairing 3개 + uncovered 3개
        forced = [r for r in records if r["source_type"] == "forced"]
        self.assertEqual(len(forced), 3)
        self.assertEqual({r["legs"][0] for r in forced}, {12, 34, 56})


class ForcedCompletionCoverageTests(unittest.TestCase):
    def test_forced_completion_does_not_count_as_policy_direct_coverage(self):
        # aggregate_by_source()에 넣었을 때 forced가 policy_direct 커버리지에 안 섞이는지 확인.
        # constraint를 안 넘겨서 validate_pairing은 안 타지만, 시간 지표 계산은 항상 도니까
        # dep_time/arr_time은 채워줘야 함.
        flights = {
            fid: {"id": fid, "origin": 0, "dest": 0, "dep_time": float(fid), "arr_time": float(fid) + 1.0}
            for fid in range(1, 100)
        }
        records = llm_output_to_pairing_records(SAMPLE_LLM_OUTPUT, include_forced_completion=True)
        report = aggregate_by_source(records, flights, n_total_flights=99)

        self.assertEqual(report["policy_direct"]["pairing_count"], 3)
        self.assertEqual(report["policy_direct"]["covered_flights"], 6)  # 1,23,45,7,88,99
        self.assertEqual(report["forced"]["pairing_count"], 3)
        self.assertEqual(report["forced"]["covered_flights"], 3)  # 12,34,56


if __name__ == "__main__":
    unittest.main()
