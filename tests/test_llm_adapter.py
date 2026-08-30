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
    LLMOutputParseError,
    parse_llm_output,
    parse_llm_solution,
    to_pairing_records,
    llm_output_to_pairing_records,
)


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

    def test_preserves_declared_base(self):
        records, _ = parse_llm_solution(SAMPLE_LLM_OUTPUT)
        self.assertEqual(records[0]["declared_base"], "ATL")
        self.assertEqual(records[1]["declared_base"], "SLC")
        self.assertIsNone(records[2]["declared_base"])

    def test_rejects_malformed_pairing_instead_of_silently_dropping_it(self):
        with self.assertRaises(LLMOutputParseError):
            parse_llm_solution("Pairing 1 (base=ATL): [1, bad, 3]\nUncovered: []")

    def test_rejects_duplicate_pairing_number(self):
        with self.assertRaises(LLMOutputParseError):
            parse_llm_solution("Pairing 1: [1,2,3]\nPairing 1: [4,5,6]")

    def test_rejects_output_without_result_records(self):
        with self.assertRaises(LLMOutputParseError):
            parse_llm_solution("no pairings here")


class RecordConversionTests(unittest.TestCase):
    def test_to_pairing_records_tags_as_policy(self):
        records = to_pairing_records([[1, 23, 45], [7, 88]])
        self.assertEqual(records, [
            {"legs": [1, 23, 45], "source_type": "policy"},
            {"legs": [7, 88], "source_type": "policy"},
        ])

    def test_llm_output_to_pairing_records(self):
        records = llm_output_to_pairing_records(SAMPLE_LLM_OUTPUT)
        self.assertEqual(len(records), 3)
        self.assertTrue(all(r["source_type"] == "policy" for r in records))

if __name__ == "__main__":
    unittest.main()
