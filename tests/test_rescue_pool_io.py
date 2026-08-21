"""
tests/test_rescue_pool_io.py -- completion/rescue_pool_io.py 단위 테스트 (F3/V2)

save_rescue_pool_json()으로 저장한 파일이 evaluation/evaluate_ip.py의
--rescue-pool-path 로더(=json.load 후 dict면 "columns"->"rescue_columns" 순으로
꺼내는 로직, evaluate_ip.py:710-713)와 실제로 호환되는지까지 확인한다.
"""

import json
import os
import sys
import tempfile
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from completion.rescue_pool_io import save_rescue_pool_json, load_rescue_pool_json  # noqa: E402


def _evaluate_ip_loader(path):
    """evaluation/evaluate_ip.py:710-713과 동일한 로딩 로직(복붙 검증용)."""
    with open(path, "r", encoding="utf-8") as handle:
        rescue_columns = json.load(handle)
    if isinstance(rescue_columns, dict):
        rescue_columns = rescue_columns.get("columns", rescue_columns.get("rescue_columns", []))
    return rescue_columns


CANDIDATE = {
    "legs": [1, 2, 3], "source_type": "rescue", "is_legal": True, "cost": 3.5,
    "repair_target_flights": [2], "validator_version": "0.1.0", "constraint_hash": "abc",
}


class SaveRescuePoolJsonTests(unittest.TestCase):
    def test_saved_file_readable_by_evaluate_ip_loader(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "rescue_pool.json")
            save_rescue_pool_json(path, [CANDIDATE], failures={5: "NO_BASE_SUFFIX"})
            loaded = _evaluate_ip_loader(path)
        self.assertEqual(loaded, [CANDIDATE])

    def test_creates_parent_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "nested", "dir", "rescue_pool.json")
            save_rescue_pool_json(path, [CANDIDATE])
            self.assertTrue(os.path.exists(path))

    def test_round_trip_via_own_loader(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "rescue_pool.json")
            save_rescue_pool_json(path, [CANDIDATE])
            self.assertEqual(load_rescue_pool_json(path), [CANDIDATE])

    def test_empty_candidates_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "rescue_pool.json")
            save_rescue_pool_json(path, [])
            self.assertEqual(_evaluate_ip_loader(path), [])


if __name__ == "__main__":
    unittest.main()
