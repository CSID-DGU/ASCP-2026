import inspect
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "RL"))

import evaluate_ip


class CppEvaluationContractTest(unittest.TestCase):
    def test_evaluate_full_has_no_base_return_opt_out(self):
        params = inspect.signature(evaluate_ip.evaluate_full).parameters
        self.assertNotIn("require_base_return", params)

    def test_collect_pool_has_no_base_return_opt_out(self):
        params = inspect.signature(evaluate_ip.collect_pool_full).parameters
        self.assertNotIn("require_base_return", params)


if __name__ == "__main__":
    unittest.main()
