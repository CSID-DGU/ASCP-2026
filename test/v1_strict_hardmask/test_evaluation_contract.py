import inspect
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "RL"))

from evaluation import evaluate_ip


class CppEvaluationContractTest(unittest.TestCase):
    def test_evaluate_full_has_no_base_return_opt_out(self):
        params = inspect.signature(evaluate_ip.evaluate_full).parameters
        self.assertNotIn("require_base_return", params)

    def test_collect_pool_has_no_base_return_opt_out(self):
        params = inspect.signature(evaluate_ip.collect_pool_full).parameters
        self.assertNotIn("require_base_return", params)


    def test_turkish_has_no_cross_base_pairing_exception(self):
        source = inspect.getsource(evaluate_ip.collect_pool_full)
        self.assertNotIn("HB1->HB2", source)
        self.assertIn("return p[\"ends_at_base\"]", source)

    def test_incomplete_coverage_fails_instead_of_reporting_cpp_solution(self):
        source = inspect.getsource(evaluate_ip.evaluate_full)
        self.assertIn("result[\"uncoverable\"] > 0", source)
        self.assertIn("CPP 해를 구성하지 못했습니다", source)

if __name__ == "__main__":
    unittest.main()
