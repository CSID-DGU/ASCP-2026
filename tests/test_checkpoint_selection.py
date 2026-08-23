import unittest

from experiments.train import _is_better_checkpoint
import config


class CheckpointSelectionTest(unittest.TestCase):
    def test_higher_coverage_wins_even_with_more_pairings(self):
        # tolerance(1.0pp)보다 확실히 큰 차이를 써야 함 -- 정확히 tolerance와 같은
        # margin을 쓰면 "동률" 분기로 빠져 이 테스트의 의도(명확한 우열)가 깨짐
        self.assertTrue(_is_better_checkpoint(100.0, 20.0, 95.0, 10.0))

    def test_fewer_pairings_only_breaks_equal_coverage_tie(self):
        self.assertTrue(_is_better_checkpoint(100.0, 9.0, 100.0, 10.0))
        self.assertFalse(_is_better_checkpoint(100.0, 11.0, 100.0, 10.0))

    def test_lower_coverage_never_wins(self):
        self.assertFalse(_is_better_checkpoint(95.0, 1.0, 100.0, 100.0))

    def test_within_tolerance_falls_back_to_pairing_tiebreak(self):
        # coverage 차이가 tolerance(1.0pp) 이내면 "사실상 동률"로 보고
        # avg_pairings 최소화로 승부를 가려야 함 (노이즈로 오판하지 않도록)
        tol = config.CHECKPOINT_COVERAGE_TOL_PCT
        self.assertTrue(_is_better_checkpoint(100.0 - tol * 0.5, 9.0, 100.0, 10.0))
        self.assertFalse(_is_better_checkpoint(100.0 - tol * 0.5, 11.0, 100.0, 10.0))

    def test_beyond_tolerance_ignores_pairing_count(self):
        # coverage 차이가 tolerance를 벗어나면 pairing 수와 무관하게 coverage로 결정
        tol = config.CHECKPOINT_COVERAGE_TOL_PCT
        self.assertFalse(_is_better_checkpoint(100.0 - tol * 2, 1.0, 100.0, 50.0))


if __name__ == "__main__":
    unittest.main()
