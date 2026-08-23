import unittest

from experiments.train import _is_better_checkpoint


class CheckpointSelectionTest(unittest.TestCase):
    def test_higher_coverage_wins_even_with_more_pairings(self):
        self.assertTrue(_is_better_checkpoint(100.0, 20.0, 99.0, 10.0))

    def test_fewer_pairings_only_breaks_equal_coverage_tie(self):
        self.assertTrue(_is_better_checkpoint(100.0, 9.0, 100.0, 10.0))
        self.assertFalse(_is_better_checkpoint(100.0, 11.0, 100.0, 10.0))

    def test_lower_coverage_never_wins(self):
        self.assertFalse(_is_better_checkpoint(99.0, 1.0, 100.0, 100.0))


if __name__ == "__main__":
    unittest.main()
