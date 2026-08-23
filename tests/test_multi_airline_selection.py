import unittest
from collections import defaultdict

from experiments.train import _airline_selection_score


class MultiAirlineSelectionTest(unittest.TestCase):
    def test_waits_until_every_airline_has_full_window(self):
        histories = defaultdict(list)
        histories["delta"] = [{"coverage_pct": 100, "n_pairings": 10}] * 25
        histories["alaska"] = [{"coverage_pct": 100, "n_pairings": 10}] * 24
        self.assertIsNone(
            _airline_selection_score(histories, ["delta", "alaska"])
        )

    def test_uses_worst_airline_coverage(self):
        histories = {
            "delta": [{"coverage_pct": 100, "n_pairings": 10}] * 25,
            "alaska": [{"coverage_pct": 80, "n_pairings": 20}] * 25,
            "jetblue": [{"coverage_pct": 90, "n_pairings": 30}] * 25,
        }
        score = _airline_selection_score(
            histories, ["delta", "alaska", "jetblue"]
        )
        self.assertEqual(score["coverage_pct"], 80)
        self.assertEqual(score["avg_pairings"], 20)


if __name__ == "__main__":
    unittest.main()
