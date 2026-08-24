import unittest

from evaluation.set_partition import solve_lp_relaxation


class Phase2ArtificialSignalTests(unittest.TestCase):
    def test_lp_reports_flights_missing_from_real_pool(self):
        result = solve_lp_relaxation(
            [{"legs": [0, 1], "cost": 1.0}],
            flight_ids=[0, 1, 2],
            artificial_cost=100.0,
        )
        self.assertEqual(result["artificial_flight_ids"], [2])

    def test_lp_reports_no_artificial_when_pool_covers_universe(self):
        result = solve_lp_relaxation(
            [{"legs": [0, 1], "cost": 1.0}, {"legs": [2], "cost": 2.0}],
            flight_ids=[0, 1, 2],
            artificial_cost=100.0,
        )
        self.assertEqual(result["artificial_flight_ids"], [])


if __name__ == "__main__":
    unittest.main()
