import unittest

from evaluation.set_partition import solve_lp_relaxation


class FullUniverseDualTest(unittest.TestCase):
    def test_empty_real_pool_still_receives_full_universe_duals(self):
        result = solve_lp_relaxation(
            [],
            flight_ids=[0, 1],
            artificial_cost=100.0,
        )
        self.assertEqual(set(result["dual_vars"]), {0, 1})
        self.assertEqual(result["reduced_costs"], [])

    def test_uncovered_flight_receives_artificial_dual(self):
        result = solve_lp_relaxation(
            [{"legs": [0], "cost": 2.0}],
            flight_ids=[0, 1],
            artificial_cost=100.0,
        )
        self.assertEqual(set(result["dual_vars"]), {0, 1})
        self.assertGreater(result["dual_vars"][1], result["dual_vars"][0])
        self.assertEqual(len(result["reduced_costs"]), 1)


if __name__ == "__main__":
    unittest.main()
