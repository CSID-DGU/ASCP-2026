import unittest

from RL import config
from RL.airline_constraints import AIRLINE_CONSTRAINTS


class AirlineWindowContractTest(unittest.TestCase):
    def test_each_window_has_one_day_completion_margin(self):
        for airline, constraint in AIRLINE_CONSTRAINTS.items():
            with self.subTest(airline=airline):
                self.assertGreaterEqual(
                    config.AIRLINE_WINDOW_DAYS[airline],
                    constraint["max_pairing_days"] + 1,
                )

    def test_multi_airline_list_is_explicit_bts_set(self):
        self.assertEqual(
            tuple(config.MULTI_AIRLINES),
            ("delta", "alaska", "jetblue"),
        )


if __name__ == "__main__":
    unittest.main()
