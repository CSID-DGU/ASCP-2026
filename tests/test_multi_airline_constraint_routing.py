import unittest

from experiments import train


class MultiAirlineConstraintRoutingTests(unittest.TestCase):
    def test_airline_specific_values_are_not_replaced_by_delta(self):
        delta = train._constraint_for_episode("delta", 10)
        alaska = train._constraint_for_episode("alaska", 20)
        jetblue = train._constraint_for_episode("jetblue", 30)

        self.assertEqual(delta["base_airport"], 10)
        self.assertEqual(alaska["base_airport"], 20)
        self.assertEqual(jetblue["base_airport"], 30)
        self.assertEqual(delta["max_legs"], 8)
        self.assertEqual(alaska["max_legs"], 6)
        self.assertEqual(jetblue["max_legs"], 7)
        self.assertEqual(delta["max_conn"], 12.0)
        self.assertEqual(alaska["max_conn"], 8.8)
        self.assertEqual(jetblue["max_conn"], 8.0)

    def test_curriculum_overrides_preserve_airline_rules(self):
        alaska_stage1 = train._constraint_for_episode(
            "alaska", 21,
            max_duty_periods=1,
            max_pairing_days=1,
            base_penalty=5.0,
        )

        self.assertEqual(alaska_stage1["base_airport"], 21)
        self.assertEqual(alaska_stage1["max_legs"], 6)
        self.assertEqual(alaska_stage1["max_conn"], 8.8)
        self.assertEqual(alaska_stage1["max_duty_periods"], 1)
        self.assertEqual(alaska_stage1["max_pairing_days"], 1)
        self.assertEqual(alaska_stage1["base_penalty"], 5.0)

    def test_multi_airline_sample_requires_airline_identity(self):
        legacy_sample = tuple(range(7))
        with self.assertRaisesRegex(ValueError, "episode 항공사"):
            train._unpack_flight_sample(
                legacy_sample,
                require_airline=True,
                default_airline="delta",
            )

    def test_explicit_sampled_airline_is_preserved(self):
        sample = (*tuple(range(7)), "jetblue")
        unpacked = train._unpack_flight_sample(
            sample,
            require_airline=True,
            default_airline="delta",
        )

        self.assertEqual(unpacked[-1], "jetblue")

    def test_unknown_airline_fails(self):
        with self.assertRaisesRegex(ValueError, "지원하지 않는"):
            train._constraint_for_episode("unknown", 0)


if __name__ == "__main__":
    unittest.main()
