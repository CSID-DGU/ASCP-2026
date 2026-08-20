import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "RL"))

import environment
from base_reach import build_base_reach
from turkish import environment_turkish


def make_state(**updates):
    value = {
        "current_airport": 0, "current_time": 0.0, "duty_start_time": 0.0,
        "legs": 0, "total_legs": 0, "pairing_start": True,
        "pairing_start_time": 0.0, "is_resting": False,
        "rest_end_time": None, "duty_period": 0,
    }
    value.update(updates)
    return value


def make_constraint(**updates):
    value = {
        "base_airport": 0, "min_conn": 0.5, "max_conn": 4.0,
        "min_rest": 8.0, "max_duty": 14.0, "max_legs": 4,
        "max_duty_periods": 2, "max_pairing_days": 2,
        "min_pairing_legs": 2,
    }
    value.update(updates)
    return value


class StrictMaskContractTest(unittest.TestCase):
    def test_cpp_requires_reachability(self):
        flights = [{"id": 0, "origin": 0, "dest": 1, "dep_time": 1.0, "arr_time": 2.0}]
        with self.assertRaisesRegex(ValueError, "_base_reach"):
            environment.get_mask(make_state(), flights, {0: False}, make_constraint())

    def test_cpp_start_never_uses_non_base_origin(self):
        flights = [
            {"id": 0, "origin": 1, "dest": 0, "dep_time": 1.0, "arr_time": 2.0},
            {"id": 1, "origin": 0, "dest": 1, "dep_time": 3.0, "arr_time": 4.0},
        ]
        rule = make_constraint()
        rule["_base_reach"] = build_base_reach(flights, 0, rule)
        mask = environment.get_mask(make_state(), flights, {0: False, 1: True}, rule)
        self.assertEqual(mask[0], 0)

    def test_cpp_end_pairing_requires_base_return(self):
        flights = [{"id": 0, "origin": 0, "dest": 1, "dep_time": 1.0, "arr_time": 2.0}]
        rule = make_constraint()
        rule["_base_reach"] = build_base_reach(flights, 0, rule)
        mask = environment.get_mask(
            make_state(current_airport=1, current_time=2.0, legs=2,
                       total_legs=2, pairing_start=False),
            flights, {0: True}, rule,
        )
        self.assertEqual(mask[-1], 0)

    def test_unreachable_flight_is_masked_before_selection(self):
        flights = [
            {"id": 0, "origin": 0, "dest": 1, "dep_time": 1.0, "arr_time": 2.0},
            {"id": 1, "origin": 0, "dest": 2, "dep_time": 1.0, "arr_time": 2.0},
            {"id": 2, "origin": 1, "dest": 0, "dep_time": 3.0, "arr_time": 4.0},
        ]
        rule = make_constraint()
        rule["_base_reach"] = build_base_reach(flights, 0, rule)
        mask = environment.get_mask(make_state(), flights, {0: False, 1: False, 2: False}, rule)
        self.assertEqual(mask[0], 1)
        self.assertEqual(mask[1], 0)

    def test_legacy_flags_cannot_disable_cpp_contract(self):
        flights = [
            {"id": 0, "origin": 1, "dest": 0, "dep_time": 1.0, "arr_time": 2.0},
        ]
        rule = make_constraint(require_base_return=False, strict_base_start=False)
        rule["_base_reach"] = build_base_reach(flights, 0, rule)
        mask = environment.get_mask(make_state(), flights, {0: False}, rule)
        self.assertEqual(mask[0], 0)

    def test_turkish_cpp_start_is_bound_to_episode_base(self):
        flights = [
            {"id": 0, "origin": 1, "dest": 0, "dep_time": 1.0, "arr_time": 2.0},
            {"id": 1, "origin": 0, "dest": 1, "dep_time": 3.0, "arr_time": 4.0},
        ]
        rule = make_constraint(base_ids=[0, 1])
        rule["_base_reach"] = build_base_reach(flights, 0, rule)
        mask = environment_turkish.get_mask(make_state(), flights, {0: False, 1: True}, rule)
        self.assertEqual(mask[0], 0)


    def test_direct_end_pairing_cannot_bypass_mask(self):
        flights = [{"id": 0, "origin": 0, "dest": 1, "dep_time": 1.0, "arr_time": 2.0}]
        rule = make_constraint()
        rule["_base_reach"] = build_base_reach(flights, 0, rule)
        state = make_state(current_airport=1, current_time=2.0, legs=2,
                           total_legs=2, pairing_start=False)
        with self.assertRaisesRegex(ValueError, "pairing"):
            environment.step(state, len(flights) + 1, flights, {0: True}, rule)
        with self.assertRaisesRegex(ValueError, "pairing"):
            environment_turkish.step(state, len(flights) + 1, flights, {0: True}, rule)

    def test_missing_base_airport_is_configuration_error(self):
        flights = []
        rule = make_constraint()
        del rule["base_airport"]
        rule["_base_reach"] = {}
        with self.assertRaises(KeyError):
            environment.get_mask(make_state(), flights, {}, rule)

    def test_direct_flight_action_cannot_bypass_mask(self):
        flights = [
            {"id": 0, "origin": 1, "dest": 0, "dep_time": 1.0, "arr_time": 2.0},
        ]
        rule = make_constraint()
        rule["_base_reach"] = build_base_reach(flights, 0, rule)
        for module in (environment, environment_turkish):
            with self.assertRaisesRegex(ValueError, "flight"):
                module.step(make_state(), 0, flights, {0: False}, rule)

    def test_out_of_range_action_fails_before_state_mutation(self):
        rule = make_constraint()
        rule["_base_reach"] = {}
        with self.assertRaises(IndexError):
            environment.step(make_state(), 2, [], {}, rule)

    def test_turkish_cross_base_return_is_legal(self):
        flights = [
            {"id": 0, "origin": 0, "dest": 2, "dep_time": 1.0, "arr_time": 2.0},
            {"id": 1, "origin": 2, "dest": 1, "dep_time": 3.0, "arr_time": 4.0},
        ]
        rule = make_constraint(
            base_airport=0, base_ids=[0, 1], allow_cross_base_return=True
        )
        rule["_base_reaches"] = {
            base: build_base_reach(flights, base, rule) for base in rule["base_ids"]
        }
        rule["_base_reach"] = rule["_base_reaches"][0]

        start_mask = environment_turkish.get_mask(
            make_state(), flights, {0: False, 1: False}, rule
        )
        self.assertEqual(start_mask[0], 1)

        end_mask = environment_turkish.get_mask(
            make_state(
                current_airport=1, current_time=4.0, legs=2,
                total_legs=2, pairing_start=False
            ),
            flights, {0: True, 1: True}, rule,
        )
        self.assertEqual(end_mask[-1], 1)

if __name__ == "__main__":
    unittest.main()
