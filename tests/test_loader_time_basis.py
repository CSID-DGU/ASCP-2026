import os
import sys
import tempfile
import unittest

import pandas as pd


ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "RL"))

from loader import load_flights_rolling, utc_offset_hours


class BtsUtcContractTests(unittest.TestCase):
    def _load(self, origin, dest):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "flights.csv")
            pd.DataFrame([{
                "ORIGIN": origin,
                "DEST": dest,
                "CRS_DEP_TIME": 1000,
                "CRS_ARR_TIME": 1200,
                "CRS_ELAPSED_TIME": 120,
                "FL_DATE": "2019-01-01",
            }]).to_csv(path, index=False)
            return load_flights_rolling(path, window_days=1)

    def test_bts_loader_always_converts_local_departure_to_utc(self):
        atl = self._load("ATL", "JFK")[0]
        lax = self._load("LAX", "SFO")[0]
        self.assertEqual(atl["dep_time"], 15.0)
        self.assertEqual(lax["dep_time"], 18.0)

    def test_unknown_bts_airport_does_not_fall_back_to_eastern(self):
        with self.assertRaisesRegex(ValueError, "UTC offset"):
            utc_offset_hours("UNKNOWN")

    def test_summer_time_uses_date_specific_utc_offset(self):
        self.assertEqual(utc_offset_hours("ATL", "2019-01-15"), -5.0)
        self.assertEqual(utc_offset_hours("ATL", "2019-08-15"), -4.0)
        self.assertEqual(utc_offset_hours("PHX", "2019-08-15"), -7.0)

    def test_all_supported_airline_edge_airports_have_offsets(self):
        expected = {
            "ADK": -10.0,
            "ADQ": -9.0,
            "BLI": -8.0,
            "BQN": -4.0,
            "PSE": -4.0,
        }
        self.assertEqual({a: utc_offset_hours(a) for a in expected}, expected)


if __name__ == "__main__":
    unittest.main()
